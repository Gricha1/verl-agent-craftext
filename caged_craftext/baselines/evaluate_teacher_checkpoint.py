#!/usr/bin/env python3
"""Reproduce the trainer's fixed-seed greedy validation for one checkpoint."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.scenarious.loader import load_scenarios
from models.actor_critic_with_text import ActorCriticConvWithBERT
from train_teacher_multitask import Config, load_checkpoint, make_fns


def make_base_env(settings: str):
    if "debug_square_16x16" in settings:
        return CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(16, 16)))
    if "debug_square_8x8" in settings:
        return CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(8, 8)))
    return CraftaxClassicPixelsEnv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--eval-envs", type=int, default=None)
    args = parser.parse_args()

    raw = serialization.msgpack_restore(Path(args.checkpoint).read_bytes())
    saved = dict(raw["config"])
    history_length = int(raw.get("architecture", {}).get("action_history_length", 0))
    if args.eval_envs is not None:
        saved["eval_envs"] = args.eval_envs
    saved["craftext_settings"] = args.settings
    saved["action_history_length"] = history_length
    cfg = Config(**saved)

    base_env = make_base_env(cfg.craftext_settings)
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, cfg.craftext_settings)
    task_names = list(load_scenarios(env.scenario_handler.config).keys())
    task_count = len(task_names)
    action_dim = env.action_space(env_params).n
    network = ActorCriticConvWithBERT(action_dim, cfg.layer_size, action_history_length=history_length)
    reset, _update, evaluate = make_fns(env, env_params, network, cfg, task_count, action_dim)
    init_keys = jax.random.split(jax.random.PRNGKey(0), cfg.eval_envs)
    obs, states = reset(init_keys)
    history = jnp.full((cfg.eval_envs, history_length), action_dim, dtype=jnp.int32)
    params = network.init(jax.random.PRNGKey(1), obs, states.instruction, history)
    template = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)),
    )
    restored = load_checkpoint(Path(args.checkpoint), template)
    validation_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 100_000), cfg.eval_envs)
    values = np.asarray(jax.device_get(evaluate(restored["params"], validation_keys)))
    successes, counts = values[3 : 3 + task_count], values[3 + task_count : 3 + 2 * task_count]
    per_task = {name: float(success / max(count, 1.0)) for name, success, count in zip(task_names, successes, counts)}
    print(json.dumps({
        "checkpoint": str(args.checkpoint), "checkpoint_step": int(raw["step"]),
        "settings": cfg.craftext_settings, "eval_envs": cfg.eval_envs,
        "history_length": history_length,
        "fixed_validation_seed": cfg.seed + 100_000,
        "mean_return": float(values[0]), "success_rate": float(values[1]),
        "episode_length": float(values[2]), "per_task_success_rate": per_task,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
