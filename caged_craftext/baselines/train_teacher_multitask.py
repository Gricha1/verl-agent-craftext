"""PPO teacher for CrafText tasks conditioned on pixels *and* DistilBERT text.

The environment owns task sampling.  ``InstructionWrapper`` performs the
frozen DistilBERT forward once for every configured instruction and carries
the selected embedding in each JAX environment state.  The Flax policy always
receives that per-environment embedding alongside pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

# Prefer the Craftax sibling shipped by this canonical checkout over a stale
# site-package/snapshot that may also be present in PYTHONPATH.
_CRAFTAX_ROOT = Path(__file__).resolve().parents[1] / "Craftax"
sys.path.insert(0, str(_CRAFTAX_ROOT))
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.scenarious.loader import load_scenarios
from models.actor_critic_with_text import ActorCriticConvWithBERT


@dataclass(frozen=True)
class Config:
    run_name: str
    out_dir: str
    craftext_settings: str
    seed: int = 7
    num_envs: int = 256
    num_steps: int = 100
    total_timesteps: int = 250_000_000
    update_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    layer_size: int = 512
    eval_every: int = 50
    checkpoint_every: int = 50
    eval_envs: int = 64
    max_episode_length: int = 1000
    action_history_length: int = 8


class Transition(NamedTuple):
    obs: jax.Array
    instruction: jax.Array
    action_history: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    success: jax.Array
    task_id: jax.Array
    ep_return: jax.Array
    ep_length: jax.Array


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()


def _logprob(pi, action):
    return pi.log_prob(action)


def make_fns(env, env_params, net, cfg: Config, task_count: int, pad_action_id: int):
    reset_v = jax.vmap(lambda key: env.reset(key, env_params))

    def reset_batch(keys):
        return reset_v(keys)

    def one_step(state, action, key, episode_return, episode_length):
        obs, next_state, reward, raw_done, _ = env.step(key, state, action, env_params)
        success = next_state.instruction_done.astype(jnp.float32)
        length = episode_length + 1
        done = jnp.logical_or(raw_done, jnp.logical_or(success.astype(bool), length >= cfg.max_episode_length))
        total_return = episode_return + reward
        reset_obs, reset_state = env.reset(jax.random.fold_in(key, 871), env_params)
        out_obs = jnp.where(done, reset_obs, obs)
        out_state = jax.tree.map(lambda x, y: jnp.where(done, y, x), next_state, reset_state)
        return out_obs, out_state, reward, done, success, total_return, length

    step_v = jax.vmap(one_step, in_axes=(0, 0, 0, 0, 0))

    def update(train_state, env_state, obs, action_history, ep_return, ep_length, rng):
        def rollout(carry, _):
            state, states, current_obs, history, returns, lengths, key = carry
            key, action_key, env_key = jax.random.split(key, 3)
            instruction = states.instruction
            pi, value = net.apply(state.params, current_obs, instruction, history)
            action = pi.sample(seed=action_key)
            log_prob = _logprob(pi, action)
            keys = jax.random.split(env_key, cfg.num_envs)
            nobs, nstates, reward, done, success, completed_return, completed_length = step_v(
                states, action, keys, returns, lengths
            )
            next_history = jnp.concatenate((history[:, 1:], action[:, None]), axis=1)
            reset_history = jnp.full_like(history, pad_action_id)
            next_history = jnp.where(done[:, None], reset_history, next_history)
            tr = Transition(current_obs, instruction, history, action, log_prob, value, reward, done,
                            success, states.idx, completed_return, completed_length)
            return (state, nstates, nobs, next_history, jnp.where(done, 0., completed_return),
                    jnp.where(done, 0, completed_length), key), tr

        (_, env_state, obs, action_history, ep_return, ep_length, rng), traj = jax.lax.scan(
            rollout, (train_state, env_state, obs, action_history, ep_return, ep_length, rng), None, cfg.num_steps
        )
        _, last_value = net.apply(train_state.params, obs, env_state.instruction, action_history)

        def gae_step(carry, tr):
            advantage, next_value = carry
            delta = tr.reward + cfg.gamma * next_value * (1. - tr.done) - tr.value
            advantage = delta + cfg.gamma * cfg.gae_lambda * (1. - tr.done) * advantage
            return (advantage, tr.value), advantage

        (_, _), advantages = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        targets = advantages + traj.value
        batch_size = cfg.num_envs * cfg.num_steps
        mini_size = batch_size // cfg.num_minibatches

        def loss_fn(params, batch):
            tr, adv, target = batch
            pi, value = net.apply(params, tr.obs, tr.instruction, tr.action_history)
            log_prob = _logprob(pi, tr.action)
            ratio = jnp.exp(log_prob - tr.log_prob)
            normalized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.minimum(ratio * normalized_adv,
                              jnp.clip(ratio, 1. - cfg.clip_eps, 1. + cfg.clip_eps) * normalized_adv).mean()
            value_clipped = tr.value + jnp.clip(value - tr.value, -cfg.clip_eps, cfg.clip_eps)
            vf = .5 * jnp.maximum((value - target) ** 2, (value_clipped - target) ** 2).mean()
            ent = pi.entropy().mean()
            clip_fraction = (jnp.abs(ratio - 1.) > cfg.clip_eps).mean()
            return pg + cfg.vf_coef * vf - cfg.ent_coef * ent, (pg, vf, ent, clip_fraction)

        flat = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj, advantages, targets))

        def epoch(carry, _):
            state, key = carry
            key, permutation_key = jax.random.split(key)
            shuffled = jax.tree.map(lambda x: x[jax.random.permutation(permutation_key, batch_size)], flat)
            minibatches = jax.tree.map(lambda x: x.reshape((cfg.num_minibatches, mini_size) + x.shape[1:]), shuffled)

            def minibatch(state, batch):
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
                grad_norm = optax.global_norm(grads)
                state = state.apply_gradients(grads=grads)
                return state, jnp.array([loss, aux[0], aux[1], aux[2], grad_norm])

            state, losses = jax.lax.scan(minibatch, state, minibatches)
            return (state, key), losses

        (train_state, rng), losses = jax.lax.scan(epoch, (train_state, rng), None, cfg.update_epochs)
        dones = traj.done.astype(jnp.float32)
        count = jnp.maximum(dones.sum(), 1.)
        task_counts = jnp.bincount(traj.task_id.reshape(-1), length=task_count)
        task_successes = jnp.bincount(traj.task_id.reshape(-1), weights=(traj.success * dones).reshape(-1), length=task_count)
        metrics = jnp.concatenate((jnp.array([
            traj.reward.mean(), (traj.success * dones).sum() / count,
            (traj.ep_return * dones).sum() / count, (traj.ep_length * dones).sum() / count,
        ]), task_successes, task_counts, losses.mean(axis=(0, 1))))
        return train_state, env_state, obs, action_history, ep_return, ep_length, rng, metrics

    def evaluate(params, keys):
        obs, states = reset_batch(keys)
        live = jnp.ones((cfg.eval_envs,), dtype=bool)
        returns = jnp.zeros((cfg.eval_envs,), dtype=jnp.float32)
        lengths = jnp.zeros((cfg.eval_envs,), dtype=jnp.int32)
        successes = jnp.zeros((cfg.eval_envs,), dtype=jnp.float32)
        task_id = states.idx
        action_history = jnp.full(
            (cfg.eval_envs, cfg.action_history_length), pad_action_id, dtype=jnp.int32
        )

        def step(carry, _):
            obs, states, history, live, returns, lengths, successes = carry
            pi, _ = net.apply(params, obs, states.instruction, history)
            actions = pi.mode()
            env_keys = jax.random.split(jax.random.PRNGKey(10), cfg.eval_envs)
            nobs, nstates, rewards, raw_done, _ = jax.vmap(lambda s, a, k: env.step(k, s, a, env_params))(states, actions, env_keys)
            success = nstates.instruction_done.astype(bool)
            done = jnp.logical_or(raw_done, jnp.logical_or(success, lengths + 1 >= cfg.max_episode_length))
            next_history = jnp.concatenate((history[:, 1:], actions[:, None]), axis=1)
            next_history = jnp.where(done[:, None], jnp.full_like(history, pad_action_id), next_history)
            return (nobs, nstates, next_history, live & ~done, returns + jnp.where(live, rewards, 0.),
                    lengths + live.astype(jnp.int32), jnp.maximum(successes, (live & success).astype(jnp.float32))), None

        (obs, states, action_history, live, returns, lengths, successes), _ = jax.lax.scan(
            step, (obs, states, action_history, live, returns, lengths, successes), None, cfg.max_episode_length
        )
        task_counts = jnp.bincount(task_id, length=task_count)
        task_successes = jnp.bincount(task_id, weights=successes, length=task_count)
        return jnp.concatenate((jnp.array([returns.mean(), successes.mean(), lengths.mean()]), task_successes, task_counts))

    return jax.jit(reset_batch), jax.jit(update), jax.jit(evaluate)


def save_checkpoint(path: Path, state: TrainState, cfg: Config, step: int, best: float, task_names: list[str]):
    payload = {"params": state.params, "opt_state": state.opt_state, "step": step,
               "best_validation_success": best, "config": asdict(cfg),
               "architecture": {"name": "ActorCriticConvWithBERT", "cnn": [32, 32, 32],
                                "layer_width": cfg.layer_size, "bert_model": "distilbert-base-uncased",
                                "bert_frozen": True, "instruction_embedding_dim": 768,
                                "past_actions_only": True,
                                "action_history_length": cfg.action_history_length},
               "tasks": task_names}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(payload))
    temporary.replace(path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({k: v for k, v in payload.items() if k not in {"params", "opt_state"}}, indent=2))


def load_checkpoint(path: Path, state_template: TrainState) -> dict[str, Any]:
    """Restore a teacher payload against the exact Flax state structure."""
    raw = serialization.msgpack_restore(path.read_bytes())
    template = {"params": state_template.params, "opt_state": state_template.opt_state,
                "step": 0, "best_validation_success": -1., "config": {},
                "architecture": {}, "tasks": ["" for _ in raw["tasks"]]}
    return serialization.from_state_dict(template, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True); parser.add_argument("--out-dir", required=True)
    parser.add_argument("--craftext-settings", required=True)
    parser.add_argument("--seed", type=int, default=7); parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=100); parser.add_argument("--total-timesteps", type=int, default=250_000_000)
    parser.add_argument("--eval-every", type=int, default=50); parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-episode-length", type=int, default=1000)
    args = parser.parse_args()
    cfg = Config(run_name=args.run_name, out_dir=args.out_dir, craftext_settings=args.craftext_settings,
                 seed=args.seed, num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=args.total_timesteps,
                 eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
                 max_episode_length=args.max_episode_length)
    if (cfg.num_envs * cfg.num_steps) % cfg.num_minibatches: raise ValueError("global batch must divide minibatches")
    out = Path(cfg.out_dir); ckpt = out / "checkpoints"; tb = out / "tensorboard"
    ckpt.mkdir(parents=True, exist_ok=True); tb.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    writer = SummaryWriter(str(tb))
    # The dataset config declares ``Craftax-Classic-Pixels-v1-Text``.  Keep
    # this identical base MDP instead of silently selecting non-Classic Craftax.
    base_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", True)
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, cfg.craftext_settings)
    scenario_map = load_scenarios(env.scenario_handler.config)
    task_names = list(scenario_map.keys())
    if not task_names: raise RuntimeError("dataset has no tasks")
    task_count = len(task_names)
    action_dim = env.action_space(env_params).n
    pad_action_id = action_dim
    network = ActorCriticConvWithBERT(
        action_dim, cfg.layer_size, action_history_length=cfg.action_history_length
    )
    reset, update, evaluate = make_fns(env, env_params, network, cfg, task_count, pad_action_id)
    rng = jax.random.PRNGKey(cfg.seed); rng, init_key, reset_key = jax.random.split(rng, 3)
    obs, states = reset(jax.random.split(reset_key, cfg.num_envs))
    action_history = jnp.full((cfg.num_envs, cfg.action_history_length), pad_action_id, dtype=jnp.int32)
    params = network.init(init_key, obs, states.instruction, action_history)
    state = TrainState.create(apply_fn=network.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5)))
    pi, value = network.apply(state.params, obs[:1], states.instruction[:1], action_history[:1])
    first = {"image_shape": list(obs[:1].shape), "instruction": env.scenario_handler.scenario_data.instructions_list[0],
             "bert_embedding_shape": list(states.instruction[:1].shape), "cnn_plus_text": "ActorCriticConvWithBERT",
             "past_actions_only": True, "action_history_length": cfg.action_history_length,
             "action_history_shape": list(action_history[:1].shape), "action_history_pad_id": pad_action_id,
             "logits_shape": list(pi.logits.shape), "value_shape": list(value.shape)}
    print(json.dumps({"event": "first_batch_shapes", **first}), flush=True)
    (out / "first_batch_shapes.json").write_text(json.dumps(first, indent=2))
    save_checkpoint(ckpt / "teacher_init.pt", state, cfg, 0, -1., task_names)
    restored_init = load_checkpoint(ckpt / "teacher_init.pt", state)
    reload_pi, reload_value = network.apply(restored_init["params"], obs[:1], states.instruction[:1], action_history[:1])
    assert reload_pi.logits.shape == pi.logits.shape and reload_value.shape == value.shape
    (out / "reload_test_init.json").write_text(json.dumps({"ok": True, "logits_shape": list(reload_pi.logits.shape),
                                                            "greedy_action": int(reload_pi.mode()[0]),
                                                            "sampled_action_supported": True}, indent=2))
    ep_return = jnp.zeros((cfg.num_envs,)); ep_length = jnp.zeros((cfg.num_envs,), dtype=jnp.int32)
    best = -1.; updates = cfg.total_timesteps // (cfg.num_envs * cfg.num_steps)
    validation_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 100_000), cfg.eval_envs)
    for step in range(1, updates + 1):
        start = time.time(); rng, key = jax.random.split(rng)
        state, states, obs, action_history, ep_return, ep_length, rng, metrics = update(
            state, states, obs, action_history, ep_return, ep_length, key
        )
        m = np.asarray(jax.device_get(metrics)); elapsed = max(time.time() - start, 1e-6)
        names = ["train/mean_reward", "train/success_rate", "train/episode_return", "train/episode_length"]
        values = {k: float(v) for k, v in zip(names, m[:4])}
        successes, counts = m[4:4+task_count], m[4+task_count:4+2*task_count]
        for name, success, count in zip(task_names, successes, counts):
            values[f"train/{safe_name(name)}_success_rate"] = float(success / max(count, 1.))
        for name, val in zip(["loss", "policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm"], m[4+2*task_count:]): values[name] = float(val)
        values["global_step"] = step; values["environment_steps"] = step * cfg.num_envs * cfg.num_steps
        values["steps_per_sec"] = cfg.num_envs * cfg.num_steps / elapsed
        for key, val in values.items(): writer.add_scalar(key, val, step)
        if step == 1 or step % cfg.eval_every == 0:
            vm = np.asarray(jax.device_get(evaluate(state.params, validation_keys)))
            validation = {"val/mean_return": float(vm[0]), "val/success_rate": float(vm[1]), "val/episode_length": float(vm[2])}
            for name, success, count in zip(task_names, vm[3:3+task_count], vm[3+task_count:]): validation[f"val/{safe_name(name)}_success_rate"] = float(success / max(count, 1.))
            validation["val/macro_success_rate"] = float(np.mean([v for k, v in validation.items() if k.endswith("_success_rate")]))
            for key, val in validation.items(): writer.add_scalar(key, val, step)
            values.update(validation)
            if validation["val/macro_success_rate"] >= best:
                best = validation["val/macro_success_rate"]; save_checkpoint(ckpt / "teacher_best.pt", state, cfg, step, best, task_names)
        if step == 1 or step % cfg.checkpoint_every == 0:
            save_checkpoint(ckpt / "teacher_latest.pt", state, cfg, step, best, task_names)
            if step == 1:
                restored_latest = load_checkpoint(ckpt / "teacher_latest.pt", state)
                test_pi, test_value = network.apply(
                    restored_latest["params"], obs[:1], states.instruction[:1], action_history[:1]
                )
                assert test_pi.logits.shape[-1] == action_dim and test_value.shape == (1,)
                (out / "reload_test_first_update.json").write_text(json.dumps({"ok": True, "step": step,
                    "logits_shape": list(test_pi.logits.shape), "value_shape": list(test_value.shape),
                    "greedy_action": int(test_pi.mode()[0])}, indent=2))
        writer.flush(); print(json.dumps({"event": "ppo_update", **values}, sort_keys=True), flush=True)
    writer.close()


if __name__ == "__main__": main()
