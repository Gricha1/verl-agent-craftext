#!/usr/bin/env python3
"""Render one deterministic trajectory of an action-history teacher checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from PIL import Image, ImageDraw

from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
from craftext.environment.craftext_wrapper import InstructionWrapper
from models.actor_critic_with_text import ActorCriticConvWithBERT
from train_teacher_multitask import Config, load_checkpoint


def frame(image, caption: str, scale: int = 5) -> np.ndarray:
    rgb = np.asarray(jax.device_get(image))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 18), fill=(0, 0, 0))
    draw.text((3, 3), caption, fill=(255, 255, 255))
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", default="debug_square_16x16")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    if "debug_square_16x16" in args.settings:
        base_env = CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(16, 16)))
    elif "debug_square_8x8" in args.settings:
        base_env = CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(8, 8)))
    else:
        base_env = CraftaxClassicPixelsEnv()
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, args.settings)
    key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(key, env_params, instruction_idx=args.task_index)
    obs_batch, instruction = obs[None, ...], state.instruction[None, ...]
    action_dim = env.action_space(env_params).n
    pad_id = action_dim
    history = jnp.full((1, 8), pad_id, dtype=jnp.int32)
    network = ActorCriticConvWithBERT(action_dim, 512, action_history_length=8)
    params = network.init(jax.random.PRNGKey(0), obs_batch, instruction, history)
    state_template = TrainState.create(
        apply_fn=network.apply, params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4, eps=1e-5)),
    )
    restored = load_checkpoint(Path(args.checkpoint), state_template)
    params = restored["params"]

    task = env.scenario_handler.scenario_data.instructions_list[args.task_index]
    frames = [frame(obs, f"{task} | start | history=PADx8")]
    total_reward = 0.0
    done = False
    for step in range(1, args.max_steps + 1):
        pi, _ = network.apply(params, obs_batch, instruction, history)
        action = int(pi.mode()[0])
        key, step_key = jax.random.split(key)
        next_obs, next_state, reward, raw_done, _ = env.step(step_key, state, jnp.int32(action), env_params)
        success = bool(next_state.instruction_done)
        done = bool(raw_done) or success
        total_reward += float(reward)
        history = jnp.concatenate((history[:, 1:], jnp.array([[action]], dtype=jnp.int32)), axis=1)
        frames.append(frame(next_obs, f"{task} | step={step} a={action} r={float(reward):+.2f} R={total_reward:+.2f} success={success}"))
        obs, state = next_obs, next_state
        obs_batch, instruction = obs[None, ...], state.instruction[None, ...]
        if done:
            break
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, duration=0.35, loop=0)
    print({"output": args.output, "task": task, "steps": len(frames) - 1,
           "success": success if frames else False, "return": total_reward, "done": done})


if __name__ == "__main__":
    main()
