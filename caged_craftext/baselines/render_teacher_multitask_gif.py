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
from craftax.craftax_classic.constants import BlockType
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


def global_debug_square_frame(env_state, caption: str, cell_size: int = 22) -> np.ndarray:
    """Render the complete tiny map, rather than the policy's 63x63 local view.

    Qwen receives this whole 16x16 state through the CrafText renderer.  The
    CNN receives a local pixel view, so using that view for a diagnostic GIF
    hides the border and made a correct debug arena look like procedural
    Craftax.  This rendering is diagnostic-only; it never enters the policy.
    """
    game_map = np.asarray(jax.device_get(env_state.map), dtype=np.int32)
    height, width = game_map.shape
    colours = {
        BlockType.GRASS.value: (51, 145, 53),
        BlockType.TREE.value: (22, 82, 25),
        BlockType.STONE.value: (132, 137, 142),
        BlockType.WATER.value: (46, 113, 197),
        BlockType.WOOD.value: (137, 88, 43),
    }
    image = Image.new("RGB", (width * cell_size, height * cell_size + 20), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for row in range(height):
        for col in range(width):
            left, top = col * cell_size, row * cell_size + 20
            colour = colours.get(int(game_map[row, col]), (190, 55, 190))
            draw.rectangle((left, top, left + cell_size - 1, top + cell_size - 1), fill=colour)
            draw.rectangle((left, top, left + cell_size - 1, top + cell_size - 1), outline=(25, 25, 25))
    row, col = map(int, np.asarray(jax.device_get(env_state.player_position)))
    left, top = col * cell_size, row * cell_size + 20
    draw.ellipse((left + 4, top + 4, left + cell_size - 5, top + cell_size - 5), fill=(250, 235, 55), outline=(0, 0, 0))
    draw.text((3, 3), caption, fill=(255, 255, 255))
    return np.asarray(image)


def diagnostic_frame(obs, state, caption: str) -> np.ndarray:
    game_map = getattr(state.env_state, "map", None)
    if game_map is not None and tuple(game_map.shape) in ((8, 8), (16, 16)):
        return global_debug_square_frame(state.env_state, caption)
    return frame(obs, caption)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", default="debug_square_16x16")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--render-mode", choices=("global_state", "craftext_pixels"), default="global_state",
        help="global_state is a diagnostic full-map view; craftext_pixels is the exact observation returned by InstructionWrapper.",
    )
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
    if args.render_mode == "craftext_pixels":
        render_frame = lambda image, _state, caption: frame(image, caption)
        view_label = "CrafText pixel observation"
    else:
        render_frame = diagnostic_frame
        view_label = "global 16x16"
    frames = [render_frame(obs, state, f"{task} | {view_label} | start | history=PADx8")]
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
        frames.append(render_frame(
            next_obs, next_state,
            f"{task} | {view_label} | step={step} a={action} r={float(reward):+.2f} R={total_reward:+.2f} success={success}",
        ))
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
