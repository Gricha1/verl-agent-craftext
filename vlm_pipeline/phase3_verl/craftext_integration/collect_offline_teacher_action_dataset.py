"""Collect frozen CNN+BERT teacher trajectories for offline action RL.

The teacher is executed only here.  The resulting Parquet rows are static and
the Qwen trainer never instantiates a CrafText environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ACTION_NAMES = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)


def state_text(state, action_history: list[str]) -> str:
    """Give Qwen a bounded symbolic view while preserving the raw pixels on disk."""
    game = state.env_state
    grid = np.asarray(game.map).astype(int)
    position = np.asarray(game.player_position).astype(int).tolist()
    row, col = position
    radius = 5
    top, bottom = max(0, row - radius), min(grid.shape[0], row + radius + 1)
    left, right = max(0, col - radius), min(grid.shape[1], col + radius + 1)
    local_grid = grid[top:bottom, left:right]
    return (
        f"You are controlling a CrafText agent on a {grid.shape[0]} by {grid.shape[1]} grid. "
        f"Your row,column position is {position}. "
        f"The following are integer tile IDs for the local window rows [{top},{bottom}) "
        f"and columns [{left},{right}), one row per line:\n"
        + "\n".join(" ".join(map(str, grid_row)) for grid_row in local_grid)
        + "\nPrevious actions (oldest to newest): "
        + (", ".join(action_history) if action_history else "none")
    )


def prompt(instruction: str, observation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful embodied CrafText agent."},
        {
            "role": "user",
            "content": (
                f"Your goal is to complete this task:\n**TASK:** {instruction}\n\n"
                f"Current state:\n{observation}\n\n"
                "Give concise step-by-step reasoning about the visible state in a non-empty "
                "<reasoning>...</reasoning> block, then output exactly one final action in a separate "
                "<action>ACTION</action> block. Both blocks are mandatory. Valid actions are NOOP, LEFT, RIGHT, UP, DOWN, DO, "
                "SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, "
                "MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, "
                "MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD."
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--craftext-settings", default="debug_square_16x16")
    parser.add_argument("--seeds", default="20261016", help="Comma-separated reset seeds")
    parser.add_argument("--task-indices", default="0", help="Comma-separated instruction indices")
    parser.add_argument("--max-steps", default=50, type=int)
    parser.add_argument("--train-fraction", default=0.8, type=float)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1)")

    # The legacy environment and its exact CNN+BERT model are deliberately
    # imported from the saved teacher checkout, never reimplemented here.
    legacy = args.legacy_root.resolve()
    sys.path[:0] = [str(legacy / "caged_craftext"), str(legacy / "caged_craftext" / "Craftax"), str(legacy / "caged_craftext" / "baselines")]
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
    from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
    from craftext.environment.craftext_wrapper import InstructionWrapper
    from models.actor_critic_with_text import ActorCriticConvWithBERT
    from train_teacher_multitask import load_checkpoint

    if "debug_square_16x16" in args.craftext_settings:
        base = CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(16, 16)))
    elif "debug_square_8x8" in args.craftext_settings:
        base = CraftaxClassicPixelsEnv(StaticEnvParams(map_size=(8, 8)))
    else:
        base = CraftaxClassicPixelsEnv()
    env_params = base.default_params
    env = InstructionWrapper(base, args.craftext_settings)
    instructions = list(env.scenario_handler.scenario_data.instructions_list)
    task_indices = [int(value) for value in args.task_indices.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not task_indices or not seeds:
        raise ValueError("--task-indices and --seeds must both be non-empty")
    invalid_indices = [index for index in task_indices if not 0 <= index < len(instructions)]
    if invalid_indices:
        raise ValueError(f"task indices {invalid_indices} are outside [0, {len(instructions)})")
    action_history_length = 8
    pad_action = len(ACTION_NAMES)
    network = ActorCriticConvWithBERT(len(ACTION_NAMES), 512, action_history_length=action_history_length)
    key = jax.random.PRNGKey(seeds[0])
    key, init_key = jax.random.split(key)
    bootstrap_obs, bootstrap_state = env.reset(
        jax.random.PRNGKey(seeds[0]), env_params, instruction_idx=task_indices[0]
    )
    bootstrap_history = jnp.full((1, action_history_length), pad_action, dtype=jnp.int32)
    template_params = network.init(
        init_key, bootstrap_obs[None, ...], bootstrap_state.instruction[None, ...], bootstrap_history
    )
    template = TrainState.create(
        apply_fn=network.apply,
        params=template_params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4, eps=1e-5)),
    )
    restored = load_checkpoint(args.checkpoint, template)

    out = args.output_dir.resolve()
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trajectories = []
    episode_id = 0
    for task_index in task_indices:
        instruction = str(instructions[task_index])
        for seed in seeds:
            reset_key = jax.random.PRNGKey(seed)
            obs, state = env.reset(reset_key, env_params, instruction_idx=task_index)
            history = jnp.full((1, action_history_length), pad_action, dtype=jnp.int32)
            history_names: list[str] = []
            success = False
            for step in range(args.max_steps):
                current_obs = np.asarray(obs)
                current_text = state_text(state, history_names)
                pi, _ = network.apply(restored["params"], obs[None, ...], state.instruction[None, ...], history)
                logits = np.asarray(pi.logits[0], dtype=np.float32)
                probabilities = np.asarray(jax.nn.softmax(pi.logits[0]), dtype=np.float32)
                action = int(pi.mode()[0])
                key, step_key = jax.random.split(key)
                next_obs, next_state, reward, done, _ = env.step(step_key, state, jnp.int32(action), env_params)
                done_flag = bool(done)
                success = bool(next_state.instruction_done)
                prefix = f"episode_{episode_id:03d}_state_{step:03d}"
                image_path = images / f"{prefix}.npz"
                next_image_path = images / f"episode_{episode_id:03d}_next_state_{step:03d}.npz"
                np.savez_compressed(image_path, observation=current_obs)
                np.savez_compressed(next_image_path, observation=np.asarray(next_obs))
                rows.append(
                    {
                        "prompt": prompt(instruction, current_text),
                        "data_source": "craftext_offline_teacher_action",
                        "reward_model": {"ground_truth": {"teacher_action": ACTION_NAMES[action]}},
                        "extra_info": {
                            "episode_id": episode_id, "seed": seed, "task_index": task_index,
                            "step": step, "instruction": instruction,
                            "teacher_action": ACTION_NAMES[action], "teacher_logits": logits.tolist(),
                            "teacher_action_probabilities": probabilities.tolist(),
                            "observation_path": str(image_path), "next_observation_path": str(next_image_path),
                            "reward": float(reward), "done": done_flag, "success": success,
                        },
                    }
                )
                if done_flag:
                    break
                obs, state = next_obs, next_state
                history = jnp.concatenate((history[:, 1:], jnp.asarray([[action]], dtype=jnp.int32)), axis=1)
                history_names = (history_names + [ACTION_NAMES[action]])[-action_history_length:]
            trajectories.append({"episode_id": episode_id, "seed": seed, "task_index": task_index,
                                 "instruction": instruction, "steps": step + 1, "success": success})
            episode_id += 1

    if len(rows) < 2:
        raise RuntimeError("Teacher trajectory was too short to split into train and validation states")
    val_rows = [row for index, row in enumerate(rows) if index % 5 == 4]
    train_rows = [row for index, row in enumerate(rows) if index % 5 != 4]
    if not val_rows:
        val_rows = [rows[-1]]
        train_rows = rows[:-1]
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(train_rows), out / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), out / "val.parquet")
    manifest = {
        "craftext_settings": args.craftext_settings,
        "seeds": seeds, "task_indices": task_indices,
        "number_of_states": len(rows),
        "train_states": len(train_rows), "val_states": len(val_rows),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "trajectories": trajectories,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
