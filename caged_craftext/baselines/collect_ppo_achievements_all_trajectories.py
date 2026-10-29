"""Collect static all-achievements CrafText trajectories from a PPO checkpoint.

The output deliberately follows the established ``wood_teacher_slm_collection``
contract: one compressed current frame per transition plus Parquet metadata.
It is a *collector*, not a PPO trainer: policy parameters are restored read-only
from the legacy Orbax checkpoint and every action is the deterministic PPO mode.
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


def prompt(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful embodied CrafText agent."},
        {
            "role": "user",
            "content": (
                f"Your goal is to complete this task:\n**TASK:** {instruction}\n\n"
                "Choose exactly one valid CrafText action from the visual observation."
            ),
        },
    ]


def parse_indices(value: str, total: int) -> list[int]:
    if value == "all":
        return list(range(total))
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 or item >= total for item in result):
        raise ValueError(f"Task indices must be within [0, {total})")
    return result


def restore_params(checkpoint_dir: Path, step: int):
    """Use the exact legacy restore path, avoiding a recreated optimizer template."""
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

    manager = CheckpointManager(
        str(checkpoint_dir), PyTreeCheckpointer(), CheckpointManagerOptions(max_to_keep=1, create=False)
    )
    restored = manager.restore(step)
    try:
        return restored["runner_state"][0]["params"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Checkpoint does not contain legacy PPO runner_state parameters") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="20261300:20261312", help="start:end (exclusive) or comma-separated")
    parser.add_argument("--task-indices", default="all", help="all or comma-separated indices")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--layer-size", type=int, default=512)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    if ":" in args.seeds:
        start, end = (int(item) for item in args.seeds.split(":", 1))
        seeds = list(range(start, end))
    else:
        seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    legacy = args.legacy_root.resolve()
    sys.path[:0] = [str(legacy), str(legacy / "Craftax"), str(legacy / "baselines")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftext.environment.craftext_wrapper import InstructionWrapper
    from models.actor_critic_with_text import ActorCriticConvWithBERT

    # Match the PPO environment construction, except without auto-reset/logging
    # wrappers: terminal transitions must remain explicit in the dataset.
    base_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, "achievements_all")
    instructions = list(env.scenario_handler.scenario_data.instructions_list)
    task_indices = parse_indices(args.task_indices, len(instructions))
    params = restore_params(args.checkpoint_dir.resolve(), args.checkpoint_step)
    network = ActorCriticConvWithBERT(base_env.action_space(env_params).n, args.layer_size)

    output = args.output_dir.resolve()
    images = output / "images"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    images.mkdir(parents=True)

    rows: list[dict] = []
    trajectories: list[dict] = []
    episode_id = 0
    for task_index in task_indices:
        instruction = str(instructions[task_index])
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            obs, state = env.reset(key, env_params, instruction_idx=task_index)
            success = False
            done_flag = False
            for step in range(args.max_steps):
                pi, _ = network.apply(params, obs[None, ...], state.instruction[None, ...])
                logits = np.asarray(pi.logits[0], dtype=np.float32)
                probabilities = np.asarray(jax.nn.softmax(pi.logits[0]), dtype=np.float32)
                action = int(pi.mode()[0])
                key, step_key = jax.random.split(key)
                next_obs, next_state, reward, done, _ = env.step(step_key, state, jnp.int32(action), env_params)
                done_flag = bool(done)
                success = bool(next_state.instruction_done)
                image_name = f"episode_{episode_id:04d}_state_{step:04d}.npz"
                image_path = images / image_name
                np.savez_compressed(image_path, observation=np.asarray(obs))
                rows.append(
                    {
                        "prompt": prompt(instruction),
                        "data_source": "craftext_ppo_achievements_all_r4",
                        "reward_model": {"ground_truth": {"teacher_action": ACTION_NAMES[action]}},
                        "extra_info": {
                            "episode_id": episode_id,
                            "seed": seed,
                            "task_index": task_index,
                            "step": step,
                            "instruction": instruction,
                            "teacher_action": ACTION_NAMES[action],
                            "teacher_logits": logits.tolist(),
                            "teacher_action_probabilities": probabilities.tolist(),
                            "observation_path": str(image_path),
                            "reward": float(reward),
                            "done": done_flag,
                            "success": success,
                        },
                    }
                )
                obs, state = next_obs, next_state
                if done_flag:
                    break
            trajectories.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "task_index": task_index,
                    "instruction": instruction,
                    "steps": step + 1,
                    "success": success,
                    "environment_done": done_flag,
                }
            )
            print(json.dumps(trajectories[-1]), flush=True)
            episode_id += 1

    if len(rows) < 2:
        raise RuntimeError("Collector produced fewer than two transitions")
    train_rows = [row for index, row in enumerate(rows) if index % 5]
    val_rows = [row for index, row in enumerate(rows) if not index % 5]
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(train_rows), output / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val_rows), output / "val.parquet")
    manifest = {
        "schema_version": 1,
        "collector": "collect_ppo_achievements_all_trajectories.py",
        "collection_policy": "deterministic PPO mode",
        "craftext_settings": "achievements_all",
        "seeds": seeds,
        "task_indices": task_indices,
        "number_of_states": len(rows),
        "train_states": len(train_rows),
        "val_states": len(val_rows),
        "max_steps": args.max_steps,
        "teacher_checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "teacher_checkpoint_step": args.checkpoint_step,
        "trajectories": trajectories,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("number_of_states", "train_states", "val_states")}), flush=True)


if __name__ == "__main__":
    main()
