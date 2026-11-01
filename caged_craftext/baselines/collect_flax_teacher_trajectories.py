"""Collect static CrafText trajectories from a Flax-serialised PPO teacher.

Same output contract as ``collect_ppo_achievements_all_trajectories.py`` (one
compressed current frame per transition, Parquet rows with teacher logits and
probabilities, deterministic PPO mode), but for ``train_teacher_multitask``
style ``teacher_*.pt`` checkpoints (Flax bytes) and any CrafText settings.

Pipeline rule: a dataset for a task set may only be collected by the teacher
trained on that same task set.  The collector refuses to run when the
checkpoint's own ``config.craftext_settings`` differs from ``--craftext-settings``.
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


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, end = (int(item) for item in value.split(":", 1))
        return list(range(start, end))
    return [int(item) for item in value.split(",") if item.strip()]


def parse_indices(value: str, total: int) -> list[int]:
    if value == "all":
        return list(range(total))
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 or item >= total for item in result):
        raise ValueError(f"Task indices must be within [0, {total})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Flax-serialised teacher_*.pt")
    parser.add_argument("--craftext-settings", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", required=True, help="start:end (exclusive) or comma-separated")
    parser.add_argument("--task-indices", default="all", help="all or comma-separated indices")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--data-source", default="craftext_ppo_flax_teacher")
    parser.add_argument("--jit", action="store_true",
                        help="jax.jit the teacher and env.step (unjitted: ~0.4 steps/s, dispatch-bound)")
    parser.add_argument("--episode-id-offset", type=int, default=0,
                        help="first episode_id: shards with disjoint ids merge by concatenation")
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")

    legacy = args.legacy_root.resolve()
    sys.path[:0] = [str(legacy), str(legacy / "Craftax"), str(legacy / "baselines")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from craftax.craftax_env import make_craftax_env_from_name
    from craftext.environment.craftext_wrapper import InstructionWrapper
    from models.actor_critic_with_text import ActorCriticConvWithBERT

    payload = serialization.msgpack_restore(args.checkpoint.resolve().read_bytes())
    teacher_config = dict(payload.get("config", {}))
    teacher_settings = teacher_config.get("craftext_settings")
    if teacher_settings != args.craftext_settings:
        raise RuntimeError(
            f"Teacher was trained on {teacher_settings!r} but collection requested {args.craftext_settings!r}; "
            "a dataset must be collected by the teacher of the same task set"
        )
    teacher_tasks = [payload["tasks"][key] for key in sorted(payload.get("tasks", {}), key=int)]
    history_length = int(teacher_config.get("action_history_length", 0) or 0)
    if history_length:
        raise NotImplementedError("Teachers with an action-history input are not supported by this collector")
    params = payload["params"]

    # Match the teacher's environment (default Craftax-Classic pixels world plus
    # the instruction wrapper), without auto-reset so terminals stay explicit.
    base_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, args.craftext_settings)
    instructions = list(env.scenario_handler.scenario_data.instructions_list)
    task_indices = parse_indices(args.task_indices, len(instructions))
    network = ActorCriticConvWithBERT(base_env.action_space(env_params).n, int(teacher_config.get("layer_size", 512)))

    output = args.output_dir.resolve()
    images = output / "images"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    images.mkdir(parents=True)

    policy, env_step = network.apply, env.step
    if args.jit:
        policy, env_step = jax.jit(network.apply), jax.jit(env.step)

    rows: list[dict] = []
    trajectories: list[dict] = []
    episode_id = args.episode_id_offset
    for task_index in task_indices:
        instruction = str(instructions[task_index])
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            obs, state = env.reset(key, env_params, instruction_idx=task_index)
            success = False
            done_flag = False
            for step in range(args.max_steps):
                pi, _ = policy(params, obs[None, ...], state.instruction[None, ...])
                logits = np.asarray(pi.logits[0], dtype=np.float32)
                probabilities = np.asarray(jax.nn.softmax(pi.logits[0]), dtype=np.float32)
                action = int(pi.mode()[0])
                key, step_key = jax.random.split(key)
                next_obs, next_state, reward, done, _ = env_step(step_key, state, jnp.int32(action), env_params)
                done_flag = bool(done)
                success = bool(next_state.instruction_done)
                image_path = images / f"episode_{episode_id:04d}_state_{step:04d}.npz"
                np.savez_compressed(image_path, observation=np.asarray(obs))
                rows.append(
                    {
                        "prompt": prompt(instruction),
                        "data_source": args.data_source,
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
        "collector": "collect_flax_teacher_trajectories.py",
        "collection_policy": "deterministic PPO mode",
        "craftext_settings": args.craftext_settings,
        "seeds": seeds,
        "task_indices": task_indices,
        "number_of_states": len(rows),
        "train_states": len(train_rows),
        "val_states": len(val_rows),
        "max_steps": args.max_steps,
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_training_config": teacher_config,
        "teacher_tasks": teacher_tasks,
        "teacher_best_validation_success": payload.get("best_validation_success"),
        "teacher_step": payload.get("step"),
        "trajectories": trajectories,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("number_of_states", "train_states", "val_states")}), flush=True)


if __name__ == "__main__":
    main()
