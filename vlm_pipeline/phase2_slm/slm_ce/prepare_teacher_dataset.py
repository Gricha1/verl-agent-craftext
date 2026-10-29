"""Convert frozen CrafText CNN+BERT-teacher parquet rows to the SLM CE schema.

The source split is a held-out-*state* split: states from one teacher episode may
occur in both files.  It is suitable for next-token CE diagnostics only, not a
closed-loop generalization claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import pyarrow.parquet as pq


ACTION_NAMES = frozenset(
    {
        "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
        "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
        "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
        "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
    }
)


# Index order of ``teacher_action_probabilities`` written by the CrafText teacher
# collectors (the Craftax-Classic action enum).
TEACHER_ACTION_ORDER = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)


def serialize_messages(messages: list[dict[str, str]]) -> str:
    """Serialize the frozen source chat without adding observations or labels."""
    rendered: list[str] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str) or not content:
            raise ValueError(f"Unexpected teacher prompt message: {message!r}")
        rendered.append(f"{role.upper()}:\n{content.strip()}")
    if not rendered:
        raise ValueError("Teacher prompt is empty")
    return "\n\n".join(rendered) + "\n\nResponse:\n"


def serialize_vision_image_only(instruction: str, action_history: list[str] | None = None) -> str:
    """Prompt protocol for a visual policy: no symbolic state leaks into text."""
    if not instruction:
        raise ValueError("Teacher row has no task instruction")
    history = ""
    if action_history is not None:
        history = "Previous executed actions (oldest to newest; up to 8): " + (", ".join(action_history) or "none") + "\n\n"
    return (
        "SYSTEM:\n"
        "You are a careful embodied CrafText agent. Infer the current navigational "
        "state solely from the accompanying visual observation.\n\n"
        "USER:\n"
        f"Your goal is to complete this task:\n**TASK:** {instruction}\n\n"
        + history
        + "Output exactly one action as ACTION: <ACTION_NAME>.\n\nResponse:\n"
    )


def convert_rows(rows: list[dict], source_image_dir: Path, destination: Path,
                 image_destination: Path, prompt_protocol: str,
                 histories: dict[tuple[int, int], list[str]], source_split: str) -> list[dict]:
    converted = []
    for row in rows:
        extra = row["extra_info"]
        action = str(extra["teacher_action"])
        if action not in ACTION_NAMES:
            raise ValueError(f"Unknown teacher action {action!r}")
        # The Parquet records contain an absolute path from the data-collection
        # container (usually /mnt/run/...).  Make the vision dataset portable by
        # copying the exact current observation beside the converted JSONL.
        image_name = Path(str(extra["observation_path"])).name
        source_image = source_image_dir / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Teacher observation is missing: {source_image}")
        image_destination.mkdir(parents=True, exist_ok=True)
        target_image = image_destination / image_name
        if not target_image.exists():
            shutil.copy2(source_image, target_image)
        if prompt_protocol == "source_chat_serialized_v1":
            prompt = serialize_messages(list(row["prompt"]))
        elif prompt_protocol == "vision_image_only_v1":
            prompt = serialize_vision_image_only(str(extra["instruction"]))
        elif prompt_protocol == "vision_image_actions8_v1":
            prompt = serialize_vision_image_only(str(extra["instruction"]),
                                                 histories[(int(extra["episode_id"]), int(extra["step"]))])
        else:
            raise ValueError(f"Unknown prompt protocol: {prompt_protocol}")
        # The full teacher policy, for distillation targets; its argmax is the
        # executed (deterministic-mode) teacher action.
        teacher_probs = [float(p) for p in extra["teacher_action_probabilities"]]
        if len(teacher_probs) != len(TEACHER_ACTION_ORDER):
            raise ValueError(f"Expected {len(TEACHER_ACTION_ORDER)} teacher probabilities, got {len(teacher_probs)}")
        if TEACHER_ACTION_ORDER[max(range(len(teacher_probs)), key=teacher_probs.__getitem__)] != action:
            raise ValueError(f"Teacher probabilities disagree with teacher_action {action!r} "
                             f"(episode {extra['episode_id']}, step {extra['step']})")
        converted.append(
            {
                "episode_seed": int(extra["seed"]),
                "step": int(extra["step"]),
                "prompt": prompt,
                "action_target": f"ACTION: {action}",
                # The action-only run must never pretend the fixed Qwen wrapper
                # is a CNN+BERT teacher rationale.
                "cot_target": f"ACTION: {action}",
                "action": action,
                "teacher_probs": teacher_probs,
                "source_episode_id": int(extra["episode_id"]),
                "source_split": source_split,
                "image_path": str(Path("images") / image_name),
            }
        )
    if not converted:
        raise ValueError(f"No rows found for {source_split}")
    destination.write_text("".join(json.dumps(row) + "\n" for row in converted), encoding="utf-8")
    return converted


def split_by_episode(rows: list[dict], successful_only: bool, val_fraction: float,
                     seed: int) -> tuple[list[dict], list[dict], dict]:
    """Return a task-stratified, episode-disjoint train/validation split."""
    episodes: dict[int, list[dict]] = {}
    for row in rows:
        episodes.setdefault(int(row["extra_info"]["episode_id"]), []).append(row)
    selected = []
    for episode_id, episode_rows in episodes.items():
        if successful_only and not any(bool(row["extra_info"]["success"]) for row in episode_rows):
            continue
        selected.append((episode_id, sorted(episode_rows, key=lambda row: int(row["extra_info"]["step"]))))
    by_task: dict[int, list[tuple[int, list[dict]]]] = {}
    for item in selected:
        by_task.setdefault(int(item[1][0]["extra_info"]["task_index"]), []).append(item)
    train, valid = [], []
    counts = {}
    for task_index, task_episodes in sorted(by_task.items()):
        task_episodes = sorted(task_episodes, key=lambda item: item[0])
        random.Random(seed + task_index).shuffle(task_episodes)
        if len(task_episodes) < 2:
            raise ValueError(f"Task {task_index} has fewer than two eligible episodes")
        val_count = max(1, round(len(task_episodes) * val_fraction))
        val_episodes = task_episodes[:val_count]
        train_episodes = task_episodes[val_count:]
        valid.extend(row for _, episode_rows in val_episodes for row in episode_rows)
        train.extend(row for _, episode_rows in train_episodes for row in episode_rows)
        counts[str(task_index)] = {
            "eligible_episodes": len(task_episodes),
            "train_episodes": len(train_episodes),
            "valid_episodes": len(val_episodes),
        }
    return train, valid, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-protocol", choices=["source_chat_serialized_v1", "vision_image_only_v1", "vision_image_actions8_v1"],
                        default="source_chat_serialized_v1")
    parser.add_argument("--split-mode", choices=["source", "episode"], default="source")
    parser.add_argument("--successful-only", action="store_true",
                        help="Keep only episodes whose final rollout state is successful (episode split only).")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_destination = args.output_dir / "images"
    source_train = pq.read_table(args.input_dir / "train.parquet").to_pylist()
    source_valid = pq.read_table(args.input_dir / "val.parquet").to_pylist()
    all_rows = source_train + source_valid
    episodes: dict[int, list[dict]] = {}
    for row in all_rows:
        episodes.setdefault(int(row["extra_info"]["episode_id"]), []).append(row)
    histories: dict[tuple[int, int], list[str]] = {}
    for episode_id, rows in episodes.items():
        prior: list[str] = []
        for row in sorted(rows, key=lambda item: int(item["extra_info"]["step"])):
            step = int(row["extra_info"]["step"])
            histories[(episode_id, step)] = prior[-8:].copy()
            prior.append(str(row["extra_info"]["teacher_action"]))
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.split_mode == "episode":
        train_rows, valid_rows, split_counts = split_by_episode(
            all_rows, args.successful_only, args.val_fraction, args.split_seed
        )
        train_label, valid_label = "episode_train", "episode_valid"
    else:
        train_rows, valid_rows = source_train, source_valid
        split_counts = None
        train_label, valid_label = "source_train", "source_valid"
    train = convert_rows(train_rows, args.input_dir / "images", args.output_dir / "train.jsonl", image_destination,
                         args.prompt_protocol, histories, train_label)
    valid = convert_rows(valid_rows, args.input_dir / "images", args.output_dir / "valid.jsonl", image_destination,
                         args.prompt_protocol, histories, valid_label)
    source_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "source_format": "modern_verl_craftext_offline_teacher_parquet",
        "source_dataset": str(args.input_dir.resolve()),
        "source_manifest": source_manifest,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "target": "action-only CE: ACTION: <CNN+BERT teacher action>",
        "prompt_protocol": args.prompt_protocol,
        "split_mode": args.split_mode,
        "successful_episodes_only": args.successful_only,
        "split_seed": args.split_seed if args.split_mode == "episode" else None,
        "split_by_task": split_counts,
        "split_caveat": (
            "episode-disjoint, task-stratified validation split"
            if args.split_mode == "episode" else
            "held-out states; teacher episodes are not disjoint across train/valid"
        ),
        "teacher_reasoning_in_input": False,
        "teacher_reasoning_target": False,
        "vision_observations": "copied current 63x63x3 float32 observations; image_path is relative to this dataset",
        "state_in_text": args.prompt_protocol == "source_chat_serialized_v1",
        "action_history_in_text": 8 if args.prompt_protocol == "vision_image_actions8_v1" else 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
