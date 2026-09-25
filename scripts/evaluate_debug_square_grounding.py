#!/usr/bin/env python3
"""Frozen-checkpoint grounding evaluation for debug_square.

The evaluator never creates an optimizer, runs backward, or alters a checkpoint.
It scores each allowed answer by teacher-forced conditional log probability.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

QUADRANTS = ("RIGHT_UP", "RIGHT_DOWN", "LEFT_UP", "LEFT_DOWN")
CONSEQUENCES = ("CLOSER", "FARTHER", "SAME")
OBJECTS = ("STONE", "WOOD", "WATER")
ACTION_DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


@dataclass(frozen=True)
class Snapshot:
    state_id: str
    seed: int
    player: tuple[int, int]
    blocks: dict[str, tuple[int, int]]
    observation: str
    task: str


def quadrant(origin: tuple[int, int], point: tuple[int, int]) -> str | None:
    dr, dc = point[0] - origin[0], point[1] - origin[1]
    if dr == 0 or dc == 0:
        return None
    return ("RIGHT" if dc > 0 else "LEFT") + ("_DOWN" if dr > 0 else "_UP")


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def consequence(player: tuple[int, int], target: tuple[int, int], action: str, size: int) -> str:
    dr, dc = ACTION_DELTAS[action]
    nxt = (player[0] + dr, player[1] + dc)
    if not (1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1):
        nxt = player
    before, after = manhattan(player, target), manhattan(nxt, target)
    return "CLOSER" if after < before else "FARTHER" if after > before else "SAME"


def make_snapshots(root: Path, size: int, seeds: Iterable[int]) -> list[Snapshot]:
    for path in (root, root / "Craftax"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import jax
    import numpy as np
    from craftax.craftax_classic.constants import BlockType
    from craftax.craftax_classic.debug_square_world_gen import generate_debug_square_world
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnvNoAutoReset
    from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
    from agent_system.environments.env_package.caged_craftext.envs import _render_craftax_text

    env = CraftaxClassicPixelsEnvNoAutoReset(static_env_params=StaticEnvParams(map_size=(size, size)))
    ids = {name: int(getattr(BlockType, name).value) for name in OBJECTS}
    snapshots = []
    for seed in seeds:
        state = generate_debug_square_world(jax.random.PRNGKey(seed), env.default_params, env.static_env_params)
        game_map = np.asarray(jax.device_get(state.map))
        blocks = {}
        for name, block_id in ids.items():
            matches = np.argwhere(game_map == block_id)
            if len(matches) != 1:
                raise RuntimeError(f"seed={seed}: expected one {name}, got {len(matches)}")
            blocks[name] = tuple(map(int, matches[0]))
        player = tuple(map(int, np.asarray(jax.device_get(state.player_position))))
        goal = OBJECTS[seed % len(OBJECTS)]
        snapshots.append(Snapshot(str(seed), seed, player, blocks, _render_craftax_text(state, "ascii"), f"Navigate to the {goal} block."))
    return snapshots


def actor_context(s: Snapshot) -> str:
    return f"""Your goal is to complete the following task:
**TASK:** {s.task}

This is what you currently see:
{s.observation}
"""


def questions(s: Snapshot, size: int):
    target = s.blocks[OBJECTS[s.seed % len(OBJECTS)]]
    label = quadrant(s.player, target)
    if label is not None:
        yield "target_direction", "Where is the target relative to your current position?", QUADRANTS, label
    distance = manhattan(s.player, target)
    if distance <= 30:
        yield "target_distance", "How many steps are required to reach the target from your current position?", tuple(map(str, range(31))), str(distance)
    for action in ACTION_DELTAS:
        yield f"action_consequence_{action.lower()}", f"Will action {action} move you closer to the target, farther from the target, or keep you at the same distance?", CONSEQUENCES, consequence(s.player, target, action, size)
    for obj in OBJECTS:
        label = quadrant(s.player, s.blocks[obj])
        if label is not None:
            yield f"object_direction_{obj.lower()}", f"Where is the {obj} block relative to your current position?", QUADRANTS, label


def load_model(checkpoint: str, base_model: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    path = Path(checkpoint)
    adapter = (path / "adapter_config.json").exists()
    model_id = base_model
    if adapter:
        if not model_id:
            model_id = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto"), checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_id or checkpoint, use_fast=True)


def choose(model, tokenizer, prompt: str, choices: tuple[str, ...]) -> str:
    import torch
    device = next(model.parameters()).device
    prefix = tokenizer(prompt + "\nAnswer with exactly one allowed answer:\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    scores = []
    with torch.inference_mode():
        for choice in choices:
            ids = tokenizer(" " + choice, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            full = torch.cat((prefix, ids), dim=1)
            logits = model(full).logits[:, prefix.size(1) - 1 : -1]
            scores.append(float(torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).sum()))
    return choices[int(np.argmax(scores))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--caged-craftext-path", required=True, type=Path)
    p.add_argument("--map-size", choices=(8, 16), type=int, default=8)
    p.add_argument("--num-seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--base-model")
    p.add_argument("--output-dir", type=Path, default=Path("grounding_results"))
    args = p.parse_args()
    snapshots = make_snapshots(args.caged_craftext_path.resolve(), args.map_size, range(args.seed_offset, args.seed_offset + args.num_seeds))
    model, tokenizer = load_model(args.checkpoint, args.base_model)
    rows = []
    for state in snapshots:
        for family, question, choices, truth in questions(state, args.map_size):
            answer = choose(model, tokenizer, actor_context(state) + "\n" + question, choices)
            rows.append({"state_id": state.state_id, "seed": state.seed, "question_family": family, "question": question, "ground_truth": truth, "model_answer": answer, "correct": answer == truth})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    def acc(xs): return sum(r["correct"] for r in xs) / len(xs) if xs else None
    aggregate = {"target_direction_accuracy": acc([r for r in rows if r["question_family"] == "target_direction"]), "target_distance_accuracy": acc([r for r in rows if r["question_family"] == "target_distance"]), "action_consequence_accuracy": acc([r for r in rows if r["question_family"].startswith("action_consequence_")]), "object_direction_accuracy": acc([r for r in rows if r["question_family"].startswith("object_direction_")])}
    aggregate["grounding_accuracy_macro"] = float(np.mean([v for v in aggregate.values() if v is not None]))
    report = {"checkpoint": args.checkpoint, "map_size": args.map_size, "num_states": len(snapshots), "num_questions": len(rows), "aggregate": aggregate}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
