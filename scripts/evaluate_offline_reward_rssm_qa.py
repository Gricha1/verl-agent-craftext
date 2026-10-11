#!/usr/bin/env python3
"""Read-only paired QA evaluation for an offline reward-RSSM on debug_square 8x8.

This is deliberately separate from training and from the live PPO evaluator.  It
loads an immutable RSSM checkpoint, rolls out model-independent, deterministic
actions to collect states, and only then asks frozen Qwen questions about those
same states.  It never creates an optimizer, calls backward, or writes a
checkpoint.

The default is the mandatory small sanity pass (one seed, at most eight states).
Use ``--full`` only after inspecting its artifacts; it evaluates eight seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAGED_CRAFTEXT_PATH", str(REPO_ROOT / "caged_craftext"))

from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT, craftext_projection
from rssm_env_validation import _chat_parts, _generate, _normalise_actor_soft
from train_qwen_transition_rssm_recon import QwenTransitionRSSMRecon, posterior, transition


MOVE_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
QUADRANTS = ("RIGHT_UP", "RIGHT_DOWN", "LEFT_UP", "LEFT_DOWN")
CONSEQUENCES = ("CLOSER", "FARTHER", "SAME")
OBJECTS = ("STONE", "WOOD", "WATER")
# Six-cell sweeps deliberately reach a debug-square border from the usual
# start locations, while the four loops guarantee reversals and revisits.
# Every schedule has exactly ten occurrences of each movement direction.
_SCHEDULE_TEMPLATE = tuple(("UP " * 6 + "DOWN " * 6 + "LEFT " * 6 + "RIGHT " * 6 +
                            "UP RIGHT DOWN LEFT " * 4).split())


def action_schedule(seed: int) -> tuple[str, ...]:
    """Seed-deterministic, balanced schedule shared by every evaluated mode."""
    directions = MOVE_ACTIONS
    shift = int(seed) % len(directions)
    rotate = {direction: directions[(index + shift) % len(directions)] for index, direction in enumerate(directions)}
    schedule = tuple(rotate[action] for action in _SCHEDULE_TEMPLATE)
    if Counter(schedule) != Counter({action: 10 for action in MOVE_ACTIONS}):
        raise AssertionError(f"unbalanced paired schedule for seed={seed}: {Counter(schedule)}")
    return schedule


@dataclass
class State:
    episode: int
    seed: int
    step: int
    task: str
    observation: str
    map_names: list[list[str]]
    player: tuple[int, int]
    target: tuple[int, int]
    history: list[dict[str, Any]]
    action_schedule: tuple[str, ...]
    h: torch.Tensor
    z: torch.Tensor


@dataclass
class Question:
    question_id: str
    group: str
    label: str
    text: str
    candidates: tuple[str, ...]
    truth: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path, help="RSSM run directory containing resolved_config.yaml")
    p.add_argument("--checkpoint", required=True, type=Path, help="immutable .pt checkpoint to evaluate")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--base-model", type=Path, help="override base_model when evaluating a checkpoint on another host")
    p.add_argument("--seed-start", type=int, default=20261001)
    p.add_argument("--candidate-batch-size", type=int, default=64)
    p.add_argument("--max-states", type=int, default=8, help="sanity limit; ignored by --full")
    p.add_argument("--full", action="store_true", help="evaluate eight seeds instead of the one-episode sanity pass")
    p.add_argument("--allow-unverified-checkpoint", action="store_true", help="only for migration of an old run without a resolved config")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(run_dir: Path, checkpoint: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Record an exact identity before any GPU work begins."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    manifest = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size, "run_dir": str(run_dir.resolve()),
        "resolved_config": str((run_dir / "resolved_config.yaml").resolve()),
        "git_commit": cfg.get("git_commit"), "run_name": cfg.get("run_name"),
        "reward_continuation_heads": cfg.get("reward_continuation_heads"),
        "h_soft_tokens": int(cfg.get("h_soft_tokens", cfg.get("soft_tokens", -1))),
        "z_soft_tokens": int(cfg.get("z_soft_tokens", cfg.get("soft_tokens", -1))),
        "latent_h_dim": cfg.get("latent_h_dim"), "latent_z_dim": cfg.get("latent_z_dim"),
    }
    # h=5,z=5 in this experiment is the interface capacity, not an ambiguous
    # shorthand for the 512-dimensional latent vectors.
    if manifest["h_soft_tokens"] != 5 or manifest["z_soft_tokens"] != 5:
        raise RuntimeError(f"expected reward-RSSM soft(h)x5 + soft(z)x5, got {manifest}")
    if not bool(manifest["reward_continuation_heads"]):
        raise RuntimeError("checkpoint config is not the reward-RSSM (reward/continuation heads are disabled)")
    if not manifest["git_commit"]:
        raise RuntimeError("resolved_config.yaml has no git_commit; refusing to call this an exact checkpoint")
    (output / "checkpoint_identity.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def actor_base_prompt(task: str, observation: str, suffix: str) -> str:
    return (f"Your goal is to complete the following task:\n**TASK:** {task}\n\n"
            f"This is what you currently see:\n{observation}\n\n{suffix}")


def actor_hz_prompt(task: str, suffix: str) -> str:
    # Do not even mention a serialized observation or history.  State arrives
    # exclusively through inputs_embeds soft tokens.
    return (f"Your goal is to complete the following task:\n**TASK:** {task}\n\n"
            "The current state is supplied only through latent tokens.\n\n" + suffix)


def qa_suffix(question: str, candidates: Iterable[str]) -> str:
    return (f"Question: {question}\nAllowed answers: {', '.join(candidates)}\n"
            "Answer with exactly one allowed answer:")


def actor_suffix() -> str:
    return ("First, think about what to do next. Then choose one available action and write it in "
            "<action>ACTION</action>. Do not use an action history.")


def assert_hz_contract(prompt: str, observation: str, history: list[dict[str, Any]], soft: torch.Tensor, cfg: dict[str, Any]) -> None:
    """Fail closed if future prompt edits leak forbidden textual information."""
    if observation and observation in prompt:
        raise AssertionError("HZ prompt leaked observation text")
    # Individual action names can legitimately occur in a question's allowed
    # answers, so detect a serialized multi-action history rather than making
    # the incorrect claim that words such as ``UP`` are forbidden.
    history_text = [str(item.get("action", "")) for item in history if item.get("action")]
    serialized_history = " -> ".join(history_text)
    if len(history_text) > 1 and serialized_history in prompt:
        raise AssertionError("HZ prompt leaked executed action history")
    forbidden = ("Actions already taken", "history of your recent actions", "Reasoning history", "This is what you currently see")
    if any(marker in prompt for marker in forbidden):
        raise AssertionError(f"HZ prompt contains forbidden textual-history marker: {prompt!r}")
    if soft.ndim != 2 or soft.shape[0] != 10:
        raise AssertionError(f"HZ must receive exactly 5 h + 5 z soft tokens, got {list(soft.shape)}")
    if int(cfg.get("h_soft_tokens", cfg.get("soft_tokens", 0))) != 5 or int(cfg.get("z_soft_tokens", cfg.get("soft_tokens", 0))) != 5:
        raise AssertionError("config is not the 5h/5z interface")


def actor_soft(qwen, model, h: torch.Tensor, z: torch.Tensor, variant: str) -> torch.Tensor:
    h_soft, z_soft = model.transition_tokens(h, z)
    if variant == "hz":
        result = torch.cat((h_soft[0], z_soft[0]), dim=0)
    elif variant == "h_only":
        result = torch.cat((h_soft[0], torch.zeros_like(z_soft[0])), dim=0)
    elif variant == "z_only":
        result = torch.cat((torch.zeros_like(h_soft[0]), z_soft[0]), dim=0)
    else:
        raise ValueError(variant)
    return _normalise_actor_soft(qwen, result)


def _choice_ids(tokenizer, choice: str) -> list[int]:
    ids = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"candidate tokenizes to nothing: {choice!r}")
    return ids


@torch.inference_mode()
def choose(qwen, tokenizer, prompt: str, candidates: tuple[str, ...], soft: torch.Tensor | None,
           batch_size: int) -> tuple[str, dict[str, float]]:
    """Exact teacher-forced candidate choice, batched over candidate strings."""
    before, boundary = _chat_parts(tokenizer, prompt)
    device, embedding = next(qwen.parameters()).device, qwen.get_input_embeddings()
    prefix_ids = before + boundary
    prefix = embedding(torch.tensor(prefix_ids, device=device))
    if soft is not None:
        split = len(before)
        prefix = torch.cat((prefix[:split], soft.to(embedding.weight.dtype), prefix[split:]), dim=0)
    all_scores: dict[str, float] = {}
    for start in range(0, len(candidates), max(1, batch_size)):
        choices = candidates[start:start + max(1, batch_size)]
        tokenized = [_choice_ids(tokenizer, item) for item in choices]
        sequences = [torch.cat((prefix, embedding(torch.tensor(ids, device=device))), dim=0) for ids in tokenized]
        width = max(seq.shape[0] for seq in sequences)
        embeds = torch.zeros(len(sequences), width, prefix.shape[-1], dtype=prefix.dtype, device=device)
        mask = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
        positions = torch.zeros_like(mask)
        offsets = []
        for index, seq in enumerate(sequences):
            pad = width - seq.shape[0]
            embeds[index, pad:] = seq; mask[index, pad:] = 1
            positions[index, pad:] = torch.arange(seq.shape[0], device=device)
            offsets.append(pad + prefix.shape[0] - 1)
        logits = qwen(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False).logits.float()
        for row, (choice, ids, offset) in enumerate(zip(choices, tokenized, offsets)):
            target = torch.tensor(ids, device=device)
            # Compare mean conditional log-probability so candidates with more
            # tokenizer pieces are not penalized merely for being longer.
            value = torch.log_softmax(logits[row, offset:offset + len(ids)], dim=-1).gather(-1, target[:, None]).mean()
            all_scores[choice] = float(value)
    return max(candidates, key=all_scores.__getitem__), all_scores


def cell_name(value: int) -> str:
    from craftax.craftax_classic.constants import BlockType
    try:
        return BlockType(int(value)).name.upper()
    except (ValueError, KeyError):
        return "UNKNOWN"


def state_map(worker: CagedCraftextWorker) -> list[list[str]]:
    import jax
    raw = np.asarray(jax.device_get(worker.state.env_state.map))
    return [[cell_name(value) for value in row] for row in raw]


def position(worker: CagedCraftextWorker) -> tuple[int, int]:
    import jax
    return tuple(map(int, np.asarray(jax.device_get(worker.state.env_state.player_position))))


def action_id(name: str) -> int:
    return int(ACTION_TO_TEXT.index(name))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quadrant(origin: tuple[int, int], point: tuple[int, int]) -> str | None:
    dr, dc = point[0] - origin[0], point[1] - origin[1]
    if dr == 0 or dc == 0:
        return None
    return ("RIGHT" if dc > 0 else "LEFT") + ("_DOWN" if dr > 0 else "_UP")


def neighbour(s: State, action: str) -> tuple[int, int]:
    dr, dc = DELTAS[action]
    return s.player[0] + dr, s.player[1] + dc


def get_cell(s: State, pos: tuple[int, int]) -> str:
    if not (0 <= pos[0] < 8 and 0 <= pos[1] < 8):
        return "OUT_OF_BOUNDS"
    return s.map_names[pos[0]][pos[1]]


def is_blocked(s: State, action: str) -> bool:
    return get_cell(s, neighbour(s, action)) in {"TREE", "OUT_OF_BOUNDS"}


def action_consequence(s: State, action: str) -> str:
    nxt = s.player if is_blocked(s, action) else neighbour(s, action)
    before, after = manhattan(s.player, s.target), manhattan(nxt, s.target)
    return "CLOSER" if after < before else "FARTHER" if after > before else "SAME"


def powerset_actions() -> tuple[str, ...]:
    values = ["NONE"]
    for n in range(1, len(MOVE_ACTIONS) + 1):
        values.extend(",".join(group) for group in combinations(MOVE_ACTIONS, n))
    return tuple(values)


BLOCKED_CANDIDATES = powerset_actions()


def questions(s: State) -> list[Question]:
    q: list[Question] = []
    target_label = quadrant(s.player, s.target)
    if target_label:
        q.append(Question("G1_target_direction", "Grounding", "target direction", "Where is the target relative to the agent?", QUADRANTS, target_label))
    q.append(Question("G2_target_shortest_path_distance", "Grounding", "target shortest-path distance", "How many shortest-path movement steps are required to reach the target?", tuple(map(str, range(31))), str(manhattan(s.player, s.target))))
    for action in MOVE_ACTIONS:
        q.append(Question(f"G{2 + MOVE_ACTIONS.index(action) + 1}_{action.lower()}_consequence", "Grounding", f"{action} consequence", f"Will action {action} move the agent CLOSER, FARTHER, or keep the SAME distance from the target?", CONSEQUENCES, action_consequence(s, action)))
    for index, obj in enumerate(OBJECTS, 7):
        coords = tuple(map(int, np.argwhere(np.asarray(s.map_names) == obj)[0]))
        label = quadrant(s.player, coords)
        if label:
            q.append(Question(f"G{index}_{obj.lower()}_direction", "Grounding", f"{obj} direction", f"Where is the {obj} relative to the agent?", QUADRANTS, label))
    cell_candidates = tuple(sorted({cell for row in s.map_names for cell in row} | {"OUT_OF_BOUNDS"}))
    for index, action in enumerate(MOVE_ACTIONS, 1):
        q.append(Question(f"Z{index}_cell_{action.lower()}", "Current-state Z", f"directly {action} cell", f"What cell type is directly {action} from the agent?", cell_candidates, get_cell(s, neighbour(s, action))))
    blocked = tuple(a for a in MOVE_ACTIONS if is_blocked(s, a))
    q.append(Question("Z5_blocked_actions", "Current-state Z", "blocked actions", "Which movement actions are currently blocked?", BLOCKED_CANDIDATES, ",".join(blocked) if blocked else "NONE"))
    q.append(Question("Z6_target_adjacent", "Current-state Z", "target adjacent", "Is the target directly adjacent to the agent?", ("YES", "NO"), "YES" if manhattan(s.player, s.target) == 1 else "NO"))
    distances = {obj: manhattan(s.player, tuple(map(int, np.argwhere(np.asarray(s.map_names) == obj)[0]))) for obj in OBJECTS}
    nearest = [obj for obj, dist in distances.items() if dist == min(distances.values())]
    if len(nearest) == 1:
        q.append(Question("Z7_nearest_object", "Current-state Z", "nearest object", "Which of STONE, WOOD, WATER is nearest to the agent?", OBJECTS, nearest[0]))
    local = [get_cell(s, (s.player[0] + dr, s.player[1] + dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    q.append(Question("Z8_local_tree_count", "Current-state Z", "TREE cells in local 3x3", "How many TREE cells are in the local 3x3 neighborhood?", tuple(map(str, range(10))), str(sum(v == "TREE" for v in local))))
    non_grass = sum(v not in {"GRASS", "OUT_OF_BOUNDS"} for v in local)
    q.append(Question("Z9_local_non_grass_count", "Current-state Z", "non-GRASS objects in local 3x3", "How many non-GRASS object cells are in the local 3x3 neighborhood?", tuple(map(str, range(10))), str(non_grass)))
    q.append(Question("Z10_target_in_local_3x3", "Current-state Z", "target in local 3x3", "Is the target inside the local 3x3 neighborhood?", ("YES", "NO"), "YES" if max(abs(s.player[0] - s.target[0]), abs(s.player[1] - s.target[1])) <= 1 else "NO"))
    h = s.history
    if len(h) >= 1:
        previous = h[-1]
        q += [Question("H1_previous_action", "Memory H", "previous action", "What action was executed immediately before this state?", MOVE_ACTIONS + ("NOOP",), previous["action"]),
              Question("H4_previous_position_change", "Memory H", "previous changed position", "Did the previous action change the agent position?", ("YES", "NO"), "YES" if previous["changed"] else "NO"),
              Question("H5_previous_target_effect", "Memory H", "previous target effect", "Did the previous action move the agent CLOSER, FARTHER, or SAME relative to the target?", CONSEQUENCES, previous["effect"]),
              Question("H8_visited_current_cell", "Memory H", "visited current cell", "Has the agent visited the current cell previously in this episode?", ("YES", "NO"), "YES" if any(tuple(x["before"]) == s.player for x in h) else "NO"),
              Question("H9_arrival_movement_direction", "Memory H", "arrival movement direction", "What movement direction brought the agent to the current cell?", MOVE_ACTIONS + ("STAY",), previous["arrival"])]
    if len(h) >= 2:
        q.append(Question("H2_action_2_steps_ago", "Memory H", "action 2 steps ago", "What action was executed 2 steps ago?", MOVE_ACTIONS + ("NOOP",), h[-2]["action"]))
    if len(h) >= 3:
        q.append(Question("H3_action_3_steps_ago", "Memory H", "action 3 steps ago", "What action was executed 3 steps ago?", MOVE_ACTIONS + ("NOOP",), h[-3]["action"]))
    if len(h) >= 5:
        last5 = h[-5:]
        q += [Question("H6_last5_position_changes", "Memory H", "last 5 position changes", "How many of the last 5 actions changed position?", tuple(map(str, range(6))), str(sum(x["changed"] for x in last5))),
              Question("H7_last5_tree_or_border_collisions", "Memory H", "last 5 TREE/border collisions", "How many of the last 5 actions hit TREE or the border and therefore did not move?", tuple(map(str, range(6))), str(sum(x["tree_or_border_collision"] for x in last5)))]
    return q


def collect_states(qwen, tokenizer, model, device: torch.device, cfg: dict[str, Any], seeds: list[int], limit: int | None) -> list[State]:
    """Generate paired trajectories before a single QA/model answer is requested."""
    states: list[State] = []
    for episode, seed in enumerate(seeds):
        schedule = action_schedule(seed)
        worker = CagedCraftextWorker(seed=seed, env_kwargs={"config_name": "debug_square_8x8", "use_debug_square_map": True, "observation_type": "ascii", "encode_form": "embedding"})
        _pixel, info = worker.reset(scenario_idx=seed % 3, return_render=False)
        task, observation = str(info["instruction"]), str(info["text_render"])
        h = torch.zeros(1, int(cfg["latent_h_dim"]), device=device)
        qm, _qs, _ql, _hidden = posterior(qwen, tokenizer, model, task, observation, h[0], device, cfg)
        z, history = qm.detach(), []
        for step in range(len(schedule) + 1):
            current_info = info
            states.append(State(episode, seed, step, task, observation, state_map(worker), position(worker), tuple(map(int, current_info["debug_goal_cell"])), list(history), schedule, h.detach().clone(), z.detach().clone()))
            if limit is not None and len(states) >= limit:
                return states
            if step == len(schedule):
                break
            action, before = schedule[step], position(worker)
            before_map = state_map(worker)
            dr_try, dc_try = DELTAS[action]
            attempt = (before[0] + dr_try, before[1] + dc_try)
            attempted_cell = "OUT_OF_BOUNDS" if not (0 <= attempt[0] < 8 and 0 <= attempt[1] < 8) else before_map[attempt[0]][attempt[1]]
            tree_or_border_collision = attempted_cell in {"TREE", "OUT_OF_BOUNDS"}
            _pixel, reward, done, info = worker.step(action_id(action), return_render=False)
            after = position(worker); changed = before != after
            target = tuple(map(int, info["debug_goal_cell"])); effect = "CLOSER" if manhattan(after, target) < manhattan(before, target) else "FARTHER" if manhattan(after, target) > manhattan(before, target) else "SAME"
            dr, dc = after[0] - before[0], after[1] - before[1]
            arrival = "STAY" if not changed else {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}[(dr, dc)]
            history.append({"action": action, "before": before, "after": after, "changed": changed, "effect": effect, "arrival": arrival, "reward": float(reward), "attempted_cell": attempted_cell, "tree_or_border_collision": tree_or_border_collision})
            h = transition(qwen, tokenizer, model, action, h, z, device, cfg).detach()
            observation = str(info["text_render"])
            qm, _qs, _ql, _hidden = posterior(qwen, tokenizer, model, task, observation, h[0], device, cfg)
            z = qm.detach()
            if done:
                break
    return states


def serialise_paired_trajectory(states: list[State]) -> list[dict[str, Any]]:
    """Persist the model-independent state/action record before QA begins."""
    return [{"episode": state.episode, "seed": state.seed, "step": state.step,
             "task": state.task, "observation": state.observation, "map": state.map_names,
             "player": state.player, "target": state.target, "executed_history": state.history,
             "action_schedule": list(state.action_schedule),
             "action_schedule_counts": dict(Counter(state.action_schedule))} for state in states]


def memory_target_distribution(states: list[State]) -> dict[str, Any]:
    """Audit label balance before scoring memory questions."""
    def tally(values: Iterable[Any], expected: Iterable[str] | None = None) -> dict[str, int]:
        counts = Counter(str(value) for value in values)
        if expected is not None:
            for value in expected: counts.setdefault(value, 0)
        return dict(sorted(counts.items()))

    eligible = [state for state in states if state.history]
    previous = [state.history[-1] for state in eligible]
    two_back = [state.history[-2]["action"] for state in states if len(state.history) >= 2]
    three_back = [state.history[-3]["action"] for state in states if len(state.history) >= 3]
    report = {
        "schedule_counts_by_seed": {str(state.seed): dict(Counter(state.action_schedule)) for state in states if state.step == 0},
        "previous_action": tally((item["action"] for item in previous), MOVE_ACTIONS),
        "action_2_steps_ago": tally(two_back, MOVE_ACTIONS),
        "action_3_steps_ago": tally(three_back, MOVE_ACTIONS),
        "changed": tally(("changed" if item["changed"] else "not_changed" for item in previous), ("changed", "not_changed")),
        "blocked": tally(("blocked" if item["tree_or_border_collision"] else "not_blocked" for item in previous), ("blocked", "not_blocked")),
        "visited": tally(("visited" if any(tuple(item["before"]) == state.player for item in state.history) else "not_visited" for state in eligible), ("visited", "not_visited")),
    }
    warnings: list[str] = []
    for name in ("previous_action", "action_2_steps_ago", "action_3_steps_ago", "changed", "blocked", "visited"):
        counts = list(report[name].values())
        observed = [count for count in counts if count > 0]
        if observed and (0 in counts or max(observed) > 3 * min(observed)):
            warnings.append(f"{name} is imbalanced: {report[name]}")
    report["warnings"] = warnings
    return report


def actor_probe(qwen, tokenizer, state: State, model, variants: tuple[str, ...], cfg: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for variant in variants:
        prompt = actor_base_prompt(state.task, state.observation, actor_suffix()) if variant == "base" else actor_hz_prompt(state.task, actor_suffix())
        soft = None if variant == "base" else actor_soft(qwen, model, state.h, state.z, variant)
        if soft is not None: assert_hz_contract(prompt, state.observation, state.history, soft, cfg)
        raw = _generate(qwen, tokenizer, prompt, soft, int(cfg.get("env_eval_actor_max_new_tokens", 64)), generation={"do_sample": False})
        ids, valid = craftext_projection([raw]); action = str(ACTION_TO_TEXT[int(ids[0])]) if valid[0] and int(ids[0]) >= 0 else None
        reasoning = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.I | re.S)
        records[variant] = {"prompt": prompt, "soft_token_count": 0 if soft is None else int(soft.shape[0]),
                            "raw_reasoning": reasoning.group(1).strip() if reasoning else "", "raw_output": raw, "parsed_action": action,
                            "closer": bool(action in MOVE_ACTIONS and action_consequence(state, action) == "CLOSER"),
                            "blocked": bool(action in MOVE_ACTIONS and is_blocked(state, action))}
    return records


def evaluate_questions(qwen, tokenizer, model, states: list[State], cfg: dict[str, Any], batch_size: int) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    variants = ("base", "hz", "h_only", "z_only")
    rows, probes = [], {}
    for state in states:
        probes[(state.episode, state.step)] = actor_probe(qwen, tokenizer, state, model, variants, cfg)
        for question in questions(state):
            row: dict[str, Any] = {"episode": state.episode, "seed": state.seed, "step": state.step, "question_id": question.question_id, "question_group": question.group, "question": question.label, "question_text": question.text, "ground_truth": question.truth, "candidates": " | ".join(question.candidates), "frame_path": f"frames/episode_{state.seed}/step_{state.step:02d}_{question.question_id}.png"}
            for variant in variants:
                prompt = actor_base_prompt(state.task, state.observation, qa_suffix(question.text, question.candidates)) if variant == "base" else actor_hz_prompt(state.task, qa_suffix(question.text, question.candidates))
                soft = None if variant == "base" else actor_soft(qwen, model, state.h, state.z, variant)
                if soft is not None: assert_hz_contract(prompt, state.observation, state.history, soft, cfg)
                answer, scores = choose(qwen, tokenizer, prompt, question.candidates, soft, batch_size)
                row[f"{variant}_answer"] = answer; row[f"{variant}_correct"] = answer == question.truth
                row[f"{variant}_prompt"] = prompt; row[f"{variant}_scores"] = json.dumps(scores)
            row["base_actor_action"] = probes[(state.episode, state.step)]["base"]["parsed_action"]
            row["hz_actor_action"] = probes[(state.episode, state.step)]["hz"]["parsed_action"]
            row["base_actor_raw_output"] = probes[(state.episode, state.step)]["base"]["raw_output"]
            row["hz_actor_raw_output"] = probes[(state.episode, state.step)]["hz"]["raw_output"]
            rows.append(row)
    return rows, probes


def short_reasoning(raw: str, limit: int = 180) -> str:
    """Compact audit text for HTML/GIF; the complete response stays in JSON."""
    value = " ".join(str(raw).replace("<think>", "").replace("</think>", "").split())
    return value[:limit] + ("..." if len(value) > limit else "")


def parse_free_generation(raw: str, candidates: Iterable[str]) -> dict[str, Any]:
    """Parse only explicit answer tags; never infer an answer from prose."""
    think = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.I | re.S)
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, re.I | re.S)
    candidate_map = {candidate.casefold(): candidate for candidate in candidates}
    answer = candidate_map.get(answers[-1].strip().casefold()) if answers else None
    return {"think": think.group(1).strip() if think else "", "answer": answer,
            "parse_success": answer is not None}


def continuation_smoke(qwen, tokenizer, model, state: State, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fail closed when generation decodes a prompt prefix instead of new tokens."""
    checks = []
    for variant in ("base", "hz"):
        prompt = actor_base_prompt(state.task, state.observation, actor_suffix()) if variant == "base" else actor_hz_prompt(state.task, actor_suffix())
        soft = None if variant == "base" else actor_soft(qwen, model, state.h, state.z, "hz")
        raw = _generate(qwen, tokenizer, prompt, soft, int(cfg.get("env_eval_actor_max_new_tokens", 64)), generation={"do_sample": False})
        normalized_raw, normalized_prompt = " ".join(raw.split()), " ".join(prompt.split())
        prompt_echo = len(normalized_raw) >= 16 and normalized_raw in normalized_prompt
        if not normalized_raw or prompt_echo:
            raise RuntimeError(f"{variant} generation smoke failed: decoded prompt text rather than a continuation: {raw!r}")
        checks.append({"model": variant, "prompt": prompt, "raw_output": raw, "prompt_echo": prompt_echo})
    return checks


def representative_free_questions(states: list[State]) -> list[tuple[State, Question]]:
    """Per episode: one G, two Z, and two H qualitative question traces."""
    selected: list[tuple[State, Question]] = []
    episodes: dict[int, list[State]] = defaultdict(list)
    for state in states:
        episodes[state.episode].append(state)
    requested = (("Grounding", ("G1_target_direction", "G2_target_shortest_path_distance"), 1),
                 ("Current-state Z", ("Z1_cell_up", "Z5_blocked_actions"), 2),
                 ("Memory H", ("H1_previous_action", "H6_last5_position_changes"), 2))
    for episode_states in episodes.values():
        for group, preferred_ids, needed in requested:
            found: list[tuple[State, Question]] = []
            for preferred in preferred_ids:
                for state in episode_states:
                    question = next((q for q in questions(state) if q.question_id == preferred), None)
                    if question is not None:
                        found.append((state, question)); break
            if len(found) < needed:
                for state in episode_states:
                    for question in questions(state):
                        if question.group == group and all(question.question_id != old.question_id or state.step != old_state.step for old_state, old in found):
                            found.append((state, question))
                            if len(found) == needed: break
                    if len(found) == needed: break
            if len(found) < needed:
                raise RuntimeError(f"sanity trajectory lacks {needed} representative {group} questions")
            selected.extend(found[:needed])
    return selected


def free_generation_examples(qwen, tokenizer, model, states: list[State], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Qualitative traces cover Grounding, current-state Z, and memory H."""
    examples = []
    for state, question in representative_free_questions(states):
        suffix = (f"Question: {question.text}\nThink in <think>...</think>, then give one allowed answer "
                  f"in <answer>...</answer>. Allowed answers: {', '.join(question.candidates)}.")
        for variant in ("base", "hz"):
            prompt = actor_base_prompt(state.task, state.observation, suffix) if variant == "base" else actor_hz_prompt(state.task, suffix)
            soft = None if variant == "base" else actor_soft(qwen, model, state.h, state.z, "hz")
            if soft is not None:
                assert_hz_contract(prompt, state.observation, state.history, soft, cfg)
            raw = _generate(qwen, tokenizer, prompt, soft, int(cfg.get("env_eval_actor_max_new_tokens", 64)), generation={"do_sample": False})
            parsed = parse_free_generation(raw, question.candidates)
            examples.append({"episode": state.episode, "seed": state.seed, "step": state.step, "question_id": question.question_id,
                             "question_group": question.group, "ground_truth": question.truth, "model": variant,
                             "prompt": prompt, "raw_output": raw, "extracted_think": parsed["think"],
                             "extracted_answer": parsed["answer"], "parse_success": parsed["parse_success"],
                             "correct": bool(parsed["answer"] == question.truth)})
    return examples


def attach_free_reasoning(rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    index = {(row["episode"], row["step"], row["question_id"]): row for row in rows}
    for row in rows:
        for variant in ("base", "hz"):
            row[f"{variant}_qa_reasoning_short"] = ""
            row[f"{variant}_freegen_answer"] = ""
            row[f"{variant}_freegen_correct"] = None
            row[f"{variant}_freegen_parse_success"] = None
    for item in examples:
        row = index[(item["episode"], item["step"], item["question_id"])]
        row[f"{item['model']}_qa_reasoning_short"] = short_reasoning(item["raw_output"])
        row[f"{item['model']}_freegen_answer"] = item["extracted_answer"] or ""
        row[f"{item['model']}_freegen_correct"] = item["correct"]
        row[f"{item['model']}_freegen_parse_success"] = item["parse_success"]


def paired_actor_probe_summary(probes: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    values = list(probes.values())
    def rate(variant: str, field: str) -> float:
        return float(np.mean([bool(item[variant][field]) for item in values])) if values else 0.0
    parsed = {variant: rate(variant, "parsed_action") for variant in ("base", "hz")}
    return {"states": len(values), "base_action_parse_rate": parsed["base"], "hz_action_parse_rate": parsed["hz"],
            "base_closer_action_rate": rate("base", "closer"), "hz_closer_action_rate": rate("hz", "closer"),
            "base_blocked_action_rate": rate("base", "blocked"), "hz_blocked_action_rate": rate("hz", "blocked"),
            "base_vs_hz_action_agreement": float(np.mean([item["base"]["parsed_action"] == item["hz"]["parsed_action"] for item in values])) if values else 0.0}


def render_map(state: State) -> str:
    symbols = {"GRASS": ".", "TREE": "T", "STONE": "S", "WOOD": "W", "WATER": "~"}
    rows = []
    for r, line in enumerate(state.map_names):
        rows.append(" ".join("@" if (r, c) == state.player else symbols.get(value, "?") for c, value in enumerate(line)))
    return "\n".join(rows)


def draw_frame(path: Path, state: State, question: Question, row: dict[str, Any]) -> Image.Image:
    base_reason = row.get("base_qa_reasoning_short") or short_reasoning(row.get("base_actor_raw_output", ""), 110)
    hz_reason = row.get("hz_qa_reasoning_short") or short_reasoning(row.get("hz_actor_raw_output", ""), 110)
    def free_line(variant: str) -> str:
        parsed = row.get(f"{variant}_freegen_parse_success")
        if parsed is None: return f"{variant.upper()} freegen: n/a"
        answer = row.get(f"{variant}_freegen_answer") or "PARSE_FAILURE"
        return f"{variant.upper()} freegen: {answer} {'OK' if row.get(f'{variant}_freegen_correct') else 'X'}"
    font = ImageFont.load_default(); lines = [f"Episode {state.seed} | step {state.step} | Task: {state.task}", "", *render_map(state).splitlines(), "", f"{question.question_id}: {question.text}", f"Ground truth: {question.truth}", f"Base candidate: {row['base_answer']} {'OK' if row['base_correct'] else 'X'}", f"HZ candidate:   {row['hz_answer']} {'OK' if row['hz_correct'] else 'X'}", free_line("base"), free_line("hz"), f"Base actor action: {row.get('base_actor_action')}", f"HZ actor action:   {row.get('hz_actor_action')}", f"Base reasoning: {base_reason}", f"HZ reasoning: {hz_reason}"]
    if question.group == "Memory H": lines.append("true last actions: " + " -> ".join(x["action"] for x in state.history[-5:]))
    image = Image.new("RGB", (920, max(260, 16 * (len(lines) + 2))), "white"); draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines): draw.text((10, 8 + 16 * i), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
    return image


def write_visuals(output: Path, states: list[State], rows: list[dict[str, Any]]) -> None:
    by_key = {(s.episode, s.step): s for s in states}; by_question: dict[tuple[int, str], list[Image.Image]] = defaultdict(list)
    for row in rows:
        state = by_key[(row["episode"], row["step"])]
        question = next(q for q in questions(state) if q.question_id == row["question_id"])
        frame = output / row["frame_path"]
        image = draw_frame(frame, state, question, row)
        by_question[(state.seed, question.question_id)].append(image)
    for (seed, question_id), frames in by_question.items():
        directory = output / "gifs" / f"episode_{seed}"; directory.mkdir(parents=True, exist_ok=True)
        frames[0].save(directory / f"{question_id}.gif", save_all=True, append_images=frames[1:], duration=650, loop=0)


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[(row["question_group"], row["question_id"], row["question"])].append(row)
    for (group, question_id, label), values in groups.items():
        free = {variant: [value for value in values if value[f"{variant}_freegen_parse_success"] is not None] for variant in ("base", "hz")}
        summaries.append({"Group": group, "Question": label, "question_id": question_id, "N": len(values),
                          **{f"{variant} Qwen acc": float(np.mean([v[f"{variant}_correct"] for v in values])) for variant in ("base", "hz", "h_only", "z_only")},
                          "Base freegen acc": float(np.mean([v["base_freegen_correct"] for v in free["base"]])) if free["base"] else None,
                          "HZ freegen acc": float(np.mean([v["hz_freegen_correct"] for v in free["hz"]])) if free["hz"] else None,
                          "Base freegen parse": float(np.mean([v["base_freegen_parse_success"] for v in free["base"]])) if free["base"] else None,
                          "HZ freegen parse": float(np.mean([v["hz_freegen_parse_success"] for v in free["hz"]])) if free["hz"] else None,
                          "Delta HZ-Base": float(np.mean([v["hz_correct"] for v in values]) - np.mean([v["base_correct"] for v in values]))})
    candidate_macro = []
    for variant in ("base", "hz", "h_only", "z_only"):
        selected = {group: [r[f"{variant}_correct"] for r in rows if r["question_group"] == group] for group in ("Grounding", "Current-state Z", "Memory H")}
        candidate_macro.append({"Model": {"base": "Base Qwen", "hz": "HZ Qwen", "h_only": "H-only", "z_only": "Z-only"}[variant], "Grounding": float(np.mean(selected["Grounding"])), "Current-state Z": float(np.mean(selected["Current-state Z"])), "Memory H": float(np.mean(selected["Memory H"])) if selected["Memory H"] else None, "Overall": float(np.mean([r[f"{variant}_correct"] for r in rows]))})
    freegen_macro = []
    for variant in ("base", "hz"):
        selected = [row for row in rows if row[f"{variant}_freegen_parse_success"] is not None]
        group_values = {group: [row for row in selected if row["question_group"] == group] for group in ("Grounding", "Current-state Z", "Memory H")}
        freegen_macro.append({"Model": "Base Qwen" if variant == "base" else "HZ Qwen",
                              **{group: float(np.mean([row[f"{variant}_freegen_correct"] for row in values])) if values else None for group, values in group_values.items()},
                              "Overall": float(np.mean([row[f"{variant}_freegen_correct"] for row in selected])) if selected else None,
                              "Parse rate": float(np.mean([row[f"{variant}_freegen_parse_success"] for row in selected])) if selected else None,
                              "N": len(selected)})
    return summaries, candidate_macro, freegen_macro


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]], candidate_macro: list[dict[str, Any]], freegen_macro: list[dict[str, Any]], actor: list[dict[str, Any]]) -> None:
    def table(values: list[dict[str, Any]]) -> str:
        if not values: return ""
        columns = list(values[0]); return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join("| " + " | ".join("" if value[c] is None else f"{value[c]:.4f}" if isinstance(value[c], float) else str(value[c]) for c in columns) + " |" for value in values)
    path.write_text("# Offline reward-RSSM paired QA evaluation\n\n## Candidate question accuracy\n\n" + table(summary) + "\n\n## Candidate group macro\n\n" + table(candidate_macro) + "\n\n## Free-generation group macro\n\n" + table(freegen_macro) + "\n\n## Autonomous actor\n\n" + table(actor) + "\n", encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["episode", "seed", "step", "question_id", "question_group", "question_text", "ground_truth", "base_answer", "base_correct", "hz_answer", "hz_correct", "base_freegen_answer", "base_freegen_correct", "base_freegen_parse_success", "hz_freegen_answer", "hz_freegen_correct", "hz_freegen_parse_success", "h_only_answer", "z_only_answer", "base_qa_reasoning_short", "hz_qa_reasoning_short", "base_actor_raw_output", "hz_actor_raw_output", "frame_path"]
    def cell(row: dict[str, Any], column: str) -> str:
        if column == "frame_path":
            return f'<td><a href="{html.escape(str(row.get(column, "")))}">frame</a></td>'
        return f"<td>{html.escape(str(row.get(column, '')))}</td>"
    body = "\n".join("<tr>" + "".join(cell(row, column) for column in columns) + "</tr>" for row in rows)
    path.write_text("<html><body><table border=\"1\"><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>" + body + "</table></body></html>", encoding="utf-8")


def autonomous(qwen, tokenizer, model, device: torch.device, cfg: dict[str, Any], seeds: list[int], output: Path) -> list[dict[str, Any]]:
    result = []
    for variant in ("base", "hz"):
        all_episodes = []
        for seed in seeds:
            worker = CagedCraftextWorker(seed=seed, env_kwargs={"config_name": "debug_square_8x8", "use_debug_square_map": True, "observation_type": "ascii", "encode_form": "embedding"})
            _pixel, info = worker.reset(scenario_idx=seed % 3, return_render=False); task, observation = str(info["instruction"]), str(info["text_render"])
            h = torch.zeros(1, int(cfg["latent_h_dim"]), device=device); qm, _qs, _ql, _hidden = posterior(qwen, tokenizer, model, task, observation, h[0], device, cfg); z = qm.detach()
            trace, previous, total = [], None, 0.0
            for step in range(50):
                before = position(worker); target = tuple(map(int, info["debug_goal_cell"])); prompt = actor_base_prompt(task, observation, actor_suffix()) if variant == "base" else actor_hz_prompt(task, actor_suffix()); soft = None if variant == "base" else actor_soft(qwen, model, h, z, "hz")
                if soft is not None: assert_hz_contract(prompt, observation, [], soft, cfg)
                raw = _generate(qwen, tokenizer, prompt, soft, int(cfg.get("env_eval_actor_max_new_tokens", 64)), generation={"do_sample": False}); ids, valid = craftext_projection([raw]); name = str(ACTION_TO_TEXT[int(ids[0])]) if valid[0] and int(ids[0]) >= 0 else "NOOP"
                _pixel, reward, done, info = worker.step(action_id(name), return_render=False); after = position(worker); total += float(reward)
                trace.append({"step": step, "prompt": prompt, "raw_output": raw, "parsed_action": name if valid[0] else None, "executed_action": name, "reward": float(reward), "blocked": bool(name in MOVE_ACTIONS and before == after), "closer": bool(name in MOVE_ACTIONS and manhattan(after, target) < manhattan(before, target)), "repeated": name == previous})
                h = transition(qwen, tokenizer, model, name, h, z, device, cfg).detach(); observation = str(info["text_render"]); qm, _qs, _ql, _hidden = posterior(qwen, tokenizer, model, task, observation, h[0], device, cfg); z = qm.detach(); previous = name
                if done: break
            all_episodes.append({"seed": seed, "steps": trace, "success": bool(info.get("won", False)), "reward": total})
        steps = [item for episode in all_episodes for item in episode["steps"]]
        result.append({"Model": "Base Qwen" if variant == "base" else "HZ Qwen", "SR": float(np.mean([e["success"] for e in all_episodes])), "Reward": float(np.mean([e["reward"] for e in all_episodes])), "Episode length": float(np.mean([len(e["steps"]) for e in all_episodes])), "Closer action": float(np.mean([x["closer"] for x in steps])) if steps else 0.0, "Blocked": float(np.mean([x["blocked"] for x in steps])) if steps else 0.0, "Parse": float(np.mean([x["parsed_action"] is not None for x in steps])) if steps else 0.0, "Repeated": float(np.mean([x["repeated"] for x in steps])) if steps else 0.0})
        (output / f"autonomous_{variant}.json").write_text(json.dumps(all_episodes, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.run_dir / "resolved_config.yaml"
    if not config_path.is_file() and not args.allow_unverified_checkpoint: raise FileNotFoundError(f"missing {config_path}; exact checkpoint identity is mandatory")
    cfg = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    if args.base_model is not None:
        cfg["base_model"] = str(args.base_model)
    cfg.update({"max_sequence_length": int(cfg.get("max_sequence_length", 3072)), "env_eval_actor_max_new_tokens": int(cfg.get("env_eval_actor_max_new_tokens", 64))})
    # An online PPO update is not an offline reconstruction checkpoint.  Refuse
    # the online trainer's ``rssm`` payload before allocating a model or GPU.
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in loaded or "rssm" in loaded:
        raise RuntimeError(f"expected offline reconstruction payload key 'model', got {sorted(loaded)}; online PPO checkpoints are forbidden")
    manifest = checkpoint_manifest(args.run_dir, args.checkpoint, cfg, args.output_dir)
    device = torch.device(args.device); tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token; tokenizer.padding_side = "left"
    qwen = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    for p in qwen.parameters(): p.requires_grad_(False)
    cfg["soft_token_target_rms"] = float(qwen.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    model = QwenTransitionRSSMRecon(cfg, qwen.config.hidden_size).to(device).eval()
    model.load_state_dict(loaded["model"], strict=True)
    manifest["checkpoint_payload_key"] = "model"
    manifest["checkpoint_step"] = loaded.get("step")
    (args.output_dir / "checkpoint_identity.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    seeds = list(range(args.seed_start, args.seed_start + (8 if args.full else 1))); states = collect_states(qwen, tokenizer, model, device, cfg, seeds, None if args.full else args.max_states)
    if not states: raise RuntimeError("fixed trajectory generated no states")
    (args.output_dir / "paired_trajectories.json").write_text(json.dumps(serialise_paired_trajectory(states), indent=2), encoding="utf-8")
    memory_distribution = memory_target_distribution(states)
    (args.output_dir / "memory_target_distribution.json").write_text(json.dumps(memory_distribution, indent=2), encoding="utf-8")
    print(json.dumps({"memory_target_distribution": memory_distribution}, indent=2), flush=True)
    smoke = continuation_smoke(qwen, tokenizer, model, states[0], cfg)
    (args.output_dir / "generation_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    rows, probes = evaluate_questions(qwen, tokenizer, model, states, cfg, args.candidate_batch_size)
    free_examples = free_generation_examples(qwen, tokenizer, model, states, cfg)
    attach_free_reasoning(rows, free_examples)
    write_csv(args.output_dir / "per_state_questions.csv", rows); write_html(args.output_dir / "per_state_questions.html", rows); write_visuals(args.output_dir, states, rows)
    (args.output_dir / "actor_probe.json").write_text(json.dumps({f"{k[0]}:{k[1]}": v for k, v in probes.items()}, indent=2), encoding="utf-8")
    (args.output_dir / "actor_probe_summary.json").write_text(json.dumps(paired_actor_probe_summary(probes), indent=2), encoding="utf-8")
    (args.output_dir / "free_generation.json").write_text(json.dumps(free_examples, indent=2), encoding="utf-8")
    summary, candidate_macro, freegen_macro = aggregate(rows)
    write_csv(args.output_dir / "question_summary.csv", summary)
    write_csv(args.output_dir / "candidate_group_macro.csv", candidate_macro)
    write_csv(args.output_dir / "freegen_group_macro.csv", freegen_macro)
    actor = autonomous(qwen, tokenizer, model, device, cfg, seeds, args.output_dir)
    write_csv(args.output_dir / "actor_summary.csv", actor); write_markdown(args.output_dir / "summary.md", summary, candidate_macro, freegen_macro, actor)
    sanity = {"mode": "full" if args.full else "sanity", "states": len(states), "seeds": seeds, "first_base_prompt": actor_base_prompt(states[0].task, states[0].observation, qa_suffix(questions(states[0])[0].text, questions(states[0])[0].candidates)), "first_hz_prompt": actor_hz_prompt(states[0].task, qa_suffix(questions(states[0])[0].text, questions(states[0])[0].candidates)), "hz_soft_tokens": 10, "hz_textual_observation": False, "hz_textual_action_history": False, "hz_textual_reasoning_history": False, "posterior_contract": "z_t=q(z_t|h_t,o_t); observation is used only here", "transition_contract": "h_(t+1)=f(h_t,z_t,a_t); z_(t+1)=q(z_(t+1)|h_(t+1),o_(t+1))", "sample_question_ids": [q.question_id for q in questions(states[0])[:3]]}
    (args.output_dir / "sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "checkpoint": manifest, "states": len(states), "questions": len(rows), "mode": sanity["mode"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
