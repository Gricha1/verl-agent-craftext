#!/usr/bin/env python3
"""Build reward-WM train/val parquet from PPO rollout buffer (transitions.jsonl)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_transitions(buffer_dir: str) -> List[dict]:
    path = os.path.join(buffer_dir, "transitions.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No transitions buffer: {path}")
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Empty buffer: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reward-horizon", type=int, default=6)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_paths()
    sys.path.insert(0, os.path.join(_repo_root(), "scripts"))
    from collect_reward_wm_dataset_debug_square import (
        Row,
        _balance_horizons,
        _write_parquet,
    )
    from agent_system.environments.env_package.caged_craftext.reward_tokens import (
        format_reward_token_sequence,
        quantize_step_reward,
    )
    from agent_system.environments.prompts.world_model_reward import format_reward_prompt

    reward_horizon = max(1, int(args.reward_horizon))
    transitions = _load_transitions(args.buffer_dir)

    by_traj: Dict[str, List[dict]] = defaultdict(list)
    for t in transitions:
        uid = str(t.get("traj_uid", "") or "")
        if not uid:
            continue
        by_traj[uid].append(t)

    rows: List[Row] = []
    for uid in sorted(by_traj.keys()):
        steps = sorted(by_traj[uid], key=lambda x: int(x.get("seq", 0)))
        episode_steps = []
        for s in steps:
            episode_steps.append(
                {
                    "state": str(s.get("curr_obs_ascii", "") or ""),
                    "instruction": str(s.get("wm_task_instruction", "") or ""),
                    "action_token": str(s.get("wm_action_token", "") or "").strip().split()[0],
                    "reward": float(s.get("wm_step_reward", 0.0)),
                    "done": int(bool(s.get("done", 0))),
                }
            )
        task_slug = str(steps[0].get("task_slug", "") or "unknown")
        instruction_idx = int(steps[0].get("instruction_idx", -1))

        for i, step in enumerate(episode_steps):
            max_h = min(reward_horizon, len(episode_steps) - i)
            for h in range(1, max_h + 1):
                future = [float(episode_steps[i + j]["reward"]) for j in range(h)]
                future_acts = [str(episode_steps[i + j]["action_token"]) for j in range(h)]
                state_after = (
                    str(episode_steps[i + 1]["state"]) if i + 1 < len(episode_steps) else ""
                )
                rows.append(
                    Row(
                        prompt=format_reward_prompt(
                            state=str(step["state"]),
                            action=str(step["action_token"]),
                            task=str(step["instruction"]),
                            horizon=h,
                            actions=future_acts,
                        ),
                        response=format_reward_token_sequence(future),
                        task_slug=task_slug,
                        instruction_idx=instruction_idx,
                        instruction=str(step["instruction"]),
                        state=str(step["state"]),
                        action_token=str(step["action_token"]),
                        state_after=state_after,
                        reward=float(future[0]),
                        reward_q=int(quantize_step_reward(future[0])),
                        done=int(step["done"]),
                        horizon=int(h),
                        future_rewards=json.dumps(future),
                        future_actions=json.dumps(future_acts),
                    )
                )

    if not rows:
        raise RuntimeError("No WM rows built from rollout buffer")

    rng = np.random.RandomState(int(args.seed))
    if reward_horizon > 1:
        rows = _balance_horizons(rows, reward_horizon, rng)
    rng.shuffle(rows)
    val_n = int(round(len(rows) * float(args.val_frac)))
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)
    print(
        f"[OK] WM parquet from PPO buffer: train={len(train_rows)} val={len(val_rows)} "
        f"trajectories={len(by_traj)} -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
