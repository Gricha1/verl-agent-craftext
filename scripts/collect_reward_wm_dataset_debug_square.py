#!/usr/bin/env python3
"""
Collect a fixed offline dataset for reward world model on debug_square_8x8.

Outputs two parquet files (train/val) with columns:
  - prompt: reward WM prompt (task + ASCII state + single-token action)
  - response: reward token(s) (i/j/k/l), concatenated for multi-step horizons

When --reward-horizon > 1, each timestep can produce up to H rows (1..H step
lookahead). Rows are balanced so horizons 1..H have equal counts.

Optional extra columns (useful for analysis):
  task_slug, instruction_idx, instruction, state, action_token, reward, reward_q,
  done, horizon, future_rewards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


DEBUG_SQUARE_TASKS: Tuple[Tuple[int, str], ...] = (
    (0, "stone"),
    (1, "wood"),
    (2, "water"),
)

# Fixed target cells for debug_square_8x8 (row, col).
_DEBUG_SQUARE_TARGET_BY_INSTRUCTION: Dict[int, Tuple[int, int]] = {
    0: (1, 1),  # stone
    1: (1, 6),  # wood
    2: (6, 1),  # water
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class Row:
    prompt: str
    response: str
    task_slug: str
    instruction_idx: int
    instruction: str
    state: str
    action_token: str
    state_after: str
    reward: float
    reward_q: int
    done: int
    horizon: int
    future_rewards: str
    future_actions: str


def _write_parquet(rows, out_path: str) -> None:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to write parquet") from exc

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not rows:
        return
    if hasattr(rows[0], "__dataclass_fields__"):
        df = pd.DataFrame([asdict(r) for r in rows])
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)


def _balance_horizons(rows: List[Row], reward_horizon: int, rng: np.random.RandomState) -> List[Row]:
    if int(reward_horizon) <= 1:
        return rows
    by_h: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        by_h[int(r.horizon)].append(r)
    counts = {h: len(by_h.get(h, [])) for h in range(1, int(reward_horizon) + 1)}
    if min(counts.values()) <= 0:
        missing = [h for h, n in counts.items() if n <= 0]
        raise RuntimeError(f"Missing horizon rows after collection: {counts}; missing={missing}")
    target = min(counts.values())
    balanced: List[Row] = []
    for h in range(1, int(reward_horizon) + 1):
        pool = list(by_h[h])
        rng.shuffle(pool)
        balanced.extend(pool[:target])
    rng.shuffle(balanced)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect offline reward-WM dataset (debug_square_8x8).")
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="Output dir for train/val parquet. Default: data/reward_wm_debug_square_8x8_h{reward_horizon}",
    )
    parser.add_argument("--episodes-per-task", default=200, type=int)
    parser.add_argument("--max-steps", default=50, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--keep-extra-columns", action="store_true", help="Keep analysis columns beyond prompt/response.")
    parser.add_argument(
        "--reward-horizon",
        default=1,
        type=int,
        help="Max reward lookahead steps. 1=single token (default). >1 emits balanced 1..H-step rows.",
    )
    parser.add_argument(
        "--policy",
        default="semi",
        choices=("random", "semi"),
        help="Action source for data collection: random or semi-greedy-to-goal to balance rewards.",
    )
    parser.add_argument(
        "--epsilon",
        default=0.15,
        type=float,
        help="Exploration probability for semi policy (pick random action).",
    )
    args = parser.parse_args()

    reward_horizon = max(1, int(args.reward_horizon))
    out_dir = str(args.out_dir or f"data/reward_wm_debug_square_8x8_h{reward_horizon}")

    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _setup_paths()

    from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
    from agent_system.environments.env_package.caged_craftext.projection import craftext_projection, ACTION_TO_TEXT
    from agent_system.environments.env_package.caged_craftext.reward_tokens import (
        format_reward_token_sequence,
        quantize_step_reward,
        reward_to_token,
    )
    from agent_system.environments.prompts.world_model_reward import format_reward_prompt

    rng = np.random.RandomState(int(args.seed))
    action_tokens = list(action_token_strings())
    if not action_tokens:
        raise RuntimeError("No action tokens found (action_token_strings() returned empty).")

    _ACTION_ID = {name: i for i, name in enumerate(ACTION_TO_TEXT)}
    MOVE_IDS = {
        "LEFT": int(_ACTION_ID["LEFT"]),
        "RIGHT": int(_ACTION_ID["RIGHT"]),
        "UP": int(_ACTION_ID["UP"]),
        "DOWN": int(_ACTION_ID["DOWN"]),
        "NOOP": int(_ACTION_ID["NOOP"]),
    }

    def choose_action_id_semi(
        *,
        rng_: np.random.RandomState,
        instruction_idx: int,
        worker: CagedCraftextWorker,
        counts: Dict[int, int],
        epsilon: float,
    ) -> int:
        if rng_.rand() < float(epsilon):
            tok = str(rng_.choice(action_tokens))
            return int(craftext_projection([tok])[0][0])

        target = _DEBUG_SQUARE_TARGET_BY_INSTRUCTION.get(int(instruction_idx))
        if target is None:
            tok = str(rng_.choice(action_tokens))
            return int(craftext_projection([tok])[0][0])

        st = worker.get_state()
        if st is None:
            return MOVE_IDS["NOOP"]
        try:
            pos = np.asarray(st.env_state.player_position).reshape(-1)
            pr, pc = int(pos[0]), int(pos[1])
        except Exception:
            return MOVE_IDS["NOOP"]

        tr, tc = int(target[0]), int(target[1])
        dr = tr - pr
        dc = tc - pc
        want = min((-1, 0, 1), key=lambda k: counts.get(k, 0))
        dist = abs(dr) + abs(dc)
        if dist <= 3 and counts.get(2, 0) < min(counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)) * 0.2:
            want = 1

        if want == 0:
            return MOVE_IDS["NOOP"]

        toward: List[int] = []
        away: List[int] = []
        if dr < 0:
            toward.append(MOVE_IDS["UP"])
            away.append(MOVE_IDS["DOWN"])
        elif dr > 0:
            toward.append(MOVE_IDS["DOWN"])
            away.append(MOVE_IDS["UP"])
        if dc < 0:
            toward.append(MOVE_IDS["LEFT"])
            away.append(MOVE_IDS["RIGHT"])
        elif dc > 0:
            toward.append(MOVE_IDS["RIGHT"])
            away.append(MOVE_IDS["LEFT"])

        pool = toward if want == 1 else away
        if not pool:
            return MOVE_IDS["NOOP"]
        return int(rng_.choice(pool))

    rows: List[Row] = []
    counts_all: Dict[int, int] = {-1: 0, 0: 0, 1: 0, 2: 0}

    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
        "encode_form": "embedding",
    }

    for instruction_idx, task_slug in DEBUG_SQUARE_TASKS:
        worker = CagedCraftextWorker(
            seed=int(args.seed) + instruction_idx * 100000,
            env_kwargs=env_kwargs,
        )
        for _ep in range(int(args.episodes_per_task)):
            _, info = worker.reset(scenario_idx=instruction_idx, return_render=False)
            episode_steps: List[dict] = []

            for _t in range(int(args.max_steps)):
                state_ascii = str(info.get("text_render", "") or "")
                instruction = str(info.get("instruction", "") or "")

                if str(args.policy) == "semi":
                    action_id = choose_action_id_semi(
                        rng_=rng,
                        instruction_idx=instruction_idx,
                        worker=worker,
                        counts=counts_all,
                        epsilon=float(args.epsilon),
                    )
                    from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_label

                    token = action_token_label(int(action_id))
                else:
                    token = str(rng.choice(action_tokens))
                    action_ids, _valids = craftext_projection([token])
                    action_id = int(action_ids[0])

                _, reward, done, next_info = worker.step(action_id, return_render=False)
                reward_f = float(reward)
                rq = int(quantize_step_reward(reward_f))
                counts_all[rq] = counts_all.get(rq, 0) + 1

                episode_steps.append(
                    {
                        "state": state_ascii,
                        "instruction": instruction,
                        "action_token": token,
                        "reward": reward_f,
                        "reward_q": rq,
                        "done": int(bool(done)),
                    }
                )
                info = next_info
                if bool(done):
                    break

            for i, step in enumerate(episode_steps):
                max_h = min(int(reward_horizon), len(episode_steps) - i)
                for h in range(1, max_h + 1):
                    future = [float(episode_steps[i + j]["reward"]) for j in range(h)]
                    future_acts = [str(episode_steps[i + j]["action_token"]) for j in range(h)]
                    state_after = (
                        str(episode_steps[i + 1]["state"]) if i + 1 < len(episode_steps) else ""
                    )
                    response = format_reward_token_sequence(future)
                    prompt = format_reward_prompt(
                        state=str(step["state"]),
                        action=str(step["action_token"]),
                        task=str(step["instruction"]),
                        horizon=h,
                        actions=future_acts,
                    )
                    rows.append(
                        Row(
                            prompt=prompt,
                            response=response,
                            task_slug=str(task_slug),
                            instruction_idx=int(instruction_idx),
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

        worker.close()

    if not rows:
        raise RuntimeError("No rows collected.")

    if reward_horizon > 1:
        rows = _balance_horizons(rows, reward_horizon, rng)

    rng.shuffle(rows)
    val_n = int(round(len(rows) * float(args.val_frac)))
    val_n = max(1, min(val_n, len(rows) - 1))
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    if not args.keep_extra_columns:
        def strip(r: Row) -> Row:
            return Row(
                prompt=r.prompt,
                response=r.response,
                task_slug="",
                instruction_idx=0,
                instruction=str(r.instruction),
                state=str(r.state),
                action_token=str(r.action_token),
                state_after=str(r.state_after),
                reward=0.0,
                reward_q=0,
                done=0,
                horizon=int(r.horizon),
                future_rewards=str(r.future_rewards),
                future_actions=str(r.future_actions),
            )

        train_rows = [strip(r) for r in train_rows]
        val_rows = [strip(r) for r in val_rows]

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "val.parquet")
    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)

    def _counts(rows_: List[Row]) -> Dict[int, int]:
        c: Dict[int, int] = {-1: 0, 0: 0, 1: 0, 2: 0}
        for r in rows_:
            try:
                q = int(getattr(r, "reward_q", 0))
            except Exception:
                q = 0
            c[q] = c.get(q, 0) + 1
        return c

    def _horizon_counts(rows_: List[Row]) -> Dict[int, int]:
        c: Dict[int, int] = defaultdict(int)
        for r in rows_:
            c[int(getattr(r, "horizon", 1))] += 1
        return dict(c)

    def _fmt_counts(c: Dict[int, int]) -> str:
        return " ".join(f"{k}:{c.get(k, 0)}" for k in (-1, 0, 1, 2))

    def _fmt_fracs(c: Dict[int, int]) -> str:
        total_ = float(sum(c.values()))
        if total_ <= 0:
            return " ".join(f"{k}:nan" for k in (-1, 0, 1, 2))
        return " ".join(f"{k}:{c.get(k, 0) / total_:.3f}" for k in (-1, 0, 1, 2))

    counts_train = _counts(train_rows)
    counts_val = _counts(val_rows)
    h_train = _horizon_counts(train_rows)
    h_val = _horizon_counts(val_rows)
    total = sum(counts_all.values())
    print(f"[OK] Wrote dataset: train={len(train_rows)} val={len(val_rows)} total={len(rows)} -> {out_dir}")
    print(f"[OK] reward_horizon={reward_horizon}  TRAIN horizon counts={dict(sorted(h_train.items()))}")
    print(f"[OK] reward_horizon={reward_horizon}  VAL   horizon counts={dict(sorted(h_val.items()))}")
    if total:
        print(f"[OK] Reward quantized distribution ALL   (counts):   {_fmt_counts(counts_all)}")
        print(f"[OK] Reward quantized distribution ALL   (fractions): {_fmt_fracs(counts_all)}")
        print(f"[OK] Reward quantized distribution TRAIN (counts):   {_fmt_counts(counts_train)}")
        print(f"[OK] Reward quantized distribution TRAIN (fractions): {_fmt_fracs(counts_train)}")
        print(f"[OK] Reward quantized distribution VAL   (counts):   {_fmt_counts(counts_val)}")
        print(f"[OK] Reward quantized distribution VAL   (fractions): {_fmt_fracs(counts_val)}")


if __name__ == "__main__":
    main()
