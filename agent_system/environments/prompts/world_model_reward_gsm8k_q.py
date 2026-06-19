"""GSM8K Monte-Carlo Q WM: (question, prefix, candidate step) -> eventual success j/k."""
from __future__ import annotations

import re
from typing import List

_GSM8K_REWARD_VALUES = (0, 1)
_GSM8K_Q_LEGEND = "j=unlikely to succeed, k=likely to succeed (eventual correct answer)"
_TOKEN_TO_REWARD = {"j": 0, "k": 1}


def _gsm8k_quantize(reward: float) -> int:
    r = float(reward)
    for v in _GSM8K_REWARD_VALUES:
        if abs(r - v) < 1e-3:
            return v
    return 1 if r >= 0.5 else 0


def format_gsm8k_reward_target(reward: float) -> str:
    return "j" if _gsm8k_quantize(reward) == 0 else "k"


def format_gsm8k_reward_target_display(reward: float) -> str:
    r = _gsm8k_quantize(reward)
    token = "j" if r == 0 else "k"
    return f"{r} ({token})"


def parse_gsm8k_reward_prediction(text: str) -> float | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    token = raw.split()[0]
    if token not in _TOKEN_TO_REWARD:
        last_line = raw.splitlines()[-1].strip()
        token = last_line.split()[0] if last_line.split() else last_line
    if token not in _TOKEN_TO_REWARD:
        return None
    return float(_TOKEN_TO_REWARD[token])


def split_gsm8k_reasoning_steps(solution: str, mode: str = "newline") -> List[str]:
    """Split a full GSM8K solution into candidate next-step strings."""
    text = str(solution or "").strip()
    if not text:
        return []

    mode = str(mode or "newline").lower()
    if mode == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if parts else [text]

    # default: newline — each line is one reasoning step
    parts = [p.strip() for p in text.splitlines() if p.strip()]
    return parts if parts else [text]


def format_gsm8k_mc_q_prompt(
    question: str,
    prefix: str,
    candidate_step: str,
    *,
    task: str = "",
) -> str:
    q = (task or question or "").strip() or "Unknown question"
    pref = str(prefix or "").strip()
    step = str(candidate_step or "").strip() or "(empty step)"
    pref_block = pref if pref else "(none — start of solution)"
    return f"""You are a Q-model for GSM8K math word problems.
Given the question, the solution written so far, and a candidate next reasoning step, predict whether following this path will eventually lead to a correct final answer.

Output tokens: {_GSM8K_Q_LEGEND}
(Use only j or k.)

Question:
{q}

Solution so far:
{pref_block}

Candidate next step:
{step}

Reply with exactly ONE token (j or k):"""


def expand_gsm8k_mc_q_examples(
    question: str,
    full_solution: str,
    final_reward: float,
    *,
    step_split: str = "newline",
    max_steps_per_traj: int = 32,
) -> List[tuple[str, str, str, float]]:
    """
    Monte-Carlo Q targets: every (prefix, step) in a rollout gets y = final episode reward.
    Returns list of (question, prefix, candidate_step, target_reward).
    """
    steps = split_gsm8k_reasoning_steps(full_solution, mode=step_split)
    if not steps:
        return []
    if max_steps_per_traj > 0:
        steps = steps[: int(max_steps_per_traj)]

    q = str(question or "").strip()
    out: List[tuple[str, str, str, float]] = []
    prefix = ""
    for step in steps:
        out.append((q, prefix, step, float(final_reward)))
        prefix = f"{prefix}\n{step}".strip() if prefix else step
    return out


__all__ = [
    "expand_gsm8k_mc_q_examples",
    "format_gsm8k_mc_q_prompt",
    "format_gsm8k_reward_target",
    "format_gsm8k_reward_target_display",
    "parse_gsm8k_reward_prediction",
    "split_gsm8k_reasoning_steps",
]
