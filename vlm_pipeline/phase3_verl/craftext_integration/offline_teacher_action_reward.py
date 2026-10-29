"""Outcome verifier for offline teacher-state reasoning-and-action RL.

Reward is granted only for a syntactically explicit, non-empty reasoning block
*and* a final action matching the frozen CNN+BERT teacher label.  This does not
claim to judge semantic quality of the reasoning; it prevents the previous
one-token-action shortcut.
"""

from __future__ import annotations

import re
from typing import Any

from craftext_integration.actions import parse_reasoning_action


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Score a Qwen response against a single teacher action label."""
    del data_source
    expected = ground_truth["teacher_action"] if isinstance(ground_truth, dict) else str(ground_truth)
    parsed = parse_reasoning_action(solution_str)
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", str(solution_str), flags=re.IGNORECASE | re.DOTALL)
    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
    has_reasoning = bool(reasoning_text)
    action_match = bool(parsed.valid and parsed.action_name == expected)
    reward = float(has_reasoning and action_match)
    probs = (extra_info or {}).get("teacher_action_probabilities", [])
    chosen_probability = float(probs[parsed.action_id]) if len(probs) > parsed.action_id else 0.0
    return {
        "score": reward,
        "action_accuracy": float(action_match),
        "parse_rate": float(parsed.valid),
        "reasoning_format_rate": float(has_reasoning),
        "reasoning_action_joint_rate": reward,
        "teacher_action_probability": chosen_probability,
        "reasoning_length": float(len(reasoning_text.split())),
        "response_length": float(len(str(solution_str).split())),
    }
