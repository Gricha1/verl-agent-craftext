"""Reward WM prompt for GSM8K: (question, full solution) -> j/k (0/1)."""
from __future__ import annotations

from agent_system.environments.env_package.caged_craftext.reward_tokens import (
    INVALID_REWARD,
    parse_reward_token,
    quantize_step_reward,
)

_GSM8K_REWARD_VALUES = (0, 1)


def _gsm8k_quantize(reward: float) -> int:
    r = quantize_step_reward(float(reward))
    if r in _GSM8K_REWARD_VALUES:
        return r
    return 1 if float(reward) >= 0.5 else 0

_GSM8K_REWARD_LEGEND = "j=0 (incorrect), k=1 (correct)"


def get_gsm8k_reward_prompt_template() -> str:
    return f"""You are a reward model for GSM8K math word problems.
Given the question and the model's full solution, predict whether the solution is correct.

Reward tokens: {_GSM8K_REWARD_LEGEND}
(Use only j or k — not digits from the solution.)

Question:
{{task}}

Solution:
{{action}}

Reply with exactly ONE reward token (j or k):"""


def format_gsm8k_reward_prompt(state: str, action: str, task: str = "") -> str:
    question = (task or state or "").strip() or "Unknown question"
    solution = str(action or "").strip() or "(empty solution)"
    return get_gsm8k_reward_prompt_template().format(task=question, action=solution)


def format_gsm8k_reward_target(reward: float) -> str:
    return "j" if _gsm8k_quantize(reward) == 0 else "k"


def format_gsm8k_reward_target_display(reward: float) -> str:
    r = _gsm8k_quantize(reward)
    token = "j" if r == 0 else "k"
    return f"{r} ({token})"


def parse_gsm8k_reward_prediction(text: str) -> float | None:
    value = parse_reward_token(text)
    if value == INVALID_REWARD or value not in _GSM8K_REWARD_VALUES:
        return None
    return float(value)
