"""Prompt template for reward world model (predict r_t token from task, s_t, a_t)."""
from __future__ import annotations

from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_legend
from agent_system.environments.env_package.caged_craftext.reward_tokens import (
    INVALID_REWARD,
    parse_reward_token,
    quantize_step_reward,
    reward_to_token,
    reward_token_legend,
)


def get_reward_prompt_template() -> str:
    action_legend = action_token_legend()
    reward_legend = reward_token_legend()
    return f"""You are a reward model. Given the current task, grid world state, and the action token taken, predict the immediate reward token for that action.

Action tokens: {action_legend}

Reward tokens: {reward_legend}

**TASK:** {{task}}

State:
{{state}}

Action taken: {{action}}

Reply with exactly ONE reward token from the reward tokens list above (i, j, k, or l — not action digits 1-9):"""


def format_reward_prompt(state: str, action: str, task: str = "") -> str:
    token = str(action).strip().split()[0] if str(action).strip() else str(action)
    return get_reward_prompt_template().format(
        state=state,
        action=token,
        task=(task or "").strip() or "Unknown task",
    )


def format_reward_target(reward: float) -> str:
    """Single-token target for SFT (i/j/k/l)."""
    return reward_to_token(reward)


def parse_reward_prediction(text: str) -> float | None:
    """Decode model output token to numeric reward, or None if invalid."""
    value = parse_reward_token(text)
    if value == INVALID_REWARD:
        return None
    return float(value)


def format_reward_target_display(reward: float) -> str:
    token = reward_to_token(reward)
    return f"{quantize_step_reward(reward)} ({token})"
