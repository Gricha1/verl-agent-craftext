"""Prompt template for inverse-action world model (predict action from s, s')."""
from __future__ import annotations

from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_legend


def get_inverse_action_prompt_template() -> str:
    legend = action_token_legend()
    return f"""You are a world model. Given the grid world state before and after one step, predict the action token that was taken.

Action tokens: {legend}

State before:
{{state_before}}

State after:
{{state_after}}

Reply with exactly ONE action token (no explanation):"""


def format_inverse_action_prompt(state_before: str, state_after: str) -> str:
    return get_inverse_action_prompt_template().format(
        state_before=state_before,
        state_after=state_after,
    )


def is_unchanged_transition(state_before: str, state_after: str) -> bool:
    """True when s_t and s_{t+1} text renders are identical (no-op / ineffective actions)."""
    return str(state_before or "").strip() == str(state_after or "").strip()
