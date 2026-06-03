"""Return-conditioned trajectory planning (Decision Transformer style) for debug_square WM."""
from __future__ import annotations

import json
from typing import Sequence

from agent_system.environments.env_package.caged_craftext.action_tokens import (
    action_token_legend,
    format_action_token_sequence,
)
from agent_system.environments.env_package.caged_craftext.reward_tokens import (
    sum_quantized_rewards,
)


def get_planning_prompt_template(*, horizon: int) -> str:
    h = max(1, int(horizon))
    return f"""You are a planning world model (return-conditioned policy). Given the task, the current grid state, and a target total reward over the next {h} steps, predict the sequence of {h} action tokens that achieves that cumulative reward.

Action tokens: {{action_legend}}

**TASK:** {{task}}

State:
{{state}}

Target cumulative reward over next {h} steps (sum of per-step rewards, each in -1, 0, 1, or 2): {{target_return}}

Reply with exactly {h} action tokens from the action list above, with no spaces between them (token for step 1, then step 2, ..., step {h}):"""


def describe_planning_prompt_schema(*, horizon: int) -> str:
    """Human-readable description of the planning prompt template (for val panels)."""
    h = max(1, int(horizon))
    return (
        f"Template: return-conditioned planning (Decision Transformer style), H={h}.\n"
        f"Fields: TASK, State (s_t), Target cumulative return R̂, Action token legend.\n"
        f"Model must output exactly {h} action tokens concatenated without spaces."
    )


def format_planning_prompt(
    state: str,
    *,
    target_return: int,
    task: str = "",
    horizon: int = 1,
) -> str:
    h = max(1, int(horizon))
    return get_planning_prompt_template(horizon=h).format(
        action_legend=action_token_legend(),
        task=(task or "").strip() or "Unknown task",
        state=state,
        target_return=int(target_return),
    )


def format_planning_target(actions: Sequence[str]) -> str:
    """Concatenated action tokens for SFT (e.g. ``324156``)."""
    return format_action_token_sequence(actions)


def parse_future_json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def cumulative_return_from_rewards(rewards: Sequence[float]) -> int:
    return sum_quantized_rewards(rewards)


def planning_row_valid(
    *,
    horizon: int,
    planning_horizon: int,
    future_actions: Sequence[str],
    future_rewards: Sequence[float],
) -> bool:
    h = int(planning_horizon)
    if int(horizon) != h:
        return False
    acts = [str(a).strip().split()[0] for a in future_actions if str(a).strip()]
    if len(acts) != h:
        return False
    rews = list(future_rewards)
    if len(rews) != h:
        return False
    return True
