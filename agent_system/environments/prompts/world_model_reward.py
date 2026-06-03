"""Prompt template for reward world model (predict reward token(s) from task, s_t, action(s))."""
from __future__ import annotations

from typing import Sequence

from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_legend
from agent_system.environments.env_package.caged_craftext.reward_tokens import (
    INVALID_REWARD,
    format_reward_token_sequence,
    parse_reward_token,
    parse_reward_token_sequence,
    quantize_step_reward,
    reward_to_token,
    reward_token_legend,
)


def _horizon_reply_instruction(horizon: int) -> str:
    h = max(1, int(horizon))
    if h == 1:
        return (
            "Reply with exactly ONE reward token from the reward tokens list above "
            "(i, j, k, or l — not action digits 1-9):"
        )
    return (
        f"Reply with exactly {h} reward tokens from the reward tokens list above "
        f"(i, j, k, or l — not action digits 1-9), with no spaces between them. "
        f"The first token is the reward after this action, the second after the next step, "
        f"and so on for {h} steps:"
    )


def get_reward_prompt_template(*, horizon: int = 1) -> str:
    action_legend = action_token_legend()
    reward_legend = reward_token_legend()
    h = max(1, int(horizon))
    if h == 1:
        task_line = (
            "You are a reward model. Given the current task, grid world state, and the action token taken, "
            "predict the immediate reward token for that action."
        )
    else:
        task_line = (
            "You are a reward model. Given the current task, grid world state, and the next "
            f"{h} action tokens from the policy, predict the reward token after each corresponding action "
            f"(rewards for step 1 through step {h})."
        )
    if h == 1:
        action_block = "Action taken: {action}"
    else:
        action_block = f"Actions (next {{h}} steps): {{actions}}"
    return f"""{task_line}

Action tokens: {action_legend}

Reward tokens: {reward_legend}

**TASK:** {{task}}

State:
{{state}}

{action_block}

{_horizon_reply_instruction(h)}"""


def _normalize_action_tokens(action: str, actions: Sequence[str] | None, *, horizon: int) -> list[str]:
    h = max(1, int(horizon))
    if actions is not None and len(actions) > 0:
        tokens = [str(a).strip().split()[0] if str(a).strip() else str(a) for a in actions]
    else:
        token = str(action).strip().split()[0] if str(action).strip() else str(action)
        tokens = [token]
    if h > 1 and len(tokens) < h:
        raise ValueError(
            f"format_reward_prompt(horizon={h}) requires {h} action tokens, got {len(tokens)}"
        )
    return tokens[:h]


def format_reward_prompt(
    state: str,
    action: str,
    task: str = "",
    *,
    horizon: int = 1,
    actions: Sequence[str] | None = None,
) -> str:
    h = max(1, int(horizon))
    tokens = _normalize_action_tokens(action, actions, horizon=h)
    return get_reward_prompt_template(horizon=h).format(
        state=state,
        action=tokens[0],
        actions=", ".join(tokens),
        h=h,
        task=(task or "").strip() or "Unknown task",
    )


def format_reward_target(reward: float) -> str:
    """Single-token target for SFT (i/j/k/l)."""
    return reward_to_token(reward)


def format_reward_target_sequence(rewards) -> str:
    """Multi-token target for SFT (e.g. ``ijk``). EOS is appended by SFTDataset."""
    return format_reward_token_sequence(rewards)


def parse_reward_prediction(text: str) -> float | None:
    """Decode model output token to numeric reward, or None if invalid."""
    value = parse_reward_token(text)
    if value == INVALID_REWARD:
        return None
    return float(value)


def parse_reward_prediction_sequence(text: str) -> tuple[float, ...]:
    """Decode concatenated model output like ``ijk`` into numeric rewards."""
    return tuple(float(v) for v in parse_reward_token_sequence(text))


def format_reward_target_display(reward: float) -> str:
    token = reward_to_token(reward)
    return f"{quantize_step_reward(reward)} ({token})"
