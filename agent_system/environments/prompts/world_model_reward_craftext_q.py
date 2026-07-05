"""Craftext Monte-Carlo Q WM: (task, state, candidate action) -> return token A..o."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

from agent_system.environments.env_package.caged_craftext.action_tokens import parse_single_token_action
from agent_system.environments.env_package.caged_craftext.projection import (
    ACTION_TO_TEXT,
    format_per_action_return_prompt,
)
from agent_system.environments.env_package.caged_craftext.return_tokens import (
    ReturnBinSpec,
    return_token_legend_for_spec,
    return_to_token,
)


def _action_name_from_token(action_token: str) -> str:
    token = str(action_token or "").strip().split()[0] if str(action_token or "").strip() else "?"
    action_id = parse_single_token_action(token)
    if 0 <= action_id < len(ACTION_TO_TEXT):
        return ACTION_TO_TEXT[action_id]
    return token


def format_craftext_mc_q_prompt(
    *,
    task: str,
    state: str,
    candidate_action: str,
    return_bin_legend: str,
    constraint: str = "",
) -> str:
    """Same layout as actor/critic (per-action return template), one candidate action."""
    token = str(candidate_action or "").strip().split()[0] if str(candidate_action or "").strip() else "?"
    return format_per_action_return_prompt(
        task_description=task,
        current_observation=state,
        action_name=_action_name_from_token(token),
        action_token=token,
        return_bin_legend=return_bin_legend,
        constraint=constraint,
    )


def compute_mc_returns(step_rewards: Sequence[float]) -> List[float]:
    """Undiscounted Monte-Carlo return G_t = sum_{k>=t} r_k for each step."""
    rewards = [float(r) for r in step_rewards]
    out: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running += r
        out.append(running)
    out.reverse()
    return out


def expand_craftext_mc_q_examples(
    *,
    traj_uids: Sequence[str],
    states: Sequence[str],
    action_tokens: Sequence[str],
    step_rewards: Sequence[float],
    task_instructions: Sequence[str],
    max_steps_per_traj: int = 50,
    return_spec: ReturnBinSpec | None = None,
) -> List[tuple[str, float]]:
    """
    Monte-Carlo Q targets from on-policy rollout transitions.
    Returns list of (prompt_text, mc_return_scalar) for SFT.
    """
    from agent_system.environments.env_package.caged_craftext.return_tokens import DEFAULT_RETURN_BIN_SPEC

    spec = return_spec or DEFAULT_RETURN_BIN_SPEC
    legend = return_token_legend_for_spec(spec)

    grouped: dict[str, list[tuple[int, str, str, float, str]]] = defaultdict(list)
    for idx, (uid, state, action, reward, task) in enumerate(
        zip(traj_uids, states, action_tokens, step_rewards, task_instructions)
    ):
        if not (uid and state and action):
            continue
        grouped[str(uid)].append((idx, str(state), str(action).strip(), float(reward), str(task or "")))

    rows: List[tuple[str, float]] = []
    for _uid, steps in grouped.items():
        steps.sort(key=lambda x: x[0])
        if max_steps_per_traj > 0:
            steps = steps[: int(max_steps_per_traj)]
        if not steps:
            continue

        rewards = [s[3] for s in steps]
        mc_returns = compute_mc_returns(rewards)
        for t, (_idx, state, action, _r, task) in enumerate(steps):
            prompt = format_craftext_mc_q_prompt(
                task=task,
                state=state,
                candidate_action=action,
                return_bin_legend=legend,
            )
            rows.append((prompt, float(mc_returns[t])))

    return rows


def format_craftext_mc_q_target(reward: float, *, spec: ReturnBinSpec | None = None) -> str:
    from agent_system.environments.env_package.caged_craftext.return_tokens import DEFAULT_RETURN_BIN_SPEC

    return return_to_token(float(reward), spec=spec or DEFAULT_RETURN_BIN_SPEC)


__all__ = [
    "compute_mc_returns",
    "expand_craftext_mc_q_examples",
    "format_craftext_mc_q_prompt",
    "format_craftext_mc_q_target",
]
