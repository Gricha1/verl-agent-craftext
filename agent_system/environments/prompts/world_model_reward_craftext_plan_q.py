"""Craftext multi-horizon plan-Q: (task, state, action plan) -> return token A..o."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

from agent_system.environments.env_package.caged_craftext.action_tokens import (
    TOKEN_TO_ACTION_ID,
    normalize_action_token,
)
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT
from agent_system.environments.env_package.caged_craftext.return_tokens import (
    ReturnBinSpec,
    compute_remaining_returns,
    return_token_legend_for_spec,
    return_to_token,
)


def format_plan_actions_display(actions: Sequence[str]) -> str:
    """Human-readable plan line: ``2=LEFT, 4=UP, 6=DO``."""
    parts: List[str] = []
    for raw in actions:
        tok = normalize_action_token(raw)
        if not tok:
            continue
        aid = TOKEN_TO_ACTION_ID.get(tok)
        name = ACTION_TO_TEXT[aid] if aid is not None and 0 <= aid < len(ACTION_TO_TEXT) else tok
        parts.append(f"{tok}={name}")
    return ", ".join(parts) if parts else "(empty plan)"


def format_craftext_plan_q_prompt(
    *,
    task: str,
    state: str,
    plan_actions: Sequence[str],
    return_bin_legend: str,
    constraint: str = "",
    executed_actions: Sequence[str] | None = None,
) -> str:
    from agent_system.environments.env_package.caged_craftext.projection import (
        format_executed_actions_history,
    )

    plan_display = format_plan_actions_display(plan_actions)
    executed_display = format_executed_actions_history(executed_actions or [])
    h = len([a for a in plan_actions if normalize_action_token(a)])
    prompt = f"""Your goal is to complete the following task:
**TASK:** {task or "No task"}

Actions already taken in this episode (oldest → newest):
{executed_display}

This is what you currently see:
{state or "(empty observation)"}

Candidate plan ({h} action{"s" if h != 1 else ""}):
{plan_display}

Estimate the expected return after executing this candidate plan, including rewards collected during the plan and the value of the resulting state.

Reply with exactly ONE token — your return estimate (no explanation):
{return_bin_legend}"""
    if constraint:
        prompt += f"\n\n**CONSTRAINT:** {constraint}"
    return prompt


def compute_nstep_plan_target(
    rewards: Sequence[float],
    start_t: int,
    horizon: int,
    *,
    gamma: float = 1.0,
) -> float:
    """
    n-step plan target:
      sum_{i=0}^{h-1} gamma^i r_{t+i} + gamma^h G_{t+h}
    where G_{t+h} is the MC remaining return from step t+h.
    """
    T = len(rewards)
    if start_t >= T or horizon <= 0:
        return 0.0

    h = min(int(horizon), T - start_t)
    nstep = 0.0
    g = 1.0
    for i in range(h):
        nstep += g * float(rewards[start_t + i])
        g *= float(gamma)

    bootstrap_t = start_t + h
    if bootstrap_t < T:
        remaining = compute_remaining_returns(rewards)
        nstep += g * float(remaining[bootstrap_t])
    return nstep


def expand_craftext_plan_q_examples(
    *,
    traj_uids: Sequence[str],
    states: Sequence[str],
    action_tokens: Sequence[str],
    step_rewards: Sequence[float],
    task_instructions: Sequence[str],
    plan_horizon: int = 6,
    gamma: float = 1.0,
    max_steps_per_traj: int = 50,
    fixed_horizon_only: bool = False,
    return_spec: ReturnBinSpec | None = None,
) -> List[tuple[str, float]]:
    """
    Multi-horizon plan-Q rows from on-policy rollouts.
    For each trajectory start t and horizon h in 1..plan_horizon:
      (s_t, [a_t, ..., a_{t+h-1}]) -> n-step return target.
    """
    from agent_system.environments.env_package.caged_craftext.return_tokens import DEFAULT_RETURN_BIN_SPEC

    spec = return_spec or DEFAULT_RETURN_BIN_SPEC
    legend = return_token_legend_for_spec(spec)
    max_h = max(1, int(plan_horizon))

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
        actions = [s[2] for s in steps]
        T = len(steps)

        for t in range(T):
            task = steps[t][4]
            if fixed_horizon_only:
                h_values = [max_h] if t + max_h <= T else []
            else:
                h_values = range(1, max_h + 1)
            for h in h_values:
                if t + h > T:
                    break
                plan = actions[t : t + h]
                target = compute_nstep_plan_target(rewards, t, h, gamma=gamma)
                prompt = format_craftext_plan_q_prompt(
                    task=task,
                    state=steps[t][1],
                    plan_actions=plan,
                    return_bin_legend=legend,
                    executed_actions=actions[:t],
                )
                rows.append((prompt, float(target)))

    return rows


def format_craftext_plan_q_target(reward: float, *, spec: ReturnBinSpec | None = None) -> str:
    from agent_system.environments.env_package.caged_craftext.return_tokens import DEFAULT_RETURN_BIN_SPEC

    return return_to_token(float(reward), spec=spec or DEFAULT_RETURN_BIN_SPEC)


__all__ = [
    "compute_nstep_plan_target",
    "expand_craftext_plan_q_examples",
    "format_craftext_plan_q_prompt",
    "format_craftext_plan_q_target",
    "format_plan_actions_display",
]
