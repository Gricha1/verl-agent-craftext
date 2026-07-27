"""AlfWorld multi-horizon plan-Q: (task, state, action plan) -> return bin token."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

from agent_system.environments.env_package.caged_craftext.return_tokens import (
    DEFAULT_RETURN_BIN_SPEC,
    ReturnBinSpec,
    return_token_legend_for_spec,
)
from agent_system.environments.prompts.world_model_reward_craftext_plan_q import (
    compute_nstep_plan_target,
)


def extract_alfworld_action(text: str) -> str:
    """Parse action text: ``<action>...</action>`` or already-projected command string."""
    s = str(text or "").strip()
    if not s:
        return ""
    lower = s.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lower.find(start_tag)
    end_idx = lower.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return s[start_idx + len(start_tag) : end_idx].strip()
    # single_token_action: rollout stores projected admissible command (no XML tags).
    return s


def format_alfworld_plan_q_prompt(
    *,
    task: str,
    state: str,
    plan_actions: Sequence[str],
    return_bin_legend: str,
) -> str:
    steps = [str(a).strip() for a in plan_actions if str(a).strip()]
    h = len(steps)
    if h == 0:
        plan_block = "(empty plan)"
    else:
        plan_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return f"""You are a plan-Q model for AlfWorld embodied tasks.
Given the task, current observation, and a candidate plan (the next {h} actions), estimate the expected return after executing this plan.

Task:
{task or "Unknown task"}

Current observation:
{state or "(empty observation)"}

Candidate plan ({h} action{"s" if h != 1 else ""}):
{plan_block}

Reply with exactly ONE token — your return estimate (no explanation):
{return_bin_legend}"""


def expand_alfworld_plan_q_examples(
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
    parsed_actions = [extract_alfworld_action(a) for a in action_tokens]
    spec = return_spec or DEFAULT_RETURN_BIN_SPEC
    legend = return_token_legend_for_spec(spec)

    grouped: dict[str, list[tuple[int, str, str, float, str]]] = defaultdict(list)
    for idx, (uid, state, action, reward, task) in enumerate(
        zip(traj_uids, states, parsed_actions, step_rewards, task_instructions)
    ):
        if not (uid and state and action):
            continue
        grouped[str(uid)].append((idx, str(state), str(action).strip(), float(reward), str(task or "")))

    rows: List[tuple[str, float]] = []
    max_h = max(1, int(plan_horizon))
    for _uid, steps in grouped.items():
        steps.sort(key=lambda x: x[0])
        if max_steps_per_traj > 0:
            steps = steps[: int(max_steps_per_traj)]
        if not steps:
            continue
        rewards = [s[3] for s in steps]
        actions = [s[2] for s in steps]
        task = steps[0][4]
        T = len(steps)
        for t in range(T):
            if fixed_horizon_only:
                h_values = [max_h] if t + max_h <= T else []
            else:
                h_values = range(1, max_h + 1)
            for h in h_values:
                if t + h > T:
                    break
                plan = actions[t : t + h]
                target = compute_nstep_plan_target(rewards, t, h, gamma=gamma)
                prompt = format_alfworld_plan_q_prompt(
                    task=task,
                    state=steps[t][1],
                    plan_actions=plan,
                    return_bin_legend=legend,
                )
                rows.append((prompt, float(target)))
    return rows


__all__ = [
    "expand_alfworld_plan_q_examples",
    "extract_alfworld_action",
    "format_alfworld_plan_q_prompt",
]
