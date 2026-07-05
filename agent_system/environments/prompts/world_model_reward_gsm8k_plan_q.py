"""GSM8K plan-Q: (question, first H reasoning steps) -> return bin token."""
from __future__ import annotations

from typing import List, Sequence

from agent_system.environments.env_package.caged_craftext.return_tokens import (
    ReturnBinSpec,
    return_token_legend_for_spec,
)
from agent_system.environments.prompts.world_model_reward_gsm8k_q import (
    split_gsm8k_reasoning_steps,
)


def format_gsm8k_plan_q_prompt(
    *,
    question: str,
    plan_steps: Sequence[str],
    return_bin_legend: str,
) -> str:
    steps = [str(s).strip() for s in plan_steps if str(s).strip()]
    h = len(steps)
    if h == 0:
        plan_block = "(empty plan)"
    else:
        plan_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    q = (question or "").strip() or "Unknown question"
    return f"""You are a plan-Q model for GSM8K math word problems.
Given the question and a candidate plan (the first {h} reasoning steps), estimate the expected return if the solution follows this plan through to completion.

Question:
{q}

Candidate plan ({h} step{"s" if h != 1 else ""}):
{plan_block}

Reply with exactly ONE token — your return estimate (no explanation):
{return_bin_legend}"""


def expand_gsm8k_plan_q_examples(
    *,
    questions: Sequence[str],
    solutions: Sequence[str],
    final_rewards: Sequence[float],
    plan_horizon: int = 100,
    step_split: str = "newline",
    fixed_horizon_only: bool = True,
    return_spec: ReturnBinSpec | None = None,
) -> List[tuple[str, float]]:
    """
    One plan-Q row per rollout sample at horizon plan_horizon (or all steps if shorter).
    Target = final episode score (0/1 or format partial).
    """
    from agent_system.environments.env_package.caged_craftext.return_tokens import DEFAULT_RETURN_BIN_SPEC

    spec = return_spec or DEFAULT_RETURN_BIN_SPEC
    legend = return_token_legend_for_spec(spec)
    max_h = max(1, int(plan_horizon))

    rows: List[tuple[str, float]] = []
    for question, solution, reward in zip(questions, solutions, final_rewards):
        q = str(question or "").strip()
        sol = str(solution or "").strip()
        if not q or not sol:
            continue
        steps = split_gsm8k_reasoning_steps(sol, mode=step_split)
        if not steps:
            continue
        if fixed_horizon_only:
            if len(steps) < max_h:
                continue
            plan = steps[:max_h]
        else:
            plan = steps[: min(max_h, len(steps))]
        if not plan:
            continue
        prompt = format_gsm8k_plan_q_prompt(
            question=q,
            plan_steps=plan,
            return_bin_legend=legend,
        )
        rows.append((prompt, float(reward)))
    return rows


__all__ = [
    "expand_gsm8k_plan_q_examples",
    "format_gsm8k_plan_q_prompt",
]
