"""Online planning WM for Craftext: (task, state) -> H action tokens from rollout windows."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

from agent_system.environments.env_package.caged_craftext.action_tokens import (
    TOKEN_TO_ACTION_ID,
    normalize_action_token,
)
from agent_system.environments.prompts.world_model_planning import (
    format_max_return_planning_prompt,
    format_planning_target,
    format_validation_planning_prompt,
)


def expand_craftext_planning_advantage_examples(
    *,
    traj_uids: Sequence[str],
    states: Sequence[str],
    action_tokens: Sequence[str],
    step_rewards: Sequence[float],
    task_instructions: Sequence[str],
    plan_horizon: int = 6,
    max_steps_per_traj: int = 50,
    prompt_style: str = "validation",
) -> List[Tuple[str, str, str, Tuple[str, ...]]]:
    """
    Planning-advantage rows: (planner_prompt, state, task, rollout_plan_tokens).
    g_data is scored later via plan-Q on ``rollout_plan_tokens`` (actual H-step window).
    """
    h = max(1, int(plan_horizon))
    style = str(prompt_style or "validation").lower()

    grouped: dict[str, list[tuple[int, str, str, float, str]]] = defaultdict(list)
    for idx, (uid, state, action, reward, task) in enumerate(
        zip(traj_uids, states, action_tokens, step_rewards, task_instructions)
    ):
        tok = normalize_action_token(action)
        if not (uid and state and tok and tok in TOKEN_TO_ACTION_ID):
            continue
        grouped[str(uid)].append((idx, str(state), tok, float(reward), str(task or "")))

    rows: List[Tuple[str, str, str, Tuple[str, ...]]] = []
    for _uid, steps in grouped.items():
        steps.sort(key=lambda x: x[0])
        if max_steps_per_traj > 0:
            steps = steps[: int(max_steps_per_traj)]
        if len(steps) < h:
            continue

        for t in range(len(steps) - h + 1):
            state = steps[t][1]
            task = steps[t][4]
            rollout_plan = tuple(steps[t + i][2] for i in range(h))
            if style == "validation":
                prompt = format_validation_planning_prompt(state, task=task, horizon=h)
            else:
                prompt = format_max_return_planning_prompt(state, task=task, horizon=h)
            rows.append((prompt, state, task, rollout_plan))

    return rows


def expand_craftext_planning_examples(
    *,
    traj_uids: Sequence[str],
    states: Sequence[str],
    action_tokens: Sequence[str],
    task_instructions: Sequence[str],
    plan_horizon: int = 6,
    max_steps_per_traj: int = 50,
    prompt_style: str = "validation",
) -> List[tuple[str, str]]:
    """
    Build SFT rows from on-policy rollouts: at each start t, predict the next H actions.
    Returns list of (prompt_text, target_action_sequence) e.g. target ``224156``.
    """
    h = max(1, int(plan_horizon))
    style = str(prompt_style or "validation").lower()

    grouped: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    for idx, (uid, state, action, task) in enumerate(
        zip(traj_uids, states, action_tokens, task_instructions)
    ):
        tok = normalize_action_token(action)
        if not (uid and state and tok and tok in TOKEN_TO_ACTION_ID):
            continue
        grouped[str(uid)].append((idx, str(state), tok, str(task or "")))

    rows: List[tuple[str, str]] = []
    for _uid, steps in grouped.items():
        steps.sort(key=lambda x: x[0])
        if max_steps_per_traj > 0:
            steps = steps[: int(max_steps_per_traj)]
        if len(steps) < h:
            continue

        for t in range(len(steps) - h + 1):
            state = steps[t][1]
            task = steps[t][3]
            plan_tokens = [steps[t + i][2] for i in range(h)]
            if style == "max_return":
                prompt = format_max_return_planning_prompt(state, task=task, horizon=h)
            else:
                prompt = format_validation_planning_prompt(state, task=task, horizon=h)
            target = format_planning_target(plan_tokens)
            rows.append((prompt, target))

    return rows


def tokenize_planning_response_ids(tokenizer, target_text: str, *, add_eos: bool = False) -> List[int]:
    """One tokenizer id per action character in the concatenated plan."""
    ids: List[int] = []
    for ch in str(target_text or ""):
        if ch not in TOKEN_TO_ACTION_ID:
            raise ValueError(f"Invalid planning action token {ch!r} in target {target_text!r}")
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Action token {ch!r} must encode to one id, got {piece!r}")
        ids.append(int(piece[0]))
    if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        ids.append(int(tokenizer.eos_token_id))
    return ids


__all__ = [
    "expand_craftext_planning_advantage_examples",
    "expand_craftext_planning_examples",
    "tokenize_planning_response_ids",
]
