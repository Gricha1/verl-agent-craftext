"""Prompt and action contract of the CE-trained Crafter visual SLM.

Copied byte-for-byte from ZeroHero ``scripts/evaluate_offline_teacher_slm_craftext.py``
(branch ``iclr2027-crafter-experiments``, base ``7a7e504``), which is the
closed-loop evaluator of the same checkpoint.  The SLM has its own BPE-4096
tokenizer and no chat template; the text is tokenised as-is and the four CNN
visual tokens precede it.  The action history is fixed at 8, as in training
(``slm5m_vision4_history8``).
"""
from __future__ import annotations

import re

from ..actions import ACTION_NAMES, ParsedAction

VISUAL_SLM_ACTION_HISTORY = 8


def vision_image_only_prompt(instruction: str, action_history: list[str] | None = None,
                             reasoning: bool = False) -> str:
    """Must match the visual-only training prompt byte-for-byte (prepare_teacher_dataset.py;
    reasoning=True is protocol vision_image_reasoning_v1)."""
    history = ""
    if action_history is not None:
        history = "Previous executed actions (oldest to newest; up to 8): " + (", ".join(action_history) or "none") + "\n\n"
    answer = ("Think in one short sentence inside <reasoning>...</reasoning>, then output exactly one action "
              "as ACTION: <ACTION_NAME>." if reasoning else "Output exactly one action as ACTION: <ACTION_NAME>.")
    return (
        "SYSTEM:\n"
        "You are a careful embodied CrafText agent. Infer the current navigational "
        "state solely from the accompanying visual observation.\n\n"
        "USER:\n"
        f"Your goal is to complete this task:\n**TASK:** {instruction}\n\n"
        + history
        + answer + "\n\nResponse:\n"
    )


def parse_slm_action(response: str) -> ParsedAction:
    """ZeroHero ``parse_action``: the text after the last ``ACTION:``; else NOOP.

    Deliberate fix: the original character filter ``[^a-zA-Z\\s:]`` also
    stripped ``_``, so PLACE_TABLE / MAKE_WOOD_PICKAXE / MAKE_WOOD_SWORD -- the
    wood-task actions -- could never parse and silently became NOOP.
    """
    cleaned = re.sub(r"[^a-zA-Z_\s:]", "", str(response or "")).split("ACTION:")[-1].strip()
    if cleaned in ACTION_NAMES:
        return ParsedAction(ACTION_NAMES.index(cleaned), cleaned, True, "")
    return ParsedAction(0, "NOOP", False, "")
