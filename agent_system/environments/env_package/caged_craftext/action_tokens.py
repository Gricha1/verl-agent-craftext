"""Single-token action labels for 17 Craftext discrete actions."""
from __future__ import annotations

from typing import Dict, Tuple

INVALID_ACTION_ID = -1

# One character per action (1-indexed in prompts: 1..9, then a..h).
_ACTION_TOKEN_LABELS: Tuple[str, ...] = tuple(str(i) for i in range(1, 10)) + tuple("abcdefghijklmnopq")[:8]

TOKEN_TO_ACTION_ID: Dict[str, int] = {label: idx for idx, label in enumerate(_ACTION_TOKEN_LABELS)}


def action_token_label(action_id: int) -> str:
    return _ACTION_TOKEN_LABELS[action_id]


def action_token_legend() -> str:
    """Prompt block: token=ACTION_NAME for all 17 actions."""
    from .projection import ACTION_TO_TEXT

    assert len(_ACTION_TOKEN_LABELS) == len(ACTION_TO_TEXT)
    lines = [f"{action_token_label(i)}={name}" for i, name in enumerate(ACTION_TO_TEXT)]
    return ", ".join(lines)


def action_token_strings() -> Tuple[str, ...]:
    return _ACTION_TOKEN_LABELS


def parse_single_token_action(text: str) -> int:
    """
    Parse model output as one action token. Returns action id or INVALID_ACTION_ID.
    """
    if not text:
        return INVALID_ACTION_ID

    raw = text.strip()
    if not raw:
        return INVALID_ACTION_ID

    if raw in TOKEN_TO_ACTION_ID:
        return TOKEN_TO_ACTION_ID[raw]

    last_line = raw.splitlines()[-1].strip()
    if last_line in TOKEN_TO_ACTION_ID:
        return TOKEN_TO_ACTION_ID[last_line]

    parts = last_line.split()
    if parts and parts[0] in TOKEN_TO_ACTION_ID:
        return TOKEN_TO_ACTION_ID[parts[0]]

    return INVALID_ACTION_ID
