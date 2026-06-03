"""Single-token action labels for 17 Craftext discrete actions."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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


def format_single_token_action_display(text: str) -> tuple[str, int]:
    """Format model output as ``NAME | raw: TOKEN``; returns (display, action_id)."""
    from .projection import ACTION_TO_TEXT

    raw = (text or "").strip().split()[0] if (text or "").strip() else ""
    action_id = parse_single_token_action(text)
    if action_id >= 0 and action_id < len(ACTION_TO_TEXT):
        name = ACTION_TO_TEXT[action_id]
    else:
        name = "?"
    display = f"{name} | raw: {raw}" if raw else ""
    return display, action_id


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


def normalize_action_token(raw: str) -> str:
    return str(raw or "").strip().split()[0] if str(raw or "").strip() else ""


def format_action_token_sequence(actions: Sequence[str]) -> str:
    """Concatenate action tokens for planning SFT targets (no spaces)."""
    parts = [normalize_action_token(a) for a in actions]
    if any(not p for p in parts):
        raise ValueError(f"Empty action in sequence: {actions!r}")
    for p in parts:
        if p not in TOKEN_TO_ACTION_ID:
            raise ValueError(f"Invalid action token {p!r}")
    return "".join(parts)


def is_action_token_sequence(text: str, *, horizon: int | None = None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if any(ch not in TOKEN_TO_ACTION_ID for ch in raw):
        return False
    if horizon is not None and len(raw) != int(horizon):
        return False
    return True


def tokenize_action_response_ids(
    tokenizer,
    response: str,
    *,
    add_eos: bool = True,
    expected_len: int | None = None,
) -> List[int]:
    """Encode each action char separately (avoid BPE merges)."""
    raw = str(response or "").strip()
    if expected_len is not None and len(raw) != int(expected_len):
        raise ValueError(f"Expected {expected_len} action chars, got {len(raw)!r}")
    ids: List[int] = []
    for ch in raw:
        if ch not in TOKEN_TO_ACTION_ID:
            break
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Action char {ch!r} must tokenize to one id, got {piece!r}")
        ids.append(int(piece[0]))
    if add_eos:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            ids.append(int(tokenizer.eos_token_id))
        elif getattr(tokenizer, "eos_token", None):
            ids.extend(tokenizer.encode(tokenizer.eos_token, add_special_tokens=False))
    return ids


def parse_action_token_sequence(text: str) -> Tuple[str, ...]:
    raw = str(text or "").strip()
    out = []
    for ch in raw:
        if ch in TOKEN_TO_ACTION_ID:
            out.append(ch)
        else:
            break
    return tuple(out)
