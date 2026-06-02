"""Single-token labels for discrete step rewards (-1, 0, 1, 2)."""
from __future__ import annotations

from typing import Dict, Tuple

# Letters i..l are unused by action tokens (actions use 1-9 and a-h).
REWARD_VALUES: Tuple[int, ...] = (-1, 0, 1, 2)
_REWARD_TOKEN_LABELS: Tuple[str, ...] = ("i", "j", "k", "l")

TOKEN_TO_REWARD: Dict[str, int] = dict(zip(_REWARD_TOKEN_LABELS, REWARD_VALUES))
REWARD_TO_TOKEN: Dict[int, str] = dict(zip(REWARD_VALUES, _REWARD_TOKEN_LABELS))

INVALID_REWARD = -999


def reward_token_legend() -> str:
    return ", ".join(f"{tok}={val}" for tok, val in zip(_REWARD_TOKEN_LABELS, REWARD_VALUES))


def reward_token_strings() -> Tuple[str, ...]:
    return _REWARD_TOKEN_LABELS


def quantize_step_reward(reward: float) -> int:
    """Map env scalar reward to one of {-1, 0, 1, 2}."""
    r = float(reward)
    for v in REWARD_VALUES:
        if abs(r - v) < 1e-3:
            return v
    rounded = int(max(-1, min(2, round(r))))
    if rounded in REWARD_TO_TOKEN:
        return rounded
    return 0


def reward_to_token(reward: float) -> str:
    return REWARD_TO_TOKEN[quantize_step_reward(reward)]


def parse_reward_token(text: str) -> int:
    if not text:
        return INVALID_REWARD
    raw = str(text).strip()
    if not raw:
        return INVALID_REWARD
    token = raw.split()[0]
    if token in TOKEN_TO_REWARD:
        return TOKEN_TO_REWARD[token]
    last_line = raw.splitlines()[-1].strip()
    if last_line in TOKEN_TO_REWARD:
        return TOKEN_TO_REWARD[last_line]
    parts = last_line.split()
    if parts and parts[0] in TOKEN_TO_REWARD:
        return TOKEN_TO_REWARD[parts[0]]
    return INVALID_REWARD


def format_reward_display(text: str) -> tuple[str, int]:
    """Format validation cell: ``VALUE (token)`` or ``invalid: TOKEN``."""
    raw = (text or "").strip().split()[0] if (text or "").strip() else ""
    value = parse_reward_token(text)
    if value == INVALID_REWARD:
        display = f"invalid: {raw}" if raw else "invalid"
    else:
        display = f"{value} ({raw})" if raw else str(value)
    return display, value
