"""Single-token labels for discretized remaining return / value (0..8)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# m..u do not overlap with action tokens (1-9, a-h) or step-reward WM (i-l).
RETURN_BINS: Tuple[int, ...] = tuple(range(9))
_RETURN_TOKEN_LABELS: Tuple[str, ...] = tuple(chr(ord("m") + i) for i in RETURN_BINS)

TOKEN_TO_RETURN_BIN: Dict[str, int] = dict(zip(_RETURN_TOKEN_LABELS, RETURN_BINS))
RETURN_BIN_TO_TOKEN: Dict[int, str] = dict(zip(RETURN_BINS, _RETURN_TOKEN_LABELS))

INVALID_RETURN_BIN = -1
MAX_RETURN_BIN = max(RETURN_BINS)


def return_token_legend() -> str:
    return ", ".join(f"{tok}={val}" for tok, val in zip(_RETURN_TOKEN_LABELS, RETURN_BINS))


def return_token_strings() -> Tuple[str, ...]:
    return _RETURN_TOKEN_LABELS


def quantize_return(value: float, *, max_bin: int = MAX_RETURN_BIN) -> int:
    """Round-clamp scalar return to an integer bin in [0, max_bin]."""
    v = float(value)
    if v < 0.0:
        return 0
    return int(min(max_bin, max(0, round(v))))


def return_to_token(value: float) -> str:
    return RETURN_BIN_TO_TOKEN[quantize_return(value)]


def decode_return_token(text: str) -> float:
    """Decode one return token to its bin center (integer bin value)."""
    b = parse_return_token(text)
    return float(b) if b >= 0 else 0.0


def parse_return_token(text: str) -> int:
    if not text:
        return INVALID_RETURN_BIN
    raw = str(text).strip()
    if not raw:
        return INVALID_RETURN_BIN
    token = raw.split()[0]
    if token in TOKEN_TO_RETURN_BIN:
        return TOKEN_TO_RETURN_BIN[token]
    last_line = raw.splitlines()[-1].strip()
    if last_line in TOKEN_TO_RETURN_BIN:
        return TOKEN_TO_RETURN_BIN[last_line]
    parts = last_line.split()
    if parts and parts[0] in TOKEN_TO_RETURN_BIN:
        return TOKEN_TO_RETURN_BIN[parts[0]]
    return INVALID_RETURN_BIN


def parse_action_return_response(text: str) -> Tuple[str, str]:
    """Split ``{action}{return}`` into two single-character tokens."""
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    head = raw.split()[0]
    if len(head) >= 2:
        return head[0], head[1]
    if len(head) == 1:
        return head[0], ""
    return "", ""


def format_action_return_response(action_token: str, return_value: float) -> str:
    act = str(action_token or "").strip()[:1]
    if not act:
        raise ValueError("Empty action token")
    return act + return_to_token(return_value)


def format_return_display(text: str) -> tuple[str, int]:
    raw = (text or "").strip().split()[0] if (text or "").strip() else ""
    value = parse_return_token(text)
    if value == INVALID_RETURN_BIN:
        display = f"invalid: {raw}" if raw else "invalid"
    else:
        display = f"{value} ({raw})" if raw else str(value)
    return display, value


def tokenize_return_response_ids(tokenizer, response: str, *, add_eos: bool = True) -> List[int]:
    """Encode action+return targets one char at a time (avoid BPE merges)."""
    act, ret = parse_action_return_response(response)
    if not act or not ret:
        raise ValueError(f"Expected 2-char action+return response, got {response!r}")
    ids: List[int] = []
    for ch in (act, ret):
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Char {ch!r} must tokenize to one id, got {piece!r}")
        ids.append(int(piece[0]))
    if add_eos:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            ids.append(int(tokenizer.eos_token_id))
    return ids


def compute_remaining_returns(step_rewards: Sequence[float]) -> List[float]:
    """MC remaining return G_t = r_t + r_{t+1} + ... for each step in a trajectory."""
    out: List[float] = []
    total = 0.0
    for r in reversed(step_rewards):
        total += float(r)
        out.append(total)
    out.reverse()
    return out
