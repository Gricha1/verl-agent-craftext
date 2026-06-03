"""Single-token labels for discrete step rewards (-1, 0, 1, 2)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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


def sum_quantized_rewards(rewards: Sequence[float]) -> int:
    """Sum of per-step quantized rewards (Decision Transformer return target)."""
    return int(sum(quantize_step_reward(float(r)) for r in rewards))


def reward_to_token(reward: float) -> str:
    return REWARD_TO_TOKEN[quantize_step_reward(reward)]


def is_reward_token_response(text: str) -> bool:
    """True if ``text`` is a non-empty concatenation of reward letters only."""
    raw = str(text or "").strip()
    return bool(raw) and all(ch in TOKEN_TO_REWARD for ch in raw)


def tokenize_reward_response_ids(tokenizer, response: str, *, add_eos: bool = True) -> List[int]:
    """
    Encode reward targets one letter at a time so ``ki`` -> two ids, not one merged BPE token.

    Qwen/BPE otherwise merges ``ki``, ``ij``, etc. into unrelated vocab ids.
    """
    ids: List[int] = []
    for ch in str(response or "").strip():
        if ch not in TOKEN_TO_REWARD:
            break
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(
                f"Reward char {ch!r} must tokenize to exactly one id, got {piece!r}"
            )
        ids.append(int(piece[0]))
    if add_eos:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            ids.append(int(tokenizer.eos_token_id))
        elif getattr(tokenizer, "eos_token", None):
            ids.extend(tokenizer.encode(tokenizer.eos_token, add_special_tokens=False))
    return ids


def format_reward_token_sequence(rewards: Sequence[float]) -> str:
    """Concatenate reward tokens, e.g. [r_t, r_{t+1}] -> ``ij`` (no spaces)."""
    return "".join(reward_to_token(float(r)) for r in rewards)


def parse_reward_token_sequence(text: str) -> Tuple[int, ...]:
    """Parse a concatenated reward-token string like ``ijk`` into reward values."""
    raw = str(text or "").strip()
    if not raw:
        return ()
    # Keep only reward-token letters before EOS / whitespace.
    head = raw.split()[0] if raw.split() else raw
    out: List[int] = []
    for ch in head:
        if ch not in TOKEN_TO_REWARD:
            break
        out.append(TOKEN_TO_REWARD[ch])
    return tuple(out)


def format_reward_sequence_display(rewards: Sequence[float]) -> str:
    """Human-readable ``-1 (i), 0 (j)`` for multi-step targets."""
    parts = [f"{quantize_step_reward(float(r))} ({reward_to_token(float(r))})" for r in rewards]
    return ", ".join(parts)


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
