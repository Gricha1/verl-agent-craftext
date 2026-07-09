"""Single-token labels for dynamic AlfWorld admissible commands (1..9, a..z)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

INVALID_ACTION_INDEX = -1
MAX_ADMISSIBLE_ACTIONS = 35

_LABEL_POOL: Tuple[str, ...] = tuple(str(i) for i in range(1, 10)) + tuple("abcdefghijklmnopqrstuvwxyz")


def action_label(action_index: int) -> str:
    if action_index < 0 or action_index >= len(_LABEL_POOL):
        raise IndexError(f"action_index out of range: {action_index}")
    return _LABEL_POOL[action_index]


def labels_for_count(n: int) -> Tuple[str, ...]:
    n = int(n)
    if n <= 0:
        return ()
    if n > MAX_ADMISSIBLE_ACTIONS:
        n = MAX_ADMISSIBLE_ACTIONS
    return _LABEL_POOL[:n]


def build_admissible_legend(commands: Sequence[str]) -> str:
    cmds = [str(c).strip() for c in commands if str(c).strip() and str(c).strip().lower() != "help"]
    if len(cmds) > MAX_ADMISSIBLE_ACTIONS:
        cmds = cmds[:MAX_ADMISSIBLE_ACTIONS]
    labels = labels_for_count(len(cmds))
    return ", ".join(f"{lab}={cmd}" for lab, cmd in zip(labels, cmds))


def parse_single_token_action(text: str, num_valid: int) -> int:
    if num_valid <= 0:
        return INVALID_ACTION_INDEX
    raw = (text or "").strip()
    if not raw:
        return INVALID_ACTION_INDEX
    tok = raw.split()[0].splitlines()[-1].strip()
    labels = labels_for_count(num_valid)
    if tok in labels:
        return labels.index(tok)
    return INVALID_ACTION_INDEX


def vocab_ids_for_labels(tokenizer, num_valid: int) -> List[int]:
    labels = labels_for_count(num_valid)
    ids: List[int] = []
    for lab in labels:
        enc = tokenizer.encode(lab, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"AlfWorld action label {lab!r} is not a single tokenizer token: {enc}")
        ids.append(int(enc[0]))
    return ids


__all__ = [
    "INVALID_ACTION_INDEX",
    "MAX_ADMISSIBLE_ACTIONS",
    "action_label",
    "build_admissible_legend",
    "labels_for_count",
    "parse_single_token_action",
    "vocab_ids_for_labels",
]
