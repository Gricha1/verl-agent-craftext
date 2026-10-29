"""Canonical parsing for free-form CrafText reasoning responses."""

from __future__ import annotations

from dataclasses import dataclass
import re


ACTION_NAMES = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)
_ACTION_IDS = {name: index for index, name in enumerate(ACTION_NAMES)}
_TAGS = (("<action>", "</action>"), ("<answer>", "</answer>"), ("<response>", "</response>"))


@dataclass(frozen=True)
class ParsedAction:
    """One deterministic environment transition derived from an LLM response."""

    action_id: int
    action_name: str
    valid: bool
    reasoning: str


def _normalise_action(value: str) -> str:
    value = value.strip().replace("-", "_")
    value = re.sub(r"\s+", "_", value)
    return re.sub(r"_+", "_", value).upper()


def parse_reasoning_action(response: str) -> ParsedAction:
    """Extract ``<action>NAME</action>`` and otherwise execute deterministic NOOP.

    Free-form reasoning never becomes part of the executed-action history.  This
    preserves the current CrafText invariant that malformed model output maps to
    one defined transition rather than an ambiguous negative index in JAX.
    """

    text = str(response or "")
    lower = text.lower()
    raw_action = None
    reasoning = text.strip()
    for start, end in _TAGS:
        start_index = lower.find(start)
        end_index = lower.find(end, start_index + len(start))
        if start_index >= 0 and end_index > start_index:
            raw_action = text[start_index + len(start) : end_index]
            reasoning = text[:start_index].strip()
            break
    if raw_action is None:
        raw_action = text.strip()

    action_name = _normalise_action(raw_action)
    if action_name in _ACTION_IDS and not re.search(r"[\u4e00-\u9fff]", text):
        return ParsedAction(_ACTION_IDS[action_name], action_name, True, reasoning)
    return ParsedAction(0, "NOOP", False, reasoning)
