"""Stable CrafText telemetry names for modern-veRL PPO runs.

The integration writes scalar telemetry into an agent-loop output's
``extra_fields``.  Keeping the reduction here makes train and fixed-seed
validation metrics use the same definitions, independent of veRL's internal
metric names.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any


def _numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            result.append(float(value))
        elif isinstance(value, int | float):
            result.append(float(value))
    return result


def _mean(values: Iterable[Any]) -> float | None:
    values = _numbers(values)
    return sum(values) / len(values) if values else None


def training_metrics(extra_fields: Iterable[Any]) -> dict[str, float]:
    """Reduce one PPO item's CrafText fields to stable train metric names."""
    rows = [row for row in extra_fields if isinstance(row, Mapping)]
    if not any("craftext_action_valid" in row or "craftext_episode_final" in row for row in rows):
        return {}

    result: dict[str, float] = {}
    action_metrics = {
        "craftext_action_valid": "train/action_parse_rate",
        "craftext_action_invalid": "train/invalid_action_rate",
        "craftext_action_noop": "train/NOOP_rate",
        "craftext_action_repeated": "train/repeated_action_rate",
    }
    for source, target in action_metrics.items():
        value = _mean(row.get(source) for row in rows)
        if value is not None:
            result[target] = value

    final_rows = [row for row in rows if row.get("craftext_episode_final")]
    for source, target in (
        ("craftext_episode_success", "train/success_rate"),
        ("craftext_episode_return", "train/episode_return_mean"),
        ("craftext_episode_steps", "train/episode_length_mean"),
    ):
        value = _mean(row.get(source) for row in final_rows)
        if value is not None:
            result[target] = value
    return result


def validation_metrics(reward_extra_info: Mapping[str, list[Any]]) -> dict[str, float]:
    """Return deterministic fixed-seed validation headline metrics.

    Values are supplied only by the final transition of each complete episode,
    hence each list entry is one evaluation episode rather than one env step.
    """
    mappings = (
        ("craftext_episode_success", "val/success_rate", "mean"),
        ("craftext_episode_return", "val/mean_episode_return", "mean"),
        ("craftext_episode_return", "val/median_episode_return", "median"),
        ("craftext_episode_steps", "val/episode_length_mean", "mean"),
        ("craftext_episode_steps", "val/episode_length_median", "median"),
        ("craftext_episode_parse_rate", "val/action_parse_rate", "mean"),
        ("craftext_episode_invalid_rate", "val/invalid_action_rate", "mean"),
        ("craftext_episode_noop_rate", "val/NOOP_rate", "mean"),
        ("craftext_episode_repeated_rate", "val/repeated_action_rate", "mean"),
    )
    result: dict[str, float] = {}
    for source, target, reducer in mappings:
        values = _numbers(reward_extra_info.get(source, []))
        if values:
            result[target] = float(median(values) if reducer == "median" else sum(values) / len(values))
    return result
