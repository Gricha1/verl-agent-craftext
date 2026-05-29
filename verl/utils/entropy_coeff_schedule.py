"""Scheduled multiplier for actor entropy_coeff (typically vs total_env_steps)."""
from __future__ import annotations

import math
from typing import Any, Mapping


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _log_decay_k(
    *,
    hold_until: int,
    target_step: int,
    target_multiplier: float,
    min_multiplier: float,
    tau: float,
) -> float:
    """Pick k so mult(target_step) ~= target_multiplier."""
    progress = max(int(target_step) - int(hold_until), 1)
    min_m = float(min_multiplier)
    target_m = float(target_multiplier)
    span = max(1.0 - min_m, 1e-8)
    if target_m <= min_m + 1e-8:
        return 0.0
    denom = target_m - min_m
    ratio = span / denom - 1.0
    log_term = math.log1p(progress / max(float(tau), 1e-8))
    if log_term <= 1e-12:
        return 0.0
    return max(ratio / log_term, 0.0)


def entropy_coeff_multiplier(
    step: int,
    *,
    hold_until: int = 0,
    decay_until: int | None = None,
    end_multiplier: float = 1.0,
    schedule: str = "linear",
    min_multiplier: float = 0.5,
    tau: float = 10000.0,
    log_k: float | None = None,
    target_step: int | None = None,
    target_multiplier: float | None = None,
) -> float:
    """
    Returns a multiplier applied to base entropy_coeff (typically in [min_multiplier, 1.0]).

    Schedules:
      - hold: step < hold_until -> 1.0; else end_multiplier
      - linear / cosine: [hold_until, decay_until] from 1.0 -> end_multiplier, then flat
      - log: after hold_until, slow log-like decay toward min_multiplier (never 0):
            mult = min + (1-min) / (1 + k * log1p((step-hold)/tau))
        If log_k is None and target_step/target_multiplier are set, k is calibrated.
    """
    step = int(max(step, 0))
    schedule = str(schedule).lower()
    end_multiplier = float(end_multiplier)
    min_multiplier = float(min_multiplier)

    if step < hold_until:
        return 1.0

    if schedule == "log":
        k = log_k
        if k is None:
            if target_step is None or target_multiplier is None:
                k = 1.0
            else:
                k = _log_decay_k(
                    hold_until=hold_until,
                    target_step=target_step,
                    target_multiplier=target_multiplier,
                    min_multiplier=min_multiplier,
                    tau=tau,
                )
        progress = float(step - hold_until)
        denom = 1.0 + float(k) * math.log1p(progress / max(float(tau), 1e-8))
        return min_multiplier + (1.0 - min_multiplier) / max(denom, 1e-8)

    if schedule == "hold" or decay_until is None or decay_until <= hold_until:
        return end_multiplier

    if step >= decay_until:
        return end_multiplier

    if schedule == "cosine":
        t = (step - hold_until) / float(decay_until - hold_until)
        t = _clamp01(t)
        return end_multiplier + (1.0 - end_multiplier) * 0.5 * (1.0 + math.cos(math.pi * t))

    # linear (default)
    t = (step - hold_until) / float(decay_until - hold_until)
    return 1.0 + (end_multiplier - 1.0) * _clamp01(t)


def scheduled_entropy_coeff(base_coeff: float, step: int, schedule_cfg: Mapping[str, Any] | None) -> float:
    if not schedule_cfg or not bool(schedule_cfg.get("enable", False)):
        return float(base_coeff)

    mult = entropy_coeff_multiplier(
        step,
        hold_until=int(schedule_cfg.get("hold_until", 0)),
        decay_until=schedule_cfg.get("decay_until"),
        end_multiplier=float(schedule_cfg.get("end_multiplier", 1.0)),
        schedule=str(schedule_cfg.get("schedule", "linear")),
        min_multiplier=float(schedule_cfg.get("min_multiplier", 0.5)),
        tau=float(schedule_cfg.get("tau", 10000.0)),
        log_k=schedule_cfg.get("log_k"),
        target_step=schedule_cfg.get("target_step"),
        target_multiplier=schedule_cfg.get("target_multiplier"),
    )
    return float(base_coeff) * mult
