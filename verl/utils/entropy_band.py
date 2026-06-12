"""Adaptive entropy coefficient (AEnt-style corridor) for action-set entropy in PPO."""
from __future__ import annotations

from typing import Any, Mapping


def entropy_band_enabled(band_cfg: Mapping[str, Any] | None) -> bool:
    return bool(band_cfg and band_cfg.get("enable", False))


def adapt_entropy_coeff(
    coef: float,
    entropy: float,
    band_cfg: Mapping[str, Any],
) -> float:
    """
    Keep mean action-set entropy in [low, high] by nudging entropy bonus coefficient.

    H < low  -> increase coef (encourage exploration)
    H > high -> decrease coef (discourage exploration)
    """
    low = float(band_cfg.get("low", 0.7))
    high = float(band_cfg.get("high", 1.4))
    coef_lr = float(band_cfg.get("coef_lr", 0.05))
    coef_low = float(band_cfg.get("coef_low", 0.0))
    coef_high = float(band_cfg.get("coef_high", 1.0))

    h = float(entropy)
    out = float(coef)
    if h < low:
        out += coef_lr * (low - h)
    elif h > high:
        out -= coef_lr * (h - high)
    return min(max(out, coef_low), coef_high)
