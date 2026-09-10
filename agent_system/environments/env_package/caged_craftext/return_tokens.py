"""Single-token labels for discretized remaining return / value."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Return bins: uppercase A..Z (26) + m,n,o — no overlap with actions (1-9,a-h) or step-reward WM (i-l).
MAX_RETURN_TOKEN_SLOTS = 29
_RETURN_TOKEN_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26)) + ("m", "n", "o")
TOKEN_TO_RETURN_BIN: Dict[str, int] = {
    tok: i for i, tok in enumerate(_RETURN_TOKEN_LABELS)
}
RETURN_BIN_TO_TOKEN: Dict[int, str] = {
    i: tok for i, tok in enumerate(_RETURN_TOKEN_LABELS)
}

INVALID_RETURN_BIN = -1


@dataclass(frozen=True)
class ReturnBinSpec:
    """Linear bin grid: scalar = vmin + bin_index * step."""

    vmin: float = 0.0
    vmax: float = 3.0
    step: float = 0.2

    def __post_init__(self) -> None:
        if self.step <= 0.0:
            raise ValueError(f"ReturnBinSpec.step must be > 0, got {self.step}")
        if self.vmax < self.vmin:
            raise ValueError(f"ReturnBinSpec.vmax must be >= vmin, got {self.vmax} < {self.vmin}")
        if self.num_bins > MAX_RETURN_TOKEN_SLOTS:
            raise ValueError(
                f"Return grid needs {self.num_bins} tokens but only {MAX_RETURN_TOKEN_SLOTS} slots "
                f"(vmin={self.vmin}, vmax={self.vmax}, step={self.step})"
            )

    @property
    def num_bins(self) -> int:
        return int(round((self.vmax - self.vmin) / self.step)) + 1

    @property
    def max_bin(self) -> int:
        return self.num_bins - 1


DEFAULT_RETURN_BIN_SPEC = ReturnBinSpec(vmin=0.0, vmax=3.0, step=0.2)
CRAFTEXT_RETURN_BIN_SPEC = ReturnBinSpec(vmin=0.0, vmax=8.0, step=0.5)


def return_bin_spec_from_meta(meta_info, *, actor_cfg=None) -> ReturnBinSpec:
    """Resolve return bins for validation Q decode (meta_info from driver overrides actor cfg)."""
    meta = meta_info or {}
    if "return_bin_vmin" in meta and "return_bin_vmax" in meta and "return_bin_step" in meta:
        return ReturnBinSpec(
            vmin=float(meta["return_bin_vmin"]),
            vmax=float(meta["return_bin_vmax"]),
            step=float(meta["return_bin_step"]),
        )
    if actor_cfg is not None and actor_cfg.get("actor_value_token", False):
        return return_bin_spec_from_actor_cfg(actor_cfg)
    # debug_square actor-value default grid (used for cross-run Q validation)
    return ReturnBinSpec(vmin=-5.0, vmax=6.0, step=0.4)


def return_bin_spec_from_env(env_cfg) -> ReturnBinSpec:
    return ReturnBinSpec(
        vmin=float(getattr(env_cfg, "value_return_min", DEFAULT_RETURN_BIN_SPEC.vmin)),
        vmax=float(getattr(env_cfg, "value_return_max", DEFAULT_RETURN_BIN_SPEC.vmax)),
        step=float(getattr(env_cfg, "value_return_bin_step", DEFAULT_RETURN_BIN_SPEC.step)),
    )


def return_bin_spec_from_actor_cfg(actor_cfg) -> ReturnBinSpec:
    return ReturnBinSpec(
        vmin=float(actor_cfg.get("actor_value_return_min", DEFAULT_RETURN_BIN_SPEC.vmin)),
        vmax=float(actor_cfg.get("actor_value_return_max", DEFAULT_RETURN_BIN_SPEC.vmax)),
        step=float(actor_cfg.get("actor_value_return_bin_step", DEFAULT_RETURN_BIN_SPEC.step)),
    )


def return_bin_specs_match(env_cfg, actor_cfg) -> bool:
    env_spec = return_bin_spec_from_env(env_cfg)
    actor_spec = return_bin_spec_from_actor_cfg(actor_cfg)
    return (
        env_spec.vmin == actor_spec.vmin
        and env_spec.vmax == actor_spec.vmax
        and env_spec.step == actor_spec.step
    )


def assert_return_bin_specs_match(env_cfg, actor_cfg, *, context: str = "actor-value") -> None:
    env_spec = return_bin_spec_from_env(env_cfg)
    actor_spec = return_bin_spec_from_actor_cfg(actor_cfg)
    if return_bin_specs_match(env_cfg, actor_cfg):
        return
    raise ValueError(
        f"{context}: env value_return grid "
        f"[{env_spec.vmin}, {env_spec.vmax}] step={env_spec.step} "
        f"does not match actor actor_value_return grid "
        f"[{actor_spec.vmin}, {actor_spec.vmax}] step={actor_spec.step}. "
        "Critic prompt legend and training bins must use the same vmin/vmax/step."
    )


def return_token_legend_for_spec(spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> str:
    parts = []
    for i in range(spec.num_bins):
        tok = RETURN_BIN_TO_TOKEN[i]
        val = bin_to_scalar(i, spec=spec)
        parts.append(f"{tok}={val:g}")
    return ", ".join(parts)


def return_token_legend_compact_for_spec(spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> str:
    lo = RETURN_BIN_TO_TOKEN[0]
    hi = RETURN_BIN_TO_TOKEN[spec.max_bin]
    return (
        f"{lo}={spec.vmin:g} .. {hi}={spec.vmax:g} "
        f"(step={spec.step:g}, {spec.num_bins} tokens A..o)"
    )


def return_token_legend() -> str:
    return return_token_legend_for_spec(DEFAULT_RETURN_BIN_SPEC)


def return_token_strings() -> Tuple[str, ...]:
    return _RETURN_TOKEN_LABELS


def bin_to_scalar(bin_index: int, *, spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> float:
    b = int(max(0, min(spec.max_bin, bin_index)))
    return float(spec.vmin) + float(b) * float(spec.step)


def quantize_return(value: float, *, spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> int:
    v = float(value)
    if v <= float(spec.vmin):
        return 0
    scaled = (v - float(spec.vmin)) / float(spec.step)
    return int(min(spec.max_bin, max(0, round(scaled))))


def return_to_token(value: float, *, spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> str:
    return RETURN_BIN_TO_TOKEN[quantize_return(value, spec=spec)]


def decode_return_token(text: str, *, spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC) -> float:
    b = parse_return_token(text)
    return bin_to_scalar(b, spec=spec) if b >= 0 else float(spec.vmin)


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


def format_action_return_response(
    action_token: str,
    return_value: float,
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> str:
    act = str(action_token or "").strip()[:1]
    if not act:
        raise ValueError("Empty action token")
    return act + return_to_token(return_value, spec=spec)


def format_return_display(
    text: str,
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> tuple[str, int]:
    raw = (text or "").strip().split()[0] if (text or "").strip() else ""
    value = parse_return_token(text)
    if value == INVALID_RETURN_BIN:
        display = f"invalid: {raw}" if raw else "invalid"
    else:
        scalar = bin_to_scalar(value, spec=spec)
        display = f"{scalar:.2f} ({raw})" if raw else f"{scalar:.2f}"
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


def compute_remaining_returns(
    step_rewards: Sequence[float],
    *,
    gamma: float = 1.0,
) -> List[float]:
    """MC remaining return for each step in a trajectory.

    gamma=1 (default, backward compatible):
        G_t = r_t + r_{t+1} + ... + r_T

    gamma in (0, 1):
        G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
    """
    g = float(gamma)
    if g < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    out: List[float] = []
    total = 0.0
    for r in reversed(list(step_rewards)):
        total = float(r) + g * total
        out.append(total)
    out.reverse()
    return out


def tokenize_return_bin_token_ids(tokenizer, token: str, *, add_eos: bool = False) -> List[int]:
    """Encode one return-bin label (A..o) as a single tokenizer id."""
    tok = str(token or "").strip()
    if len(tok) != 1 or tok not in TOKEN_TO_RETURN_BIN:
        raise ValueError(f"Expected one return-bin token, got {token!r}")
    piece = tokenizer.encode(tok, add_special_tokens=False)
    if len(piece) != 1:
        raise ValueError(f"Return token {tok!r} must encode to one id, got {piece!r}")
    ids = [int(piece[0])]
    if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        ids.append(int(tokenizer.eos_token_id))
    return ids
