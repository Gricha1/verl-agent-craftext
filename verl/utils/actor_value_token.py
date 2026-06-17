"""Dual-prompt actor-value PPO: actor prompt -> action token; critic prompt -> return bin."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings
from agent_system.environments.env_package.caged_craftext.return_tokens import (
    DEFAULT_RETURN_BIN_SPEC,
    RETURN_BIN_TO_TOKEN,
    ReturnBinSpec,
    bin_to_scalar,
    compute_remaining_returns,
    decode_return_token,
    parse_return_token,
    quantize_return,
    return_bin_spec_from_actor_cfg,
)


def action_token_id_map(tokenizer) -> List[int]:
    out: List[int] = []
    for ch in action_token_strings():
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Action char {ch!r} must tokenize to one id, got {piece!r}")
        out.append(int(piece[0]))
    return out


def action_token_id_tensor(tokenizer, device: torch.device) -> torch.Tensor:
    return torch.tensor(action_token_id_map(tokenizer), dtype=torch.long, device=device)


def return_token_id_map(tokenizer) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for bin_val, ch in RETURN_BIN_TO_TOKEN.items():
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Return char {ch!r} must tokenize to one id, got {piece!r}")
        out[int(bin_val)] = int(piece[0])
    return out


def return_bin_id_tensor(tokenizer, device: torch.device, num_bins: int) -> torch.Tensor:
    id_map = return_token_id_map(tokenizer)
    ids = [id_map[b] for b in range(int(num_bins))]
    return torch.tensor(ids, dtype=torch.long, device=device)


def vocab_id_to_return_bin_tensor(tokenizer, device: torch.device) -> torch.Tensor:
    id_map = return_token_id_map(tokenizer)
    max_vocab = max(int(v) for v in id_map.values()) + 1
    table = torch.zeros(max_vocab, dtype=torch.long, device=device)
    for bin_val, vocab_id in id_map.items():
        table[int(vocab_id)] = int(bin_val)
    return table


def mask_logits_to_allowed(logits: torch.Tensor, allowed_ids: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    masked = torch.full_like(logits, float("-inf"))
    masked[:, allowed_ids] = logits[:, allowed_ids]
    return masked


def constrained_sample_token_ids(
    logits: torch.Tensor,
    allowed_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    scaled = logits / max(float(temperature), 1e-8)
    masked = mask_logits_to_allowed(scaled, allowed_ids)
    if do_sample:
        probs = F.softmax(masked, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        token_ids = masked.argmax(dim=-1)
    return token_ids, masked


def constrained_log_probs(
    masked_logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _forward_last_logits(
    actor_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Next-token logits at the last input position only (logits_to_keep=1 when supported)."""
    mrope = position_ids.dim() == 3
    pos = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0) * attention_mask
    if mrope:
        pos = pos.unsqueeze(0).expand(3, -1, -1)
    try:
        output = actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            use_cache=False,
            logits_to_keep=1,
            **kwargs,
        )
    except TypeError:
        output = actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            use_cache=False,
            **kwargs,
        )
    if getattr(output, "logits", None) is None:
        raise RuntimeError("actor_value forward returned no logits")
    logits = output.logits
    if logits.dim() == 3:
        return logits[:, -1, :]
    return logits


@torch.no_grad()
def constrained_generate_return_token(
    actor_module,
    value_prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    tokenizer,
    *,
    temperature: float = 1.0,
    do_sample: bool = True,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One constrained return-bin token per critic prompt. Returns (token_ids B,), (log_probs B,)."""
    device = value_prompts.device
    return_ids_allowed = return_bin_id_tensor(tokenizer, device, spec.num_bins)
    actor_module.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = _forward_last_logits(actor_module, value_prompts, attention_mask, position_ids)
        token_ids, masked = constrained_sample_token_ids(
            logits, return_ids_allowed, temperature=temperature, do_sample=do_sample
        )
        log_probs = constrained_log_probs(masked, token_ids)
    return token_ids, log_probs


@torch.no_grad()
def constrained_decode_return_values(
    actor_module,
    value_prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    tokenizer,
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> torch.Tensor:
    """Greedy constrained decode of return bin from critic prompt only. Returns (B,) scalars."""
    device = value_prompts.device
    return_ids_allowed = return_bin_id_tensor(tokenizer, device, spec.num_bins)
    vocab_to_bin = vocab_id_to_return_bin_tensor(tokenizer, device)
    actor_module.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = _forward_last_logits(actor_module, value_prompts, attention_mask, position_ids)
        masked = mask_logits_to_allowed(logits, return_ids_allowed)
        pred_vocab_ids = masked.argmax(dim=-1)
    pred_bins = vocab_to_bin[pred_vocab_ids.clamp(max=vocab_to_bin.numel() - 1)]
    return torch.tensor(
        [bin_to_scalar(int(b), spec=spec) for b in pred_bins],
        dtype=torch.float32,
        device=pred_bins.device,
    )


def decode_return_from_token_ids(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    texts = tokenizer.batch_decode(token_ids.unsqueeze(-1), skip_special_tokens=True)
    values = [decode_return_token(t) for t in texts]
    return torch.tensor(values, dtype=torch.float32, device=token_ids.device)


def concat_value_prompt_and_response(
    value_input_ids: torch.Tensor,
    value_attention_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing sequence: critic value prompt + actor response tokens."""
    combined_ids = torch.cat([value_input_ids, responses], dim=-1)
    combined_mask = torch.cat([value_attention_mask, response_mask.to(value_attention_mask.dtype)], dim=-1)
    return combined_ids, combined_mask


def expected_return_from_bin_logits(
    bin_logits: torch.Tensor,
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> torch.Tensor:
    """Expected scalar return from bin logits; supports (..., num_bins)."""
    num_bins = bin_logits.shape[-1]
    bins = torch.arange(num_bins, device=bin_logits.device, dtype=torch.float32)
    probs = F.softmax(bin_logits.float(), dim=-1)
    bin_expectation = (probs * bins).sum(dim=-1)
    return float(spec.vmin) + bin_expectation * float(spec.step)


def forward_value_bin_logits_per_response_token(
    actor_module,
    *,
    value_input_ids: torch.Tensor,
    value_attention_mask: torch.Tensor,
    value_position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> torch.Tensor:
    """
    Per-response-token return-bin logits from critic prompt + teacher-forced response.
    Mirrors separate critic V(s) at each token: input is value_prompt || response[:t] before token t.

    Returns:
        (B, response_length, num_bins)
    """
    device = value_input_ids.device
    combined_ids, combined_mask = concat_value_prompt_and_response(
        value_input_ids, value_attention_mask, responses, response_mask
    )
    response_length = responses.size(1)
    return_ids_allowed = return_bin_id_tensor(tokenizer, device, spec.num_bins)

    mrope = value_position_ids.dim() == 3
    pos = (combined_mask.cumsum(dim=-1) - 1).clamp(min=0) * combined_mask
    temp = max(float(temperature), 1e-8)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if mrope:
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        try:
            output = actor_module(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                position_ids=pos,
                use_cache=False,
            )
        except TypeError:
            output = actor_module(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                position_ids=pos,
                use_cache=False,
            )
        if getattr(output, "logits", None) is None:
            raise RuntimeError("actor_value per-token forward requires model logits")
        logits = output.logits.float() / temp
        if logits.dim() != 3:
            raise RuntimeError(f"Expected logits (B, S, V), got shape {tuple(logits.shape)}")
        next_token_logits = logits[:, -response_length - 1 : -1, :]
        bin_logits = next_token_logits.index_select(-1, return_ids_allowed)
    return bin_logits


def per_token_values_from_critic_prompt(
    actor_module,
    *,
    value_input_ids: torch.Tensor,
    value_attention_mask: torch.Tensor,
    value_position_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    temperature: float = 1.0,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> torch.Tensor:
    """GAE values (B, response_length) — expected return at each response position."""
    bin_logits = forward_value_bin_logits_per_response_token(
        actor_module,
        value_input_ids=value_input_ids,
        value_attention_mask=value_attention_mask,
        value_position_ids=value_position_ids,
        responses=responses,
        response_mask=response_mask,
        tokenizer=tokenizer,
        temperature=temperature,
        spec=spec,
    )
    values = expected_return_from_bin_logits(bin_logits, spec=spec)
    return values * response_mask.to(values.dtype)


def build_actor_token_values(
    value_scalars: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Deprecated: broadcast scalar to all tokens. Use per_token_values_from_critic_prompt."""
    values = value_scalars.to(dtype=torch.float32).unsqueeze(1).expand(-1, response_length)
    return values.contiguous()


def build_step_reward_tensor(
    responses: torch.Tensor,
    step_rewards: Sequence[float],
) -> torch.Tensor:
    reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
    for i, r in enumerate(step_rewards):
        reward_tensor[i, responses.size(1) - 1] = float(r)
    return reward_tensor


def compute_remaining_return_bins(
    traj_uids: Sequence[str],
    step_rewards: Sequence[float],
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> np.ndarray:
    scalars = compute_remaining_return_scalars(traj_uids, step_rewards)
    return np.array([quantize_return(float(g), spec=spec) for g in scalars], dtype=np.int64)


def compute_remaining_return_scalars(
    traj_uids: Sequence[str],
    step_rewards: Sequence[float],
) -> np.ndarray:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, uid in enumerate(traj_uids):
        groups[str(uid)].append(idx)

    targets = np.zeros(len(traj_uids), dtype=np.float64)
    for indices in groups.values():
        rewards = [float(step_rewards[i]) for i in indices]
        remaining = compute_remaining_returns(rewards)
        for idx, g in zip(indices, remaining):
            targets[idx] = float(g)
    return targets


def two_hot_return_distribution(
    values: torch.Tensor,
    *,
    max_bin: int,
) -> torch.Tensor:
    """Linear two-hot distribution over bins [0, max_bin] for each scalar return."""
    num_bins = max_bin + 1
    v = values.to(dtype=torch.float32).clamp(min=0.0, max=float(max_bin))
    k_low = v.floor().to(dtype=torch.long).clamp(max=max_bin)
    k_high = (k_low + 1).clamp(max=max_bin)
    w_high = v - k_low.to(dtype=torch.float32)
    w_low = 1.0 - w_high
    at_boundary = k_low >= max_bin
    w_low = torch.where(at_boundary, torch.ones_like(w_low), w_low)
    w_high = torch.where(at_boundary, torch.zeros_like(w_high), w_high)

    out = torch.zeros(v.size(0), num_bins, device=v.device, dtype=torch.float32)
    out.scatter_add_(1, k_low.unsqueeze(1), w_low.unsqueeze(1))
    out.scatter_add_(1, k_high.unsqueeze(1), w_high.unsqueeze(1))
    return out


def return_token_value_loss(
    bin_logits: torch.Tensor,
    target_returns: torch.Tensor,
    *,
    target_encoding: str = "one_hot",
    response_mask: torch.Tensor | None = None,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over return bins. Targets are scalar returns in [vmin, vmax]."""
    encoding = str(target_encoding).lower()
    max_bin = bin_logits.shape[-1] - 1
    if bin_logits.dim() == 3:
        batch_size, response_length, _ = bin_logits.shape
        if target_returns.dim() == 1:
            target_returns = target_returns.unsqueeze(1).expand(batch_size, response_length)
        flat_logits = bin_logits.reshape(-1, bin_logits.size(-1))
        flat_targets = target_returns.reshape(-1).to(dtype=torch.float32)
        if response_mask is not None:
            flat_mask = response_mask.reshape(-1).bool()
            flat_logits = flat_logits[flat_mask]
            flat_targets = flat_targets[flat_mask]
        if flat_logits.numel() == 0:
            zero = bin_logits.sum() * 0.0
            return zero, torch.zeros(0, dtype=torch.long, device=bin_logits.device)
        return return_token_value_loss(
            flat_logits,
            flat_targets,
            target_encoding=target_encoding,
            response_mask=None,
            spec=spec,
        )

    target_bins = (
        target_returns.to(dtype=torch.float32).clamp(min=float(spec.vmin), max=float(spec.vmax))
        - float(spec.vmin)
    ) / float(spec.step)
    target_bins = target_bins.clamp(min=0.0, max=float(max_bin))
    hard_bins = target_bins.round().to(dtype=torch.long)
    log_probs = F.log_softmax(bin_logits, dim=-1)

    if encoding == "one_hot":
        loss = F.nll_loss(log_probs, hard_bins, reduction="mean")
    elif encoding == "two_hot":
        target_probs = two_hot_return_distribution(target_bins, max_bin=max_bin)
        loss = -(target_probs * log_probs).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unsupported actor_value_target_encoding: {target_encoding!r}")

    return loss, hard_bins


def return_bin_distribution_entropy(bin_logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy over return bins. bin_logits: (B, num_bins). Returns (B,) entropy."""
    log_probs = F.log_softmax(bin_logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)
