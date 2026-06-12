"""Dual-prompt actor-value PPO: actor prompt -> action token; critic prompt -> return bin."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings
from agent_system.environments.env_package.caged_craftext.return_tokens import (
    MAX_RETURN_BIN,
    RETURN_BIN_TO_TOKEN,
    compute_remaining_returns,
    decode_return_token,
    parse_return_token,
    quantize_return,
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


def return_bin_id_tensor(tokenizer, device: torch.device) -> torch.Tensor:
    id_map = return_token_id_map(tokenizer)
    max_bin = max(id_map.keys())
    ids = torch.zeros(max_bin + 1, dtype=torch.long, device=device)
    for b, tid in id_map.items():
        ids[b] = tid
    return ids


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One constrained return-bin token per critic prompt. Returns (token_ids B,), (log_probs B,)."""
    device = value_prompts.device
    return_ids_allowed = return_bin_id_tensor(tokenizer, device)
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
) -> torch.Tensor:
    """Greedy constrained decode of return bin from critic prompt only. Returns (B,) float bins."""
    device = value_prompts.device
    return_ids_allowed = return_bin_id_tensor(tokenizer, device)
    vocab_to_bin = vocab_id_to_return_bin_tensor(tokenizer, device)
    actor_module.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = _forward_last_logits(actor_module, value_prompts, attention_mask, position_ids)
        masked = mask_logits_to_allowed(logits, return_ids_allowed)
        pred_vocab_ids = masked.argmax(dim=-1)
    pred_bins = vocab_to_bin[pred_vocab_ids.clamp(max=vocab_to_bin.numel() - 1)]
    return pred_bins.to(dtype=torch.float32)


def decode_return_from_token_ids(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    texts = tokenizer.batch_decode(token_ids.unsqueeze(-1), skip_special_tokens=True)
    values = [decode_return_token(t) for t in texts]
    return torch.tensor(values, dtype=torch.float32, device=token_ids.device)


def build_actor_token_values(
    value_scalars: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """GAE values tensor (B, response_length); scalar V(s) on last response token."""
    values = torch.zeros(
        value_scalars.size(0), response_length, dtype=torch.float32, device=value_scalars.device
    )
    values[:, -1] = value_scalars
    return values


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
) -> np.ndarray:
    scalars = compute_remaining_return_scalars(traj_uids, step_rewards)
    return np.array([quantize_return(float(g)) for g in scalars], dtype=np.int64)


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
    max_bin: int = MAX_RETURN_BIN,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over return bins. target_encoding: one_hot | two_hot."""
    encoding = str(target_encoding).lower()
    v = target_returns.to(dtype=torch.float32).clamp(min=0.0, max=float(MAX_RETURN_BIN))
    hard_bins = v.round().to(dtype=torch.long)
    log_probs = F.log_softmax(bin_logits, dim=-1)

    if encoding == "one_hot":
        loss = F.nll_loss(log_probs, hard_bins, reduction="mean")
    elif encoding == "two_hot":
        target_probs = two_hot_return_distribution(target_returns, max_bin=bin_logits.size(-1) - 1)
        loss = -(target_probs * log_probs).sum(dim=-1).mean()
    else:
        raise ValueError(f"Unsupported actor_value_target_encoding: {target_encoding!r}")

    return loss, hard_bins


def return_bin_distribution_entropy(bin_logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy over return bins. bin_logits: (B, num_bins). Returns (B,) entropy."""
    log_probs = F.log_softmax(bin_logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)
