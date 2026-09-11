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


def _module_device(actor_module) -> torch.device:
    return next(actor_module.parameters()).device


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
    device = _module_device(actor_module)
    value_prompts = value_prompts.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
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
def constrained_generate_return_token_sequence(
    actor_module,
    value_prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    tokenizer,
    *,
    num_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = True,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive decode of num_tokens return-bin chars (one per action). Returns (B, T), (B, T) log_probs."""
    from verl.utils.model import compute_position_id_with_mask

    if int(num_tokens) <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")

    device = _module_device(actor_module)
    value_prompts = value_prompts.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    return_ids_allowed = return_bin_id_tensor(tokenizer, device, spec.num_bins)
    input_ids = value_prompts
    attn = attention_mask
    pos = position_ids
    token_rows: List[torch.Tensor] = []
    log_prob_rows: List[torch.Tensor] = []
    actor_module.eval()
    for _ in range(int(num_tokens)):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = _forward_last_logits(actor_module, input_ids, attn, pos)
            token_ids, masked = constrained_sample_token_ids(
                logits, return_ids_allowed, temperature=temperature, do_sample=do_sample
            )
            log_probs = constrained_log_probs(masked, token_ids)
        token_rows.append(token_ids.unsqueeze(-1))
        log_prob_rows.append(log_probs)
        input_ids = torch.cat([input_ids, token_ids.unsqueeze(-1)], dim=-1)
        attn = torch.cat(
            [attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=device)],
            dim=-1,
        )
        pos = compute_position_id_with_mask(attn).to(device)
    out_ids = torch.cat(token_rows, dim=-1)
    out_log_probs = torch.stack(log_prob_rows, dim=-1)
    return out_ids, out_log_probs


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
    device = _module_device(actor_module)
    value_prompts = value_prompts.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
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


def _gather_response_logits_rmpad(
    logits_rmpad: torch.Tensor,
    attention_mask: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Map varlen logits (total_nnz, V) to (B, response_length, V) at pre-response positions."""
    batch_size, seqlen = attention_mask.shape
    vocab_size = logits_rmpad.size(-1)
    valid_lens = attention_mask.sum(dim=1).to(torch.long)
    flat_cumsum = attention_mask.reshape(-1).cumsum(0)
    out = torch.empty(
        batch_size,
        response_length,
        vocab_size,
        device=logits_rmpad.device,
        dtype=logits_rmpad.dtype,
    )
    for b in range(batch_size):
        valid_len = int(valid_lens[b].item())
        for t in range(response_length):
            col = valid_len - response_length - 1 + t
            flat_idx = b * seqlen + col
            rmpad_idx = int(flat_cumsum[flat_idx].item()) - 1
            out[b, t] = logits_rmpad[rmpad_idx]
    return out


def _response_bin_logits(
    logits: torch.Tensor,
    response_length: int,
    return_ids_allowed: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """(B, response_length, V) or (B, S, V) -> (B, response_length, num_bins)."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    elif logits.size(1) != response_length:
        logits = logits[:, -response_length - 1 : -1, :]
    temp = max(float(temperature), 1e-8)
    return logits.index_select(-1, return_ids_allowed).float() / temp


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
    multi_modal_inputs: dict | None = None,
    use_remove_padding: bool = False,
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
    logits_to_keep = int(response_length) + 1
    mm_kwargs = multi_modal_inputs or {}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if use_remove_padding:
            try:
                from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input
            except ImportError:
                from transformers.integrations.npu_flash_attention import (
                    index_first_axis,
                    rearrange,
                    unpad_input,
                )

            if mrope:
                pos = pos.unsqueeze(0).expand(3, -1, -1)
            input_ids_rmpad, indices, *_ = unpad_input(combined_ids.unsqueeze(-1), combined_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            if mrope:
                position_ids_rmpad = index_first_axis(
                    rearrange(pos, "c b s ... -> (b s) c ..."),
                    indices,
                ).transpose(0, 1).unsqueeze(1)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(pos.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)
            output = actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
                **mm_kwargs,
            )
            if getattr(output, "logits", None) is None:
                raise RuntimeError("actor_value per-token forward requires model logits")
            logits_rmpad = output.logits.squeeze(0)
            if logits_rmpad.dim() != 2:
                raise RuntimeError(f"Expected rmpad logits (total_nnz, V), got {tuple(logits_rmpad.shape)}")
            response_logits = _gather_response_logits_rmpad(logits_rmpad, combined_mask, response_length)
            return _response_bin_logits(response_logits, response_length, return_ids_allowed, temperature)

        if mrope:
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        try:
            output = actor_module(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                position_ids=pos,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                **mm_kwargs,
            )
        except TypeError:
            output = actor_module(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                position_ids=pos,
                use_cache=False,
                **mm_kwargs,
            )
        if getattr(output, "logits", None) is None:
            raise RuntimeError("actor_value per-token forward requires model logits")
        logits = output.logits
        if logits.dim() != 3:
            raise RuntimeError(f"Expected logits (B, S, V), got shape {tuple(logits.shape)}")
        return _response_bin_logits(logits, response_length, return_ids_allowed, temperature)


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
    multi_modal_inputs: dict | None = None,
    use_remove_padding: bool = False,
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
        multi_modal_inputs=multi_modal_inputs,
        use_remove_padding=use_remove_padding,
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
    episode_step_idx: Sequence[int] | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    scalars = compute_remaining_return_scalars(
        traj_uids, step_rewards, episode_step_idx=episode_step_idx, gamma=gamma
    )
    return np.array([quantize_return(float(g), spec=spec) for g in scalars], dtype=np.int64)


def compute_remaining_return_scalars(
    traj_uids: Sequence[str],
    step_rewards: Sequence[float],
    episode_step_idx: Sequence[int] | None = None,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """MC remaining return G_t^gamma per row, grouped by traj_uid.

    gamma=1 => undiscounted suffix sum (legacy). gamma<1 => discounted return-to-go.

    If ``episode_step_idx`` is provided, steps are ordered by it within each trajectory
    (safe after batch shuffle / adjust_batch duplicates).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, uid in enumerate(traj_uids):
        groups[str(uid)].append(idx)

    step_idx = None
    if episode_step_idx is not None:
        step_idx = np.asarray(episode_step_idx, dtype=np.int64).reshape(-1)

    targets = np.zeros(len(traj_uids), dtype=np.float64)
    for indices in groups.values():
        if step_idx is not None:
            indices_sorted = sorted(indices, key=lambda i: int(step_idx[i]))
            # Unique chronological steps (adjust_batch may duplicate rows).
            uniq_indices: List[int] = []
            seen_steps = set()
            for i in indices_sorted:
                key = int(step_idx[i])
                if key in seen_steps:
                    continue
                seen_steps.add(key)
                uniq_indices.append(i)
            rewards = [float(step_rewards[i]) for i in uniq_indices]
            remaining = compute_remaining_returns(rewards, gamma=gamma)
            by_step = {
                int(step_idx[i]): float(g) for i, g in zip(uniq_indices, remaining)
            }
            for i in indices:
                targets[i] = by_step[int(step_idx[i])]
        else:
            rewards = [float(step_rewards[i]) for i in indices]
            remaining = compute_remaining_returns(rewards, gamma=gamma)
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



def two_hot_target_mass_diagnostics(
    bin_logits: torch.Tensor,
    target_returns: torch.Tensor,
    *,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> dict[str, float]:
    """Diagnostics for two-hot CE: p_left/p_right/p_target_mass (NOT exp(-CE)).

    For two-hot targets with mass on bins (k, k+1) and weights (alpha, 1-alpha):
        CE = -alpha log p_left - (1-alpha) log p_right
    so exp(-CE) is NOT a probability of the target support.
    """
    if bin_logits.numel() == 0:
        return {}
    max_bin = bin_logits.shape[-1] - 1
    logits = bin_logits
    targets = target_returns
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))

    raw = targets.to(dtype=torch.float32)
    clipped = raw.clamp(min=float(spec.vmin), max=float(spec.vmax))
    clip_frac = float((raw != clipped).float().mean().item()) if raw.numel() else 0.0
    target_bins = ((clipped - float(spec.vmin)) / float(spec.step)).clamp(0.0, float(max_bin))
    k_low = target_bins.floor().to(dtype=torch.long).clamp(max=max_bin)
    k_high = (k_low + 1).clamp(max=max_bin)
    alpha = 1.0 - (target_bins - k_low.to(dtype=torch.float32))  # mass on left
    at_boundary = k_low >= max_bin
    alpha = torch.where(at_boundary, torch.ones_like(alpha), alpha)
    # two_hot_return_distribution uses w_low = 1 - (v - k_low) = alpha here

    gather_left = probs.gather(1, k_low.unsqueeze(1)).squeeze(1)
    gather_right = probs.gather(1, k_high.unsqueeze(1)).squeeze(1)
    p_target_mass = gather_left + gather_right
    # when left==right (edge), mass counted once in scatter but twice here — fix:
    same = k_low == k_high
    p_target_mass = torch.where(same, gather_left, p_target_mass)

    target_probs = two_hot_return_distribution(target_bins, max_bin=max_bin)
    two_hot_ce = -(target_probs * log_probs).sum(dim=-1)

    pred_argmax = probs.argmax(dim=-1)
    edge_frac = float(((k_low == 0) | (k_low >= max_bin) | (k_high >= max_bin)).float().mean().item())

    def _stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
        if x.numel() == 0:
            return {}
        xs = x.detach().float().cpu()
        return {
            f"{prefix}/mean": float(xs.mean().item()),
            f"{prefix}/median": float(xs.median().item()),
            f"{prefix}/p01": float(xs.quantile(0.01).item()),
            f"{prefix}/p05": float(xs.quantile(0.05).item()),
            f"{prefix}/p95": float(xs.quantile(0.95).item()),
        }

    out: dict[str, float] = {}
    out.update(_stats(gather_left, "actor_value/p_left"))
    out.update(_stats(gather_right, "actor_value/p_right"))
    out.update(_stats(p_target_mass, "actor_value/p_target_mass"))
    out.update(_stats(alpha, "actor_value/two_hot_alpha"))
    out.update(_stats(two_hot_ce, "actor_value/two_hot_CE"))
    out.update(_stats(k_low.float(), "actor_value/target_left_bin"))
    out.update(_stats(k_high.float(), "actor_value/target_right_bin"))
    out.update(_stats(pred_argmax.float(), "actor_value/pred_argmax_bin"))
    out["actor_value/edge_bin_fraction"] = edge_frac
    out["actor_value/target_clipping_fraction"] = clip_frac
    for thr in (1e-4, 1e-3, 0.01, 0.05, 0.1):
        out[f"actor_value/p_target_mass_lt_{thr:g}"] = float((p_target_mass < thr).float().mean().item())
    for thr in (0.5, 0.9):
        out[f"actor_value/p_target_mass_gt_{thr:g}"] = float((p_target_mass > thr).float().mean().item())
    # Explicitly log that exp(-CE) is NOT p_target (debug only mean).
    out["actor_value/exp_neg_CE_mean_NOT_p_target"] = float(torch.exp(-two_hot_ce).mean().item())
    return out



def categorical_mae_loss(
    probs: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """PaW-style categorical MAE: 0.5 * L1(p, y) averaged over batch.

    For one-hot y with mass on class k: 0.5 * ||p-y||_1 = 1 - p_k.
    For two-hot y the same formula is the natural generalization.
    probs/target_probs: (N, num_bins). Returns scalar mean loss.
    """
    if probs.numel() == 0:
        return probs.sum() * 0.0
    per = 0.5 * (probs - target_probs).abs().sum(dim=-1)
    return per.mean()


def return_token_value_loss(
    bin_logits: torch.Tensor,
    target_returns: torch.Tensor,
    *,
    target_encoding: str = "one_hot",
    value_loss_type: str = "ce",
    response_mask: torch.Tensor | None = None,
    spec: ReturnBinSpec = DEFAULT_RETURN_BIN_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Value loss over return bins (CE or categorical MAE). Targets are scalar returns in [vmin, vmax].

    value_loss_type:
      - "ce": NLL / soft CE (default; preserves historical behavior)
      - "mae": 0.5 * ||softmax(logits) - target||_1  (PaW-style; equals 1-p_y for one-hot)
    """
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
            value_loss_type=value_loss_type,
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

    loss_type = str(value_loss_type).lower()
    if loss_type not in ("ce", "mae"):
        raise ValueError(f"Unsupported actor_value_loss_type: {value_loss_type!r}")

    if encoding == "one_hot":
        num_bins = max_bin + 1
        target_probs = torch.zeros(
            hard_bins.size(0), num_bins, device=bin_logits.device, dtype=torch.float32
        )
        target_probs.scatter_(1, hard_bins.unsqueeze(1), 1.0)
        probs = log_probs.exp()
        if loss_type == "ce":
            loss = F.nll_loss(log_probs, hard_bins, reduction="mean")
        else:
            loss = categorical_mae_loss(probs, target_probs)
        diag = {
            "actor_value/value_loss_type": 0.0 if loss_type == "ce" else 1.0,
            "actor_value/mae_mean": float(categorical_mae_loss(probs, target_probs).detach().item()),
        }
    elif encoding == "two_hot":
        target_probs = two_hot_return_distribution(target_bins, max_bin=max_bin)
        per = -(target_probs * log_probs).sum(dim=-1)
        probs = log_probs.exp()
        if loss_type == "ce":
            loss = per.mean()
        else:
            loss = categorical_mae_loss(probs, target_probs)
        # Real target-support probabilities (not exp(-CE)).
        probs = log_probs.exp()
        k_low = target_bins.floor().to(dtype=torch.long).clamp(max=max_bin)
        k_high = (k_low + 1).clamp(max=max_bin)
        alpha = 1.0 - (target_bins - k_low.to(dtype=torch.float32))
        at_boundary = k_low >= max_bin
        alpha = torch.where(at_boundary, torch.ones_like(alpha), alpha)
        p_left = probs.gather(1, k_low.unsqueeze(1)).squeeze(1)
        p_right = probs.gather(1, k_high.unsqueeze(1)).squeeze(1)
        p_target_mass = torch.where(k_low == k_high, p_left, p_left + p_right)
        pred_argmax = probs.argmax(dim=-1)
        raw_t = target_returns.to(dtype=torch.float32)
        clipped = raw_t.clamp(min=float(spec.vmin), max=float(spec.vmax))
        edge_frac = float(((k_low == 0) | (k_high >= max_bin)).float().mean().item())
        clip_frac = float((raw_t != clipped).float().mean().item())
        bin_centers = (
            float(spec.vmin)
            + torch.arange(max_bin + 1, device=probs.device, dtype=torch.float32) * float(spec.step)
        )
        pred_value = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)
        # Pearson corr / explained var vs scalar targets (batch-local).
        t_c = raw_t - raw_t.mean()
        p_c = pred_value - pred_value.mean()
        denom = torch.sqrt((t_c * t_c).sum() * (p_c * p_c).sum()).clamp_min(1e-8)
        corr = float((t_c * p_c).sum().item() / denom.item()) if raw_t.numel() > 1 else 0.0
        resid = pred_value - raw_t
        var_t = float(torch.var(raw_t, unbiased=False).item()) if raw_t.numel() > 1 else 0.0
        var_r = float(torch.var(resid, unbiased=False).item()) if resid.numel() > 1 else 0.0
        explained = float(1.0 - (var_r / var_t)) if var_t > 1e-12 else 0.0
        mae_scalar = float(resid.abs().mean().item()) if resid.numel() else 0.0

        def _pct(x: torch.Tensor, q: float) -> float:
            if x.numel() == 0:
                return 0.0
            return float(torch.quantile(x.detach().float(), q).item())

        diag = {
            "actor_value/value_loss_type": 0.0 if loss_type == "ce" else 1.0,
            "actor_value/mae_mean": float(categorical_mae_loss(probs, target_probs).detach().item()),
            "actor_value/p_left_mean": float(p_left.mean().item()),
            "actor_value/p_right_mean": float(p_right.mean().item()),
            "actor_value/p_target_mass_mean": float(p_target_mass.mean().item()),
            "actor_value/p_target_mass_median": float(p_target_mass.median().item()),
            "actor_value/two_hot_alpha_mean": float(alpha.mean().item()),
            "actor_value/two_hot_CE_mean": float(per.mean().item()),
            "actor_value/target_left_bin_mean": float(k_low.float().mean().item()),
            "actor_value/target_right_bin_mean": float(k_high.float().mean().item()),
            "actor_value/pred_argmax_bin_mean": float(pred_argmax.float().mean().item()),
            "actor_value/edge_bin_fraction": edge_frac,
            "actor_value/target_clipping_fraction": clip_frac,
            "actor_value/exp_neg_CE_mean_NOT_p_target": float(torch.exp(-per).mean().item()),
            # User-facing aliases for Comet / dashboard.
            "value_target/min": float(raw_t.min().item()) if raw_t.numel() else 0.0,
            "value_target/max": float(raw_t.max().item()) if raw_t.numel() else 0.0,
            "value_target/mean": float(raw_t.mean().item()) if raw_t.numel() else 0.0,
            "value_target/std": float(raw_t.std(unbiased=False).item()) if raw_t.numel() else 0.0,
            "value_target/p01": _pct(raw_t, 0.01),
            "value_target/p99": _pct(raw_t, 0.99),
            "predicted_value/min": float(pred_value.min().item()) if pred_value.numel() else 0.0,
            "predicted_value/max": float(pred_value.max().item()) if pred_value.numel() else 0.0,
            "predicted_value/mean": float(pred_value.mean().item()) if pred_value.numel() else 0.0,
            "predicted_value/std": float(pred_value.std(unbiased=False).item()) if pred_value.numel() else 0.0,
            "edge_bin_fraction": edge_frac,
            "clipping_fraction": clip_frac,
            "target_left_bin": float(k_low.float().mean().item()),
            "target_right_bin": float(k_high.float().mean().item()),
            "target_alpha": float(alpha.mean().item()),
            "scalar_value_mae": mae_scalar,
            "corr_pred_target": corr,
            "explained_variance_env": explained,
        }
        for thr in (1e-4, 1e-3, 0.01, 0.05, 0.1):
            diag[f"actor_value/p_target_mass_lt_{thr:g}"] = float((p_target_mass < thr).float().mean().item())
        for thr in (0.5, 0.9):
            diag[f"actor_value/p_target_mass_gt_{thr:g}"] = float((p_target_mass > thr).float().mean().item())
        # richer percentiles via helper (keeps one source of truth)
        diag.update(
            {
                k: v
                for k, v in two_hot_target_mass_diagnostics(
                    bin_logits, target_returns, spec=spec
                ).items()
                if k not in diag or k.endswith(("/p01", "/p05", "/p95", "/median", "/mean"))
            }
        )
    else:
        raise ValueError(f"Unsupported actor_value_target_encoding: {target_encoding!r}")

    # Attach diagnostics on the loss tensor for optional upstream logging.
    try:
        loss._actor_value_diag = diag  # type: ignore[attr-defined]
    except Exception:
        pass
    return loss, hard_bins


def return_bin_distribution_entropy(bin_logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy over return bins. bin_logits: (B, num_bins). Returns (B,) entropy."""
    log_probs = F.log_softmax(bin_logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)
