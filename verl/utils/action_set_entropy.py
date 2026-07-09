"""
Entropy over the discrete environment action set (e.g. 17 Craftext actions).

For text policies we score canonical strings ``<action>DOWN</action>`` via teacher-forced
token log-probs, then compute categorical entropy over those scores only:

    log_scores[b, a] = sum_t log p(token_t | prompt, canonical_action_a)
    H_b = -sum_a softmax(log_scores / T)_a * log softmax(...)_a
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from verl.utils.torch_functional import logprobs_from_logits


def canonical_action_text(action_name: str, *, single_token: bool = False) -> str:
    if single_token:
        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            action_token_label,
        )
        from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT

        idx = ACTION_TO_TEXT.index(action_name)
        return action_token_label(idx)
    return f"<action>{action_name}</action>"


def default_action_names() -> Tuple[str, ...]:
    from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT

    return ACTION_TO_TEXT


@lru_cache(maxsize=8)
def _load_canonical_action_token_cache(
    tokenizer_path: str,
    trust_remote_code: bool,
    single_token_actions: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns:
        padded_ids: (num_actions, max_len)
        lengths: (num_actions,) int64
        num_actions: int
    """
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    token_rows: List[List[int]] = []
    for name in default_action_names():
        text = canonical_action_text(name, single_token=single_token_actions)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Empty tokenization for canonical action: {text!r}")
        if single_token_actions and len(ids) != 1:
            raise ValueError(
                f"Expected single token for action {name!r}, got {len(ids)} tokens for {text!r}"
            )
        token_rows.append(ids)

    num_actions = len(token_rows)
    max_len = max(len(r) for r in token_rows)
    padded = torch.full((num_actions, max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(num_actions, dtype=torch.long)
    for j, row in enumerate(token_rows):
        lengths[j] = len(row)
        padded[j, : len(row)] = torch.tensor(row, dtype=torch.long)
    return padded, lengths, num_actions


def build_prompt_action_sequences(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    canonical_padded: torch.Tensor,
    canonical_lengths: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build a batch of size (B * num_actions) with prompt + each canonical action response.

    Returns:
        input_ids, attention_mask, position_ids, prompt_length
    """
    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    num_actions, max_action_len = canonical_padded.shape
    seq_len = prompt_length + max_action_len

    flat_batch = batch_size * num_actions
    input_ids = torch.full((flat_batch, seq_len), pad_token_id, dtype=prompt_ids.dtype, device=device)
    attention_mask = torch.zeros(flat_batch, seq_len, dtype=prompt_mask.dtype, device=device)

    for i in range(batch_size):
        p_ids = prompt_ids[i]
        p_valid = prompt_mask[i].bool()
        for j in range(num_actions):
            action_len = int(canonical_lengths[j].item())
            row = i * num_actions + j
            input_ids[row, :prompt_length] = p_ids
            input_ids[row, prompt_length : prompt_length + action_len] = canonical_padded[j, :action_len]
            attention_mask[row, :prompt_length] = p_valid.to(attention_mask.dtype)
            attention_mask[row, prompt_length : prompt_length + action_len] = 1

    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    position_ids = position_ids * attention_mask
    return input_ids, attention_mask, position_ids, prompt_length


def build_prompt_action_batch_for_one_action(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    action_token_ids: torch.Tensor,
    action_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch size B with one shared canonical action response. seq_len = prompt_len + action_len."""
    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    seq_len = prompt_length + action_len
    input_ids = torch.full(
        (batch_size, seq_len), pad_token_id, dtype=prompt_ids.dtype, device=device
    )
    attention_mask = torch.zeros(batch_size, seq_len, dtype=prompt_mask.dtype, device=device)
    input_ids[:, :prompt_length] = prompt_ids
    input_ids[:, prompt_length : prompt_length + action_len] = action_token_ids[:action_len]
    attention_mask[:, :prompt_length] = prompt_mask
    attention_mask[:, prompt_length : prompt_length + action_len] = 1
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask
    return input_ids, attention_mask, position_ids


def action_vocab_ids_from_canonical(
    canonical_padded: torch.Tensor,
    canonical_lengths: torch.Tensor,
) -> torch.Tensor:
    """Vocab ids for each action (single-token actions only). Shape (num_actions,)."""
    if int(canonical_lengths.max().item()) != 1:
        raise ValueError("action_vocab_ids_from_canonical requires single-token actions")
    rows = []
    for j in range(canonical_lengths.shape[0]):
        rows.append(int(canonical_padded[j, 0].item()))
    return torch.tensor(rows, dtype=torch.long, device=canonical_padded.device)


def action_log_scores_from_next_token_logits(
    next_token_logits: torch.Tensor,
    action_vocab_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Fast path for single-token actions: one forward, logits at last prompt position.

    log_score[b, a] = log p(action_token_a | prompt) under the full-vocab distribution.
    Same values as teacher-forced log-score when each action is one token.

    Args:
        next_token_logits: (B, vocab_size)
        action_vocab_ids: (num_actions,) token ids aligned with canonical action order

    Returns:
        log_scores: (B, num_actions)
    """
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    return log_probs.index_select(dim=-1, index=action_vocab_ids.to(log_probs.device))


def action_log_scores_from_dynamic_admissible_vocab(
    next_token_logits: torch.Tensor,
    admissible_vocab_ids: torch.Tensor,
    admissible_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample admissible action sets (AlfWorld). Invalid padded slots use -1e9."""
    log_probs = F.log_softmax(next_token_logits, dim=-1)
    gathered = log_probs.gather(-1, admissible_vocab_ids.to(log_probs.device))
    return gathered.masked_fill(~admissible_mask.to(gathered.device), -1e9)


def action_log_scores_one_action_batch(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    action_len: int,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Per-sample log-score for one canonical action. Returns (batch_size,)."""
    if action_len == 0:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    start = prompt_length - 1
    end = prompt_length + action_len - 1
    resp_logits = logits[:, start:end, :]
    resp_labels = input_ids[:, prompt_length : prompt_length + action_len]
    token_log_probs = logprobs_from_logits(
        resp_logits,
        resp_labels,
        inplace_backward=False,
    )
    scores = token_log_probs.sum(dim=-1)
    if length_normalize:
        scores = scores / action_len
    return scores


def action_log_scores_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    canonical_lengths: torch.Tensor,
    batch_size: int,
    length_normalize: bool = True,
) -> torch.Tensor:
    """
    Sum token log-probs for each canonical action string (teacher forcing).

    Args:
        logits: (B * A, seq_len, vocab)
        input_ids: same batch layout
        canonical_lengths: (A,)
        batch_size: B

    Returns:
        log_scores: (B, A)
    """
    num_actions = canonical_lengths.shape[0]
    device = logits.device
    log_scores = torch.zeros(batch_size, num_actions, device=device, dtype=logits.dtype)

    for i in range(batch_size):
        for j in range(num_actions):
            row = i * num_actions + j
            action_len = int(canonical_lengths[j].item())
            if action_len == 0:
                continue
            # Predict token at t using logits at t-1.
            start = prompt_length - 1
            end = prompt_length + action_len - 1
            resp_logits = logits[row, start:end, :]
            resp_labels = input_ids[row, prompt_length : prompt_length + action_len]
            token_log_probs = logprobs_from_logits(
                resp_logits.unsqueeze(0),
                resp_labels.unsqueeze(0),
                inplace_backward=False,
            ).squeeze(0)
            score = token_log_probs.sum()
            if length_normalize:
                score = score / action_len
            log_scores[i, j] = score
    return log_scores


def entropy_loss_over_action_scores(
    log_scores: torch.Tensor,
    temperature: float = 1.0,
    sample_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Categorical entropy over ``num_actions`` candidates (not summed per-action entropies).

    Returns:
        scalar loss (mean or masked mean over batch),
        per-sample entropy (B,)
    """
    temp = max(float(temperature), 1e-8)
    log_probs = F.log_softmax(log_scores / temp, dim=-1)
    probs = log_probs.exp()
    entropy_per_sample = -(probs * log_probs).sum(dim=-1)

    if sample_mask is not None:
        mask = sample_mask.to(dtype=entropy_per_sample.dtype, device=entropy_per_sample.device)
        denom = mask.sum().clamp(min=1.0)
        loss = (entropy_per_sample * mask).sum() / denom
    else:
        loss = entropy_per_sample.mean()
    return loss, entropy_per_sample
