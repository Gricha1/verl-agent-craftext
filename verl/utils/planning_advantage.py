"""Planning advantage utilities (ported from fsdp_sft_trainer for online PPO)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from verl.utils.model import compute_position_id_with_mask


def init_action_token_cache(tokenizer, device: torch.device):
    """Return (global_action_ids, global_to_local, id_to_char)."""
    from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings

    ids: list[int] = []
    id_to_char: dict[int, str] = {}
    for ch in action_token_strings():
        piece = tokenizer.encode(ch, add_special_tokens=False)
        if len(piece) != 1:
            raise ValueError(f"Action token {ch!r} must encode to one id, got {piece!r}")
        tid = int(piece[0])
        ids.append(tid)
        id_to_char[tid] = ch
    global_ids = torch.tensor(ids, dtype=torch.long, device=device)
    global_to_local = {tid: i for i, tid in enumerate(ids)}
    return global_ids, global_to_local, id_to_char


def tokenize_planning_prompt_only(
    tokenizer,
    prompt_text: str,
    *,
    max_length: int,
    reserve_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Left-pad-free prompt with trailing pad; leave ``reserve_tokens`` slots for in-place actions."""
    token_budget = int(max_length) - int(reserve_tokens)
    if token_budget <= 0:
        raise ValueError(f"max_length={max_length} too small for reserve_tokens={reserve_tokens}")

    chat = [{"role": "user", "content": str(prompt_text)}]
    prompt_str = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    tok = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    input_ids = tok["input_ids"][0]
    attention_mask = tok["attention_mask"][0]
    if int(input_ids.shape[0]) > token_budget:
        input_ids = input_ids[:token_budget]
        attention_mask = attention_mask[:token_budget]
    seq_len = int(input_ids.shape[0])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    if seq_len < max_length:
        pad_n = max_length - seq_len
        input_ids = torch.cat([input_ids, torch.full((pad_n,), int(pad_id), dtype=input_ids.dtype)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_n, dtype=attention_mask.dtype)])
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    position_ids = compute_position_id_with_mask(attention_mask).to(device)
    return input_ids, attention_mask, position_ids, seq_len


def sample_plan_actions_no_grad(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    action_token_ids: torch.Tensor,
    num_actions: int,
) -> torch.Tensor:
    """Sample H action tokens autoregressively (no grad). Returns (B, H) global token ids."""
    bsz = input_ids.shape[0]
    device = input_ids.device
    ids = input_ids.clone()
    attn = attention_mask.clone()
    pos = position_ids.clone()
    chosen: list[torch.Tensor] = []

    for _ in range(max(1, int(num_actions))):
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        logits = out.logits
        last_idx = attn.sum(dim=1).long() - 1
        step_logits = logits[torch.arange(bsz, device=device), last_idx, :]
        masked = torch.full_like(step_logits, float("-inf"))
        masked[:, action_token_ids] = step_logits[:, action_token_ids]
        probs = F.softmax(masked, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        chosen.append(token)

        write_pos = attn.sum(dim=1).long()
        if (write_pos >= ids.shape[1]).any():
            raise RuntimeError(
                f"Planning sample ran out of sequence space (seq_len={ids.shape[1]}). "
                "Increase max_prompt_length or reduce horizon."
            )
        rows = torch.arange(bsz, device=device)
        ids[rows, write_pos] = token
        attn[rows, write_pos] = 1
        pos = compute_position_id_with_mask(attn)

    return torch.stack(chosen, dim=1) if chosen else torch.zeros(bsz, 0, device=device, dtype=torch.long)


def plan_logprob_on_actions(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    action_token_ids: torch.Tensor,
    *,
    global_action_ids: torch.Tensor,
    global_to_local: dict[int, int],
    return_masked_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sum log p(fixed action tokens | prompt); optional mean per-step masked action entropy."""
    bsz, h = action_token_ids.shape
    device = input_ids.device
    ids = input_ids.clone()
    attn = attention_mask.clone()
    prompt_lens = attn.sum(dim=1).long()
    rows = torch.arange(bsz, device=device)

    seq_cap = ids.shape[1]
    if (prompt_lens + h > seq_cap).any():
        raise RuntimeError(
            f"Planning prompt fills max_length; cannot append {h} action tokens (seq_cap={seq_cap})."
        )

    for k in range(h):
        write_pos = attn.sum(dim=1).long()
        ids[rows, write_pos] = action_token_ids[:, k]
        attn[rows, write_pos] = 1

    pos = compute_position_id_with_mask(attn)
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
    logits = out.logits

    logp_sum = torch.zeros(bsz, device=device, dtype=torch.float32)
    ent_steps: list[torch.Tensor] = []
    for k in range(h):
        logit_idx = (prompt_lens + k - 1).clamp(0, logits.shape[1] - 1)
        step_logits = logits[rows, logit_idx, :]
        action_logits = step_logits[:, global_action_ids]
        log_probs_a = F.log_softmax(action_logits, dim=-1)
        local_idx = torch.tensor(
            [global_to_local[int(tid)] for tid in action_token_ids[:, k].tolist()],
            device=device,
            dtype=torch.long,
        )
        logp_sum = logp_sum + log_probs_a[rows, local_idx]
        if return_masked_entropy:
            probs = log_probs_a.exp()
            ent_steps.append(-(probs * (log_probs_a + 1e-8)).sum(dim=-1))

    if return_masked_entropy:
        ent_stack = torch.stack(ent_steps, dim=1)
        return logp_sum, ent_stack.mean()
    return logp_sum


def step_action_entropy_topm(action_logits: torch.Tensor, *, top_m: int = 8) -> torch.Tensor:
    top_m = min(int(top_m), int(action_logits.shape[-1]))
    top_m = max(1, top_m)
    top_vals, _ = torch.topk(action_logits, k=top_m, dim=-1)
    log_probs = F.log_softmax(top_vals, dim=-1)
    probs = log_probs.exp()
    return -(probs * (log_probs + 1e-8)).sum(dim=-1)


@torch.no_grad()
def score_plans_with_plan_q(
    model,
    tokenizer,
    *,
    states: Sequence[str],
    tasks: Sequence[str],
    action_token_ids: torch.Tensor,
    id_to_char: dict[int, str],
    return_spec,
    max_prompt_length: int,
    device: torch.device,
    executed_actions: Sequence[Sequence[str]] | None = None,
) -> torch.Tensor:
    """Greedy plan-Q decode: (state, plan) -> scalar return estimate."""
    from agent_system.environments.env_package.caged_craftext.return_tokens import decode_return_token
    from agent_system.environments.prompts.world_model_reward_craftext_plan_q import (
        format_craftext_plan_q_prompt,
    )
    from verl.utils.actor_value_token import constrained_generate_return_token

    h = int(action_token_ids.shape[1])
    from agent_system.environments.env_package.caged_craftext.return_tokens import (
        return_token_legend_compact_for_spec,
    )

    legend = return_token_legend_compact_for_spec(return_spec)

    scores: list[float] = []
    for i in range(action_token_ids.shape[0]):
        plan_chars = [
            id_to_char.get(int(tid), "") for tid in action_token_ids[i].tolist()
        ]
        plan_chars = [c for c in plan_chars if c]
        if len(plan_chars) < h:
            scores.append(0.0)
            continue
        past = []
        if executed_actions is not None and i < len(executed_actions):
            past = list(executed_actions[i] or [])
        prompt = format_craftext_plan_q_prompt(
            task=str(tasks[i] or ""),
            state=str(states[i] or ""),
            plan_actions=plan_chars,
            return_bin_legend=legend,
            executed_actions=past,
        )
        chat = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        tok = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        position_ids = compute_position_id_with_mask(attention_mask).to(device)
        token_id, _ = constrained_generate_return_token(
            model,
            input_ids,
            attention_mask,
            position_ids,
            tokenizer,
            temperature=0.1,
            do_sample=False,
            spec=return_spec,
        )
        tok_str = tokenizer.decode([int(token_id.item())], skip_special_tokens=True).strip()
        try:
            scores.append(float(decode_return_token(tok_str, spec=return_spec)))
        except Exception:
            scores.append(0.0)
    return torch.tensor(scores, dtype=torch.float32, device=action_token_ids.device)


@torch.no_grad()
def score_plans_with_reward_wm(
    model,
    tokenizer,
    *,
    states: Sequence[str],
    tasks: Sequence[str],
    action_token_ids: torch.Tensor,
    id_to_char: dict[int, str],
    horizon: int,
    max_prompt_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Greedy reward WM: sum of H predicted step rewards (matches offline SFT g_hat)."""
    from agent_system.environments.env_package.caged_craftext.reward_tokens import (
        parse_reward_token_sequence,
        reward_token_strings,
        sum_quantized_rewards,
    )
    from agent_system.environments.prompts.world_model_reward import format_reward_prompt
    from verl.utils.model import compute_position_id_with_mask

    reward_toks = list(reward_token_strings())
    reward_tok_ids = []
    id_to_reward_tok: dict[int, str] = {}
    for t in reward_toks:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            reward_tok_ids.append(int(ids[0]))
            id_to_reward_tok[int(ids[0])] = t
    reward_ids_tensor = torch.tensor(reward_tok_ids, dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id

    h = int(horizon)
    scores: list[float] = []

    def _masked_greedy(step_logits: torch.Tensor) -> int:
        masked = torch.full_like(step_logits, float("-inf"))
        masked[reward_ids_tensor] = step_logits[reward_ids_tensor]
        return int(masked.argmax().item())

    for i in range(action_token_ids.shape[0]):
        acts = [id_to_char.get(int(tid), "") for tid in action_token_ids[i].tolist()]
        acts = [a for a in acts if a]
        if len(acts) < h:
            scores.append(0.0)
            continue
        prompt = format_reward_prompt(
            state=str(states[i] or ""),
            action=acts[0],
            task=str(tasks[i] or ""),
            horizon=h,
            actions=acts[:h],
        )
        chat = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        tok = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        position_ids = compute_position_id_with_mask(attention_mask).to(device)
        decoded_chars: list[str] = []
        for _ in range(h):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            seq_len = int(attention_mask.sum(dim=1).item()) - 1
            step_logits = out.logits[0, seq_len, :]
            next_id = _masked_greedy(step_logits)
            if eos_id is not None and next_id == int(eos_id):
                break
            decoded_chars.append(id_to_reward_tok[next_id])
            next_t = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_t], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = compute_position_id_with_mask(attention_mask).to(device)
        pred = "".join(decoded_chars)
        vals = parse_reward_token_sequence(pred)
        if len(vals) >= h:
            scores.append(float(sum_quantized_rewards(vals[:h])))
        else:
            scores.append(float(sum(vals)) if vals else 0.0)
    return torch.tensor(scores, dtype=torch.float32, device=action_token_ids.device)
