#!/usr/bin/env python3
"""HF LoRA checkpoint as single-token action policy (PPO / SFT rollout)."""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_base_model(checkpoint: str, base_model: str | None) -> str:
    if base_model:
        return base_model
    adapter_cfg = os.path.join(checkpoint, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("base_model_name_or_path")
        if name:
            return str(name)
    raise ValueError(f"Cannot infer base model from {checkpoint}; pass base_model.")


def _attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _masked_greedy_token_id(step_logits: torch.Tensor, allowed_ids: torch.Tensor) -> int:
    masked = torch.full_like(step_logits, float("-inf"))
    masked[allowed_ids] = step_logits[allowed_ids]
    return int(masked.argmax(dim=-1).item())


class HfSingleTokenPolicy:
    """Greedy or sampled one action token (masked to 17 Craftext labels)."""

    def __init__(
        self,
        *,
        checkpoint: str,
        base_model: str | None = None,
        device: str = "cuda:0",
        temperature: float = 1.0,
    ) -> None:
        from peft import PeftModel

        from verl.utils.model import compute_position_id_with_mask

        self.device = torch.device(device)
        self.temperature = max(0.0, float(temperature))
        self._compute_position_id_with_mask = compute_position_id_with_mask

        base = _resolve_base_model(checkpoint, base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype,
            attn_implementation=_attn_implementation(),
            trust_remote_code=True,
        )
        adapter_files = (
            os.path.join(checkpoint, "adapter_model.safetensors"),
            os.path.join(checkpoint, "adapter_model.bin"),
        )
        if any(os.path.isfile(p) for p in adapter_files):
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        elif os.path.isfile(os.path.join(checkpoint, "config.json")):
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=dtype,
                attn_implementation=_attn_implementation(),
                trust_remote_code=True,
            )
        else:
            raise FileNotFoundError(f"No HF weights in {checkpoint}")

        self.model = model.to(self.device).eval()

        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            action_token_strings,
        )

        action_toks = list(action_token_strings())
        action_tok_ids: list[int] = []
        self._id_to_action_tok: dict[int, str] = {}
        for t in action_toks:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                tid = int(ids[0])
                action_tok_ids.append(tid)
                self._id_to_action_tok[tid] = t
        self._action_ids_tensor = torch.tensor(action_tok_ids, dtype=torch.long, device=self.device)
        self._eos_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def action_token(self, prompt_text: str) -> str:
        chat = [{"role": "user", "content": str(prompt_text)}]
        prompt_str = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        position_ids = self._compute_position_id_with_mask(attention_mask).to(self.device)

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )
        with ctx:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            seq_len = int(attention_mask.sum(dim=1).item()) - 1
            step_logits = out.logits[0, seq_len, :]

        masked = torch.full_like(step_logits, float("-inf"))
        masked[self._action_ids_tensor] = step_logits[self._action_ids_tensor]

        if self.temperature <= 1e-6:
            next_id = int(masked.argmax(dim=-1).item())
        else:
            probs = torch.softmax(masked / self.temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if self._eos_id is not None and next_id == int(self._eos_id):
            return ""
        return self._id_to_action_tok.get(next_id, "")
