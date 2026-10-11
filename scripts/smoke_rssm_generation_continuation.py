#!/usr/bin/env python3
"""Verify that Base and soft-conditioned HZ decoding return new continuations.

This is intentionally checkpoint-free: its purpose is to test the shared
``_generate`` prefix slicing implementation before a costly QA run.  HZ uses
ten ordinary Qwen embedding rows only as shape-valid stand-ins for 5 h + 5 z
soft tokens; it is not an RSSM-quality evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from rssm_env_validation import _generate


def prompt(task: str, observation: str | None) -> str:
    state = f"This is what you currently see:\n{observation}\n\n" if observation else "The current state is supplied only through latent tokens.\n\n"
    return (f"Your goal is to complete the following task:\n**TASK:** {task}\n\n{state}"
            "First, think about what to do next. Then choose one available action and write it in <action>ACTION</action>.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    task = "Go to the wooden block."
    base_prompt = prompt(task, "[8x8 map omitted in this decoder-only smoke]")
    hz_prompt = prompt(task, None)
    embedding_rows = model.get_input_embeddings().weight.detach()[:10]
    records = []
    for name, text, soft in (("base", base_prompt, None), ("hz", hz_prompt, embedding_rows)):
        assistant_prefix = "<think>\n"
        raw = _generate(model, tokenizer, text, soft, args.max_new_tokens, assistant_prefix=assistant_prefix, generation={"do_sample": False})
        normalized = " ".join(raw[len(assistant_prefix):].split())
        prompt_echo = len(normalized) >= 16 and normalized in " ".join(text.split())
        if not normalized or prompt_echo:
            raise RuntimeError(f"{name} decoded a prompt prefix instead of new output: {raw!r}")
        records.append({"model": name, "prompt": text, "raw_output": raw, "prompt_echo": prompt_echo})
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
