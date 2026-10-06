#!/usr/bin/env python3
"""Read-only checkpoint evaluator for the RSSM reconstruction experiment.

It never constructs an optimizer and never writes the checkpoint.  The JSON
artifact contains every environment interaction, including invalid output and
parser fallback, so headline success metrics remain auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from rssm_env_validation import run_env_validation
from train_qwen_transition_rssm_recon import QwenTransitionRSSMRecon, transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=20261001)
    parser.add_argument("--settings", default="debug_square_8x8")
    parser.add_argument("--modes", default="base,z_actor,plan_only,plan6")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--torch-seed", type=int, default=20261001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    # Historical map/seeds make all four modes directly comparable.  This
    # evaluator is intentionally separate from the trainer's live 8-episode
    # telemetry and does not alter a running training job.
    cfg.update({
        "env_eval_enabled": True,
        "env_eval_num_episodes": args.episodes,
        "env_eval_seed": args.seed_start,
        "env_eval_craftext_settings": args.settings,
        "env_episode_max_steps": 50,
        "env_eval_history_length": 50,
        "env_eval_reasoning_history_length": 3,
        "env_eval_store_raw_reasoning_on_missing_action_tag": False,
        "env_eval_actor_max_new_tokens": args.max_new_tokens,
        "env_eval_do_sample": args.do_sample,
        "env_eval_temperature": args.temperature,
        "env_eval_top_p": 1.0,
        "env_eval_top_k": 0,
    })
    torch.manual_seed(args.torch_seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    qwen = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()
    for parameter in qwen.parameters():
        parameter.requires_grad_(False)
    cfg["soft_token_target_rms"] = float(qwen.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    model = QwenTransitionRSSMRecon(cfg, qwen.config.hidden_size).to(args.device).eval()
    loaded = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(loaded["model"], strict=True)
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    allowed = {"base", "z_actor", "plan_only", "plan6"}
    if not modes or set(modes) - allowed:
        raise ValueError(f"modes must be a nonempty subset of {sorted(allowed)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = run_env_validation(qwen, tokenizer, model, args.device, cfg, transition, debug_path=output, modes=modes)
    print(json.dumps({"checkpoint": str(args.checkpoint), "step": loaded.get("step"), "modes": modes, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
