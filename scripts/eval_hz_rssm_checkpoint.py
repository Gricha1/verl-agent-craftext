#!/usr/bin/env python3
"""Read-only HZ-actor / Plan6-HZ evaluation of a saved RSSM checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from rssm_env_validation import run_env_validation
from train_qwen_transition_rssm_recon import QwenTransitionRSSMRecon, posterior, transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.run_dir)
    cfg = yaml.safe_load((root / 'resolved_config.yaml').read_text())
    cfg.update({'env_eval_enabled': True, 'env_eval_num_episodes': int(args.episodes),
                'env_eval_full_latent_trace_episodes': min(3, int(args.episodes))})
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    qwen = AutoModelForCausalLM.from_pretrained(cfg['base_model'], torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True).to(device).eval()
    for parameter in qwen.parameters(): parameter.requires_grad_(False)
    model = QwenTransitionRSSMRecon(cfg, qwen.config.hidden_size).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    metrics = run_env_validation(qwen, tokenizer, model, device, cfg, transition, episodes=args.episodes,
                                 debug_path=root / f'eval_hz_{Path(args.checkpoint).stem}_{args.episodes}seeds.json',
                                 modes=('base', 'hz_actor', 'plan6_hz'), posterior_fn=posterior)
    output = root / f'eval_hz_{Path(args.checkpoint).stem}_{args.episodes}seeds_metrics.json'
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
