#!/usr/bin/env python3
"""Fast zero-shot A/B for Craftext reasoning-memory rendering.

This intentionally bypasses PPO, FSDP, critic and gradient updates.  It keeps
the actor prompt, environment manager and action parser used by training, so
the only switch is whether a raw reply without ``<action>`` is remembered.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.environments import make_envs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-reasoning-on-missing-action", action="store_true")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Use temperature 0 instead of sampling.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_config(args):
    return OmegaConf.create(
        {
            "env": {
                "env_name": "caged_craftext/CagedCraftextEnv",
                "craftext_settings": "debug_square_16x16",
                "seed": args.seed,
                "max_steps": args.max_steps,
                "history_length": 50,
                "reasoning_history_length": 3,
                "enable_reasoning": True,
                "prompt_template_type": "single_token_action_reasoning",
                "store_raw_reasoning_on_missing_action_tag": args.raw_reasoning_on_missing_action,
                "observation_type": "ascii",
                "rollout": {"n": 1},
                "resources_per_worker": {"num_cpus": 0.03},
                "use_jax_gpu": False,
                "use_optimistic_parallel": True,
                "optimistic_reset_ratio": 8,
                "use_ray_text_render_workers": False,
            },
            "data": {"train_batch_size": args.num_envs, "val_batch_size": 1},
        }
    )


def chat_prompts(tokenizer, texts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        for text in texts
    ]


def main():
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    config = build_config(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    torch.manual_seed(args.seed)

    # The environment uses JAX/Ray.  Initialise it after the actor model so no
    # environment worker competes for the actor GPU.
    envs, _ = make_envs(config)

    observations, _ = envs.reset({})
    active = np.ones(args.num_envs, dtype=bool)
    rewards_total = np.zeros(args.num_envs, dtype=np.float64)
    lengths = np.zeros(args.num_envs, dtype=np.int32)
    wins = np.zeros(args.num_envs, dtype=np.float64)
    valid_count = 0
    action_count = 0
    no_action_tag_count = 0
    raw_samples = []

    for step in range(args.max_steps):
        active_idx = np.flatnonzero(active)
        if not len(active_idx):
            break
        prompts = chat_prompts(tokenizer, [observations["text"][i] for i in active_idx])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=320,
                do_sample=not args.greedy,
                temperature=None if args.greedy else args.temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        responses = [response.strip() for response in responses]
        actions = ["NOOP"] * args.num_envs
        for env_idx, response in zip(active_idx, responses):
            actions[env_idx] = response
            no_action_tag_count += int("<action>" not in response.lower())
            if len(raw_samples) < 8:
                raw_samples.append({"step": step, "env": int(env_idx), "response": response})

        next_observations, rewards, dones, infos = envs.step(actions)
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        valid = np.asarray([bool(info.get("is_action_valid", False)) for info in infos], dtype=bool)
        valid_count += int(valid[active].sum())
        action_count += int(active.sum())
        rewards_total[active] += rewards[active]
        lengths[active] += 1
        for i in active_idx:
            wins[i] = float(bool(infos[i].get("won", False)))
        active &= ~dones
        observations = next_observations

    result = {
        "raw_reasoning_on_missing_action": args.raw_reasoning_on_missing_action,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "valid_action_rate": valid_count / action_count if action_count else 0.0,
        "success_rate": float(wins.mean()),
        "reward_mean": float(rewards_total.mean()),
        "episode_length_mean": float(lengths.mean()),
        "responses_without_action_tag_rate": no_action_tag_count / action_count if action_count else 0.0,
        "raw_samples": raw_samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
