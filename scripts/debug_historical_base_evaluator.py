#!/usr/bin/env python3
"""Reproduce and inspect the last known-good frozen-Qwen CrafText Base evaluator.

This is intentionally evaluation-only: it has no RSSM dependency, performs no
optimisation, and writes complete raw trajectories before any aggregate metric.
The defaults are the evaluator that produced ``env_base/success_rate=0.25`` in
the historical ``comet_b6ab78...`` export (commit b6304ff).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CAGED_CRAFTEXT_PATH", str(ROOT / "caged_craftext"))

from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT, craftext_projection


HISTORICAL_SEEDS = list(range(20261001, 20261033))


class DebugLocalVector:
    """Evaluator vector with either historical or r6 scenario sampling."""

    def __init__(self, cfg: Any, seeds: list[int], per_seed_reset_rng: bool):
        kwargs = {
            "config_name": str(cfg.env.craftext_settings),
            "use_debug_square_map": True,
            "observation_type": "ascii",
            "encode_form": "embedding",
        }
        self.workers = [CagedCraftextWorker(seed=int(seed), env_kwargs=kwargs) for seed in seeds]
        self.rngs = [np.random.RandomState(int(seed)) for seed in seeds]
        self.global_rng = np.random.RandomState(int(cfg.env.seed))
        self.per_seed_reset_rng = per_seed_reset_rng

    def reset(self):
        if self.per_seed_reset_rng:
            choices = [int(rng.choice(np.arange(3))) for rng in self.rngs]
        else:
            choices = self.global_rng.choice(np.arange(3), size=len(self.workers), replace=True).tolist()
        pairs = [worker.reset(scenario_idx=choice, return_render=False)
                 for worker, choice in zip(self.workers, choices)]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def step(self, actions):
        pairs = [worker.step(int(action), return_render=False) for worker, action in zip(self.workers, actions)]
        return ([pair[0] for pair in pairs], np.asarray([pair[1] for pair in pairs]),
                np.asarray([pair[2] for pair in pairs]), [pair[3] for pair in pairs])

    def close(self):
        for worker in self.workers:
            close = getattr(worker, "close", None)
            if close:
                close()


def environment_config(seed: int, max_steps: int, settings: str, store_raw_reasoning: bool) -> Any:
    return OmegaConf.create({"env": {
        "env_name": "caged_craftext/CagedCraftextEnv",
        "craftext_settings": settings,
        "seed": seed,
        "max_steps": max_steps,
        "history_length": 50,
        "reasoning_history_length": 3,
        "enable_reasoning": True,
        "prompt_template_type": "single_token_action_reasoning",
        "store_raw_reasoning_on_missing_action_tag": store_raw_reasoning,
        "observation_type": "ascii",
        "auto_reset": False,
        "use_jax_gpu": False,
        "jax_gpu_fraction": 0.0,
        "use_optimistic_parallel": False,
        "use_ray_text_render_workers": False,
    }, "data": {"train_batch_size": 1, "val_batch_size": 1}})


def parse_action(raw: str) -> tuple[str | None, bool]:
    ids, valid = craftext_projection([raw])
    ok = bool(valid[0]) and int(ids[0]) >= 0
    return (str(ACTION_TO_TEXT[int(ids[0])]) if ok else None), ok


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


@torch.no_grad()
def evaluate(model, tokenizer, device: torch.device, seeds: list[int], max_steps: int,
             max_new_tokens: int, temperature: float, settings: str, do_sample: bool,
             per_seed_reset_rng: bool, store_raw_reasoning: bool) -> dict[str, Any]:
    """Historical batched sampled rollout with a complete trace for every action."""
    cfg = environment_config(seeds[0], max_steps, settings, store_raw_reasoning)
    vector = DebugLocalVector(cfg, seeds, per_seed_reset_rng)
    env = CagedCraftextEnvironmentManager(vector, craftext_projection, cfg)
    model.eval()
    records = [{"seed": int(seed), "steps": [], "cumulative_reward": 0.0} for seed in seeds]
    actions: list[list[str | None]] = [[] for _ in seeds]
    try:
        observations, reset_infos = env.reset({})
        for record, info, prompt in zip(records, reset_infos, observations["text"]):
            record["reset_info"] = jsonable(info)
            record["initial_observation"] = str(prompt)
        active = np.ones(len(seeds), dtype=bool)
        for step in range(max_steps):
            indexes = np.flatnonzero(active)
            if not len(indexes):
                break
            user_prompts = [str(observations["text"][index]) for index in indexes]
            rendered = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
                        for prompt in user_prompts]
            batch = tokenizer(rendered, return_tensors="pt", padding=True,
                              add_special_tokens=False).to(device)
            # The historical evaluator used one batch-wide RNG stream per
            # step.  Keeping it for greedy runs too makes the trace comparable.
            torch.manual_seed(int(seeds[0]) * 1000 + step)
            generated = model.generate(
                **batch, do_sample=do_sample, temperature=temperature, top_p=1.0, top_k=0,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            raw_outputs = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
            responses = ["<action>NOOP</action>"] * len(seeds)
            parsed: dict[int, tuple[str | None, bool]] = {}
            for index, raw in zip(indexes, raw_outputs):
                parsed[index] = parse_action(raw)
                responses[index] = raw
            next_observations, rewards, dones, infos = env.step(responses)
            for local, index in enumerate(indexes):
                parsed_action, parse_ok = parsed[index]
                info = infos[index]
                executed = str(info.get("action_name", "NOOP"))
                record = records[index]
                record["cumulative_reward"] += float(rewards[index])
                record["steps"].append({
                    "step": int(step),
                    "observation": user_prompts[local],
                    "messages": [{"role": "user", "content": user_prompts[local]}],
                    "rendered_chat_prompt": rendered[local],
                    "assistant_prefix": "",
                    "raw_qwen_generation": raw_outputs[local],
                    "parsed_action": parsed_action,
                    "parser_valid": parse_ok,
                    "fallback_used": not parse_ok,
                    "executed_action": executed,
                    "repeated_action": bool(actions[index] and actions[index][-1] == executed),
                    "next_observation": str(next_observations["text"][index]),
                    "reward": float(rewards[index]),
                    "terminated": bool(dones[index]),
                    "truncated": bool(info.get("truncated", False) or info.get("TimeLimit.truncated", False)),
                    "success_flag": bool(info.get("won", False) or info.get("instruction_done", False)),
                    "env_info": jsonable(info),
                })
                actions[index].append(executed)
                if bool(dones[index]):
                    active[index] = False
            observations = next_observations
    finally:
        env.close()
    for record, history in zip(records, actions):
        final = record["steps"][-1] if record["steps"] else {}
        record.update({
            "length": len(history),
            "success": float(bool(final.get("success_flag", False))),
            "final_reward": float(final.get("reward", 0.0)),
            "terminated": bool(final.get("terminated", False)),
            "truncated": bool(final.get("truncated", False)),
            "final_observation": final.get("next_observation", record.get("initial_observation", "")),
        })
    total_steps = max(sum(record["length"] for record in records), 1)
    return {
        "profile": "historical_base_b6304ff" if per_seed_reset_rng and do_sample else "r6_base_compat",
        "environment": {"settings": settings, "max_steps": max_steps, "seeds": seeds,
                        "per_seed_scenario_rng": per_seed_reset_rng,
                        "store_raw_reasoning_on_missing_action_tag": store_raw_reasoning},
        "generation": {"do_sample": do_sample, "temperature": temperature, "top_p": 1.0,
                       "top_k": 0, "max_new_tokens": max_new_tokens, "assistant_prefix": ""},
        "metrics": {
            "success_rate": float(np.mean([record["success"] for record in records])),
            "mean_reward": float(np.mean([record["cumulative_reward"] for record in records])),
            "mean_episode_length": float(np.mean([record["length"] for record in records])),
            "raw_action_parse_rate": sum(step["parser_valid"] for record in records for step in record["steps"]) / total_steps,
            "fallback_rate": sum(step["fallback_used"] for record in records for step in record["steps"]) / total_steps,
            "executed_action_valid_rate": sum(bool(step["env_info"].get("is_action_valid", False)) for record in records for step in record["steps"]) / total_steps,
            "repeated_action_rate": sum(step["repeated_action"] for record in records for step in record["steps"]) / total_steps,
        },
        "episodes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=20261001)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--settings", default="debug_square_8x8")
    parser.add_argument("--greedy", action="store_true", help="Match r6's do_sample=False generation.")
    parser.add_argument("--global-reset-rng", action="store_true", help="Match r6's single reset RNG.")
    parser.add_argument("--store-raw-reasoning", action="store_true", help="Match r6 missing-action history behaviour.")
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True).to(device).eval()
    result = evaluate(model, tokenizer, device, seeds, args.max_steps, args.max_new_tokens,
                      args.temperature, args.settings, not args.greedy,
                      not args.global_reset_rng, args.store_raw_reasoning)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
