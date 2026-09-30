#!/usr/bin/env python3
"""Collect reward-free CrafText transitions from the untrained Qwen actor.

The collector deliberately uses ``make_envs`` and therefore the production
``CagedCraftextEnvironmentManager`` for prompt construction, action parsing,
reasoning memory, and invalid-action behaviour.  It only persists the actor
trajectory; environment rewards are neither written nor consulted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

# ``python scripts/foo.py`` puts ``scripts/`` rather than the repository root
# on sys.path.  Keep this standalone entry point importable on every host.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CAGED_CRAFTEXT_PATH", str(REPO_ROOT / "caged_craftext"))

from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import craftext_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--fixed-layout", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Any:
    """Resolved subset of the active 16x16 DUAL reasoning rollout config."""
    env = {
        "env_name": "caged_craftext/CagedCraftextEnv",
        "craftext_settings": "debug_square_16x16",
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "history_length": 50,
        "reasoning_history_length": 3,
        "enable_reasoning": True,
        "prompt_template_type": "single_token_action_reasoning",
        "store_raw_reasoning_on_missing_action_tag": True,
        "observation_type": "ascii",
        "auto_reset": False,
        "rollout": {"n": 1},
        "resources_per_worker": {"num_cpus": 0.03},
        "use_jax_gpu": False,
        "jax_gpu_fraction": 0.15,
        "use_optimistic_parallel": True,
        "optimistic_reset_ratio": 8,
        "use_ray_text_render_workers": False,
    }
    if args.fixed_layout:
        env["fixed_debug_square_layout"] = True
    return OmegaConf.create({"env": env, "data": {"train_batch_size": int(args.num_envs), "val_batch_size": 1}})


def chat_prompts(tokenizer: AutoTokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        for prompt in prompts
    ]


class LocalCagedVector:
    """Local transport for the production manager, avoiding JAX-after-Ray fork.

    PPO normally puts these same workers behind Ray.  That is unsafe after this
    standalone process has initialized torch/JAX threads; prompt construction,
    parsing and stepping remain the production implementations.
    """

    def __init__(self, config: Any, num_envs: int):
        env_kwargs = {
            "config_name": str(config.env.craftext_settings),
            "use_debug_square_map": True,
            "observation_type": str(config.env.observation_type),
            "encode_form": "embedding",
        }
        if bool(getattr(config.env, "fixed_debug_square_layout", False)):
            env_kwargs["fixed_debug_square_layout"] = True
        self.workers = [
            CagedCraftextWorker(seed=int(config.env.seed) + index, env_kwargs=env_kwargs)
            for index in range(num_envs)
        ]
        # Same seeded sampling used by CagedCraftextMultiProcessEnv.reset().
        self._rng = np.random.RandomState(int(config.env.seed))

    def reset(self):
        # Production CagedCraftextMultiProcessEnv samples among all three
        # debug-square scenarios with its seeded NumPy RNG before calling each
        # worker.  Reproduce that selection exactly without Ray transport.
        scenario_indices = self._rng.choice(np.arange(3), size=len(self.workers), replace=True)
        pairs = [
            worker.reset(scenario_idx=int(scenario_idx), return_render=False)
            for worker, scenario_idx in zip(self.workers, scenario_indices)
        ]
        observations, infos = zip(*pairs)
        return list(observations), list(infos)

    def step(self, actions):
        results = [
            worker.step(int(action), return_render=False)
            for worker, action in zip(self.workers, actions)
        ]
        observations, rewards, dones, infos = zip(*results)
        return list(observations), np.asarray(rewards), np.asarray(dones), list(infos)

    def close(self):
        for worker in self.workers:
            close = getattr(worker, "close", None)
            if close is not None:
                close()


def response_metadata(tokenizer: AutoTokenizer, token_ids: torch.Tensor, response: str) -> dict[str, Any]:
    eos_id = tokenizer.eos_token_id
    ids = token_ids.tolist()
    return {
        "response_token_count": len(ids),
        "ended_with_eos": bool(eos_id is not None and eos_id in ids),
        "truncated_at_max_new_tokens": len(ids) >= 1,
        "raw_response": response,
    }


def jsonl_append(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.num_envs <= 0 or args.max_steps <= 0:
        raise SystemExit("--episodes, --num-envs, and --max-steps must be positive")
    if args.temperature <= 0:
        raise SystemExit("This collector is sampling-only; --temperature must be > 0")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    (args.output_dir / "collector_config.json").write_text(
        json.dumps(
            {
                "dataset_kind": "reward_free_latent_wm_transitions",
                "model": args.model,
                "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
                "environment": OmegaConf.to_container(config.env, resolve=True),
                "episodes_requested": args.episodes,
                "collector_uses_production_manager": True,
                "reward_stored_or_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    # This is the production manager and parser, not a hand-built prompt.  The
    # worker transport is local because Ray forks after JAX initialisation here.
    envs = CagedCraftextEnvironmentManager(
        LocalCagedVector(config, args.num_envs), craftext_projection, config
    )
    transitions_path = args.output_dir / "transitions.jsonl"
    episodes_path = args.output_dir / "episodes.jsonl"
    started = time.monotonic()
    episode_id = 0
    transition_count = 0
    try:
        with transitions_path.open("w", encoding="utf-8") as transitions, episodes_path.open("w", encoding="utf-8") as episodes:
            while episode_id < args.episodes:
                observations, infos = envs.reset({})
                active = np.ones(args.num_envs, dtype=bool)
                local_episode_ids = np.full(args.num_envs, -1, dtype=np.int64)
                local_steps = np.zeros(args.num_envs, dtype=np.int64)
                for i in range(args.num_envs):
                    if episode_id >= args.episodes:
                        active[i] = False
                        continue
                    local_episode_ids[i] = episode_id
                    episode_id += 1

                while bool(active.any()):
                    active_indices = np.flatnonzero(active).tolist()
                    actor_prompts = [str(observations["text"][i]) for i in active_indices]
                    encoded = tokenizer(
                        chat_prompts(tokenizer, actor_prompts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=3072,
                        add_special_tokens=False,
                    ).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            do_sample=True,
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            top_k=0 if int(args.top_k) < 0 else int(args.top_k),
                            max_new_tokens=int(args.max_new_tokens),
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    continuation = generated[:, encoded["input_ids"].shape[1] :]
                    raw_responses = tokenizer.batch_decode(continuation, skip_special_tokens=True)
                    # Supply empty strings for completed slots. The manager itself owns parsing.
                    responses = [""] * args.num_envs
                    metadata: dict[int, dict[str, Any]] = {}
                    prompt_by_index = dict(zip(active_indices, actor_prompts))
                    for j, index in enumerate(active_indices):
                        raw = raw_responses[j]
                        responses[index] = raw
                        meta = response_metadata(tokenizer, continuation[j], raw)
                        meta["truncated_at_max_new_tokens"] = bool(
                            meta["response_token_count"] >= int(args.max_new_tokens) and not meta["ended_with_eos"]
                        )
                        metadata[index] = meta

                    current_obs = [str(observations["anchor"][i]) for i in range(args.num_envs)]
                    next_observations, _rewards, dones, step_infos = envs.step(responses)
                    dones = np.asarray(dones, dtype=bool)
                    for index in active_indices:
                        info = step_infos[index]
                        action_id = int(info.get("action_id", -1))
                        parsed_action = str(info.get("action_name", "")) or None
                        row = {
                            "episode_id": int(local_episode_ids[index]),
                            "seed": int(args.seed + index),
                            "environment_worker_seed": int(args.seed + index),
                            "step": int(local_steps[index]),
                            "actor_prompt_t": prompt_by_index[index],
                            "observation_t": current_obs[index],
                            **metadata[index],
                            "reasoning_t": str(info.get("action_text", "")) if False else None,
                            "parsed_action_t": parsed_action,
                            "parsed_action_id_t": action_id,
                            "executed_action_t": parsed_action if action_id >= 0 else "INVALID_ACTION",
                            "action_parse_error": bool(not info.get("is_action_valid", False)),
                            "observation_t_plus_1": str(next_observations["anchor"][index]),
                            "env_done": bool(dones[index]),
                            "terminated": bool(dones[index] and (local_steps[index] + 1) < int(args.max_steps)),
                            "truncated": bool((local_steps[index] + 1) >= int(args.max_steps)),
                        }
                        # Reasoning is exactly what production stores for its next prompt.
                        try:
                            row["reasoning_t"] = str(envs.memory[index][-1].get("reasoning", ""))
                        except Exception:
                            row["reasoning_t"] = ""
                        jsonl_append(transitions, row)
                        transition_count += 1
                        local_steps[index] += 1
                        if bool(dones[index]) or local_steps[index] >= int(args.max_steps):
                            jsonl_append(
                                episodes,
                                {
                                    "episode_id": int(local_episode_ids[index]),
                                    "seed": int(args.seed + index),
                                    "environment_worker_seed": int(args.seed + index),
                                    "steps": int(local_steps[index]),
                                    "env_done": bool(dones[index]),
                                    "terminated": bool(dones[index] and local_steps[index] < int(args.max_steps)),
                                    "truncated": bool(local_steps[index] >= int(args.max_steps)),
                                },
                            )
                            active[index] = False
                    observations = next_observations
    finally:
        if hasattr(envs, "close"):
            envs.close()

    summary = {
        "episodes_collected": int(episode_id),
        "transitions_collected": int(transition_count),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "transitions_path": str(transitions_path),
        "episodes_path": str(episodes_path),
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
