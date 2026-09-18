#!/usr/bin/env python3
"""
Minimal VL rollout: one debug_square task, N env steps, vLLM generate each step.

  bash examples/ppo_trainer/debug_vl_single_trajectory.sh
  STEPS=5 TASK=wood bash examples/ppo_trainer/debug_vl_single_trajectory.sh

Exit 0 = all generates OK (same path as PPO train rollout).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

TASK_TO_SCENARIO = {"stone": 0, "wood": 1, "water": 2}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    scripts = os.path.join(root, "scripts")
    for p in (craftax, caged, root, scripts):
        if p not in sys.path:
            sys.path.insert(0, p)


def _frame_from_reset(obs, info) -> np.ndarray:
    frame = info.get("render_frame") if info else None
    if frame is None:
        frame = obs
    if frame is None:
        raise RuntimeError("env reset returned no pixel frame (use return_render=True)")
    return np.asarray(frame)


def run_trajectory(*, task: str, steps: int, seed: int) -> bool:
    from debug_qwen_vl_vllm_mm_tokens import (
        PpoVlVllmParams,
        _build_rollout_sample,
        _prepare_vllm_inputs_like_rollout_spmd,
    )
    from agent_system.environments.env_package.caged_craftext.action_tokens import (
        format_single_token_action_display,
    )
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
    from agent_system.environments.env_package.caged_craftext.projection import (
        craftext_projection,
        get_single_token_action_vl_template_no_his,
    )
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if task not in TASK_TO_SCENARIO:
        raise ValueError(f"unknown task {task!r}, choose from {list(TASK_TO_SCENARIO)}")

    params = PpoVlVllmParams.from_env()
    scenario_idx = TASK_TO_SCENARIO[task]

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")

    print(f"\n=== VL single trajectory: task={task} steps={steps} seed={seed} ===")
    print(
        f"  model={params.model} max_model_len={params.max_model_len} "
        f"mm_pixels=[{params.min_pixels}, {params.max_pixels}]"
    )

    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
        "encode_form": "embedding",
    }
    worker = CagedCraftextWorker(seed=seed, env_kwargs=env_kwargs)
    template = get_single_token_action_vl_template_no_his()

    obs, info = worker.reset(scenario_idx=scenario_idx, return_render=True)
    instruction = str(info.get("instruction") or task)
    frame = _frame_from_reset(obs, info)
    print(f"  task text: {instruction}")

    processor = AutoProcessor.from_pretrained(params.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    t0 = time.time()
    llm = LLM(
        model=params.model,
        trust_remote_code=True,
        enforce_eager=params.enforce_eager,
        gpu_memory_utilization=params.gpu_memory_utilization,
        tensor_parallel_size=params.tensor_parallel_size,
        max_model_len=params.max_model_len,
        max_num_batched_tokens=params.max_num_batched_tokens,
        enable_chunked_prefill=False,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": params.min_pixels, "max_pixels": params.max_pixels},
    )
    print(f"  vLLM loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(max_tokens=1, temperature=1.0, detokenize=False)
    done = False
    total_reward = 0.0

    for step_idx in range(steps):
        if done:
            print(f"  step {step_idx}: episode already done, stop")
            break

        obs_text = template.format(task_description=instruction)
        sample = _build_rollout_sample(
            processor=processor,
            tokenizer=tokenizer,
            obs_image=frame,
            obs_text=obs_text,
            max_prompt_length=params.max_prompt_length,
            max_pixels=params.max_pixels,
            min_pixels=params.min_pixels,
        )
        vllm_input = _prepare_vllm_inputs_like_rollout_spmd([sample], tokenizer, processor)[0]

        t_gen = time.time()
        try:
            outputs = llm.generate([vllm_input], sampling_params=sampling, use_tqdm=False)
        except ValueError as exc:
            print(f"  FAIL step {step_idx} in {time.time() - t_gen:.1f}s: {exc}")
            if "assign" in str(exc).lower():
                print("  >>> merge_multimodal_embeddings — same as PPO train rollout")
            return False
        except Exception:
            traceback.print_exc()
            return False

        token_ids = outputs[0].outputs[0].token_ids
        raw_action = tokenizer.decode(token_ids, skip_special_tokens=True)
        action_id, _valid = craftext_projection([raw_action])
        action_id = action_id[0]
        display, _ = format_single_token_action_display(raw_action)
        print(
            f"  step {step_idx}: generate {time.time() - t_gen:.1f}s "
            f"action={display!r} id={action_id} reward_before_step"
        )

        if action_id < 0:
            print(f"  WARN: invalid action {raw_action!r}, using NOOP (0)")
            action_id = 0

        _, reward, done, info = worker.step(action_id, return_render=True)
        total_reward += reward
        frame = _frame_from_reset(None, info)
        print(f"           env reward={reward:.3f} done={done} cum_reward={total_reward:.3f}")

    print(f"  OK: {steps} step(s) requested, trajectory finished")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=os.environ.get("TASK", "stone"), choices=list(TASK_TO_SCENARIO))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "3")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    args = parser.parse_args()
    _setup_paths()
    ok = run_trajectory(task=args.task, steps=args.steps, seed=args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
