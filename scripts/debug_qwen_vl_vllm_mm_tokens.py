#!/usr/bin/env python3
"""
Smoke test: PPO train rollout vLLM path (should PASS after fix).

  rollout_loop.preprocess_single_sample  -> raw_prompt_ids + multi_modal_data
  vllm_rollout_spmd.generate_sequences   -> dedup pads -> vLLM.generate

  bash examples/ppo_trainer/debug_vl_vllm_mm_tokens.sh

Exit 0 = rollout generate OK (safe to run PPO).
Exit 1 = same crash as training (fix not applied or PYTHONPATH wrong).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PpoVlVllmParams:
    model: str
    batch_size: int
    max_prompt_length: int
    max_response_length: int
    max_pixels: int
    min_pixels: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    enforce_eager: bool
    enable_chunked_prefill: bool
    max_num_batched_tokens: int

    @property
    def max_model_len(self) -> int:
        return self.max_prompt_length + self.max_response_length

    @classmethod
    def from_env(cls) -> PpoVlVllmParams:
        return cls(
            model=os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
            batch_size=int(os.environ.get("DEBUG_VL_BATCH_SIZE", "17")),
            max_prompt_length=int(os.environ.get("VL_MAX_PROMPT_LENGTH", "768")),
            max_response_length=1,
            max_pixels=int(os.environ.get("VL_MM_MAX_PIXELS", "280000")),
            min_pixels=int(os.environ.get("VL_MM_MIN_PIXELS", "65536")),
            gpu_memory_utilization=float(os.environ.get("VL_GPU_MEMORY_UTIL", "0.50")),
            tensor_parallel_size=int(
                os.environ.get("DEBUG_VL_TENSOR_PARALLEL", os.environ.get("VL_TENSOR_PARALLEL", "1"))
            ),
            enforce_eager=True,
            enable_chunked_prefill=os.environ.get("VL_ENABLE_CHUNKED_PREFILL", "false").lower()
            in ("1", "true", "yes"),
            max_num_batched_tokens=int(os.environ.get("VL_MAX_NUM_BATCHED_TOKENS", "8192")),
        )


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_craftax_env_batch(batch_size: int, seed: int) -> list[tuple[np.ndarray, str]]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
    from agent_system.environments.env_package.caged_craftext.projection import (
        get_single_token_action_vl_template_no_his,
    )

    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
        "encode_form": "embedding",
    }
    template = get_single_token_action_vl_template_no_his()
    out = []
    for i in range(batch_size):
        worker = CagedCraftextWorker(seed=seed + i * 997, env_kwargs=env_kwargs)
        frame, info = worker.reset(scenario_idx=i % 3, return_render=True)
        if frame is None:
            raise RuntimeError("craftax reset returned no frame")
        task = str(info.get("instruction") or "Collect stone.")
        out.append((np.asarray(frame), template.format(task_description=task)))
    return out


def _build_rollout_sample(
    *,
    processor,
    tokenizer,
    obs_image: np.ndarray,
    obs_text: str,
    max_prompt_length: int,
    max_pixels: int,
    min_pixels: int,
) -> dict:
    """Same VL branch as TrajectoryCollector.preprocess_single_sample."""
    from verl.utils.dataset.vision_utils import (
        qwen_vl_pil_for_mm,
        qwen_vl_image_processor_call,
        qwen_vl_prepare_prompt_ids_for_vllm,
        resolve_qwen_image_pad_token_ids,
    )
    import verl.utils.torch_functional as verl_F

    chat = [{"role": "user", "content": obs_text}]
    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )

    row: dict = {}
    pil_mm = qwen_vl_pil_for_mm(obs_image, max_pixels=max_pixels, min_pixels=min_pixels)
    row["multi_modal_data"] = {"image": [pil_mm]}
    image_inputs = qwen_vl_image_processor_call(
        processor, [pil_mm],
        max_pixels=max_pixels, min_pixels=min_pixels,
    )
    image_grid_thw = image_inputs["image_grid_thw"]
    merge_length = processor.image_processor.merge_size**2
    vllm_prompt_text = prompt_with_chat_template
    index = 0
    while "<image>" in prompt_with_chat_template:
        n_image_tokens = int(image_grid_thw[index].prod().item() // merge_length)
        vision_span = (
            "<|vision_start|>" + "<|placeholder|>" * n_image_tokens + "<|vision_end|>"
        )
        prompt_with_chat_template = prompt_with_chat_template.replace("<image>", vision_span, 1)
        if "<image>" in vllm_prompt_text:
            vllm_prompt_text = vllm_prompt_text.replace(
                "<image>",
                "<|vision_start|><|image_pad|><|vision_end|>",
                1,
            )
        index += 1
    prompt_with_chat_template = prompt_with_chat_template.replace(
        "<|placeholder|>", processor.image_token
    )

    input_ids, _ = verl_F.tokenize_and_postprocess_data(
        prompt=prompt_with_chat_template,
        tokenizer=tokenizer,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="error",
    )
    raw_prompt_ids = tokenizer.encode(vllm_prompt_text, add_special_tokens=False)
    raw_prompt_ids = qwen_vl_prepare_prompt_ids_for_vllm(
        raw_prompt_ids, tokenizer, processor=processor
    )

    pad_ids = set(resolve_qwen_image_pad_token_ids(tokenizer, processor=processor))
    row["input_ids"] = input_ids[0].tolist()
    row["raw_prompt_ids"] = raw_prompt_ids
    row["vllm_prompt_text"] = vllm_prompt_text
    row["n_hf_pads"] = sum(1 for t in row["input_ids"] if t in pad_ids)
    row["n_vllm_pads"] = sum(1 for t in raw_prompt_ids if t in pad_ids)
    return row


def _prepare_vllm_inputs_like_rollout_spmd(samples: list[dict], tokenizer, processor) -> list[dict]:
    """Mirror vllm_rollout_spmd.generate_sequences (VL text prompt path)."""
    from verl.utils.dataset.vision_utils import (
        qwen_vl_prepare_prompt_ids_for_vllm,
        resolve_qwen_image_pad_token_ids,
    )

    vllm_inputs = []
    pad_ids = resolve_qwen_image_pad_token_ids(tokenizer, processor=processor)
    pad_id_set = set(pad_ids)

    for s in samples:
        if s.get("vllm_prompt_text"):
            vllm_inputs.append(
                {"prompt": s["vllm_prompt_text"], "multi_modal_data": s["multi_modal_data"]}
            )
            continue
        prompt_token_ids = qwen_vl_prepare_prompt_ids_for_vllm(
            list(s["raw_prompt_ids"]), tokenizer, processor=processor
        )
        n_pads = sum(1 for tok in prompt_token_ids if tok in pad_id_set)
        n_images = len(s["multi_modal_data"].get("image") or [])
        if n_images and n_pads != n_images:
            raise RuntimeError(
                f"Qwen-VL vLLM prompt has {n_pads} image_pad tokens but {n_images} image(s). "
                f"pad_token_ids={pad_ids}"
            )
        vllm_inputs.append(
            {"prompt_token_ids": prompt_token_ids, "multi_modal_data": s["multi_modal_data"]}
        )
    return vllm_inputs


def run_rollout_smoke(params: PpoVlVllmParams, seed: int) -> bool:
    print("\n=== PPO rollout smoke: preprocess -> vLLM.generate ===")
    print(
        f"  max_model_len={params.max_model_len} batch={params.batch_size} "
        f"chunked_prefill={params.enable_chunked_prefill} "
        f"mm_pixels=[{params.min_pixels}, {params.max_pixels}] gpu_mem={params.gpu_memory_utilization}"
    )

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(params.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print(f"  Craftax env batch ({params.batch_size}) …")
    t0 = time.time()
    env_batch = _load_craftax_env_batch(params.batch_size, seed)
    samples = [
        _build_rollout_sample(
            processor=processor,
            tokenizer=tokenizer,
            obs_image=frame,
            obs_text=text,
            max_prompt_length=params.max_prompt_length,
            max_pixels=params.max_pixels,
            min_pixels=params.min_pixels,
        )
        for frame, text in env_batch
    ]
    print(f"  preprocess done in {time.time() - t0:.1f}s")
    s0 = samples[0]
    print(
        f"  sample[0]: hf_pads_in_input_ids={s0['n_hf_pads']} "
        f"vllm_pads_in_raw_prompt_ids={s0['n_vllm_pads']} (expect 1)"
    )

    vllm_inputs = _prepare_vllm_inputs_like_rollout_spmd(samples, tokenizer, processor)
    from verl.utils.dataset.vision_utils import resolve_qwen_image_pad_token_ids

    pad_id_set = set(resolve_qwen_image_pad_token_ids(tokenizer, processor=processor))
    n_text = sum(1 for inp in vllm_inputs if "prompt" in inp)
    total_vllm_pads = sum(
        sum(1 for t in inp.get("prompt_token_ids") or [] if t in pad_id_set) for inp in vllm_inputs
    )
    print(f"  vLLM batch: {n_text} text prompts, {total_vllm_pads} pad tokens in token-id fallback")

    t1 = time.time()
    llm = LLM(
        model=params.model,
        trust_remote_code=True,
        enforce_eager=params.enforce_eager,
        gpu_memory_utilization=params.gpu_memory_utilization,
        tensor_parallel_size=params.tensor_parallel_size,
        max_model_len=params.max_model_len,
        max_num_batched_tokens=params.max_num_batched_tokens,
        enable_chunked_prefill=params.enable_chunked_prefill,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": params.min_pixels, "max_pixels": params.max_pixels},
    )
    print(f"  vLLM loaded in {time.time() - t1:.1f}s")

    sampling = SamplingParams(max_tokens=params.max_response_length, temperature=1.0, detokenize=False)
    t2 = time.time()
    try:
        llm.generate(vllm_inputs, sampling_params=sampling, use_tqdm=True)
        print(f"  OK: rollout generate passed in {time.time() - t2:.1f}s")
        return True
    except ValueError as exc:
        print(f"  FAIL in {time.time() - t2:.1f}s: {exc}")
        if "assign" in str(exc).lower():
            print("  >>> merge_multimodal_embeddings — same as PPO train rollout")
        return False
    except Exception:
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _setup_paths()
    params = PpoVlVllmParams.from_env()
    ok = run_rollout_smoke(params, args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
