#!/usr/bin/env python3
"""Run the exported offline-teacher Qwen actor in real CrafText and save raw traces."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imageio.v2 as imageio
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_legacy_paths(root: Path) -> None:
    for path in (root / "caged_craftext" / "Craftax", root / "caged_craftext", root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def messages(instruction: str, observation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful embodied CrafText agent."},
        {
            "role": "user",
            "content": (
                f"Your goal is to complete this task:\n**TASK:** {instruction}\n\n"
                f"Current state:\n{observation}\n\n"
                "Think about the next move, then output exactly one final action as "
                "<action>ACTION</action>. Valid actions are NOOP, LEFT, RIGHT, UP, DOWN, DO, "
                "SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, "
                "MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, "
                "MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD."
            ),
        },
    ]


def save_gif(frames: list[np.ndarray], prompts: list[str], actions: list[str], path: Path) -> None:
    from agent_system.environments.env_package.caged_craftext.utility import composite_frame_with_prompt_text

    composed = [composite_frame_with_prompt_text(frame, prompt, action) for frame, prompt, action in zip(frames, prompts, actions)]
    if not composed:
        return
    imageio.mimsave(path, composed, duration=0.85, loop=0)


@torch.inference_mode()
def generate(model, tokenizer, instruction: str, observation: str, *, max_new_tokens: int, seed: int) -> str:
    rendered = tokenizer.apply_chat_template(messages(instruction, observation), tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    # This transformers version does not accept a per-call ``generator``.
    # Keep rollout sampling reproducible through the CUDA global RNG instead.
    torch.cuda.manual_seed_all(seed)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="20261016,20261017,20261018,20261019")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    add_legacy_paths(args.legacy_root)
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
    from craftext_integration.actions import parse_reasoning_action

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).to("cuda").eval()

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for episode, seed in enumerate(int(value) for value in args.seeds.split(",") if value.strip()):
        worker = CagedCraftextWorker(
            seed=seed,
            env_kwargs={"config_name": "debug_square_16x16", "use_debug_square_map": True, "observation_type": "ascii", "encode_form": "embedding"},
        )
        _, info = worker.reset(scenario_idx=args.task_index, return_render=False)
        instruction = str(info["instruction"])
        observation = str(info["text_render"])
        frames: list[np.ndarray] = []
        prompts: list[str] = []
        overlays: list[str] = []
        total_reward = 0.0
        success = False
        done = False
        for step in range(args.max_steps):
            raw = generate(model, tokenizer, instruction, observation, max_new_tokens=args.max_new_tokens, seed=seed * 1000 + step)
            parsed = parse_reasoning_action(raw)
            _, reward, done, info = worker.step(parsed.action_id, return_render=True)
            total_reward += float(reward)
            success = bool(info.get("instruction_done", False) or info.get("won", False))
            row = {
                "episode": episode, "seed": seed, "step": step, "instruction": instruction,
                "state": observation, "raw_response": raw, "reasoning": parsed.reasoning,
                "executed_action": parsed.action_name, "valid_action": parsed.valid,
                "reward": float(reward), "return_so_far": total_reward, "done": bool(done), "success": success,
            }
            all_rows.append(row)
            frame = info.get("render_frame")
            if frame is not None:
                frames.append(np.asarray(frame))
                prompts.append(f"Task: {instruction}\nState: {observation}")
                compact = raw.replace("\n", " ").strip()
                overlays.append(f"step={step} action={parsed.action_name} valid={parsed.valid} reward={float(reward):+.2f}\nraw: {compact[:420]}")
            observation = str(info.get("text_render", observation))
            if done or success:
                break
        worker.close()
        gif_path = args.output_dir / f"episode_{episode:02d}_seed_{seed}_success_{int(success)}.gif"
        save_gif(frames, prompts, overlays, gif_path)
        summaries.append({"episode": episode, "seed": seed, "instruction": instruction, "steps": len(frames), "return": total_reward, "success": success, "gif": str(gif_path)})

    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"episodes": summaries, "success_rate": sum(row["success"] for row in summaries) / max(1, len(summaries))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
