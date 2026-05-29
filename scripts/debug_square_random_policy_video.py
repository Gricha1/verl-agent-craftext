#!/usr/bin/env python3
"""
Random single-token policy on debug_square_8x8 — no vLLM, no model.

Runs one env slot, prints the full agent prompt, saves a validation-style GIF
with observation + action overlay (same idea as trainer val video).

Usage:
  python scripts/debug_square_random_policy_video.py
  python scripts/debug_square_random_policy_video.py --steps 30 --seed 1
  bash examples/ppo_trainer/debug_square_random_policy_video.sh
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _make_config(
    *,
    prompt_template_type: str,
    max_steps: int,
    seed: int,
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env": {
                "env_name": "caged_craftext/CagedCraftextEnv",
                "craftext_settings": "debug_square_8x8",
                "observation_type": "ascii",
                "prompt_template_type": prompt_template_type,
                "enable_reasoning": False,
                "history_length": 0,
                "max_steps": max_steps,
                "seed": seed,
                "use_optimistic_parallel": True,
                "optimistic_reset_ratio": 1,
                "use_ray_text_render_workers": False,
                "use_jax_gpu": False,
                "auto_reset": False,
                "rollout": {"n": 0},
                "resources_per_worker": {"num_cpus": 1},
            },
            "data": {
                "train_batch_size": 1,
                "val_batch_size": 1,
            },
        }
    )


def _save_gif(
    frames: List[np.ndarray],
    prompts: List[str],
    actions: List[str],
    out_path: str,
) -> None:
    import imageio

    from agent_system.environments.env_package.caged_craftext.utility import (
        composite_frame_with_prompt_text,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    composed = []
    for i, frame in enumerate(frames):
        p = prompts[i] if i < len(prompts) else ""
        a = actions[i] if i < len(actions) else ""
        composed.append(composite_frame_with_prompt_text(frame, p, a))
    imageio.mimsave(out_path, composed, duration=0.35, loop=0)
    print(f"[INFO] Saved GIF ({len(composed)} frames): {out_path}")


def _log_to_comet(
    *,
    project_name: str,
    experiment_name: str,
    config: dict,
    gif_path: str,
    prompt_file: str,
    prompt: str,
    metrics: dict,
) -> None:
    try:
        from verl.utils.tracking import CometMLLogger
    except ImportError as exc:
        raise RuntimeError("comet_ml / verl not available — pip install comet_ml") from exc

    logger = CometMLLogger(project_name=project_name, experiment_name=experiment_name, config=config)
    logger.log(metrics, step=0)
    logger.log_video(gif_path, step=0, name="random_policy_trajectory")
    if os.path.isfile(prompt_file):
        logger.experiment.log_asset(prompt_file, file_name=os.path.basename(prompt_file))
    if prompt:
        logger.experiment.log_text(prompt, metadata={"type": "full_prompt"})
    logger.finish()
    print(f"[INFO] Logged to Comet ML: project={project_name!r} experiment={experiment_name!r}")


def run_random_rollout(
    *,
    steps: int,
    seed: int,
    prompt_template_type: str,
    gif_path: str,
    print_prompt: bool,
) -> Tuple[str, int, int, float, bool, Optional[str]]:
    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    _setup_paths()
    from agent_system.environments.env_manager import make_envs
    from agent_system.environments.env_package.caged_craftext.action_tokens import (
        action_token_strings,
    )

    config = _make_config(
        prompt_template_type=prompt_template_type,
        max_steps=steps,
        seed=seed,
    )
    _envs, val_envs = make_envs(config)
    val_envs.set_record_video(True, env_idx=0)

    rng = np.random.RandomState(seed)
    labels = list(action_token_strings())

    obs, _infos = val_envs.reset(kwargs=None)
    prompt0 = obs["text"][0]
    if print_prompt:
        print("=" * 72)
        print("FULL PROMPT (env_idx=0, before step 0):")
        print("=" * 72)
        print(prompt0)
        print("=" * 72)

    frames: List[np.ndarray] = []
    prompts: List[str] = []
    actions: List[str] = []
    valid = 0
    total = 0
    total_reward = 0.0
    instruction_done = False
    instruction_text: Optional[str] = None
    is_done = False

    for t in range(steps):
        if is_done:
            break
        prompt_text = obs["text"][0]
        token = str(rng.choice(labels))
        text_actions = [token]

        next_obs, rewards, dones, infos = val_envs.step(text_actions)
        info = infos[0]
        total += 1
        total_reward += float(rewards[0])
        if info.get("is_action_valid"):
            valid += 1
        if instruction_text is None:
            instruction_text = info.get("instruction") or info.get("instruction_text")
        if info.get("instruction_done"):
            instruction_done = True
        elif bool(dones[0]):
            instruction_done = True

        frame = info.get("render_frame")
        if frame is not None:
            frames.append(np.asarray(frame))
            prompts.append(prompt_text)
            name = info.get("action_name") or "?"
            raw = info.get("action_text") or token
            actions.append(f"{name} | raw: {raw} | r={float(rewards[0]):.3f}")

        if print_prompt and t == 0:
            print(f"[step 0] token={token!r} action_id={info.get('action_id')} "
                  f"valid={info.get('is_action_valid')} reward={rewards[0]:.3f} done={dones[0]}")

        obs = next_obs
        is_done = bool(dones[0])

    if not frames:
        raise RuntimeError("No render_frame collected — check set_record_video / optimistic env.")

    _save_gif(frames, prompts, actions, gif_path)
    val_envs.close()
    return prompt0, valid, total, total_reward, instruction_done, instruction_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Random single-token policy + val-style GIF (no vLLM)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        default="single_token_action",
        choices=("single_token_action", "default_template"),
    )
    parser.add_argument(
        "--gif-path",
        default="gif/random_policy_debug_square.gif",
    )
    parser.add_argument("--no-print-prompt", action="store_true")
    parser.add_argument(
        "--comet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log GIF, prompt, and metrics to Comet ML (needs COMET_API_KEY)",
    )
    parser.add_argument(
        "--comet-project",
        default=os.environ.get("COMET_PROJECT", "verl_agent_caged_craftext"),
    )
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    gif_path = args.gif_path if os.path.isabs(args.gif_path) else os.path.join(root, args.gif_path)

    prompt, valid, total, total_reward, instruction_done, instruction_text = run_random_rollout(
        steps=args.steps,
        seed=args.seed,
        prompt_template_type=args.prompt_template,
        gif_path=gif_path,
        print_prompt=not args.no_print_prompt,
    )

    prompt_file = os.path.splitext(gif_path)[0] + "_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"[INFO] Wrote prompt to {prompt_file}")
    valid_rate = valid / total if total else 0.0
    print(f"[INFO] valid_action_rate={valid}/{total} ({valid_rate:.1%})")
    print(f"[INFO] total_reward={total_reward:.3f} instruction_done={instruction_done}")

    if args.comet:
        if not os.environ.get("COMET_API_KEY"):
            print("[WARN] COMET_API_KEY not set — skip Comet logging (use --no-comet to silence)")
        else:
            experiment_name = os.environ.get(
                "RUN_NAME",
                f"random_policy_debug_square_{args.seed}",
            )
            _log_to_comet(
                project_name=args.comet_project,
                experiment_name=experiment_name,
                config={
                    "script": "debug_square_random_policy_video",
                    "env": "debug_square_8x8",
                    "prompt_template": args.prompt_template,
                    "steps": args.steps,
                    "seed": args.seed,
                    "instruction": instruction_text,
                },
                gif_path=gif_path,
                prompt_file=prompt_file,
                prompt=prompt,
                metrics={
                    "debug/valid_action_rate": valid_rate,
                    "debug/total_steps": total,
                    "debug/total_reward": total_reward,
                    "debug/instruction_done": float(instruction_done),
                },
            )


if __name__ == "__main__":
    main()
