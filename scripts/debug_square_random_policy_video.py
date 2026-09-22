#!/usr/bin/env python3
"""
Random single-token policy on debug_square_8x8 — no vLLM, no model.

Runs one episode per task (stone / wood / water), prints prompts, saves GIFs
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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# (instruction_idx, slug) — debug_square_8x8 easy scenarios
DEBUG_SQUARE_TASKS: Tuple[Tuple[int, str], ...] = (
    (0, "stone"),
    (1, "wood"),
    (2, "water"),
)


@dataclass
class TaskRolloutResult:
    slug: str
    instruction_idx: int
    instruction_text: Optional[str]
    prompt: str
    gif_path: str
    prompt_file: str
    valid: int
    total: int
    total_reward: float
    instruction_done: bool


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.path.join(root, "caged_craftext")
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _task_gif_path(base_gif_path: str, slug: str) -> str:
    root, ext = os.path.splitext(base_gif_path)
    if root.endswith(f"_{slug}"):
        return base_gif_path
    return f"{root}_{slug}{ext}"


def _format_prompt(template: str, task: str, text_render: str) -> str:
    return template.format(task_description=task, action_history=[], current_observation=text_render)


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


def _log_tasks_to_comet(
    *,
    project_name: str,
    experiment_name: str,
    config: dict,
    results: List[TaskRolloutResult],
) -> None:
    try:
        from verl.utils.tracking import CometMLLogger
    except ImportError as exc:
        raise RuntimeError("comet_ml / verl not available — pip install comet_ml") from exc

    logger = CometMLLogger(project_name=project_name, experiment_name=experiment_name, config=config)
    for i, result in enumerate(results):
        prefix = f"debug/{result.slug}"
        valid_rate = result.valid / result.total if result.total else 0.0
        logger.log(
            {
                f"{prefix}/valid_action_rate": valid_rate,
                f"{prefix}/total_steps": result.total,
                f"{prefix}/total_reward": result.total_reward,
                f"{prefix}/instruction_done": float(result.instruction_done),
                f"{prefix}/instruction_idx": float(result.instruction_idx),
            },
            step=i,
        )
        logger.log_video(result.gif_path, step=i, name=f"random_policy_{result.slug}")
        if os.path.isfile(result.prompt_file):
            logger.experiment.log_asset(
                result.prompt_file,
                file_name=os.path.basename(result.prompt_file),
            )
        if result.prompt:
            logger.experiment.log_text(
                result.prompt,
                metadata={"type": "full_prompt", "task": result.slug},
            )
    logger.finish()
    print(
        f"[INFO] Logged {len(results)} GIFs to Comet ML: "
        f"project={project_name!r} experiment={experiment_name!r}"
    )


def run_random_rollout_for_task(
    *,
    steps: int,
    seed: int,
    instruction_idx: int,
    task_slug: str,
    gif_path: str,
    print_prompt: bool,
) -> TaskRolloutResult:
    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    _setup_paths()
    from agent_system.environments.env_package.caged_craftext.action_tokens import (
        action_token_strings,
    )
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
    from agent_system.environments.env_package.caged_craftext.projection import (
        craftext_projection,
        get_single_token_action_template_no_his,
    )

    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
        "encode_form": "embedding",
    }
    worker = CagedCraftextWorker(seed=seed + instruction_idx * 1000, env_kwargs=env_kwargs)
    template = get_single_token_action_template_no_his()

    _, info = worker.reset(scenario_idx=instruction_idx, return_render=False)
    instruction_text = info.get("instruction")
    prompt_text = _format_prompt(template, instruction_text, info["text_render"])

    if print_prompt:
        print("=" * 72)
        print(f"FULL PROMPT (task={task_slug}, instruction_idx={instruction_idx}, before step 0):")
        print("=" * 72)
        print(prompt_text)
        print("=" * 72)

    rng = np.random.RandomState(seed + instruction_idx)
    labels = list(action_token_strings())

    frames: List[np.ndarray] = []
    prompts: List[str] = []
    actions: List[str] = []
    valid = 0
    total = 0
    total_reward = 0.0
    instruction_done = False
    is_done = False

    for t in range(steps):
        if is_done:
            break
        token = str(rng.choice(labels))
        action_ids, valids = craftext_projection([token])
        action_id = action_ids[0]

        _, reward, done, info = worker.step(int(action_id), return_render=True)
        total += 1
        total_reward += float(reward)
        if valids[0]:
            valid += 1
        if info.get("instruction_done"):
            instruction_done = True
        elif bool(done):
            instruction_done = True

        frame = info.get("render_frame")
        if frame is not None:
            frames.append(np.asarray(frame))
            prompts.append(prompt_text)
            name = info.get("action_name") or "?"
            raw = token
            actions.append(f"{name} | raw: {raw} | r={float(reward):.3f}")

        if print_prompt and t == 0:
            print(
                f"[{task_slug} step 0] token={token!r} action_id={info.get('action_id')} "
                f"valid={bool(valids[0])} reward={reward:.3f} done={done}"
            )

        prompt_text = _format_prompt(template, info.get("instruction", instruction_text), info["text_render"])
        is_done = bool(done)

    worker.close()

    if not frames:
        raise RuntimeError(
            f"No render_frame collected for task {task_slug} — check return_render=True on step()."
        )

    _save_gif(frames, prompts, actions, gif_path)

    prompt_file = os.path.splitext(gif_path)[0] + "_prompt.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompts[0] if prompts else prompt_text)

    return TaskRolloutResult(
        slug=task_slug,
        instruction_idx=instruction_idx,
        instruction_text=instruction_text,
        prompt=prompts[0] if prompts else prompt_text,
        gif_path=gif_path,
        prompt_file=prompt_file,
        valid=valid,
        total=total,
        total_reward=total_reward,
        instruction_done=instruction_done,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random single-token policy + val-style GIFs for all debug_square tasks (no vLLM)"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        default="single_token_action",
        choices=("single_token_action", "default_template"),
        help="Only single_token_action is supported in this script",
    )
    parser.add_argument(
        "--gif-path",
        default="gif/random_policy_debug_square.gif",
        help="Base path; outputs gif/random_policy_debug_square_{stone,wood,water}.gif",
    )
    parser.add_argument("--no-print-prompt", action="store_true")
    parser.add_argument(
        "--comet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log GIFs and metrics to Comet ML (needs COMET_API_KEY)",
    )
    parser.add_argument(
        "--comet-project",
        default=os.environ.get("COMET_PROJECT", "verl_agent_caged_craftext"),
    )
    args = parser.parse_args()

    if args.prompt_template != "single_token_action":
        raise SystemExit("This script only supports --prompt-template single_token_action")

    root = _repo_root()
    os.chdir(root)
    base_gif = args.gif_path if os.path.isabs(args.gif_path) else os.path.join(root, args.gif_path)

    results: List[TaskRolloutResult] = []
    for instruction_idx, slug in DEBUG_SQUARE_TASKS:
        gif_path = _task_gif_path(base_gif, slug)
        print(f"\n--- Task: {slug} (instruction_idx={instruction_idx}) -> {gif_path} ---")
        result = run_random_rollout_for_task(
            steps=args.steps,
            seed=args.seed,
            instruction_idx=instruction_idx,
            task_slug=slug,
            gif_path=gif_path,
            print_prompt=not args.no_print_prompt,
        )
        results.append(result)
        valid_rate = result.valid / result.total if result.total else 0.0
        print(f"[INFO] {slug}: valid_action_rate={result.valid}/{result.total} ({valid_rate:.1%})")
        print(
            f"[INFO] {slug}: total_reward={result.total_reward:.3f} "
            f"instruction_done={result.instruction_done}"
        )
        print(f"[INFO] {slug}: prompt -> {result.prompt_file}")

    if args.comet:
        if not os.environ.get("COMET_API_KEY"):
            print("[WARN] COMET_API_KEY not set — skip Comet logging (use --no-comet to silence)")
        else:
            experiment_name = os.environ.get(
                "RUN_NAME",
                f"random_policy_debug_square_{args.seed}",
            )
            _log_tasks_to_comet(
                project_name=args.comet_project,
                experiment_name=experiment_name,
                config={
                    "script": "debug_square_random_policy_video",
                    "env": "debug_square_8x8",
                    "prompt_template": args.prompt_template,
                    "steps": args.steps,
                    "seed": args.seed,
                    "tasks": [slug for _, slug in DEBUG_SQUARE_TASKS],
                },
                results=results,
            )


if __name__ == "__main__":
    main()
