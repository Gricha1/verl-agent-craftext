#!/usr/bin/env python3
"""
Validate reward-WM SFT checkpoint as a receding-horizon planner on debug_square_8x8.

Each env step:
  1. Build max-return planning prompt (H action tokens).
  2. Greedy-decode H actions with the LLM.
  3. Execute only the first action; re-plan on the next step.

Runs all 3 debug_square tasks (stone / wood / water), reports success rate and
total reward, saves one GIF per task.

Usage:
  python scripts/validate_debug_square_llm_planning.py \\
    --checkpoint /path/to/latest --horizon 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEBUG_SQUARE_TASKS: Tuple[Tuple[int, str], ...] = (
    (0, "stone"),
    (1, "wood"),
    (2, "water"),
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class EpisodeResult:
    slug: str
    instruction_idx: int
    episode_idx: int
    total_steps: int
    total_reward: float
    instruction_done: bool
    valid_actions: int
    gif_path: str = ""
    frames: List[np.ndarray] = field(default_factory=list)
    prompts_overlay: List[str] = field(default_factory=list)
    actions_overlay: List[str] = field(default_factory=list)
    action_ids: List[int] = field(default_factory=list)


@dataclass
class TaskSummary:
    slug: str
    instruction_idx: int
    episodes: int
    successes: int
    total_reward_sum: float
    mean_reward: float
    success_rate: float
    gif_path: str
    action_hist_path: str = ""


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


def _task_action_hist_path(base_gif_path: str, slug: str) -> str:
    gif_path = _task_gif_path(base_gif_path, slug)
    root, _ext = os.path.splitext(gif_path)
    return f"{root}_action_hist.png"


def _resolve_base_model(checkpoint: str, base_model: str | None) -> str:
    if base_model:
        return base_model
    adapter_cfg = os.path.join(checkpoint, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("base_model_name_or_path")
        if name:
            return str(name)
    raise ValueError(
        f"Cannot infer base model from {checkpoint}. Pass --base-model explicitly."
    )


def _attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _masked_greedy_token_id(step_logits: torch.Tensor, allowed_ids: torch.Tensor) -> int:
    masked = torch.full_like(step_logits, float("-inf"))
    masked[allowed_ids] = step_logits[allowed_ids]
    return int(masked.argmax(dim=-1).item())


class PlanningPolicy:
    """Greedy H-step planner or single-token PPO actor policy (HF / LoRA checkpoint)."""

    def __init__(
        self,
        *,
        checkpoint: str | None,
        base_model: str | None,
        device: str,
        horizon: int,
        planning_mode: str,
        target_return: int,
        ppo_actor_prompt: bool = False,
        base_model_only: bool = False,
    ) -> None:
        from verl.utils.model import compute_position_id_with_mask

        self.horizon = max(1, int(horizon))
        self.planning_mode = str(planning_mode)
        self.target_return = int(target_return)
        self.ppo_actor_prompt = bool(ppo_actor_prompt)
        self.base_model_only = bool(base_model_only)
        self.device = torch.device(device)
        self._compute_position_id_with_mask = compute_position_id_with_mask

        if self.base_model_only:
            base = base_model or DEFAULT_BASE_MODEL
            print(f"[INFO] base_model_only=True — loading default HF weights from {base!r} (no SFT/LoRA)")
        else:
            if not checkpoint:
                raise ValueError("checkpoint is required unless base_model_only=True")
            base = _resolve_base_model(checkpoint, base_model)

        self.tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        if self.base_model_only:
            model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=dtype,
                attn_implementation=_attn_implementation(),
                trust_remote_code=True,
            )
        else:
            from peft import PeftModel

            assert checkpoint is not None
            model = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=dtype,
                attn_implementation=_attn_implementation(),
                trust_remote_code=True,
            )
            adapter_files = (
                os.path.join(checkpoint, "adapter_model.safetensors"),
                os.path.join(checkpoint, "adapter_model.bin"),
            )
            if any(os.path.isfile(p) for p in adapter_files):
                model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
            elif os.path.isfile(os.path.join(checkpoint, "config.json")):
                # Merged or full HF save at checkpoint root.
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    torch_dtype=dtype,
                    attn_implementation=_attn_implementation(),
                    trust_remote_code=True,
                )
            else:
                raise FileNotFoundError(
                    f"No adapter or config.json under checkpoint: {checkpoint}"
                )

        self.model = model.to(self.device).eval()

        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            action_token_strings,
        )

        action_toks = list(action_token_strings())
        action_tok_ids: list[int] = []
        self._id_to_action_tok: dict[int, str] = {}
        for t in action_toks:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                tid = int(ids[0])
                action_tok_ids.append(tid)
                self._id_to_action_tok[tid] = t
        self._action_ids_tensor = torch.tensor(action_tok_ids, dtype=torch.long, device=self.device)
        self._eos_id = self.tokenizer.eos_token_id

        if self.ppo_actor_prompt:
            from agent_system.environments.env_package.caged_craftext.projection import (
                get_single_token_action_template_no_his,
            )

            self._ppo_actor_template = get_single_token_action_template_no_his()

    def _build_ppo_actor_prompt(self, state: str, task: str) -> str:
        return self._ppo_actor_template.format(
            task_description=task,
            current_observation=state,
        )

    def build_user_prompt(self, state: str, task: str) -> str:
        """Filled template text (task + observation) before chat wrapping."""
        if self.ppo_actor_prompt:
            return self._build_ppo_actor_prompt(state, task)
        return self._build_planning_prompt(state, task)

    def build_model_input(self, state: str, task: str) -> str:
        """Full string tokenized by the model (chat template + generation prompt)."""
        prompt_text = self.build_user_prompt(state, task)
        chat = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

    def _build_planning_prompt(self, state: str, task: str) -> str:
        from agent_system.environments.prompts.world_model_planning import (
            format_max_return_planning_prompt,
            format_planning_prompt,
        )

        if self.planning_mode == "return_conditioned":
            return format_planning_prompt(
                state,
                target_return=self.target_return,
                task=task,
                horizon=self.horizon,
            )
        return format_max_return_planning_prompt(
            state,
            task=task,
            horizon=self.horizon,
        )

    @torch.no_grad()
    def greedy_plan(self, state: str, task: str) -> str:
        """Return concatenated H action token chars (greedy)."""
        prompt_str = self.build_model_input(state, task)
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        position_ids = self._compute_position_id_with_mask(attention_mask).to(self.device)

        decoded: list[str] = []

        def _decode_loop() -> None:
            nonlocal input_ids, attention_mask, position_ids
            for _ in range(self.horizon):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                seq_len = int(attention_mask.sum(dim=1).item()) - 1
                step_logits = out.logits[0, seq_len, :]
                next_id = _masked_greedy_token_id(step_logits, self._action_ids_tensor)
                if self._eos_id is not None and next_id == int(self._eos_id):
                    break
                decoded.append(self._id_to_action_tok[next_id])
                next_t = torch.tensor([[next_id]], device=self.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                position_ids = self._compute_position_id_with_mask(attention_mask).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _decode_loop()
            else:
                _decode_loop()
        return "".join(decoded)

    @torch.no_grad()
    def greedy_single_action(self, state: str, task: str) -> str:
        """Greedy one action token (PPO single_token_action prompt)."""
        prompt_str = self.build_model_input(state, task)
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        position_ids = self._compute_position_id_with_mask(attention_mask).to(self.device)

        def _forward() -> str:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            seq_len = int(attention_mask.sum(dim=1).item()) - 1
            step_logits = out.logits[0, seq_len, :]
            next_id = _masked_greedy_token_id(step_logits, self._action_ids_tensor)
            if self._eos_id is not None and next_id == int(self._eos_id):
                return ""
            return self._id_to_action_tok.get(next_id, "")

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return _forward()
            return _forward()

    def first_action_token(self, state: str, task: str) -> tuple[str, str]:
        """Return (first_action_token, full_plan_string)."""
        if self.ppo_actor_prompt:
            tok = self.greedy_single_action(state, task)
            return tok, tok

        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            parse_action_token_sequence,
        )

        plan = self.greedy_plan(state, task)
        toks = parse_action_token_sequence(plan)
        if not toks:
            return "", plan
        return toks[0], plan


def _save_gif(
    frames: Sequence[np.ndarray],
    prompts: Sequence[str],
    actions: Sequence[str],
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


def _save_action_histogram(
    action_ids: Sequence[int],
    out_path: str,
    *,
    title: str,
) -> None:
    """Bar chart of action counts for one recorded episode (matches GIF)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT

    ids = np.asarray(list(action_ids), dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(ACTION_TO_TEXT))]
    num_actions = len(ACTION_TO_TEXT)
    counts = (
        np.bincount(ids, minlength=num_actions)
        if ids.size > 0
        else np.zeros(num_actions, dtype=np.int64)
    )
    action_names = list(ACTION_TO_TEXT)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    x = np.arange(num_actions, dtype=np.int64)
    fig_w = max(10.0, 0.6 * num_actions)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5), dpi=160)
    bars = ax.bar(x, counts, color="#4C78A8", edgecolor="#2F3B4A", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("action")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(action_names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ymax = int(counts.max()) if counts.size > 0 else 0
    ax.set_ylim(0, max(1, ymax + max(1, int(0.15 * ymax))))
    for rect, c in zip(bars, counts):
        if int(c) == 0:
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            str(int(c)),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1B1F24",
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved action histogram ({int(ids.size)} steps): {out_path}")


def make_debug_square_worker(seed: int):
    """Create one env worker (heavy init). Reuse across episodes via reset()."""
    from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker

    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
        "encode_form": "embedding",
    }
    return CagedCraftextWorker(seed=int(seed), env_kwargs=env_kwargs)


def run_episode(
    *,
    policy: PlanningPolicy,
    worker,
    instruction_idx: int,
    task_slug: str,
    episode_idx: int,
    max_steps: int,
    capture_frames: bool,
) -> EpisodeResult:
    from agent_system.environments.env_package.caged_craftext.action_tokens import (
        format_single_token_action_display,
        parse_single_token_action,
    )
    from agent_system.environments.env_package.caged_craftext.projection import craftext_projection

    _, info = worker.reset(scenario_idx=instruction_idx, return_render=False)
    instruction = str(info.get("instruction", "") or "")

    frames: list[np.ndarray] = []
    prompts_overlay: list[str] = []
    actions_overlay: list[str] = []
    episode_action_ids: list[int] = []
    total_reward = 0.0
    valid_actions = 0
    total_steps = 0
    instruction_done = False

    for _t in range(max_steps):
        state_ascii = str(info.get("text_render", "") or "")
        first_tok, full_plan = policy.first_action_token(state_ascii, instruction)
        if parse_single_token_action(first_tok) < 0:
            first_tok = "5"  # NOOP fallback

        action_ids, valids = craftext_projection([first_tok])
        action_id = int(action_ids[0])
        episode_action_ids.append(action_id)

        _, reward, done, info = worker.step(action_id, return_render=capture_frames)
        total_steps += 1
        total_reward += float(reward)
        if valids[0]:
            valid_actions += 1
        if info.get("instruction_done"):
            instruction_done = True
        elif bool(done):
            instruction_done = True

        if capture_frames:
            frame = info.get("render_frame")
            if frame is not None:
                frames.append(np.asarray(frame))
                model_input = policy.build_model_input(state_ascii, instruction).strip()
                if getattr(policy, "ppo_actor_prompt", False):
                    mode_line = "mode=ppo_actor (single_token_action)"
                else:
                    mode_line = f"mode=planner H={policy.horizon} ({policy.planning_mode})"
                overlay_prompt = (
                    f"task={task_slug} ep={episode_idx} step={_t}\n"
                    f"{mode_line}\n"
                    f"--- LLM input (chat template) ---\n"
                    f"{model_input}"
                )
                display, _ = format_single_token_action_display(first_tok or "?")
                if getattr(policy, "ppo_actor_prompt", False):
                    action_line = f"{display} | r={float(reward):.3f} | done={bool(done)}"
                else:
                    action_line = (
                        f"plan={full_plan or '?'} | exec={display} | "
                        f"r={float(reward):.3f} | done={bool(done)}"
                    )
                actions_overlay.append(action_line)
                prompts_overlay.append(overlay_prompt)

        if bool(done):
            break

    return EpisodeResult(
        slug=task_slug,
        instruction_idx=instruction_idx,
        episode_idx=episode_idx,
        total_steps=total_steps,
        total_reward=total_reward,
        instruction_done=instruction_done,
        valid_actions=valid_actions,
        frames=frames,
        prompts_overlay=prompts_overlay,
        actions_overlay=actions_overlay,
        action_ids=episode_action_ids,
    )


def _summarize_task(
    slug: str,
    instruction_idx: int,
    episodes: List[EpisodeResult],
    gif_path: str,
    action_hist_path: str = "",
) -> TaskSummary:
    n = len(episodes)
    successes = sum(1 for e in episodes if e.instruction_done)
    reward_sum = sum(e.total_reward for e in episodes)
    return TaskSummary(
        slug=slug,
        instruction_idx=instruction_idx,
        episodes=n,
        successes=successes,
        total_reward_sum=reward_sum,
        mean_reward=reward_sum / n if n else 0.0,
        success_rate=successes / n if n else 0.0,
        gif_path=gif_path,
        action_hist_path=action_hist_path,
    )


class CometValidationSession:
    """Incremental Comet logging: experiment visible after init, metrics per task."""

    def __init__(self, *, project_name: str, experiment_name: str, config: dict) -> None:
        from verl.utils.tracking import CometMLLogger

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.logger = CometMLLogger(
            project_name=project_name,
            experiment_name=experiment_name,
            config=config,
        )
        exp = self.logger.experiment
        url = getattr(exp, "url", None)
        if not url and hasattr(exp, "get_url"):
            try:
                url = exp.get_url()
            except Exception:
                url = None
        if url:
            print(f"[INFO] Comet run (live): {url}", flush=True)
        else:
            print(
                f"[INFO] Comet experiment started: project={project_name!r} "
                f"name={experiment_name!r}",
                flush=True,
            )

    def log_task(self, task_idx: int, summary: TaskSummary) -> None:
        prefix = f"val/{summary.slug}"
        self.logger.log(
            {
                f"{prefix}/success_rate": summary.success_rate,
                f"{prefix}/successes": float(summary.successes),
                f"{prefix}/episodes": float(summary.episodes),
                f"{prefix}/total_reward_sum": summary.total_reward_sum,
                f"{prefix}/mean_reward": summary.mean_reward,
            },
            step=task_idx,
        )
        if summary.gif_path and os.path.isfile(summary.gif_path):
            self.logger.log_video(
                summary.gif_path,
                step=task_idx,
                name=f"llm_planning_{summary.slug}",
            )
        if summary.action_hist_path and os.path.isfile(summary.action_hist_path):
            self.logger.log_image(
                summary.action_hist_path,
                step=task_idx,
                name=f"llm_planning_{summary.slug}_action_hist",
            )
        print(f"[INFO] Comet: logged task {summary.slug} (step={task_idx})", flush=True)

    def log_overall(
        self,
        *,
        overall_success_rate: float,
        overall_reward_sum: float,
        num_tasks: int,
        total_episodes: int,
    ) -> None:
        self.logger.log(
            {
                "val/success_rate": overall_success_rate,
                "val/total_reward_sum": overall_reward_sum,
                "val/num_tasks": float(num_tasks),
                "val/total_episodes": float(total_episodes),
            },
            step=num_tasks,
        )

    def finish(self) -> None:
        self.logger.finish()
        print(
            f"[INFO] Comet finished: project={self.project_name!r} "
            f"experiment={self.experiment_name!r}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receding-horizon LLM planner validation on debug_square_8x8"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="HF LoRA checkpoint dir (e.g. .../latest). Omit with --base-model-only.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"Base HF model (default: infer from checkpoint, or {DEFAULT_BASE_MODEL} with --base-model-only)",
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Load only the base HF model (factory weights), skip SFT/LoRA checkpoint",
    )
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--planning-mode",
        choices=("max_return", "return_conditioned"),
        default="max_return",
        help="max_return matches PLANNING_ADVANTAGE_WM training",
    )
    parser.add_argument(
        "--target-return",
        type=int,
        default=5,
        help="Only for return_conditioned planning prompt",
    )
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gif-path",
        default="gif/llm_planning_debug_square.gif",
        help="Base path; outputs ..._stone.gif, ..._wood.gif, ..._water.gif",
    )
    parser.add_argument(
        "--gif-episode",
        type=int,
        default=0,
        help="Which episode index to record as GIF per task",
    )
    parser.add_argument(
        "--ppo-actor-prompt",
        action="store_true",
        help="Use PPO single_token_action prompt (one greedy action token per step, no H-plan)",
    )
    parser.add_argument(
        "--comet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--comet-project",
        default=os.environ.get("COMET_PROJECT", "verl_agent_caged_craftext"),
    )
    args = parser.parse_args()

    if args.base_model_only:
        if not args.base_model:
            args.base_model = DEFAULT_BASE_MODEL
    elif not args.checkpoint:
        raise SystemExit("[ERROR] --checkpoint is required unless --base-model-only is set")
    elif not os.path.isdir(args.checkpoint):
        raise SystemExit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _setup_paths()
    root = _repo_root()
    os.chdir(root)

    print("=" * 72)
    print("LLM receding-horizon planning validation (debug_square_8x8)")
    print("=" * 72)
    print(f"[INFO] checkpoint={args.checkpoint or '(none — base model only)'}")
    print(f"[INFO] base_model={args.base_model or '(infer from checkpoint)'}")
    print(f"[INFO] base_model_only={args.base_model_only}")
    print(f"[INFO] horizon={args.horizon}  planning_mode={args.planning_mode}")
    print(f"[INFO] ppo_actor_prompt={args.ppo_actor_prompt}")
    print(f"[INFO] episodes_per_task={args.episodes_per_task}  max_steps={args.max_steps}")
    if args.ppo_actor_prompt:
        print("[INFO] strategy: PPO actor prompt -> greedy 1 action token per step")
    else:
        print("[INFO] strategy: plan H actions, execute 1st token, re-plan each step")

    comet_session: CometValidationSession | None = None
    if args.comet:
        if not os.environ.get("COMET_API_KEY"):
            print("[WARN] COMET_API_KEY not set — skip Comet logging")
        else:
            experiment_name = os.environ.get(
                "RUN_NAME",
                f"llm_planning_debug_square_h{args.horizon}",
            )
            comet_session = CometValidationSession(
                project_name=args.comet_project,
                experiment_name=experiment_name,
                config={
                    "script": "validate_debug_square_llm_planning",
                    "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
                    "base_model": args.base_model,
                    "base_model_only": args.base_model_only,
                    "horizon": args.horizon,
                    "planning_mode": args.planning_mode,
                    "ppo_actor_prompt": args.ppo_actor_prompt,
                    "episodes_per_task": args.episodes_per_task,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                    "tasks": [slug for _, slug in DEBUG_SQUARE_TASKS],
                },
            )

    policy = PlanningPolicy(
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else None,
        base_model=args.base_model,
        device=args.device,
        horizon=args.horizon,
        planning_mode=args.planning_mode,
        target_return=args.target_return,
        ppo_actor_prompt=args.ppo_actor_prompt,
        base_model_only=args.base_model_only,
    )

    base_gif = args.gif_path if os.path.isabs(args.gif_path) else os.path.join(root, args.gif_path)
    summaries: List[TaskSummary] = []
    all_episodes: List[EpisodeResult] = []

    print("[INFO] Creating env worker (one-time init)...", flush=True)
    worker = make_debug_square_worker(int(args.seed))
    try:
        for task_idx, (instruction_idx, slug) in enumerate(DEBUG_SQUARE_TASKS):
            gif_path = _task_gif_path(base_gif, slug)
            task_episodes: List[EpisodeResult] = []
            print(f"\n--- Task: {slug} (instruction_idx={instruction_idx}) ---")

            for ep in range(int(args.episodes_per_task)):
                capture = ep == int(args.gif_episode)
                result = run_episode(
                    policy=policy,
                    worker=worker,
                    instruction_idx=instruction_idx,
                    task_slug=slug,
                    episode_idx=ep,
                    max_steps=int(args.max_steps),
                    capture_frames=capture,
                )
                task_episodes.append(result)
                all_episodes.append(result)
                print(
                    f"  ep {ep}: steps={result.total_steps} reward={result.total_reward:.3f} "
                    f"success={result.instruction_done}",
                    flush=True,
                )

            rec = task_episodes[int(args.gif_episode)]
            hist_path = _task_action_hist_path(base_gif, slug)
            if rec.frames:
                _save_gif(rec.frames, rec.prompts_overlay, rec.actions_overlay, gif_path)
            else:
                print(f"[WARN] No frames for GIF ({slug}); skip video")
            if rec.action_ids:
                _save_action_histogram(
                    rec.action_ids,
                    hist_path,
                    title=f"Action histogram — {slug} ep {int(args.gif_episode)}",
                )
            else:
                hist_path = ""
                print(f"[WARN] No actions for histogram ({slug}); skip hist")

            summary = _summarize_task(
                slug, instruction_idx, task_episodes, gif_path, action_hist_path=hist_path
            )
            summaries.append(summary)
            print(
                f"[TASK {slug}] success_rate={summary.successes}/{summary.episodes} "
                f"({summary.success_rate:.1%})  total_reward_sum={summary.total_reward_sum:.3f} "
                f"mean_reward={summary.mean_reward:.3f}"
            )
            if comet_session is not None:
                comet_session.log_task(task_idx, summary)
    finally:
        worker.close()

    total_episodes = len(all_episodes)
    total_successes = sum(1 for e in all_episodes if e.instruction_done)
    overall_success_rate = total_successes / total_episodes if total_episodes else 0.0
    overall_reward_sum = sum(e.total_reward for e in all_episodes)

    print("\n" + "=" * 72)
    print("OVERALL")
    print("=" * 72)
    print(f"  episodes={total_episodes}  successes={total_successes}")
    print(f"  success_rate={overall_success_rate:.1%}")
    print(f"  total_reward_sum={overall_reward_sum:.3f}")
    print(f"  mean_reward_per_episode={overall_reward_sum / total_episodes if total_episodes else 0:.3f}")

    if comet_session is not None:
        comet_session.log_overall(
            overall_success_rate=overall_success_rate,
            overall_reward_sum=overall_reward_sum,
            num_tasks=len(summaries),
            total_episodes=total_episodes,
        )
        comet_session.finish()


if __name__ == "__main__":
    main()
