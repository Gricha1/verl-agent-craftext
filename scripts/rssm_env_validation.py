"""Read-only CrafText validation.  Episode length and MPC horizon are separate."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CAGED_CRAFTEXT_PATH", str(REPO_ROOT / "caged_craftext"))
from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT, craftext_projection

PLAN_ACTIONS = frozenset(("UP", "DOWN", "LEFT", "RIGHT"))


class _LocalCagedVector:
    """Same worker implementation as production, without Ray transport."""
    def __init__(self, config: Any, seeds: list[int]):
        kwargs = {"config_name": str(config.env.craftext_settings), "use_debug_square_map": True,
                  "observation_type": str(config.env.observation_type), "encode_form": "embedding"}
        self.seeds = [int(seed) for seed in seeds]
        self.workers = [CagedCraftextWorker(seed=seed, env_kwargs=kwargs) for seed in self.seeds]
        # Match the historical actor evaluator: each environment's scenario
        # sampling is driven by its own fixed evaluation seed.  A single global
        # RNG makes a 32-seed evaluation depend on batch size/order.
        self.rngs = [np.random.RandomState(seed) for seed in self.seeds]
    def reset(self):
        choices = [rng.randint(3) for rng in self.rngs]
        pairs = [worker.reset(scenario_idx=int(choice), return_render=False) for worker, choice in zip(self.workers, choices)]
        observations, infos = zip(*pairs)
        return list(observations), list(infos)
    def step(self, actions):
        values = [worker.step(int(action), return_render=False) for worker, action in zip(self.workers, actions)]
        obs, rewards, dones, infos = zip(*values)
        return list(obs), np.asarray(rewards), np.asarray(dones), list(infos)
    def close(self):
        for worker in self.workers:
            close = getattr(worker, "close", None)
            if close: close()


def _env_config(cfg: dict[str, Any]) -> Any:
    return OmegaConf.create({"env": {"env_name": "caged_craftext/CagedCraftextEnv",
        "craftext_settings": str(cfg.get("env_eval_craftext_settings", "debug_square_16x16")),
        "seed": int(cfg["env_eval_seed"]), "max_steps": int(cfg["env_episode_max_steps"]),
        "history_length": int(cfg.get("env_eval_history_length", 50)),
        "reasoning_history_length": int(cfg.get("env_eval_reasoning_history_length", 3)),
        "enable_reasoning": True, "prompt_template_type": "single_token_action_reasoning",
        "store_raw_reasoning_on_missing_action_tag": bool(cfg.get("env_eval_store_raw_reasoning_on_missing_action_tag", False)), "observation_type": "ascii",
        "auto_reset": False, "use_jax_gpu": False, "use_ray_text_render_workers": False},
        "data": {"train_batch_size": 1, "val_batch_size": 1}})


def _chat_parts(tokenizer, prompt: str) -> tuple[list[int], list[int]]:
    messages = [{"role": "user", "content": str(prompt)}]
    before = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before_ids = tokenizer(before, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(before_ids)] != before_ids:
        raise RuntimeError("chat generation boundary is not an exact token suffix")
    return before_ids, full_ids[len(before_ids):]


def _render_chat_prompt(tokenizer, prompt: str, assistant_prefix: str = "") -> str:
    """The exact textual chat prompt sent to Qwen, for trajectory auditing."""
    base = tokenizer.apply_chat_template([{"role": "user", "content": str(prompt)}], tokenize=False,
                                         add_generation_prompt=True)
    return base + assistant_prefix


def _normalise_actor_soft(qwen, soft: torch.Tensor) -> torch.Tensor:
    """Deterministically match every soft-token L2 norm to normal Qwen tokens."""
    normal = qwen.get_input_embeddings().weight.detach().float().norm(dim=-1).mean()
    return soft * (normal.to(soft.dtype) / soft.float().norm(dim=-1, keepdim=True).clamp_min(1e-8).to(soft.dtype))


def _actor_z_soft(qwen, model, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Exactly the transition z branch's five tokens. h is deliberately zeroed.
    soft = model.transition_tokens(torch.zeros_like(h), z)[1][0]
    if soft.shape[0] != model.k:
        raise RuntimeError("actor z interface must contain exactly soft_tokens")
    return _normalise_actor_soft(qwen, soft)


def _actor_hz_soft(qwen, model, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The HZ policy interface: five state tokens followed by five z tokens."""
    h_soft, z_soft = model.transition_tokens(h, z)
    if h_soft.shape[1] != model.k or z_soft.shape[1] != model.k:
        raise RuntimeError("HZ actor interface must contain five h and five z soft tokens")
    return torch.cat((_normalise_actor_soft(qwen, h_soft[0]), _normalise_actor_soft(qwen, z_soft[0])), 0)


@torch.no_grad()
def _generate(qwen, tokenizer, prompt: str, soft: torch.Tensor | None, max_new_tokens: int,
              assistant_prefix: str = "", generation: dict[str, Any] | None = None) -> str:
    """Soft conditioning is inserted before, never after, the assistant boundary."""
    before_ids, assistant_ids = _chat_parts(tokenizer, prompt)
    device = next(qwen.parameters()).device
    prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    generation = generation or {}
    do_sample = bool(generation.get("do_sample", False))
    kwargs: dict[str, Any] = {"do_sample": do_sample, "max_new_tokens": max_new_tokens,
                              "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if do_sample:
        kwargs.update({"temperature": float(generation.get("temperature", .7)),
                       "top_p": float(generation.get("top_p", 1.0)),
                       "top_k": int(generation.get("top_k", 0))})
    if soft is None:
        ids = torch.tensor(before_ids + assistant_ids + prefix_ids, device=device)[None]
        result = qwen.generate(input_ids=ids, **kwargs)
        new_ids = result[0, ids.shape[1]:]
    else:
        embedding = qwen.get_input_embeddings()
        prefix = embedding(torch.tensor(before_ids, device=device))
        boundary = embedding(torch.tensor(assistant_ids + prefix_ids, device=device))
        inputs = torch.cat((prefix, soft.to(embedding.weight.dtype), boundary), 0)[None]
        result = qwen.generate(inputs_embeds=inputs,
                               attention_mask=torch.ones(inputs.shape[:2], device=device, dtype=torch.long), **kwargs)
        # HF versions differ on whether they prepend an inputs_embeds prefix.
        new_ids = result[0, inputs.shape[1]:] if result.shape[1] > inputs.shape[1] else result[0]
    return assistant_prefix + tokenizer.decode(new_ids, skip_special_tokens=True)


def _action_name(response: str) -> str | None:
    ids, valid = craftext_projection([response])
    return str(ACTION_TO_TEXT[int(ids[0])]) if valid[0] and int(ids[0]) >= 0 else None


def _plan_actions(response: str, horizon: int) -> list[str] | None:
    match = re.search(r"<plan>\s*(.*?)\s*</plan>", response, re.I | re.S)
    if not match: return None
    names = [part.strip().upper() for part in re.split(r"[\s,]+", match.group(1).strip()) if part.strip()]
    return names if len(names) == horizon and all(name in PLAN_ACTIONS for name in names) else None


def _fallback_response() -> str: return "<action>NOOP</action>"


def _planning_context(prompt: str) -> str:
    """Keep the production state/task text but replace its actor-only suffix."""
    for marker in ("\nFirst, think about what to do next.", "\nChoose one of the available actions"):
        position = prompt.find(marker)
        if position >= 0:
            return prompt[:position]
    return prompt


def _plan_prompt(prompt: str, horizon: int) -> str:
    return (_planning_context(prompt) + "\n\nYou are planning, not choosing a single action. "
            "Output only one tag, exactly in this format: <plan>UP,LEFT,LEFT,DOWN,RIGHT,UP</plan>. "
            f"It must contain exactly {horizon} comma-separated actions from UP, DOWN, LEFT, RIGHT. "
            "Do not use an <action> tag and do not add an explanation.")


def _task_actor_prompt(task: str) -> str:
    """Policy text deliberately contains task only: no real or decoded observation."""
    action_names = ", ".join(str(name) for name in ACTION_TO_TEXT)
    return (f"Your goal is to complete the following task:\n**TASK:** {task}\n\n"
            "The current world state is supplied only as latent soft tokens. "
            f"Choose exactly one of these available actions: {action_names}. "
            "Output exactly one tag in this form: <action>ACTION</action>. "
            "Do not add an explanation.")


def _task_plan_prompt(task: str, horizon: int) -> str:
    return (_task_actor_prompt(task) + "\nPlan from the supplied latent state, not from an observation. "
            "Output only one tag exactly in this format: "
            "<plan>UP,LEFT,LEFT,DOWN,RIGHT,UP</plan>. "
            f"It must contain exactly {horizon} comma-separated actions from UP, DOWN, LEFT, RIGHT.")


def _revised_prompt(prompt: str, plan: list[str], horizon: int) -> str:
    candidate = ",".join(plan)
    return (_planning_context(prompt) + f"\n\nCandidate plan: <plan>{candidate}</plan>. "
            "Based on the supplied latent imagined rollout, output only these two tags on separate lines:\n"
            f"<plan>{candidate}</plan>\n<action>UP</action>\n"
            f"The plan must contain exactly {horizon} comma-separated actions from UP, DOWN, LEFT, RIGHT. "
            "The action must be one valid action. Do not add an explanation.")


def _task_revised_prompt(task: str, plan: list[str], horizon: int) -> str:
    candidate = ",".join(plan)
    return (_task_actor_prompt(task) + f"\nCandidate plan: <plan>{candidate}</plan>. "
            "Using only the supplied sequence of latent imagined states, output these two tags:\n"
            f"<plan>{candidate}</plan>\n<action>UP</action>\n"
            f"The plan must have exactly {horizon} actions from UP, DOWN, LEFT, RIGHT.")


def _norm_metrics(qwen, softs: list[torch.Tensor]) -> dict[str, float]:
    normal = qwen.get_input_embeddings().weight.detach().float().norm(dim=-1)
    values = torch.cat([soft.detach().float().norm(dim=-1).reshape(-1) for soft in softs])
    return {"env_interface/normal_token_embedding_norm_mean": float(normal.mean()),
            "env_interface/z_soft_token_embedding_norm_mean": float(values.mean()),
            "env_interface/z_soft_token_embedding_norm_std": float(values.std(unbiased=False)),
            "env_interface/z_soft_token_embedding_norm_max": float(values.max())}


@torch.no_grad()
def run_interface_sanity(qwen, tokenizer, model, device, cfg: dict[str, Any], prompts: list[str]) -> tuple[dict[str, float], dict[str, list[str]]]:
    """32-prompt raw-format check, independent of any environment rollout."""
    count = int(cfg.get("env_interface_smoke_prompts", 32)); prompts = prompts[:count]
    if len(prompts) != count: raise RuntimeError(f"need {count} production actor prompts for interface smoke")
    h = torch.zeros(1, int(cfg["latent_h_dim"]), device=device)
    zero_z = torch.zeros(1, int(cfg["latent_z_dim"]), device=device)
    predicted_z = model.prior_head(h).chunk(2, dim=-1)[0]
    zero_soft, predicted_soft = _actor_z_soft(qwen, model, h, zero_z), _actor_z_soft(qwen, model, h, predicted_z)
    max_new, horizon = int(cfg["env_eval_actor_max_new_tokens"]), int(cfg["plan_horizon"])
    raw: dict[str, list[str]] = {"base": [], "zero_z": [], "predicted_z": [], "initial_plan": [], "zero_z_initial_plan": [], "revised_plan": [], "zero_z_revised_plan": []}
    for prompt in prompts:
        raw["base"].append(_generate(qwen, tokenizer, prompt, None, max_new))
        raw["zero_z"].append(_generate(qwen, tokenizer, prompt, zero_soft, max_new))
        raw["predicted_z"].append(_generate(qwen, tokenizer, prompt, predicted_soft, max_new))
        initial_prompt = _plan_prompt(prompt, horizon)
        initial, zero_initial = (_generate(qwen, tokenizer, initial_prompt, None, max_new, "<plan>"),
                                 _generate(qwen, tokenizer, initial_prompt, zero_soft, max_new, "<plan>"))
        raw["initial_plan"].append(initial); raw["zero_z_initial_plan"].append(zero_initial)
        plan = _plan_actions(initial, horizon) or ["UP"] * horizon
        zero_plan = _plan_actions(zero_initial, horizon) or ["UP"] * horizon
        raw["revised_plan"].append(_generate(qwen, tokenizer, _revised_prompt(prompt, plan, horizon), None, max_new, "<plan>"))
        raw["zero_z_revised_plan"].append(_generate(qwen, tokenizer, _revised_prompt(prompt, zero_plan, horizon), zero_soft, max_new, "<plan>"))
    action_rate = lambda values: sum(_action_name(value) is not None for value in values) / len(values)
    plan_rate = lambda values: sum(_plan_actions(value, horizon) is not None for value in values) / len(values)
    revised_rate = lambda values: sum(_plan_actions(value, horizon) is not None and _action_name(value) is not None for value in values) / len(values)
    metrics = {"env_interface/base_raw_action_parse_rate": action_rate(raw["base"]),
        "env_interface/zero_z_raw_action_parse_rate": action_rate(raw["zero_z"]),
        "env_interface/predicted_z_raw_action_parse_rate": action_rate(raw["predicted_z"]),
        "env_interface/initial_plan_parse_rate": plan_rate(raw["initial_plan"]),
        "env_interface/revised_plan_parse_rate": revised_rate(raw["revised_plan"]),
        "env_interface/zero_z_initial_plan_parse_rate": plan_rate(raw["zero_z_initial_plan"]),
        "env_interface/zero_z_revised_plan_parse_rate": revised_rate(raw["zero_z_revised_plan"]),
        "env_interface/env_episode_max_steps": float(cfg["env_episode_max_steps"]),
        "env_interface/plan_horizon": float(horizon)}
    metrics.update(_norm_metrics(qwen, [zero_soft, predicted_soft]))
    return metrics, raw


def assert_interface_sanity(metrics: dict[str, float]) -> None:
    required = ("env_interface/initial_plan_parse_rate", "env_interface/revised_plan_parse_rate",
                "env_interface/zero_z_initial_plan_parse_rate", "env_interface/zero_z_revised_plan_parse_rate")
    failures = []
    if int(metrics["env_interface/env_episode_max_steps"]) == 6: failures.append("env_episode_max_steps == 6")
    if int(metrics["env_interface/plan_horizon"]) != 6: failures.append("plan_horizon != 6")
    if abs(metrics["env_interface/base_raw_action_parse_rate"] - metrics["env_interface/zero_z_raw_action_parse_rate"]) > .05:
        failures.append("zero-z raw parse differs from base by > 5pp")
    failures += [f"{key} < .95" for key in required if metrics[key] < .95]
    if failures: raise RuntimeError("interface sanity failed: " + "; ".join(failures))


def _summary(prefix: str, values: dict[str, float], episodes: int) -> dict[str, float]:
    steps = max(values["steps"], 1.)
    return {f"{prefix}/success_rate": values["success"] / max(episodes, 1),
            f"{prefix}/mean_reward": values["reward"] / max(episodes, 1),
            f"{prefix}/mean_episode_length": values["steps"] / max(episodes, 1),
            f"{prefix}/raw_action_parse_rate": values["raw_parsed"] / steps,
            f"{prefix}/fallback_rate": values["fallback"] / steps,
            f"{prefix}/executed_action_valid_rate": values["executed_valid"] / steps,
            f"{prefix}/repeated_action_rate": values["repeated"] / steps}


def _json_safe(value: Any) -> Any:
    """JSON conversion for worker info, which can contain NumPy scalar/arrays."""
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        return _json_safe(converted) if value.ndim == 0 else [_json_safe(item) for item in converted]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    return value


def _generation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"do_sample": bool(cfg.get("env_eval_do_sample", False)),
            "temperature": float(cfg.get("env_eval_temperature", .7)),
            "top_p": float(cfg.get("env_eval_top_p", 1.0)),
            "top_k": int(cfg.get("env_eval_top_k", 0))}


def _is_success(info: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Use the historical success definition unless an experiment opts in."""
    return bool(info.get("won", False) or
                (bool(cfg.get("env_eval_count_instruction_done_as_success", False)) and info.get("instruction_done", False)))


@torch.no_grad()
def _run_mode(mode: str, qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable,
              posterior_fn: Callable | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    episodes, horizon = int(cfg["env_eval_num_episodes"]), int(cfg["plan_horizon"])
    seeds = [int(cfg["env_eval_seed"]) + index for index in range(episodes)]
    env_cfg = _env_config(cfg); env = CagedCraftextEnvironmentManager(_LocalCagedVector(env_cfg, seeds), craftext_projection, env_cfg)
    values: dict[str, float] = defaultdict(float)
    traces: list[dict[str, Any]] = [{"mode": mode, "episode": index, "seed": seed, "steps": []}
                                    for index, seed in enumerate(seeds)]
    generation = _generation_cfg(cfg)
    try:
        observations, _ = env.reset({}); active = np.ones(episodes, dtype=bool); lengths = np.zeros(episodes, dtype=np.int64)
        h = torch.zeros(episodes, int(cfg["latent_h_dim"]), device=device); z = torch.zeros(episodes, int(cfg["latent_z_dim"]), device=device)
        if mode in ("hz_actor", "plan6_hz"):
            if posterior_fn is None:
                raise RuntimeError("HZ evaluation requires posterior_fn for z_t | h_t,o_t")
            for i in range(episodes):
                qm, _qs, _ql, _hidden = posterior_fn(qwen, tokenizer, model, str(env.tasks[i]), str(observations["anchor"][i]), h[i], device, cfg)
                z[i] = qm[0]
        previous: list[str | None] = [None] * episodes
        while active.any():
            responses = [_fallback_response()] * episodes; parsed = [False] * episodes; candidates: list[str | None] = [None] * episodes
            initial_ok = [False] * episodes; revised_ok = [False] * episodes; changed = [False] * episodes
            raw: dict[int, dict[str, Any]] = {}
            for i in np.flatnonzero(active):
                prompt = str(observations["text"][i])
                if mode == "base":
                    response = _generate(qwen, tokenizer, prompt, None, int(cfg["env_eval_actor_max_new_tokens"]), generation=generation)
                    raw[i] = {"action": response, "action_prompt": prompt, "assistant_prefix": "", "soft_conditioned": False}
                elif mode == "z_actor":
                    soft = None if previous[i] is None else _actor_z_soft(qwen, model, h[i:i + 1], z[i:i + 1])
                    response = _generate(qwen, tokenizer, prompt, soft, int(cfg["env_eval_actor_max_new_tokens"]), generation=generation)
                    raw[i] = {"action": response, "action_prompt": prompt, "assistant_prefix": "", "soft_conditioned": soft is not None,
                              "latent_h_l2": float(h[i].float().norm()), "latent_z_l2": float(z[i].float().norm()),
                              "soft_token_norms": [] if soft is None else [float(value) for value in soft.float().norm(dim=-1)]}
                elif mode == "hz_actor":
                    task = str(env.tasks[i]); soft = _actor_hz_soft(qwen, model, h[i:i + 1], z[i:i + 1])
                    actor_prompt = _task_actor_prompt(task)
                    response = _generate(qwen, tokenizer, actor_prompt, soft, int(cfg["env_eval_actor_max_new_tokens"]), generation=generation)
                    raw[i] = {"task": task, "action": response, "action_prompt": actor_prompt, "assistant_prefix": "", "soft_conditioned": True,
                              "actor_observation_tokens": 0, "latent_h_l2": float(h[i].float().norm()), "latent_z_l2": float(z[i].float().norm()),
                              "soft_token_norms": [float(value) for value in soft.float().norm(dim=-1)]}
                elif mode == "plan_only":
                    plan_prompt = _plan_prompt(prompt, horizon)
                    initial = _generate(qwen, tokenizer, plan_prompt, None, int(cfg["env_eval_actor_max_new_tokens"]), "<plan>", generation)
                    plan = _plan_actions(initial, horizon); initial_ok[i] = plan is not None; plan = plan or ["UP"] * horizon
                    response = f"<action>{plan[0]}</action>"
                    parsed[i] = initial_ok[i]
                    raw[i] = {"initial_plan": initial, "initial_plan_prompt": plan_prompt, "plan": plan,
                              "plan_fallback": not initial_ok[i], "action": response, "action_prompt": None,
                              "assistant_prefix": "<plan>", "soft_conditioned": False,
                              "policy_action_source": "initial_plan[0]"}
                elif mode in ("plan6", "plan6_hz"):
                    task = str(env.tasks[i]); hz = mode == "plan6_hz"
                    plan_prompt = _task_plan_prompt(task, horizon) if hz else _plan_prompt(prompt, horizon)
                    initial_soft = _actor_hz_soft(qwen, model, h[i:i + 1], z[i:i + 1]) if hz else None
                    initial = _generate(qwen, tokenizer, plan_prompt, initial_soft, int(cfg["env_eval_actor_max_new_tokens"]), "<plan>", generation)
                    plan = _plan_actions(initial, horizon); initial_ok[i] = plan is not None; plan = plan or ["UP"] * horizon
                    # The imagined world model is intentionally latent-only:
                    # (h,z,a) -> (h_next,z_next).  Decoding observations here
                    # would create an observation feedback path into dynamics.
                    ih, iz, imagined = h[i:i + 1], z[i:i + 1], []
                    imagined_steps: list[dict[str, Any]] = []
                    for j, action in enumerate(plan):
                        ih = transition_fn(qwen, tokenizer, model, action, ih, iz, device, cfg); iz = model.prior_head(ih).chunk(2, dim=-1)[0]; imagined.append(iz[0])
                        imagined_row = {"index": j, "action": action, "transition_input": "soft(h),soft(z),action",
                                        "h_l2": float(ih[0].float().norm()), "z_l2": float(iz[0].float().norm()),
                                        "z_mean": float(iz[0].float().mean()), "z_std": float(iz[0].float().std(unbiased=False))}
                        # Full predicted latents are needed only for the small
                        # fixed audit sample, not every 32x50x6 MPC step.
                        if i < int(cfg.get("env_eval_full_latent_trace_episodes", 0)):
                            imagined_row["predicted_h"] = [float(value) for value in ih[0].float().cpu()]
                            imagined_row["predicted_z"] = [float(value) for value in iz[0].float().cpu()]
                        imagined_steps.append(imagined_row)
                    if hz:
                        # Preserve every pair (h_0,z_0) ... (h_6,z_6), not decoded text.
                        pairs = [(h[i:i + 1], z[i:i + 1])]
                        ih2, iz2 = h[i:i + 1], z[i:i + 1]
                        for action in plan:
                            ih2 = transition_fn(qwen, tokenizer, model, action, ih2, iz2, device, cfg)
                            iz2 = model.prior_head(ih2).chunk(2, dim=-1)[0]
                            pairs.append((ih2, iz2))
                        soft = torch.cat([_actor_hz_soft(qwen, model, state_h, state_z) for state_h, state_z in pairs], 0)
                        expected = (horizon + 1) * 2 * int(cfg["soft_tokens"])
                        if soft.shape[0] != expected: raise RuntimeError("HZ plan latent count mismatch")
                        revised_prompt = _task_revised_prompt(task, plan, horizon)
                    else:
                        soft = torch.cat([_actor_z_soft(qwen, model, torch.zeros_like(state)[None], state[None]) for state in imagined], 0)
                        if soft.shape[0] != horizon * int(cfg["soft_tokens"]): raise RuntimeError("plan latent count mismatch")
                        revised_prompt = _revised_prompt(prompt, plan, horizon)
                    response = _generate(qwen, tokenizer, revised_prompt, soft, int(cfg["env_eval_actor_max_new_tokens"]), "<plan>", generation)
                    revised_ok[i] = _plan_actions(response, horizon) is not None; candidate = _action_name(response); changed[i] = candidate is not None and candidate != plan[0]; raw[i] = {"initial_plan": initial, "revised": response}
                    raw[i].update({"initial_plan_prompt": plan_prompt, "revised_prompt": revised_prompt, "plan": plan,
                                   "initial_plan_fallback": not initial_ok[i], "imagined_steps": imagined_steps,
                                   "soft_token_count": int(soft.shape[0]), "soft_conditioned": True,
                                   "soft_token_norms": [float(value) for value in soft.float().norm(dim=-1)],
                                   "revised_plan": _plan_actions(response, horizon), "revised_action": _action_name(response),
                                   "assistant_prefix": "<plan>", "task": task, "actor_observation_tokens": 0 if hz else None})
                    if hz and i < int(cfg.get("env_eval_full_latent_trace_episodes", 0)):
                        raw[i]["current_h"] = [float(value) for value in h[i].float().cpu()]
                        raw[i]["current_z"] = [float(value) for value in z[i].float().cpu()]
                else:
                    raise RuntimeError(f"unknown validation mode: {mode}")
                candidate = _action_name(response); candidates[i] = candidate
                if mode != "plan_only": parsed[i] = candidate is not None
                if parsed[i]: responses[i] = response
            previous_observations = list(observations["text"]); next_obs, rewards, dones, infos = env.step(responses)
            for i in np.flatnonzero(active):
                info = infos[i]; valid = bool(info.get("is_action_valid", False)); action = str(info.get("action_name", "")) if valid else None
                values["steps"] += 1; lengths[i] += 1; values["reward"] += float(rewards[i]); values["raw_parsed"] += float(parsed[i]); values["fallback"] += float(not parsed[i]); values["executed_valid"] += float(valid); values["repeated"] += float(action is not None and action == previous[i])
                if mode in ("plan_only", "plan6", "plan6_hz"): values["initial_plan_ok"] += float(initial_ok[i])
                if mode in ("plan6", "plan6_hz"): values["revised_plan_ok"] += float(revised_ok[i]); values["action_changed"] += float(changed[i])
                trace = {"step": int(lengths[i] - 1), "observation": previous_observations[i],
                         "messages": [{"role": "user", "content": previous_observations[i]}],
                         "raw_model_output": raw[i].get("action", raw[i].get("revised", "")),
                         "parsed_action": candidates[i] if parsed[i] else None, "parser_valid": bool(parsed[i]),
                         "fallback_used": bool(not parsed[i]), "environment_response": responses[i],
                         "executed_action": action, "previous_executed_action": previous[i],
                         "repeated_action": bool(action is not None and action == previous[i]),
                         "next_observation": str(next_obs["text"][i]), "reward": float(rewards[i]),
                         "terminated": bool(dones[i]), "truncated": bool(lengths[i] >= int(cfg["env_episode_max_steps"]) and not dones[i]),
                         "success": _is_success(info, cfg), "info": _json_safe(info),
                         "generation": _json_safe(raw[i])}
                action_prompt = raw[i].get("action_prompt") or raw[i].get("revised_prompt") or raw[i].get("initial_plan_prompt")
                if action_prompt is not None:
                    trace["rendered_chat_prompt"] = _render_chat_prompt(tokenizer, action_prompt, raw[i].get("assistant_prefix", ""))
                traces[i]["steps"].append(trace)
                actual = action or "NOOP"; next_h = transition_fn(qwen, tokenizer, model, actual, h[i:i + 1], z[i:i + 1], device, cfg); h[i] = next_h[0]
                if mode in ("hz_actor", "plan6_hz"):
                    qm, _qs, _ql, _hidden = posterior_fn(qwen, tokenizer, model, str(env.tasks[i]), str(next_obs["anchor"][i]), next_h[0], device, cfg)
                    z[i] = qm[0]
                else:
                    z[i] = model.prior_head(next_h).chunk(2, dim=-1)[0][0]
                previous[i] = actual
                if bool(dones[i]) or lengths[i] >= int(cfg["env_episode_max_steps"]):
                    success = _is_success(info, cfg); values["success"] += float(success); active[i] = False
                    traces[i].update({"final_reward": float(sum(step["reward"] for step in traces[i]["steps"])),
                                      "final_length": int(lengths[i]), "success": success,
                                      "termination_info": _json_safe(info)})
            observations = next_obs
            # Variable-length prompts and generation caches otherwise remain
            # reserved by the CUDA allocator across a long read-only rollout.
            # This is deliberately enabled by the checkpoint evaluator so it
            # can coexist safely with a live training job on the same GPU.
            if int(cfg.get("env_eval_empty_cache_every_steps", 0)) > 0 and torch.cuda.is_available():
                if int(values["steps"]) % int(cfg["env_eval_empty_cache_every_steps"]) == 0:
                    torch.cuda.empty_cache()
    finally:
        env.close()
    prefix = {"base": "env/base", "z_actor": "env/z_actor", "hz_actor": "env/hz_actor", "plan_only": "env/plan_only", "plan6": "env/plan6", "plan6_hz": "env/plan6_hz"}[mode]; metrics = _summary(prefix, values, episodes)
    if mode in ("plan_only", "plan6", "plan6_hz"):
        metrics[f"{prefix}/initial_plan_parse_rate"] = values["initial_plan_ok"] / max(values["steps"], 1.)
    if mode in ("plan6", "plan6_hz"):
        denom = max(values["steps"], 1.); metrics.update({f"{prefix}/initial_plan_parse_rate": values["initial_plan_ok"] / denom, f"{prefix}/revised_plan_parse_rate": values["revised_plan_ok"] / denom, f"{prefix}/revised_action_parse_rate": values["raw_parsed"] / denom, f"{prefix}/action_changed_rate": values["action_changed"] / denom})
    return metrics, traces


@torch.no_grad()
def run_env_validation(qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable,
                       episodes: int | None = None, debug_path: Path | None = None,
                       modes: tuple[str, ...] = ("base", "z_actor", "plan6"),
                       posterior_fn: Callable | None = None) -> dict[str, float]:
    if not bool(cfg.get("env_eval_enabled", False)): return {}
    cfg = dict(cfg)
    if episodes is not None: cfg["env_eval_num_episodes"] = int(episodes)
    was_training = model.training; model.eval()
    try:
        output, traces = {}, []
        for mode in modes:
            metrics, mode_traces = _run_mode(mode, qwen, tokenizer, model, device, cfg, transition_fn, posterior_fn)
            output.update(metrics); traces += mode_traces
            # Persist completed modes independently: an expensive later Plan6
            # pass must not hide already-audited Base/Z/PlanOnly evidence.
            if debug_path is not None:
                mode_path = debug_path.with_name(f"{debug_path.stem}_{mode}{debug_path.suffix}")
                mode_path.write_text(json.dumps({"metrics": metrics, "config": _json_safe(cfg), "episodes": mode_traces},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
        if debug_path is not None:
            debug_path.write_text(json.dumps({"metrics": output, "config": _json_safe(cfg), "episodes": traces}, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
    finally:
        if was_training: model.train()
