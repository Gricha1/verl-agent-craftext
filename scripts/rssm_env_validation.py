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
        self.workers = [CagedCraftextWorker(seed=int(seed), env_kwargs=kwargs) for seed in seeds]
        self.rng = np.random.RandomState(int(config.env.seed))
    def reset(self):
        choices = self.rng.choice(np.arange(3), size=len(self.workers), replace=True)
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
        "store_raw_reasoning_on_missing_action_tag": True, "observation_type": "ascii",
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


@torch.no_grad()
def _generate(qwen, tokenizer, prompt: str, soft: torch.Tensor | None, max_new_tokens: int) -> str:
    """Soft conditioning is inserted before, never after, the assistant boundary."""
    before_ids, assistant_ids = _chat_parts(tokenizer, prompt)
    device = next(qwen.parameters()).device
    if soft is None:
        ids = torch.tensor(before_ids + assistant_ids, device=device)[None]
        result = qwen.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_ids = result[0, ids.shape[1]:]
    else:
        embedding = qwen.get_input_embeddings()
        prefix = embedding(torch.tensor(before_ids, device=device))
        boundary = embedding(torch.tensor(assistant_ids, device=device))
        inputs = torch.cat((prefix, soft.to(embedding.weight.dtype), boundary), 0)[None]
        result = qwen.generate(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=device, dtype=torch.long),
                               do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)
        # HF versions differ on whether they prepend an inputs_embeds prefix.
        new_ids = result[0, inputs.shape[1]:] if result.shape[1] > inputs.shape[1] else result[0]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _action_name(response: str) -> str | None:
    ids, valid = craftext_projection([response])
    return str(ACTION_TO_TEXT[int(ids[0])]) if valid[0] and int(ids[0]) >= 0 else None


def _plan_actions(response: str, horizon: int) -> list[str] | None:
    match = re.search(r"<plan>\s*(.*?)\s*</plan>", response, re.I | re.S)
    if not match: return None
    names = [part.strip().upper() for part in re.split(r"[\s,]+", match.group(1).strip()) if part.strip()]
    return names if len(names) == horizon and all(name in PLAN_ACTIONS for name in names) else None


def _fallback_response() -> str: return "<action>NOOP</action>"
def _transition_row(prompt: str, action: str) -> dict[str, str]: return {"actor_prompt_t": str(prompt), "action_t": str(action)}


def _plan_prompt(prompt: str, horizon: int) -> str:
    return (f"{prompt}\n\nReturn exactly <plan>UP,LEFT,LEFT,DOWN,RIGHT,UP</plan>. "
            f"The tag must contain exactly {horizon} comma-separated actions from UP, DOWN, LEFT, RIGHT. No explanation.")


def _revised_prompt(prompt: str, plan: list[str], horizon: int) -> str:
    candidate = ",".join(plan)
    return (f"{prompt}\n\nCandidate plan: <plan>{candidate}</plan>. Based on imagined future states, "
            f"return exactly <plan>{candidate}</plan><action>UP</action>. The plan has exactly {horizon} "
            "comma-separated actions from UP, DOWN, LEFT, RIGHT. No explanation.")


def _imagined_prompt(task: str, observation: str, history: list[str]) -> str:
    previous = ", ".join(history) if history else "(none)"
    return (f"Your goal is to complete the following task:\n**TASK:** {task}\n\nImagined prior actions: {previous}\n\n"
            f"This is what you currently see:\n{observation}\n\nChoose one action in <action> tags.")


@torch.no_grad()
def _decode_observation(qwen, tokenizer, model, z: torch.Tensor, device, cfg: dict[str, Any]) -> str:
    from train_qwen_transition_rssm_recon import DECODER_INSTRUCTION
    ids = tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)["input_ids"]
    instruction = qwen.get_input_embeddings()(torch.tensor(ids, device=device))
    inputs = torch.cat((instruction, model.decoder_tokens(z[None])[0].to(instruction.dtype)), 0)[None]
    result = qwen.generate(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=device, dtype=torch.long),
                           do_sample=False, max_new_tokens=int(cfg["env_eval_decoder_max_new_tokens"]),
                           eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    new_ids = result[0, inputs.shape[1]:] if result.shape[1] > inputs.shape[1] else result[0]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


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
        initial, zero_initial = _generate(qwen, tokenizer, initial_prompt, None, max_new), _generate(qwen, tokenizer, initial_prompt, zero_soft, max_new)
        raw["initial_plan"].append(initial); raw["zero_z_initial_plan"].append(zero_initial)
        plan = _plan_actions(initial, horizon) or ["UP"] * horizon
        zero_plan = _plan_actions(zero_initial, horizon) or ["UP"] * horizon
        raw["revised_plan"].append(_generate(qwen, tokenizer, _revised_prompt(prompt, plan, horizon), None, max_new))
        raw["zero_z_revised_plan"].append(_generate(qwen, tokenizer, _revised_prompt(prompt, zero_plan, horizon), zero_soft, max_new))
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


@torch.no_grad()
def _run_mode(mode: str, qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable) -> tuple[dict[str, float], list[dict[str, Any]]]:
    episodes, horizon = int(cfg["env_eval_num_episodes"]), int(cfg["plan_horizon"])
    seeds = [int(cfg["env_eval_seed"]) + index for index in range(episodes)]
    env_cfg = _env_config(cfg); env = CagedCraftextEnvironmentManager(_LocalCagedVector(env_cfg, seeds), craftext_projection, env_cfg)
    values: dict[str, float] = defaultdict(float); debug: list[dict[str, Any]] = []
    try:
        observations, _ = env.reset({}); active = np.ones(episodes, dtype=bool); lengths = np.zeros(episodes, dtype=np.int64)
        h = torch.zeros(episodes, int(cfg["latent_h_dim"]), device=device); z = torch.zeros(episodes, int(cfg["latent_z_dim"]), device=device)
        previous: list[str | None] = [None] * episodes
        while active.any():
            responses = [_fallback_response()] * episodes; parsed = [False] * episodes; initial_ok = [False] * episodes; revised_ok = [False] * episodes; changed = [False] * episodes; raw = {}
            for i in np.flatnonzero(active):
                prompt = str(observations["text"][i])
                if mode == "base":
                    response = _generate(qwen, tokenizer, prompt, None, int(cfg["env_eval_actor_max_new_tokens"])); raw[i] = {"action": response}
                elif mode == "z_actor":
                    soft = None if previous[i] is None else _actor_z_soft(qwen, model, h[i:i + 1], z[i:i + 1])
                    response = _generate(qwen, tokenizer, prompt, soft, int(cfg["env_eval_actor_max_new_tokens"])); raw[i] = {"action": response}
                else:
                    initial = _generate(qwen, tokenizer, _plan_prompt(prompt, horizon), None, int(cfg["env_eval_actor_max_new_tokens"]))
                    plan = _plan_actions(initial, horizon); initial_ok[i] = plan is not None; plan = plan or ["UP"] * horizon
                    ih, iz, imagined = h[i:i + 1], z[i:i + 1], []; imagined_obs = str(observations["anchor"][i])
                    for j, action in enumerate(plan):
                        row = _transition_row(prompt if j == 0 else _imagined_prompt(str(env.tasks[i]), imagined_obs, plan[:j]), action)
                        ih = transition_fn(qwen, tokenizer, model, row, ih, iz, device, cfg); iz = model.prior_head(ih).chunk(2, dim=-1)[0]; imagined.append(iz[0])
                        if j + 1 < horizon: imagined_obs = _decode_observation(qwen, tokenizer, model, iz[0], device, cfg)
                    soft = torch.cat([_actor_z_soft(qwen, model, torch.zeros_like(state)[None], state[None]) for state in imagined], 0)
                    if soft.shape[0] != horizon * int(cfg["soft_tokens"]): raise RuntimeError("plan latent count mismatch")
                    response = _generate(qwen, tokenizer, _revised_prompt(prompt, plan, horizon), soft, int(cfg["env_eval_actor_max_new_tokens"]))
                    revised_ok[i] = _plan_actions(response, horizon) is not None; candidate = _action_name(response); changed[i] = candidate is not None and candidate != plan[0]; raw[i] = {"initial_plan": initial, "revised": response}
                candidate = _action_name(response); parsed[i] = candidate is not None
                if parsed[i]: responses[i] = response
            previous_observations = list(observations["text"]); next_obs, rewards, dones, infos = env.step(responses)
            for i in np.flatnonzero(active):
                info = infos[i]; valid = bool(info.get("is_action_valid", False)); action = str(info.get("action_name", "")) if valid else None
                values["steps"] += 1; lengths[i] += 1; values["reward"] += float(rewards[i]); values["raw_parsed"] += float(parsed[i]); values["fallback"] += float(not parsed[i]); values["executed_valid"] += float(valid); values["repeated"] += float(action is not None and action == previous[i])
                if mode == "plan6": values["initial_plan_ok"] += float(initial_ok[i]); values["revised_plan_ok"] += float(revised_ok[i]); values["action_changed"] += float(changed[i])
                if not parsed[i]: debug.append({"mode": mode, "episode": int(i), "step": int(lengths[i] - 1), **raw[i]})
                actual = action or "NOOP"; next_h = transition_fn(qwen, tokenizer, model, _transition_row(previous_observations[i], actual), h[i:i + 1], z[i:i + 1], device, cfg); h[i] = next_h[0]; z[i] = model.prior_head(next_h).chunk(2, dim=-1)[0][0]; previous[i] = actual
                if bool(dones[i]) or lengths[i] >= int(cfg["env_episode_max_steps"]): values["success"] += float(bool(info.get("won", False) or info.get("instruction_done", False))); active[i] = False
            observations = next_obs
    finally:
        env.close()
    prefix = {"base": "env/base", "z_actor": "env/z_actor", "plan6": "env/plan6"}[mode]; metrics = _summary(prefix, values, episodes)
    if mode == "plan6":
        denom = max(values["steps"], 1.); metrics.update({f"{prefix}/initial_plan_parse_rate": values["initial_plan_ok"] / denom, f"{prefix}/revised_plan_parse_rate": values["revised_plan_ok"] / denom, f"{prefix}/revised_action_parse_rate": values["raw_parsed"] / denom, f"{prefix}/action_changed_rate": values["action_changed"] / denom})
    return metrics, debug


@torch.no_grad()
def run_env_validation(qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable, episodes: int | None = None, debug_path: Path | None = None) -> dict[str, float]:
    if not bool(cfg.get("env_eval_enabled", False)): return {}
    cfg = dict(cfg)
    if episodes is not None: cfg["env_eval_num_episodes"] = int(episodes)
    was_training = model.training; model.eval()
    try:
        output, debug = {}, []
        for mode in ("base", "z_actor", "plan6"):
            metrics, failures = _run_mode(mode, qwen, tokenizer, model, device, cfg, transition_fn); output.update(metrics); debug += failures
        if debug_path is not None: debug_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in debug) + ("\n" if debug else ""), encoding="utf-8")
        return output
    finally:
        if was_training: model.train()
