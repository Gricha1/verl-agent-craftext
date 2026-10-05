"""Causal, read-only CrafText validation for the RSSM reconstruction experiment.

The module deliberately owns neither an optimizer nor a replay buffer.  It uses
the production CrafText manager/parser and is called under ``torch.no_grad()``
from the offline trainer.  Three policies are evaluated on the same fixed seeds:
base Qwen, base Qwen augmented with predicted prior-z, and plan-6 imagination.
"""
from __future__ import annotations

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


class _LocalCagedVector:
    """The production worker transport without Ray (safe after torch/JAX init)."""

    def __init__(self, config: Any, seeds: list[int]):
        kwargs = {
            "config_name": str(config.env.craftext_settings),
            "use_debug_square_map": True,
            "observation_type": str(config.env.observation_type),
            "encode_form": "embedding",
        }
        self.workers = [CagedCraftextWorker(seed=int(seed), env_kwargs=kwargs) for seed in seeds]
        self._rng = np.random.RandomState(int(config.env.seed))

    def reset(self):
        # This is the same seeded scenario selection used by the normal vector env.
        scenarios = self._rng.choice(np.arange(3), size=len(self.workers), replace=True)
        pairs = [worker.reset(scenario_idx=int(scenario), return_render=False)
                 for worker, scenario in zip(self.workers, scenarios)]
        observations, infos = zip(*pairs)
        return list(observations), list(infos)

    def step(self, actions):
        result = [worker.step(int(action), return_render=False) for worker, action in zip(self.workers, actions)]
        obs, rewards, dones, infos = zip(*result)
        return list(obs), np.asarray(rewards), np.asarray(dones), list(infos)

    def close(self):
        for worker in self.workers:
            close = getattr(worker, "close", None)
            if close is not None:
                close()


def _env_config(cfg: dict[str, Any]) -> Any:
    return OmegaConf.create({"env": {
        "env_name": "caged_craftext/CagedCraftextEnv",
        "craftext_settings": str(cfg.get("env_eval_craftext_settings", "debug_square_16x16")),
        "seed": int(cfg["env_eval_seed"]),
        "max_steps": int(cfg.get("env_eval_max_steps", 50)),
        "history_length": int(cfg.get("env_eval_history_length", 50)),
        "reasoning_history_length": int(cfg.get("env_eval_reasoning_history_length", 3)),
        "enable_reasoning": True,
        "prompt_template_type": "single_token_action_reasoning",
        "store_raw_reasoning_on_missing_action_tag": True,
        "observation_type": "ascii",
        "auto_reset": False,
        "use_jax_gpu": False,
        "use_ray_text_render_workers": False,
    }, "data": {"train_batch_size": 1, "val_batch_size": 1}})


def _chat_input(tokenizer, prompt: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template([{"role": "user", "content": str(prompt)}], add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], device=tokenizer._rssm_device, dtype=torch.long)


@torch.no_grad()
def _generate(qwen, tokenizer, prompt: str, soft: torch.Tensor | None, max_new_tokens: int) -> str:
    """Greedy generation with optional continuous tokens immediately before output."""
    ids = _chat_input(tokenizer, prompt)
    if soft is None:
        generated = qwen.generate(input_ids=ids[None], do_sample=False, max_new_tokens=max_new_tokens,
                                  temperature=1.0, top_p=1.0, top_k=None,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_ids = generated[0, len(ids):]
    else:
        embed = qwen.get_input_embeddings()
        inputs = torch.cat((embed(ids), soft.to(embed.weight.dtype)), 0)[None]
        generated = qwen.generate(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
                                  do_sample=False, max_new_tokens=max_new_tokens,
                                  temperature=1.0, top_p=1.0, top_k=None,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        # HF returns only generated ids when generation starts from inputs_embeds.
        new_ids = generated[0]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _action_name(response: str) -> str | None:
    ids, valid = craftext_projection([response])
    if not valid[0] or int(ids[0]) < 0:
        return None
    return str(ACTION_TO_TEXT[int(ids[0])])


def _plan_actions(response: str, horizon: int) -> list[str] | None:
    match = re.search(r"<plan>\s*(.*?)\s*</plan>", response, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    names = [line.strip().upper().replace(" ", "_") for line in match.group(1).splitlines() if line.strip()]
    return names if len(names) == horizon and all(name in ACTION_TO_TEXT for name in names) else None


def _fallback_response() -> str:
    return "<action>NOOP</action>"


def _summary(prefix: str, aggregate: dict[str, float], episodes: int) -> dict[str, float]:
    denom = max(episodes, 1)
    steps = max(aggregate["steps"], 1.0)
    return {
        f"{prefix}/success_rate": aggregate["success"] / denom,
        f"{prefix}/mean_reward": aggregate["reward"] / denom,
        f"{prefix}/mean_episode_length": aggregate["steps"] / denom,
        f"{prefix}/valid_action_rate": aggregate["valid"] / steps,
        f"{prefix}/parse_error_rate": aggregate["parse_error"] / steps,
        f"{prefix}/repeated_action_rate": aggregate["repeated"] / steps,
    }


def _transition_row(prompt: str, action: str) -> dict[str, str]:
    # transition() uses the exact production prompt serialized as actor_prompt_t.
    return {"actor_prompt_t": str(prompt), "action_t": str(action)}


@torch.no_grad()
def _decode_observation(qwen, tokenizer, model, z: torch.Tensor, device, cfg: dict[str, Any]) -> str:
    from train_qwen_transition_rssm_recon import DECODER_INSTRUCTION
    instruction = qwen.get_input_embeddings()(torch.tensor(tokenizer(DECODER_INSTRUCTION, add_special_tokens=False)["input_ids"], device=device))
    initial = torch.cat((instruction, model.decoder_tokens(z[None])[0].to(instruction.dtype)), 0)[None]
    out = qwen.generate(inputs_embeds=initial, attention_mask=torch.ones(initial.shape[:2], device=device, dtype=torch.long),
                        do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                        temperature=1.0, top_p=1.0, top_k=None,
                        max_new_tokens=int(cfg.get("env_eval_decoder_max_new_tokens", 256)))
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _imagined_prompt(task: str, observation: str, plan_prefix: list[str]) -> str:
    history = ", ".join(plan_prefix) if plan_prefix else "(none)"
    return f"Your goal is to complete the following task:\n**TASK:** {task}\n\nImagined prior actions: {history}\n\nThis is what you currently see:\n{observation}\n\nChoose one action in <action> tags."


@torch.no_grad()
def _run_mode(mode: str, qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable) -> dict[str, float]:
    episodes = int(cfg["env_eval_num_episodes"])
    horizon = int(cfg["env_eval_plan_horizon"])
    seeds = [int(cfg["env_eval_seed"]) + index for index in range(episodes)]
    config = _env_config(cfg)
    env = CagedCraftextEnvironmentManager(_LocalCagedVector(config, seeds), craftext_projection, config)
    values: dict[str, float] = defaultdict(float)
    try:
        observations, _ = env.reset({})
        active = np.ones(episodes, dtype=bool)
        episode_steps = np.zeros(episodes, dtype=np.int64)
        max_steps = int(cfg.get("env_eval_max_steps", 50))
        h = torch.zeros(episodes, int(cfg["latent_h_dim"]), device=device)
        z = torch.zeros(episodes, int(cfg["latent_z_dim"]), device=device)
        previous_actions: list[str | None] = [None] * episodes
        while bool(active.any()):
            responses = [_fallback_response()] * episodes
            executed_candidates: list[str | None] = [None] * episodes
            plan_ok = [False] * episodes
            revised_ok = [False] * episodes
            changed = [False] * episodes
            for i in np.flatnonzero(active):
                prompt = str(observations["text"][i])
                if mode == "base":
                    response = _generate(qwen, tokenizer, prompt, None, int(cfg["env_eval_actor_max_new_tokens"]))
                elif mode == "z_actor":
                    soft = None if previous_actions[i] is None else model.transition_tokens(h[i:i + 1], z[i:i + 1])[1][0]
                    if soft is not None and soft.shape[0] != int(cfg["soft_tokens"]):
                        raise RuntimeError("z actor must receive exactly transition_z_projector soft_tokens")
                    response = _generate(qwen, tokenizer, prompt, soft, int(cfg["env_eval_actor_max_new_tokens"]))
                else:
                    initial_prompt = prompt + f"\n\nReturn exactly <plan> with {horizon} lines, each one valid action name, then </plan>."
                    initial_raw = _generate(qwen, tokenizer, initial_prompt, None, int(cfg["env_eval_actor_max_new_tokens"]))
                    plan = _plan_actions(initial_raw, horizon)
                    plan_ok[i] = plan is not None
                    if plan is None:
                        plan = ["NOOP"] * horizon
                    ih, iz = h[i:i + 1], z[i:i + 1]
                    imagined = []
                    imagined_obs = str(observations["anchor"][i])
                    for j, planned_action in enumerate(plan):
                        imagined_row = _transition_row(prompt if j == 0 else _imagined_prompt(str(env.tasks[i]), imagined_obs, plan[:j]), planned_action)
                        ih = transition_fn(qwen, tokenizer, model, imagined_row, ih, iz, device, cfg)
                        iz = model.prior_head(ih).chunk(2, dim=-1)[0]  # deterministic prior mean; no posterior.
                        imagined.append(iz[0])
                        if j + 1 < horizon:
                            imagined_obs = _decode_observation(qwen, tokenizer, model, iz[0], device, cfg)
                    soft = torch.cat([model.transition_tokens(torch.zeros_like(state)[None], state[None])[1][0] for state in imagined], dim=0)
                    if soft.shape[0] != horizon * int(cfg["soft_tokens"]):
                        raise RuntimeError("plan6 refinement must receive exactly horizon * soft_tokens latent embeddings")
                    revise_prompt = (prompt + "\n\nCandidate plan:\n" + "\n".join(f"Step {j + 1}: {action}" for j, action in enumerate(plan)) +
                                     "\n\nBased on imagined future states, return <revised_plan> with exactly " + str(horizon) +
                                     " action lines and <action>ACTION</action>.")
                    response = _generate(qwen, tokenizer, revise_prompt, soft, int(cfg["env_eval_actor_max_new_tokens"]))
                    revised_plan = _plan_actions(response.replace("revised_plan", "plan"), horizon)
                    revised_ok[i] = revised_plan is not None
                    candidate = _action_name(response)
                    changed[i] = candidate is not None and candidate != plan[0]
                candidate = _action_name(response)
                if candidate is None:
                    responses[i] = _fallback_response()
                else:
                    responses[i] = response
                    executed_candidates[i] = candidate
            previous_observations = list(observations["text"])
            next_obs, rewards, dones, infos = env.step(responses)
            for i in np.flatnonzero(active):
                info = infos[i]
                valid = bool(info.get("is_action_valid", False))
                action_name = str(info.get("action_name", "")) if valid else None
                values["steps"] += 1
                episode_steps[i] += 1
                values["reward"] += float(rewards[i])
                values["valid"] += float(valid)
                values["parse_error"] += float(not valid or executed_candidates[i] is None)
                values["repeated"] += float(action_name is not None and action_name == previous_actions[i])
                if mode == "plan6":
                    values["initial_plan_ok"] += float(plan_ok[i])
                    values["revised_plan_ok"] += float(revised_ok[i])
                    values["action_changed"] += float(changed[i])
                # Update only from the real current observation/action. No posterior is used.
                actual = action_name or "NOOP"
                hi = transition_fn(qwen, tokenizer, model, _transition_row(previous_observations[i], actual), h[i:i + 1], z[i:i + 1], device, cfg)
                h[i] = hi[0]
                z[i] = model.prior_head(hi).chunk(2, dim=-1)[0][0]
                previous_actions[i] = actual
                if bool(dones[i]) or int(episode_steps[i]) >= max_steps:
                    values["success"] += float(bool(info.get("won", False) or info.get("instruction_done", False)))
                    active[i] = False
            observations = next_obs
    finally:
        env.close()
    prefix = {"base": "env/base", "z_actor": "env/z_actor", "plan6": "env/plan6"}[mode]
    metrics = _summary(prefix, values, episodes)
    if mode == "plan6":
        denom = max(values["steps"], 1.0)
        metrics.update({f"{prefix}/initial_plan_parse_rate": values["initial_plan_ok"] / denom,
                        f"{prefix}/revised_plan_parse_rate": values["revised_plan_ok"] / denom,
                        f"{prefix}/action_changed_rate": values["action_changed"] / denom})
    return metrics


@torch.no_grad()
def run_env_validation(qwen, tokenizer, model, device, cfg: dict[str, Any], transition_fn: Callable, episodes: int | None = None) -> dict[str, float]:
    """Run the three strictly causal policy evaluations and restore train mode."""
    if not bool(cfg.get("env_eval_enabled", False)):
        return {}
    cfg = dict(cfg)
    if episodes is not None:
        cfg["env_eval_num_episodes"] = int(episodes)
    was_training = model.training
    model.eval()
    tokenizer._rssm_device = device
    try:
        out = {}
        for mode in ("base", "z_actor", "plan6"):
            out.update(_run_mode(mode, qwen, tokenizer, model, device, cfg, transition_fn))
        return out
    finally:
        if was_training:
            model.train()
        if hasattr(tokenizer, "_rssm_device"):
            delattr(tokenizer, "_rssm_device")
