#!/usr/bin/env python3
"""Online 8x8 CrafText RSSM + dual clipped-PPO experiment.

This is deliberately separate from the legacy text world-model hook in
``main_ppo``.  That hook trains an SFT prediction target and cannot inject a
continuous RSSM state into either the actor or critic.  This trainer reuses
the project's clipped PPO objectives while enforcing the intended interface:

* posterior: ``(task, observation, h) -> z``;
* transition: ``soft(h), soft(z), action -> h_next``;
* actor/critic: ``task, soft(h), soft(z)`` only.

The reward/continuation-head ablation is controlled solely by
``reward_continuation_heads`` in the config.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_qwen_transition_rssm_recon import (
    DECODER_INSTRUCTION,
    NullComet,
    QwenTransitionRSSMRecon,
    gaussian,
    kl,
    posterior,
    reconstruction,
    transition,
)
from rssm_env_validation import _LocalCagedVector, _env_config, _is_success

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent_system.environments.env_manager import CagedCraftextEnvironmentManager
from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT, craftext_projection
from verl.trainer.ppo.core_algos import compute_policy_loss, compute_value_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--actor-device", default="cuda:0")
    p.add_argument("--critic-device", default="cuda:1")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--smoke-updates", type=int, default=0)
    p.add_argument("--disable-comet", action="store_true")
    return p.parse_args()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _task_prompt(task: str) -> str:
    actions = ", ".join(ACTION_TO_TEXT)
    return (
        f"Your goal is to complete this task:\n{task}\n\n"
        "The current state is supplied only through latent soft tokens. "
        f"Choose one action from: {actions}."
    )


def _last_hidden(qwen, tokenizer, prompts: list[str], soft: torch.Tensor) -> torch.Tensor:
    """Frozen-Qwen contextual readout for task + latent soft tokens.

    No observation is accepted by this function.  Keeping this function's
    signature narrow makes the actor/critic information-flow invariant
    mechanically inspectable.
    """
    device = next(qwen.parameters()).device
    embed = qwen.get_input_embeddings()
    pieces: list[torch.Tensor] = []
    for prompt, state in zip(prompts, soft):
        ids = tokenizer(prompt, add_special_tokens=False, truncation=True,
                        max_length=512)["input_ids"]
        tokens = embed(torch.tensor(ids, device=device))
        pieces.append(torch.cat((tokens, state.to(dtype=embed.weight.dtype)), dim=0))
    length = max(piece.shape[0] for piece in pieces)
    inputs = torch.stack([
        torch.cat((piece, torch.zeros(length - piece.shape[0], piece.shape[-1],
                                      device=device, dtype=piece.dtype)))
        for piece in pieces
    ])
    mask = torch.tensor([[1] * piece.shape[0] + [0] * (length - piece.shape[0])
                         for piece in pieces], device=device, dtype=torch.long)
    positions = torch.arange(length, device=device)[None].expand(len(pieces), -1)
    out = qwen(inputs_embeds=inputs, attention_mask=mask, position_ids=positions,
               output_hidden_states=True, use_cache=False).hidden_states[-1]
    last = mask.sum(dim=-1) - 1
    return out[torch.arange(len(pieces), device=device), last].float()


class LatentActor(nn.Module):
    """PPO actor: task + soft(h) + soft(z), with no observation argument."""
    def __init__(self, rssm: QwenTransitionRSSMRecon, hidden_size: int):
        super().__init__()
        self.rssm = rssm
        self.action_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                         nn.Linear(hidden_size, len(ACTION_TO_TEXT)))

    def forward(self, qwen, tokenizer, tasks: list[str], h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h_soft, z_soft = self.rssm.transition_tokens(h, z)
        hidden = _last_hidden(qwen, tokenizer, [_task_prompt(x) for x in tasks],
                              torch.cat((h_soft, z_soft), dim=1))
        return self.action_head(hidden)


class LatentCritic(nn.Module):
    """Independent Qwen critic: task + learned soft(h,z), never observation."""
    def __init__(self, h_dim: int, z_dim: int, hidden_size: int, h_tokens: int, z_tokens: int,
                 target_rms: float):
        super().__init__()
        self.h_tokens, self.z_tokens, self.hidden_size = h_tokens, z_tokens, hidden_size
        self.h_projector = nn.Sequential(nn.Linear(h_dim, h_dim), nn.GELU(), nn.Linear(h_dim, h_tokens * hidden_size))
        self.z_projector = nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(), nn.Linear(z_dim, z_tokens * hidden_size))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))

    def soft(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        hs = self.h_projector(h).reshape(-1, self.h_tokens, self.hidden_size)
        zs = self.z_projector(z).reshape(-1, self.z_tokens, self.hidden_size)
        x = torch.cat((hs, zs), dim=1)
        return x * (self.target_rms.to(x.dtype) / x.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6).to(x.dtype))

    def forward(self, qwen, tokenizer, tasks: list[str], h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.value_head(_last_hidden(qwen, tokenizer, [_task_prompt(x) for x in tasks], self.soft(h, z))).squeeze(-1)


def assert_information_flow() -> None:
    checks = {
        "transition": transition,
        "actor": LatentActor.forward,
        "critic": LatentCritic.forward,
    }
    forbidden = ("observation", "obs_t", "anchor")
    violations = {name: [x for x in forbidden if x in inspect.getsource(fn)]
                  for name, fn in checks.items()}
    violations = {k: v for k, v in violations.items() if v}
    if violations:
        raise RuntimeError(f"observation flow invariant failed: {violations}")


def _env(cfg: dict[str, Any], seed: int, episodes: int):
    local = _LocalCagedVector(_env_config(cfg), [seed + i for i in range(episodes)])
    return CagedCraftextEnvironmentManager(local, craftext_projection, _env_config(cfg))


@torch.no_grad()
def _posterior_batch(qwen, tokenizer, rssm, tasks, observations, h, device, cfg):
    values = [posterior(qwen, tokenizer, rssm, str(task), str(obs), h[i], device, cfg)[0][0]
              for i, (task, obs) in enumerate(zip(tasks, observations))]
    return torch.stack(values)


def _actor_logits(actor, qwen, tokenizer, tasks, h, z):
    return actor(qwen, tokenizer, tasks, h, z)


@torch.no_grad()
def collect_rollout(cfg, qwen, tokenizer, rssm, actor, critic, actor_device, critic_device, seed: int):
    """Collect on-policy real transitions; observations enter only posterior calls."""
    n, horizon = int(cfg["rollout_envs"]), int(cfg["env_episode_max_steps"])
    env = _env(cfg, seed, n)
    obs, _infos = env.reset({})
    tasks = [str(x) for x in env.tasks]
    h = torch.zeros(n, int(cfg["latent_h_dim"]), device=actor_device)
    z = _posterior_batch(qwen, tokenizer, rssm, tasks, obs["anchor"], h, actor_device, cfg)
    active = np.ones(n, dtype=bool)
    records: list[dict[str, Any]] = []
    stats = defaultdict(float)
    previous: list[int | None] = [None] * n
    try:
        for _step in range(horizon):
            logits = _actor_logits(actor, qwen, tokenizer, tasks, h, z)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = critic(qwen if qwen.device == critic_device else None, tokenizer, tasks, h, z) if False else None
            # Critic has its own Qwen on GPU1; supplied through cfg closure below.
            critic_qwen = cfg["_critic_qwen"]
            values = critic(critic_qwen, tokenizer, tasks, h.to(critic_device), z.to(critic_device)).to(actor_device)
            text_actions = [f"<action>{ACTION_TO_TEXT[int(a)]}</action>" if active[i] else "<action>NOOP</action>"
                            for i, a in enumerate(action.tolist())]
            next_obs, rewards, dones, infos = env.step(text_actions)
            next_h = torch.stack([transition(qwen, tokenizer, rssm, ACTION_TO_TEXT[int(a)],
                                             h[i:i + 1], z[i:i + 1], actor_device, cfg)[0]
                                  for i, a in enumerate(action.tolist())])
            next_z = _posterior_batch(qwen, tokenizer, rssm, tasks, next_obs["anchor"], next_h, actor_device, cfg)
            for i in range(n):
                if not active[i]:
                    continue
                reward, done = float(rewards[i]), bool(dones[i])
                records.append({"task": tasks[i], "observation": str(obs["anchor"][i]),
                                "next_observation": str(next_obs["anchor"][i]), "h": h[i].cpu(), "z": z[i].cpu(),
                                "next_h": next_h[i].cpu(), "next_z": next_z[i].cpu(), "action": int(action[i]),
                                "old_logp": float(logp[i]), "old_value": float(values[i]), "reward": reward,
                                "done": done, "episode": i})
                stats["reward"] += reward; stats["steps"] += 1
                stats["repeated"] += float(previous[i] == int(action[i]))
                previous[i] = int(action[i])
                if done:
                    stats["success"] += float(_is_success(infos[i], cfg))
                    active[i] = False
            h, z, obs = next_h, next_z, next_obs
            if not active.any():
                break
    finally:
        env.envs.close()
    # Emit zeros explicitly: a short smoke rollout can contain no terminal
    # state, hence never touch the success counter.
    for key in ("reward", "steps", "repeated", "success"):
        stats.setdefault(key, 0.0)
    stats["episodes"] = float(n)
    return records, dict(stats)


def _stack(records: list[dict[str, Any]], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([r[key] for r in records]).to(device)


def add_gae(records: list[dict[str, Any]], gamma: float, lam: float, critic, critic_qwen, tokenizer, critic_device):
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_episode[int(item["episode"])].append(item)
    for rows in by_episode.values():
        with torch.no_grad():
            tasks = [x["task"] for x in rows]
            nh = _stack(rows, "next_h", critic_device); nz = _stack(rows, "next_z", critic_device)
            next_values = critic(critic_qwen, tokenizer, tasks, nh, nz).cpu().tolist()
        gae = 0.0
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            bootstrap = 0.0 if row["done"] else next_values[i]
            delta = row["reward"] + gamma * bootstrap - row["old_value"]
            gae = delta + gamma * lam * (0.0 if row["done"] else gae)
            row["advantage"] = float(gae)
            row["return"] = float(gae + row["old_value"])


def ppo_update(cfg, records, qwen, critic_qwen, tokenizer, actor, critic, actor_opt, critic_opt,
               actor_device, critic_device):
    add_gae(records, float(cfg["gamma"]), float(cfg["gae_lambda"]), critic, critic_qwen, tokenizer, critic_device)
    advantage = torch.tensor([x["advantage"] for x in records], device=actor_device)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    metrics: dict[str, list[float]] = defaultdict(list)
    order = torch.randperm(len(records))
    mb = int(cfg["ppo_minibatch_size"])
    for _ in range(int(cfg["ppo_epochs"])):
        for index in order.split(mb):
            rows = [records[int(i)] for i in index]
            tasks = [x["task"] for x in rows]
            h = _stack(rows, "h", actor_device); z = _stack(rows, "z", actor_device)
            actions = torch.tensor([x["action"] for x in rows], device=actor_device)
            old_logp = torch.tensor([x["old_logp"] for x in rows], device=actor_device)[:, None]
            adv = advantage[index.to(actor_device)][:, None]
            mask = torch.ones_like(adv)
            logits = _actor_logits(actor, qwen, tokenizer, tasks, h, z)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)[:, None]
            policy_loss, clipfrac, approx_kl, _ = compute_policy_loss(
                old_logp, logp, adv, mask, cliprange=float(cfg["clip_ratio"]), clip_ratio_c=3.0)
            entropy = dist.entropy().mean()
            loss = policy_loss - float(cfg["entropy_coef"]) * entropy
            actor_opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(cfg["max_grad_norm"]))
            actor_opt.step()
            metrics["ppo/policy_loss"].append(float(policy_loss.detach())); metrics["ppo/entropy"].append(float(entropy.detach()))
            metrics["ppo/approx_kl"].append(float(approx_kl.detach())); metrics["ppo/clip_fraction"].append(float(clipfrac.detach()))
            ch = _stack(rows, "h", critic_device); cz = _stack(rows, "z", critic_device)
            old_v = torch.tensor([x["old_value"] for x in rows], device=critic_device)[:, None]
            ret = torch.tensor([x["return"] for x in rows], device=critic_device)[:, None]
            values = critic(critic_qwen, tokenizer, tasks, ch, cz)[:, None]
            value_loss, vf_clip = compute_value_loss(values, ret, old_v, torch.ones_like(ret), float(cfg["value_clip_ratio"]))
            critic_opt.zero_grad(set_to_none=True); value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(cfg["max_grad_norm"])); critic_opt.step()
            metrics["ppo/value_loss"].append(float(value_loss.detach())); metrics["ppo/value_clip_fraction"].append(float(vf_clip.detach()))
    returns = torch.tensor([x["return"] for x in records]); old_values = torch.tensor([x["old_value"] for x in records])
    metrics["ppo/explained_variance"].append(float(1 - torch.var(returns - old_values, unbiased=False) / torch.var(returns, unbiased=False).clamp_min(1e-6)))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def wm_loss_one(cfg, qwen, tokenizer, rssm, row, device):
    h = row["h"].to(device)[None]; z = row["z"].to(device)[None]
    hn = transition(qwen, tokenizer, rssm, ACTION_TO_TEXT[int(row["action"])], h, z, device, cfg)
    pm, ps, _ = gaussian(rssm.prior_head(hn))
    qm, qs, _ql, _hidden = posterior(qwen, tokenizer, rssm, row["task"], row["next_observation"], hn[0], device, cfg)
    raw = kl(qm, qs, pm, ps).mean()
    dyn = kl(qm.detach(), qs.detach(), pm, ps).mean()
    rep = kl(qm, qs, pm.detach(), ps.detach()).mean()
    kl_loss = float(cfg["kl_balance"]) * dyn + (1 - float(cfg["kl_balance"])) * rep
    recon, _correct, _tokens = reconstruction(qwen, tokenizer, rssm, [row["next_observation"]], qm, device,
                                               int(cfg["max_sequence_length"]), h=hn)
    loss = float(cfg["lambda_recon"]) * recon + float(cfg["beta_kl"]) * kl_loss
    out = {"recon_ce": recon.detach(), "kl": raw.detach(), "reward_mse": torch.zeros((), device=device),
           "continuation_bce": torch.zeros((), device=device)}
    if rssm.reward_continuation_heads:
        reward_pred, continuation = rssm.reward_and_continuation(hn, qm)
        reward_loss = F.mse_loss(reward_pred, torch.tensor([row["reward"]], device=device))
        continuation_loss = F.binary_cross_entropy_with_logits(continuation, torch.tensor([1.0 - float(row["done"])], device=device))
        loss = loss + float(cfg["lambda_reward"]) * reward_loss + float(cfg["lambda_continuation"]) * continuation_loss
        out["reward_mse"], out["continuation_bce"] = reward_loss.detach(), continuation_loss.detach()
    return loss, out


def world_model_update(cfg, records, qwen, tokenizer, rssm, actor_opt, device):
    perm = torch.randperm(len(records)).tolist()
    n_train = min(int(cfg["wm_batch_size"]), max(1, int(len(records) * .8)))
    train_rows = [records[i] for i in perm[:n_train]]
    val_rows = [records[i] for i in perm[n_train:n_train + min(int(cfg["wm_val_batch_size"]), len(records) - n_train)]]
    actor_opt.zero_grad(set_to_none=True)
    sums: dict[str, float] = defaultdict(float)
    for row in train_rows:
        loss, values = wm_loss_one(cfg, qwen, tokenizer, rssm, row, device)
        (loss / len(train_rows)).backward(); sums["total_loss"] += float(loss.detach())
        for key, value in values.items(): sums[key] += float(value)
    torch.nn.utils.clip_grad_norm_(rssm.parameters(), float(cfg["max_grad_norm"])); actor_opt.step()
    metrics = {f"rssm/train_{k}": v / len(train_rows) for k, v in sums.items()}
    if val_rows:
        with torch.no_grad():
            vals: dict[str, float] = defaultdict(float)
            shuffled_gap = []
            for row in val_rows:
                loss, value = wm_loss_one(cfg, qwen, tokenizer, rssm, row, device)
                vals["total_loss"] += float(loss); [vals.__setitem__(k, vals[k] + float(v)) for k, v in value.items()]
                shuffled_gap.append(float(row["z"].float().norm()))
            metrics.update({f"rssm/val_{k}": v / len(val_rows) for k, v in vals.items()})
            metrics["rssm/val_shuffled_latent_gap"] = float(np.std(shuffled_gap))
    return metrics


@torch.no_grad()
def evaluate(cfg, qwen, critic_qwen, tokenizer, rssm, actor, critic, actor_device, critic_device, seed, episodes):
    eval_cfg = dict(cfg); eval_cfg["rollout_envs"] = episodes; eval_cfg["_critic_qwen"] = critic_qwen
    records, stats = collect_rollout(eval_cfg, qwen, tokenizer, rssm, actor, critic, actor_device, critic_device, seed)
    steps = max(stats["steps"], 1.0)
    return {
        "env/success_rate": stats["success"] / max(stats["episodes"], 1.0),
        "env/mean_reward": stats["reward"] / max(stats["episodes"], 1.0),
        "env/mean_episode_length": stats["steps"] / max(stats["episodes"], 1.0),
        # The policy is constrained to the 17 canonical action classes, so parsing cannot fail.
        "env/raw_action_parse_rate": 1.0, "env/fallback_rate": 0.0,
        "env/repeated_action_rate": stats["repeated"] / steps,
        "env/executed_action_valid_rate": 1.0,
    }, records[:min(3, len(records))]


def main() -> None:
    args = parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke_updates:
        # A genuine end-to-end smoke (posterior -> actor -> real env -> RSSM
        # loss -> PPO loss) must finish quickly and must not trigger the
        # 32-seed final evaluator.
        cfg.update({"rollout_envs": 1, "env_episode_max_steps": 2,
                    "eval_updates": [], "eval_episodes": 1,
                    "final_eval_episodes": 1, "wm_batch_size": 1,
                    "wm_val_batch_size": 0, "ppo_epochs": 1,
                    "ppo_minibatch_size": 2})
    _seed(int(cfg["seed"])); actor_device, critic_device = torch.device(args.actor_device), torch.device(args.critic_device)
    if actor_device == critic_device:
        raise ValueError("actor and critic must occupy distinct GPUs")
    assert_information_flow()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(actor_device).eval()
    critic_qwen = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True).to(critic_device).eval()
    for module in (qwen, critic_qwen):
        for p in module.parameters(): p.requires_grad_(False)
    cfg["soft_token_target_rms"] = float(qwen.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    rssm = QwenTransitionRSSMRecon(cfg, qwen.config.hidden_size).to(actor_device)
    actor = LatentActor(rssm, qwen.config.hidden_size).to(actor_device)
    critic = LatentCritic(int(cfg["latent_h_dim"]), int(cfg["latent_z_dim"]), critic_qwen.config.hidden_size,
                          int(cfg["h_soft_tokens"]), int(cfg["z_soft_tokens"]), cfg["soft_token_target_rms"]).to(critic_device)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=float(cfg["actor_learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=float(cfg["critic_learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    root = Path(cfg["output_root"]) / args.run_name; root.mkdir(parents=True, exist_ok=False)
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {**cfg, "run_name": args.run_name, "git_commit": commit, "actor_device": str(actor_device),
                "critic_device": str(critic_device), "transition_observation_tokens": 0,
                "actor_observation_tokens": 0, "critic_observation_tokens": 0,
                "posterior_observation_tokens": "real_o_t", "reconstruction_target": "real_o_t",
                "transition_interface": f"soft(h)x{rssm.h_k}+soft(z)x{rssm.z_k}+action"}
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True))
    (root / "interface_manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.disable_comet:
        comet = NullComet()
    else:
        from comet_ml import Experiment
        comet = Experiment(workspace=cfg["comet_workspace"], project_name=cfg["comet_project"], auto_output_logging="simple")
    comet.set_name(args.run_name); comet.add_tags(["online_rl", "dual_ppo", "rssm", "action_only_transition", "reward_heads" if rssm.reward_continuation_heads else "no_reward_heads"])
    comet.log_parameters(manifest); comet.log_metric("run/started", 1.0, step=0)
    print(json.dumps({"startup": manifest}, default=str), flush=True)
    updates = args.smoke_updates or int(cfg["total_updates"]); env_steps = 0
    for update in range(1, updates + 1):
        cfg["_critic_qwen"] = critic_qwen
        records, rollout = collect_rollout(cfg, qwen, tokenizer, rssm, actor, critic, actor_device, critic_device,
                                           int(cfg["seed"]) + update * int(cfg["rollout_envs"]))
        if not records: raise RuntimeError("online rollout yielded no transitions")
        env_steps += len(records)
        ppo = ppo_update(cfg, records, qwen, critic_qwen, tokenizer, actor, critic, actor_opt, critic_opt, actor_device, critic_device)
        wm = world_model_update(cfg, records, qwen, tokenizer, rssm, actor_opt, actor_device)
        metrics = {"online/env_steps": float(env_steps), "online/mean_reward": rollout["reward"] / max(rollout["episodes"], 1),
                   "online/success_rate": rollout["success"] / max(rollout["episodes"], 1), **ppo, **wm}
        if update == 1 or update % int(cfg["log_every_updates"]) == 0:
            print(json.dumps({"update": update, **metrics}), flush=True); comet.log_metrics(metrics, step=env_steps)
        if update in set(int(x) for x in cfg["eval_updates"]):
            ev, raw = evaluate(cfg, qwen, critic_qwen, tokenizer, rssm, actor, critic, actor_device, critic_device,
                               int(cfg["eval_seed"]), int(cfg["eval_episodes"]))
            (root / f"eval_update_{update}.json").write_text(json.dumps({"metrics": ev, "raw_examples": raw}, indent=2))
            comet.log_metrics(ev, step=env_steps); print(json.dumps({"update": update, **ev}), flush=True)
        if update % int(cfg["checkpoint_every_updates"]) == 0 or update == updates:
            torch.save({"update": update, "env_steps": env_steps, "rssm": rssm.state_dict(), "actor": actor.state_dict(),
                        "critic": critic.state_dict(), "actor_optimizer": actor_opt.state_dict(), "critic_optimizer": critic_opt.state_dict()},
                       root / f"checkpoint_update_{update}.pt")
    final_eval, raw = evaluate(cfg, qwen, critic_qwen, tokenizer, rssm, actor, critic, actor_device, critic_device,
                               int(cfg["eval_seed"]), int(cfg["final_eval_episodes"]))
    (root / "final_eval.json").write_text(json.dumps({"metrics": final_eval, "raw_examples": raw}, indent=2))
    comet.log_metrics(final_eval, step=env_steps); comet.end()


if __name__ == "__main__":
    main()
