#!/usr/bin/env python3
"""GRPO of the SLM plan policy inside the Dreamer 4 world model, from dataset states only (no environment).

Hypothesis H-WM-GRPO (user, 2026-10-28) and the 2026-11-19 diagnosis: every failed episode of the teacher (344 of 2165) and
almost every failed closed-loop episode of the SLM ends in death, and behaviour cloning cannot beat the teacher's ~84%; the
world model has to tell the policy which plans die. Here the world model is a Dreamer 4 phase-2 checkpoint trained on reward
v2 (build_wood_d4_heads.py death: +1 one frame before the task is done, -1 one frame before death; |r| >= 0.5 ends the
episode), with a value head (gamma 0.99) and the task as text. Since 2026-11-19 (user: the environment's reward must not be
changed) the heads learn the native reward (build_wood_d4_heads.py native, Craftax / 50) and the end of the episode from a
continue head (patch_dreamer4_heads_continue.py); p_k below is then that head, and death / success in the logs are terminals
with negative / positive arriving reward (the killing hit costs health, the finishing step pays its achievement).

One update:
  starts   B frames t of training episodes (uniform, or --death-frac of them from the last --death-window frames of episodes
           that ended in death): Z_t = z_{t-3..t}, task g, the actions a_{t-3..t-1} that were taken
  group    G answers of the SLM at --temperature: "<reasoning>..</reasoning><plan>p_1, .., p_K</plan>"
  dream    d_1..d_H = Dreamer dynamics from context z_{t-2..t} under (a_{t-2}, a_{t-1}, p_1..p_H) (windowed, K denoising steps;
           Noop after a short plan)
  score    R = sum_{k<=L} gamma^(k-1) A_k r_k + gamma^L A_{L+1} V_L, r_k / p_k expected reward / terminal probability of
           the reward head at d_k, A_k = prod_{j<k} (1 - p_j), V_L the Dreamer value head at d_L, L = min(plan length, H);
           an unparsable answer scores --bad-score
  update   A_i = (R_i - mean_G R) / (std_G R + 1e-4);  L = - mean_i A_i * mean_tokens log pi(answer_i | Z_t, g)
           + --bc-weight * CE(teacher reasoning + next K teacher actions | Z_t, g) on labelled starts
           (the BC anchor keeps the policy near the demonstrations it started from)
  PPO / KL (user 2026-11-19; --clip, --kl-coef, --ppo-epochs, --minibatches): log pi_old of the sampled answers is taken once,
           then --ppo-epochs passes of --minibatches steps over the group batch with, per token,
             rho = pi_theta / pi_old,  surrogate = min(rho A_i, clip(rho, 1 - eps, 1 + eps) A_i)
             L = - mean_i mean_tok surrogate + beta mean_i mean_tok [pi_ref/pi_theta - log(pi_ref/pi_theta) - 1] + bc CE
           (the k3 KL estimator of GRPO / DeepSeekMath), pi_ref = --ref (frozen; the BC SLM in the Dyna loop). With the
           defaults (1 pass, 1 step, eps = beta = 0) this is the plain update above (rho = 1, same gradient).
Online Dyna loop (dyna_loop.sh, user 2026-11-19): --online <traj dirs> makes --online-frac of the starts the states the
planner itself reached in the environment (eval_slm_d4_env.py --save-traj; --death-frac of them from the last
--death-window frames of its failed episodes), the rest stay dataset starts with the BC anchor; --critic fixes the value
SLM (the first SLM of the loop) while the policy moves from iteration to iteration.
The SLM keeps its reconstruction / value heads untouched (not in the loss). Checkpoints in the train_slm_reason_act_d4.py
format (eval_slm_d4_env.py reads them). Every --eval-every updates: on held-out validation starts, mean score, death share
(sum_k A_k p_k with r_k < 0) and success share of the greedy plan, and the same for the start checkpoint at update 0.

--frame-labels DIR (Qwen-teacher Crafter data of Alexey, 2026-11-19, the SLM of train_slm_reason_act_d4.py --frame-labels):
dataset starts are the frames of --train / --val (build_qwen_crafter64.py dirs: actions.npy, terminal_in.npy, manifest.json),
the BC target of a start is the teacher Qwen's reasoning (DIR/{train,val}.jsonl of build_qwen_slm_labels.py) + its next K
actions, and "near death" are the last --death-window frames of the episodes a death ended; --raw-data / --reasoning unused.
The rows are built lazily (3M frames). Answers are decoded up to the SLM's --gen-tokens (Qwen reasoning is longer).

ZeroHero recipe (user 2026-11-21: their critic, and the planner learning from real steps while it keeps planning in Dreamer):
  --critic an HL-Gauss critic (hl_critic.py; stage 1 train_slm_reason_act_d4.py --critic-from, stage 2
           finetune_slm_critic_real.py on the real returns of the planner): its mean V is the tail value of every dream.
  --ratio seq  the GSPO ratio of a whole answer, rho_i = exp(mean_tok(log pi_theta - log pi_old)) (their length-normalised
           response ratio; the per-token product of ~100 ratios explodes), clipped at --clip (theirs 0.02);
  --target-kl  once the k3 estimate mean(rho - 1 - log rho) of a minibatch exceeds it (theirs 5e-4), the rest of the batch
           makes no policy step.
  --real-weight w  the real-step term: every played frame t of the --online episodes carries the plan the Dreamer planner
           chose there (plans.json of eval_slm_d4_env.py --save-traj) and its real advantage A_t = G^lambda_t - V(x_t)
           (hl_critic.online_targets with the --critic; real rewards, real deaths); the loss adds
              w * mean_i  min(exp(A_i / (beta std A)), --real-wmax) * CE(chosen plan_i | Z_i, g)      (AWR, Peng et al. 2019)
           i.e. the proposal SLM is distilled toward the planner's choices that turned out well in the real game and away
           from the ones that did not (weights below 1), while GRPO in imagination keeps training it on dataset and online
           starts.

  python train_slm_imagine_grpo.py --slm <run>/run/checkpoints/step_060000.pt --dyn <phase-1 dyn> --heads <phase-2 latest.pt>
     --raw-data <wood2200 raw> --reasoning <dir> --train <wood2200_v3> --val <wood2200_d4/val> --latent-cache <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    for k in ("--slm", "--dyn", "--heads", "--train", "--val", "--latent-cache", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--raw-data", default="")
    p.add_argument("--reasoning", default="")
    p.add_argument("--frame-labels", default="", help="dir with {train,val}.jsonl of build_qwen_slm_labels.py (Qwen Crafter data)")
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--starts", type=int, default=32)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bc-weight", type=float, default=0.1)
    p.add_argument("--bad-score", type=float, default=-1.0)
    p.add_argument("--death-frac", type=float, default=0.5, help="share of starts taken near the end of death episodes")
    p.add_argument("--death-window", type=int, default=30)
    p.add_argument("--tail", choices=("dreamer", "slm"), default="dreamer",
                   help="value of the last imagined state: Dreamer's value head, or the frozen start SLM's value (with a "
                        "continue head also at every imagined end: the native reward does not pay for the task)")
    p.add_argument("--critic", default="", help="--tail slm: SLM checkpoint whose value is the critic (default: a frozen copy "
                   "of --slm); the Dyna loop keeps the first SLM's value through all iterations")
    p.add_argument("--online", default="", help="comma-separated eval_slm_d4_env.py --save-traj dirs: --online-frac of the "
                   "starts are the states the planner reached in the environment (no BC label on them)")
    p.add_argument("--online-frac", type=float, default=0.75)
    p.add_argument("--reencode", choices=("", "online", "all"), default="", help="encode frames with the tokenizer of --dyn "
                   "(cached in <out>/latents) instead of the traj dirs' z.pt (online) and also --latent-cache (all): the Dyna loop "
                   "with tokenizer finetuning; 'online' for the 3M-frame Qwen data, whose latents the anchor keeps")
    p.add_argument("--clip", type=float, default=0.0, help="PPO clip epsilon of the ratio pi/pi_old (0: off)")
    p.add_argument("--ratio", choices=("token", "seq"), default="token", help="per-token ratio, or GSPO length-normalised "
                   "ratio of the whole answer (ZeroHero)")
    p.add_argument("--target-kl", type=float, default=0.0, help="stop the batch's policy steps once the k3 KL estimate of the "
                   "ratio exceeds this (0: off; ZeroHero 5e-4)")
    p.add_argument("--real-weight", type=float, default=0.0, help="weight of the AWR term on the planner's real chosen plans")
    p.add_argument("--real-beta", type=float, default=1.0, help="AWR temperature, in units of the std of the real advantages")
    p.add_argument("--real-wmax", type=float, default=20.0)
    p.add_argument("--real-batch", type=int, default=32)
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda of the real returns")
    p.add_argument("--kl-coef", type=float, default=0.0, help="weight of the per-token k3 KL(pi || pi_ref) (0: off)")
    p.add_argument("--ref", default="", help="reference SLM checkpoint of the KL (default --slm; the Dyna loop: the BC SLM)")
    p.add_argument("--ppo-epochs", type=int, default=1, help="passes over each sampled group batch")
    p.add_argument("--minibatches", type=int, default=1, help="gradient steps per pass")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-starts", type=int, default=512)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=7950)
    p.add_argument("--diag-out", default="", help="diagnostic only: sample GRPO groups on held-out starts, write their raw "
                   "scores, advantages and plan diversity at every --diag-horizons to this json, no training")
    p.add_argument("--diag-starts", type=int, default=256)
    p.add_argument("--diag-horizons", default="8,16")
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from train_slm_reason_act_d4 import EOS, NAMES, SYSTEM_PLAN, SLMReasonAct, parse_plan
    from train_slm_recon_d4 import load_split, windows
    from train_slm_world_model import load_episodes
    from dreamer4.dynamics_eval import windowed_rollout
    from dreamer4.models.agent_heads import build_agent_heads, terminal_bins
    from dreamer4.models.dynamics import _log2_int
    from dreamer4.models.world_model import WorldModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    device = torch.device("cuda")
    out = Path(a.out); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    # ---------------- SLM ----------------
    base = torch.load(a.slm, map_location="cpu", weights_only=False)
    cfg = base["args"]; K = cfg["plan"]; tasks = base["tasks"]
    assert K, "needs a --plan checkpoint"
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    n_h = int(base["model"]["h_slot"].shape[0]) if "h_slot" in base["model"] else 0
    assert n_h == 0, "this script trains SLMs that read z only"
    model = SLMReasonAct(lm, cfg["readout"]).to(device)
    model.load_state_dict(base["model"])
    pre = torch.tensor([enc("System:\n" + SYSTEM_PLAN.format(K) + "\nUser:\n")], device=device)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=device); post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    max_tokens = cfg.get("gen_tokens") or 80 + 6 * K
    params = [q for n, q in model.named_parameters() if not n.startswith(("head.", "value_head."))]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    critic = copy.deepcopy(model).eval().requires_grad_(False) if a.tail == "slm" else None
    if critic is not None and a.critic:
        cst = torch.load(a.critic, map_location="cpu", weights_only=False)
        assert cst["tasks"] == tasks, "critic and policy SLMs have different task lists"
        from hl_critic import attach_for
        attach_for(critic, cst).load_state_dict(cst["model"]); critic.eval().requires_grad_(False)
        print("critic", a.critic, flush=True)
    ref = None
    if a.kl_coef > 0:   # frozen reference policy of the KL term
        ref = copy.deepcopy(model).eval().requires_grad_(False)
        if a.ref:
            rst = torch.load(a.ref, map_location="cpu", weights_only=False)
            ref.load_state_dict(rst["model"]); ref.eval().requires_grad_(False)
        print("KL reference", a.ref or a.slm, "coef", a.kl_coef, "clip", a.clip, "epochs", a.ppo_epochs, "minibatches", a.minibatches,
              flush=True)

    @torch.no_grad()
    def critic_value(W, g):
        """SLM value V(g, W) of the frozen start checkpoint (G = gamma^(T-1-t) on success, 0 on failure)."""
        emb, mask, _ = critic.prefix(W, pre, post_ids[g], post_mask[g])
        h = critic.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, -1]
        return critic.value_head(h).squeeze(-1).float()

    # ---------------- world model ----------------
    hck = torch.load(a.heads, map_location="cpu", weights_only=False)
    world = WorldModel(a.dyn, device=device, n_agent=int(hck["n_agent"]))
    world.load_dynamics_weights(hck["dyn"])
    dyn = world.dyn.eval()

    @torch.no_grad()
    def latents(root, tag, fallback):
        """z of every frame of a dataset / traj dir: the cache (fallback path), or with --reencode the --dyn tokenizer."""
        if not a.reencode or (a.reencode == "online" and not tag.startswith("online_")):
            return torch.load(fallback)
        path = out / "latents" / f"z_{tag}.pt"
        if path.exists():
            return torch.load(path)
        frames, st, ln = load_split(root)
        zs = []
        for s0, n in zip(st, ln):
            x = torch.from_numpy(np.ascontiguousarray(frames[s0:s0 + n, :63, :63]))
            zs.append(torch.cat([world.tok.encode_frames(x[None, max(0, c - 3):c + 256].to(device))[0, min(c, 3):]
                                 for c in range(0, n, 256)]).half().cpu())
        z = torch.cat(zs)
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(z, path)
        print(f"re-encoded {root} ({len(z)} frames) with the tokenizer of --dyn", flush=True)
        return z
    heads = build_agent_heads(hck["agent_cfg"]).to(device).eval()
    heads.load_state_dict(hck["agent"])
    term_bin = terminal_bins(heads.reward, world.continues_from_reward, device)
    head_task = ({g: i for i, g in enumerate(world.dataset.manifest["task_names"])} if "task_names" in world.dataset.manifest
                 else {g: 0 for g in tasks})   # Qwen Crafter data: one task, no names
    k_max, tau = world.k_max, world.tau_ctx
    window = int(world.window); ctx = window - 1
    sig = min(int(round((1.0 - tau) * k_max)), k_max)
    H = a.horizon

    # ---------------- data ----------------
    episodes = None if a.frame_labels else load_episodes(Path(a.raw_data), None, None)
    reasoning = {}
    for split in (() if a.frame_labels else ("train", "valid")):
        for line in open(Path(a.reasoning) / f"{split}.jsonl"):
            r = json.loads(line)
            m = re.match(r"<reasoning>(.*)</reasoning>\nACTION: (\S+)$", r["cot_target"], re.S)
            if m:
                reasoning[(int(r["source_episode_id"]), int(r["step"]))] = m.group(1).strip()

    def split(root, tag):
        frames, starts, lengths = load_split(root)
        z = latents(root, tag, Path(a.latent_cache) / f"z_{tag}.pt").to(device)
        man = json.loads(((Path(root) / "manifest.json") if (Path(root) / "manifest.json").exists()
                          else (Path(root) / "wood_manifest.json")).read_text())
        rows = []          # (global frame, task, a_{t-3..t-1}, BC target ids or None, near death)
        first = 0
        for key, n in zip(man["episode_ids"], man["lengths"]):
            ep = episodes[int(key)]
            died = (not ep[-1]["success"]) and bool(ep[-1]["done"]) and n < 1000
            moves = [it["action"] for it in ep]
            for t in range(n):
                lab = reasoning.get((int(key), t))
                tgt = None
                if lab is not None:
                    plan = ", ".join(NAMES[x] for x in moves[t:t + K])
                    tgt = enc(f"<reasoning>\n{lab}\n</reasoning>\n<plan>{plan}</plan>") + [EOS]
                prev = [moves[t - 3 + i] if t - 3 + i >= 0 else 0 for i in range(3)]
                rows.append((first + t, tasks.index(ep[0]["instruction"]), prev, tgt, died and t >= n - a.death_window, t))
            first += n
        ep_first = torch.tensor([r[0] - r[5] for r in rows], device=device)
        return z, ep_first, rows

    class FrameRows:   # --frame-labels: rows (frame, task, a_{t-3..t-1}, BC target ids or None, near death, t), built on access
        def __init__(self, root):
            root = Path(root)
            man = json.loads((root / "manifest.json").read_text())
            self.acts = np.load(root / "actions.npy").astype(np.int64)
            died = np.load(root / "terminal_in.npy")
            N = len(self.acts)
            self.t = np.zeros(N, np.int64); self.end = np.zeros(N, np.int64); near = np.zeros(N, bool)
            for st, n in zip(man["starts"], man["lengths"]):
                self.t[st:st + n] = np.arange(n); self.end[st:st + n] = st + n
                if died[st + n - 1]:
                    near[st + max(0, n - a.death_window):st + n] = True
            self.near = np.flatnonzero(near).tolist()
            self.lab = {}
            for line in open(Path(a.frame_labels) / f"{root.name}.jsonl"):
                r = json.loads(line); self.lab[r["row"]] = r["reasoning"]

        def __len__(self):
            return len(self.acts)

        def __getitem__(self, i):
            i = int(i); t = int(self.t[i])
            prev = [int(self.acts[i - 3 + j]) if t - 3 + j >= 0 else 0 for j in range(3)]
            tgt = None
            if i in self.lab:
                plan = ", ".join(NAMES[x] for x in self.acts[i:min(i + K, int(self.end[i]) - 1)])
                tgt = enc(f"<reasoning>\n{self.lab[i]}\n</reasoning>\n<plan>{plan}</plan>") + [EOS]
                if len(tgt) > cfg.get("max_target", 10 ** 9):
                    tgt = None
            return (i, 0, prev, tgt, False, t)

    def frame_split(root, tag):
        rows = FrameRows(root)
        z = latents(root, tag, Path(a.latent_cache) / f"z_{tag}.pt").to(device)
        assert len(z) == len(rows), (len(z), len(rows))
        return z, torch.from_numpy(np.arange(len(rows)) - rows.t).to(device), rows

    if a.frame_labels:
        (zT, fT, rowsT), (zV, fV, rowsV) = frame_split(a.train, "train"), frame_split(a.val, "val")
    else:
        zT, fT, rowsT = split(a.train, "train")
        zV, fV, rowsV = split(a.val, "val")
    near = rowsT.near if a.frame_labels else [i for i, r in enumerate(rowsT) if r[4]]
    print(f"train starts {len(rowsT)} ({len(near)} near death), val {len(rowsV)}", flush=True)
    rowsO, nearO, metaO = [], [], []
    if a.online:   # states the planner reached in the environment: (frame, task, a_{t-3..t-1}, no BC label, near death, t)
        zs_, f0 = [], 0
        for d in a.online.split(","):
            d = Path(d)
            m = json.loads((d / "wood_manifest.json").read_text())
            eps = json.loads((d / "episodes.json").read_text())
            acts = np.load(d / "actions.npy")
            z = latents(d, "online_" + "_".join(d.parts[-2:]), d / "z.pt")
            assert len(z) == len(acts) == sum(m["lengths"]) and len(eps) == len(m["lengths"]), d
            for s, n, e in zip(m["starts"], m["lengths"], eps):
                moves = acts[s:s + n].tolist()
                failed = (not e["success"]) and e["end"] != "timeout"
                for t in range(n):
                    prev = [moves[t - 3 + i] if t - 3 + i >= 0 else 0 for i in range(3)]
                    rowsO.append((f0 + s + t, tasks.index(e["task"]), prev, None, failed and t >= n - a.death_window, t))
            pf = d / "plans.json"
            metaO.append(dict(f0=f0, starts=m["starts"], lengths=m["lengths"], eps=eps, rew=np.load(d / "rewards.npy"),
                              plans=json.loads(pf.read_text()) if pf.exists() else None))
            zs_.append(z); f0 += len(z)
        zO = torch.cat(zs_).to(device)
        fO = torch.tensor([r[0] - r[5] for r in rowsO], device=device)
        nearO = [i for i, r in enumerate(rowsO) if r[4]]
        print(f"online starts {len(rowsO)} ({len(nearO)} near a failed end) from {a.online}", flush=True)
    realO = []   # (online frame, ids of the plan the planner chose there, its real advantage)
    if a.real_weight > 0:
        from hl_critic import online_targets
        assert critic is not None and all(r[0] == i for i, r in enumerate(rowsO))
        gall = torch.tensor([r[1] for r in rowsO], device=device)
        vO = torch.cat([critic_value(windows(zO, fO, torch.arange(s, min(s + 512, len(rowsO)), device=device)),
                                     gall[s:s + 512]) for s in range(0, len(rowsO), 512)]).cpu()
        for md in metaO:
            if md["plans"] is None:
                print("no plans.json in an online dir: its frames have no real-step term", flush=True); continue
            n_all = sum(md["lengths"]); f0 = md["f0"]
            G = online_targets(vO[f0:f0 + n_all], md["rew"], md["starts"], md["lengths"], md["eps"],
                               "crafter" if a.frame_labels else "wood", a.gamma, a.lam)
            Adv = G - vO[f0:f0 + n_all]
            for k, txt in enumerate(md["plans"]):
                if txt and parse_plan(txt):
                    realO.append((f0 + k, enc(txt) + [EOS], float(Adv[k])))
        Ar = torch.tensor([x[2] for x in realO])
        sd_A = float(Ar.std()) + 1e-6
        wR = (Ar / (a.real_beta * sd_A)).exp().clamp(max=a.real_wmax)
        print(f"real-step term: {len(realO)} chosen plans, advantage mean {float(Ar.mean()):.4f} std {sd_A:.4f}, "
              f"AWR weight mean {float(wR.mean()):.3f} max {float(wR.max()):.2f}", flush=True)

    def real_loss():
        pick = rng.choice(len(realO), a.real_batch)
        idx = torch.tensor([realO[i][0] for i in pick], device=device)
        tg_ = [realO[i][1] for i in pick]
        Wr = max(map(len, tg_))
        tb_ = torch.zeros(len(tg_), Wr, dtype=torch.long, device=device); mb_ = torch.zeros_like(tb_)
        for i, t_ in enumerate(tg_):
            tb_[i, :len(t_)] = torch.tensor(t_, device=device); mb_[i, :len(t_)] = 1
        gr = gall[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, lgr, _ = model(windows(zO, fO, idx), pre, post_ids[gr], post_mask[gr], tb_, mb_)
        ce = F.cross_entropy(lgr.float().transpose(1, 2), tb_, reduction="none")
        ce = (ce * mb_).sum(1) / mb_.sum(1)
        return (wR[torch.as_tensor(pick)].to(device) * ce).mean()

    def choose(near_rows, n_rows, k):
        nd = int(round(a.death_frac * k)) if near_rows else 0
        return list(rng.choice(near_rows, nd)) + list(rng.choice(n_rows, k - nd)) if k else []
    val_pick = rng.choice(len(rowsV), min(a.eval_starts, len(rowsV)), replace=False)

    # ---------------- helpers ----------------
    @torch.no_grad()
    def sample(Z, g, greedy):
        emb, mask, _ = model.prefix(Z, pre, post_ids[g], post_mask[g])
        o = model.backbone(inputs_embeds=emb, attention_mask=mask, use_cache=True)
        past, h = o.past_key_values, o.last_hidden_state[:, -1]
        B = Z.shape[0]; done = torch.zeros(B, dtype=torch.bool, device=device); written = []
        for _ in range(max_tokens):
            lg = model.lm_head(h).float()
            tk = lg.argmax(-1) if greedy else torch.multinomial((lg / a.temperature).softmax(-1), 1)[:, 0]
            tk = torch.where(done, torch.zeros_like(tk), tk); written.append(tk); done |= tk == EOS
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=device)], 1)
            o = model.backbone(inputs_embeds=model.embed(tk[:, None]).float(), attention_mask=mask, past_key_values=past, use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[:, -1]
        ids = torch.stack(written, 1)
        lens = []
        for row in ids.tolist():
            lens.append(row.index(EOS) + 1 if EOS in row else len(row))
        return ids, lens

    @torch.no_grad()
    def score(Zs, prev, g, plans, H=H):
        """Zs [C,4,64,16], prev [C,3] actions a_{t-3..t-1}, g [C] SLM task ids, plans list of action lists, H the dream
        horizon (--horizon; --diag-out also scores at other horizons) -> score [C], death mass [C], success mass [C]."""
        C = Zs.shape[0]
        L = torch.tensor([min(len(pl), H) for pl in plans], device=device)
        pa = torch.tensor([pl[:H] + [0] * (H - min(len(pl), H)) for pl in plans], device=device)
        acts = torch.cat([prev[:, 1:], pa], 1)                                # a_{t-2}, a_{t-1}, p_1..p_H
        zf = torch.cat([Zs[:, 4 - ctx:], Zs[:, -1:].expand(-1, H, -1, -1)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, _, _ = windowed_rollout(dyn, z_full=zf, actions=F.one_hot(acts, 17).float(), window=window, ctx=ctx, K=a.K,
                                       k_max=k_max, tau_ctx=tau)
        S = torch.cat([Zs, d.float()], 1)                                    # [C, 4 + H]
        into = torch.cat([prev, pa], 1)                                       # action into S[i + 1]
        task = torch.tensor([head_task[tasks[int(x)]] for x in g.tolist()], device=device)
        rk, pk, vk = [], [], []
        for k in range(1, H + 1):
            av = torch.zeros(C, 4, 18, device=device); av[:, 0, 17] = 1.0
            av[:, 1:, :17] = F.one_hot(into[:, k:k + 3], 17).float()
            st = torch.full((C, 4), _log2_int(k_max), device=device, dtype=torch.long)
            sg = torch.full((C, 4), sig, device=device, dtype=torch.long)
            ag = heads.tokens((C, 4), task_ids=task.reshape(C, 1).expand(C, 4), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = dyn(av, st, sg, (1 - tau) * S[:, k:k + 4], agent_tokens=ag)
            hh = o[-1][:, -1].float().flatten(1)
            lr_ = heads.reward.at(hh, 0).float()
            rk.append(heads.reward.mean(lr_))
            pk.append(torch.sigmoid(heads.cont(hh).squeeze(-1).float()) if hasattr(heads, "cont")
                      else (lr_.softmax(-1) * term_bin).sum(-1))   # continue head (native reward) or reward bins
            vk.append(heads.value.mean(heads.value.at(hh, 0).float()))
        rk, pk, vk = torch.stack(rk, 1), torch.stack(pk, 1), torch.stack(vk, 1)
        valid = torch.arange(H, device=device)[None] < L[:, None]
        alive = torch.cumprod(torch.cat([torch.ones_like(pk[:, :1]), 1 - pk], 1), 1)
        disc = a.gamma ** torch.arange(H, device=device, dtype=torch.float32)
        last = (L - 1).clamp_min(0)
        if a.tail == "slm":   # value of the frozen start SLM on the imagined windows S[k..k+3] instead of Dreamer's value
            W = torch.stack([S[:, k:k + 4] for k in range(1, H + 1)], 1).flatten(0, 1)
            vk = torch.cat([critic_value(W[i:i + 512], g.repeat_interleave(H)[i:i + 512]) for i in range(0, W.shape[0], 512)]).view(C, H)
        sc = (disc * alive[:, :H] * rk * valid).sum(1) + a.gamma ** L.float() * alive.gather(1, L[:, None])[:, 0] * vk.gather(1, last[:, None])[:, 0]
        if a.tail == "slm" and hasattr(heads, "cont"):   # an imagined end is worth the SLM value of its frame (~1 success, ~0 death)
            sc = sc + (a.gamma * disc * alive[:, :H] * pk * vk * valid).sum(1)
        dead = (alive[:, :H] * pk * (rk < 0) * valid).sum(1)
        won = (alive[:, :H] * pk * (rk > 0) * valid).sum(1)
        return sc, dead, won

    def batch_of(rows, z, ep_first, pick):
        idx = torch.tensor([rows[i][0] for i in pick], device=device)
        Zs = windows(z, ep_first, idx)
        g = torch.tensor([rows[i][1] for i in pick], device=device)
        prev = torch.tensor([rows[i][2] for i in pick], device=device)
        return Zs, g, prev

    def parse(ids, lens):
        plans, ok = [], []
        for row, n in zip(ids.tolist(), lens):
            pl = [x for x in (parse_plan(text.decode(row[:n])) or []) if x is not None]
            plans.append(pl or [0]); ok.append(bool(pl))
        return plans, ok

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        res = dict(score=0.0, death=0.0, success=0.0, parsed=0.0); n = 0
        for s0 in range(0, len(val_pick), 128):
            pick = val_pick[s0:s0 + 128]
            Zs, g, prev = batch_of(rowsV, zV, fV, pick)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids, lens = sample(Zs, g, greedy=True)
            plans, ok = parse(ids, lens)
            sc, dead, won = score(Zs, prev, g, plans)
            res["score"] += float(sc.sum()); res["death"] += float(dead.sum()); res["success"] += float(won.sum())
            res["parsed"] += sum(ok); n += len(pick)
        model.train()
        return {k: v / n for k, v in res.items()} | {"step": step}

    def save(step, res):
        args = dict(cfg); args.update(grpo=vars(a), grpo_from=a.slm)
        torch.save(dict(model=model.state_dict(), step=step, args=args, val=res, tasks=tasks), out / "checkpoints" / f"step_{step:06d}.pt")

    if a.diag_out:   # how different are the G answers of a group, and how much of their score difference is real
        G, Hs = a.group, [int(x) for x in a.diag_horizons.split(",")]
        pick_all = val_pick[:a.diag_starts]
        Rs = {h: [] for h in Hs}; okm, plan_rows = [], []
        model.eval()
        for s0 in range(0, len(pick_all), 32):
            Zs, g, prev = batch_of(rowsV, zV, fV, pick_all[s0:s0 + 32])
            ZG, gG, pG = Zs.repeat_interleave(G, 0), g.repeat_interleave(G, 0), prev.repeat_interleave(G, 0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ids, lens = sample(ZG, gG, greedy=False)
            plans, ok = parse(ids, lens)
            for h in Hs:
                sc, _, _ = score(ZG, pG, gG, plans, H=h)
                sc = torch.where(torch.tensor(ok, device=device), sc, torch.full_like(sc, a.bad_score))
                Rs[h] += sc.view(-1, G).cpu().tolist()
            okm += torch.tensor(ok).view(-1, G).tolist(); plan_rows += [plans[i:i + G] for i in range(0, len(plans), G)]
            print(f"diag {s0 + len(Zs)}/{len(pick_all)} starts", flush=True)
        okm = np.array(okm, bool)
        res = dict(slm=a.slm, heads=a.heads, starts=len(okm), group=G, temperature=a.temperature, parsed=float(okm.mean()),
                   distinct_first_action=float(np.mean([len({p[0] for p in ps}) for ps in plan_rows])),
                   distinct_first8=float(np.mean([len({tuple(p[:8]) for p in ps}) for ps in plan_rows])),
                   distinct_plan16=float(np.mean([len({tuple(p) for p in ps}) for ps in plan_rows])), horizons={})
        for h in Hs:
            R = np.array(Rs[h]); full = okm.all(1)   # groups with all G answers parsed: their spread is the plans' own
            sd, rng_ = R.std(1, ddof=1), R.max(1) - R.min(1)
            adv = (R - R.mean(1, keepdims=True)) / (sd[:, None] + 1e-4)
            res["horizons"][h] = dict(
                score_mean=float(R[okm].mean()), group_std_all=sd.tolist(), group_std_parsed_groups=sd[full].tolist(),
                group_range_parsed_groups=rng_[full].tolist(),
                share_std_below={str(t): float((sd[full] < t).mean()) for t in (0.001, 0.01, 0.03, 0.1)},
                adv_abs_mean=float(np.abs(adv).mean()), R=R.tolist())
            print(f"H={h}", json.dumps({k: v for k, v in res["horizons"][h].items() if not isinstance(v, list)}), flush=True)
        if len(Hs) > 1:   # does the longer dream pick the same best plan?
            b = [np.array(Rs[h]).argmax(1) for h in Hs]
            res["same_best_plan"] = float(np.mean(b[0] == b[-1]))
        print("diversity", {k: res[k] for k in ("parsed", "distinct_first_action", "distinct_first8", "distinct_plan16")},
              "same best", res.get("same_best_plan"), flush=True)
        Path(a.diag_out).write_text(json.dumps(res))
        return

    log = open(out / "metrics.jsonl", "a")
    res = evaluate(0); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res), flush=True)
    model.train()
    t0 = time.time(); run = dict(score=0.0, adv=0.0, pg=0.0, bc=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, real=0.0, kl_old=0.0, stopped=0.0, death=0.0, parsed=0.0, n=0)
    for step in range(1, a.updates + 1):
        nO = int(round(a.online_frac * a.starts)) if rowsO else 0
        pick = choose(near, len(rowsT), a.starts - nO)        # dataset starts first: the BC rows index into them
        Zs, g, prev = batch_of(rowsT, zT, fT, pick)
        if nO:
            Zo, go, po = batch_of(rowsO, zO, fO, choose(nearO, len(rowsO), nO))
            Zs, g, prev = torch.cat([Zs, Zo]), torch.cat([g, go]), torch.cat([prev, po])
        G = a.group
        ZG, gG, pG = Zs.repeat_interleave(G, 0), g.repeat_interleave(G, 0), prev.repeat_interleave(G, 0)
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids, lens = sample(ZG, gG, greedy=False)
        model.train()
        plans, ok = parse(ids, lens)
        sc, dead, _ = score(ZG, pG, gG, plans)
        sc = torch.where(torch.tensor(ok, device=device), sc, torch.full_like(sc, a.bad_score))
        R = sc.view(-1, G)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).flatten()
        W = max(lens)
        tg = ids[:, :W]
        tm = (torch.arange(W, device=device)[None] < torch.tensor(lens, device=device)[:, None]).long()

        def token_lp(net, sel):   # log pi(y_tok | prefix) of the sampled answers, [len(sel), W]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lg_, _ = net(ZG[sel], pre, post_ids[gG[sel]], post_mask[gG[sel]], tg[sel], tm[sel])
            return -F.cross_entropy(lg_.float().transpose(1, 2), tg[sel], reduction="none")

        bc_rows = [i for i, j in enumerate(pick) if rowsT[j][3] is not None]
        if a.bc_weight > 0 and bc_rows:
            tgts = [rowsT[pick[i]][3] for i in bc_rows]
            Wb = max(map(len, tgts))
            tb = torch.zeros(len(tgts), Wb, dtype=torch.long, device=device); mb = torch.zeros_like(tb)
            for i, t_ in enumerate(tgts):
                tb[i, :len(t_)] = torch.tensor(t_, device=device); mb[i, :len(t_)] = 1
            sel_bc = torch.tensor(bc_rows, device=device)

        def bc_loss():
            if not (a.bc_weight > 0 and bc_rows):
                return torch.zeros((), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lgb, _ = model(Zs[sel_bc], pre, post_ids[g[sel_bc]], post_mask[g[sel_bc]], tb, mb)
            ce = F.cross_entropy(lgb.float().transpose(1, 2), tb, reduction="none")
            return (ce * mb).sum() / mb.sum()

        n_rows = ZG.shape[0]
        everything = torch.arange(n_rows, device=device)
        with torch.no_grad():   # the sampling policy and the reference, fixed for all the epochs of this batch
            model.eval()
            lp_old = torch.cat([token_lp(model, everything[i:i + 128]) for i in range(0, n_rows, 128)])
            lp_ref = (torch.cat([token_lp(ref, everything[i:i + 128]) for i in range(0, n_rows, 128)])
                      if ref is not None else None)
            model.train()
        st = dict(pg=0.0, bc=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, real=0.0, kl_old=0.0, stopped=0.0); n_steps = 0
        stopped = False
        for _ep in range(a.ppo_epochs):
            if stopped:
                break
            perm = everything[torch.randperm(n_rows, device=device)]
            for mbi in perm.chunk(a.minibatches):
                lp = token_lp(model, mbi)
                m = tm[mbi].float(); A = adv[mbi].detach()[:, None]
                per_seq = lambda x: (x * m).sum(1) / m.sum(1).clamp_min(1)   # mean over the tokens of an answer
                if a.ratio == "seq":   # GSPO: one length-normalised ratio per answer
                    dl = per_seq(lp - lp_old[mbi])
                    ratio = torch.exp(dl)[:, None].expand_as(lp)
                else:
                    dl = per_seq(lp - lp_old[mbi]).detach()
                    ratio = torch.exp(lp - lp_old[mbi])
                kl_old = float((torch.exp(dl.detach()) - 1 - dl.detach()).mean())
                if a.target_kl > 0 and kl_old > a.target_kl:   # ZeroHero early stop: no more policy steps on this batch
                    stopped = True; st["stopped"] = 1.0
                    break
                surr = ratio * A
                if a.clip > 0:   # PPO clipped surrogate
                    surr = torch.minimum(surr, ratio.clamp(1 - a.clip, 1 + a.clip) * A)
                pg = -per_seq(surr).mean()
                loss = pg
                kl = torch.zeros((), device=device)
                if lp_ref is not None:   # k3 estimator of KL(pi_theta || pi_ref) per token (GRPO, DeepSeekMath)
                    dlt = lp_ref[mbi] - lp
                    kl = per_seq(torch.exp(dlt) - dlt - 1).mean()
                    loss = loss + a.kl_coef * kl
                bc = bc_loss()
                loss = loss + a.bc_weight * bc
                rl_ = real_loss() if realO else torch.zeros((), device=device)
                loss = loss + a.real_weight * rl_
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                with torch.no_grad():
                    st["pg"] += float(pg); st["bc"] += float(bc); st["kl"] += float(kl); st["real"] += float(rl_)
                    st["kl_old"] += kl_old
                    st["clipfrac"] += float(per_seq(((ratio - 1).abs() > a.clip).float()).mean()) if a.clip > 0 else 0.0
                    st["ratio"] += float(per_seq(ratio).mean())
                n_steps += 1
        n_steps = max(n_steps, 1)
        pg, bc = st["pg"] / n_steps, st["bc"] / n_steps
        run["score"] += float(sc.mean()); run["adv"] += float(adv.abs().mean()); run["pg"] += pg; run["bc"] += bc
        run["kl"] += st["kl"] / n_steps; run["clipfrac"] += st["clipfrac"] / n_steps; run["ratio"] += st["ratio"] / n_steps
        run["real"] += st["real"] / n_steps; run["kl_old"] += st["kl_old"] / n_steps; run["stopped"] += st["stopped"]
        run["death"] += float(dead.mean()); run["parsed"] += float(np.mean(ok)); run["n"] += 1
        if step % 10 == 0:
            n = run.pop("n")
            print(f"update {step} " + " ".join(f"{k} {v / n:.4f}" for k, v in run.items()) + f" {step / (time.time() - t0):.3f} upd/s",
                  flush=True)
            run = dict(score=0.0, adv=0.0, pg=0.0, bc=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, real=0.0, kl_old=0.0, stopped=0.0, death=0.0, parsed=0.0, n=0)
        if step % a.eval_every == 0 or step == a.updates:
            res = evaluate(step); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res), flush=True)
        if step % a.save_every == 0 or step == a.updates:
            save(step, res)
    print("GRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
