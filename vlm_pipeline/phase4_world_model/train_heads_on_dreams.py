#!/usr/bin/env python3
"""Value-equivalent training of the plan scorer on dreams (user 2026-11-22: "нам общий алгоритм нужен"; experiments of
2026-11-21: the dream draws the right frames, but the reward / continue heads and the critic, trained on latents of REAL frames,
do not see on DREAMED latents which of 8 plans dies or finishes the task -- rank correlation dream vs real 0.26).

As in MuZero / EfficientZero (the model is trained on its own unrolls against real outcomes): from a real frame t of a recorded
episode, the actions that were really taken a_t .. a_{t+H-1} are dreamed open-loop by the frozen Dreamer 4 dynamics,
    z^_{t+1..t+H} = dream(z_{t-2..t}, a_{t-2..t+H-1}),
and the scorer's three parts are trained on the dreamed windows W_k = (S_k..S_{k+3}), S = (z_{t-3..t}, z^_{t+1..t+H}),
against what really happened after those actions:
    reward head   r^(W_k) -> r_{t+k}        (two-hot CE, the reward that arrived with real frame t+k)
    continue head p^(W_k) -> done_{t+k}     (BCE, the real episode ended with that transition)
    critic        V(W_k)  -> G_{t+k}        (HL-Gauss CE or MSE: the value target of real frame t+k)
for k = 1..H while the real episode lasts. Nothing about the environment is used beyond its recorded rewards / ends / value
targets, so the same procedure runs on the BC dataset (Stage 1) and on the planner's own episodes (Stage 2, --online).
The dynamics stay frozen (the scorer learns to read the dreams the planner will see); the planner then scores with the new
heads (--heads <out>/heads.pt) and critic (--critic <out>/critic.pt).

  python train_heads_on_dreams.py --dyn <dyn> --heads <phase-2 heads> --critic <SLM / HL critic .pt> --game wood|crafter
     --data <dataset dir> [--latents <z cache .pt>] [--online <traj>[,<traj>]] [--val <dataset dir> --val-latents <.pt>] --out <dir>
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    for k in ("--dyn", "--heads", "--critic", "--data", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--game", choices=("wood", "crafter"), required=True)
    p.add_argument("--latents", default="", help="z cache of --data (default: encode with the tokenizer of --dyn, cached in --out)")
    p.add_argument("--val", default="")
    p.add_argument("--val-latents", default="")
    p.add_argument("--online", default="", help="traj dirs of eval_slm_d4_env.py --save-traj (Stage 2), --online-frac of the starts")
    p.add_argument("--online-frac", type=float, default=0.5)
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr-heads", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=5e-5)
    p.add_argument("--end-frac", type=float, default=0.25, help="share of starts within --horizon frames of an episode end")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--value-horizon", type=int, default=300)
    p.add_argument("--eval-batches", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from hl_critic import attach_for
    from train_slm_reason_act_d4 import SYSTEM_PLAN, SLMReasonAct
    from train_slm_recon_d4 import load_split
    from dreamer4.dynamics_eval import windowed_rollout
    from dreamer4.models.agent_heads import build_agent_heads
    from dreamer4.models.dynamics import _log2_int
    from dreamer4.models.world_model import WorldModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = a.horizon

    # ---------------- world model + heads (dynamics frozen) ----------------
    hck = torch.load(a.heads, map_location="cpu", weights_only=False)
    world = WorldModel(a.dyn, device=dev, n_agent=int(hck["n_agent"]))
    world.load_dynamics_weights(hck["dyn"])
    dyn = world.dyn.eval().requires_grad_(False)
    agent_cfg = dict(hck["agent_cfg"])
    new_cont = not agent_cfg.get("cont_head")
    if new_cont:   # heads without a continue head (Qwen-data Stage 1): a fresh one, learnt here on the dreams
        agent_cfg["cont_head"] = True
    heads = build_agent_heads(agent_cfg).to(dev)
    missing, unexpected = heads.load_state_dict(hck["agent"], strict=False)
    assert not unexpected and all(k.startswith("cont") for k in missing), (missing, unexpected)
    if new_cont:
        print("new continue head", missing, flush=True)
        hck = dict(hck, agent_cfg=agent_cfg)
    heads0 = copy.deepcopy(heads).eval().requires_grad_(False)
    k_max, tau = world.k_max, world.tau_ctx
    window = int(world.window); ctx = window - 1
    sig = min(int(round((1.0 - tau) * k_max)), k_max)
    head_names = list(world.dataset.manifest.get("task_names", []))

    # ---------------- critic (SLM copy, HL-Gauss or scalar value head) ----------------
    cst = torch.load(a.critic, map_location="cpu", weights_only=False)
    HL = cst.get("critic_hl") is not None
    cfg = cst["args"]; tasks = cst["tasks"]
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    critic = attach_for(SLMReasonAct(lm, cfg["readout"]).to(dev), cst)
    critic.load_state_dict(cst["model"])
    critic0 = copy.deepcopy(critic).eval().requires_grad_(False)
    pre = torch.tensor([enc("System:\n" + SYSTEM_PLAN.format(cfg["plan"]) + "\nUser:\n")], device=dev)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=dev); post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    print("critic", a.critic, "HL-Gauss" if HL else "scalar value head", "| heads", a.heads, flush=True)

    # ---------------- data: per frame z, action, arriving reward / terminal, value target, task ----------------
    @torch.no_grad()
    def encode(root, cache):
        if cache and Path(cache).exists():
            return torch.load(cache)
        frames, st, ln = load_split(root)
        zs = []
        for s0, n in zip(st, ln):
            x = torch.from_numpy(np.ascontiguousarray(frames[s0:s0 + n, :63, :63]))
            zs.append(torch.cat([world.tok.encode_frames(x[None, max(0, c - 3):c + 256].to(dev))[0, min(c, 3):]
                                 for c in range(0, n, 256)]).half().cpu())
        z = torch.cat(zs)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True); torch.save(z, cache)
        return z

    def load_dir(root, zpath, online=False):
        root = Path(root)
        mf = root / "wood_manifest.json" if (root / "wood_manifest.json").exists() else root / "manifest.json"
        man = json.loads(mf.read_text())
        st, ln = man["starts"], man["lengths"]
        N = int(sum(ln))
        act = np.load(root / "actions.npy").astype(np.int64)
        if act.ndim > 1:
            act = act.argmax(-1)
        rew = np.load(root / "rewards.npy").astype(np.float32)
        term_tr = np.zeros(N, np.float32)                     # the transition j -> j+1 ended the episode
        if (root / "terminals.npy").exists():
            term_tr[:] = np.load(root / "terminals.npy").astype(np.float32)[:N]
        else:                                                 # Qwen crafter64: a death in the last frame
            died = np.load(root / "terminal_in.npy")
            for s, n in zip(st, ln):
                if n >= 2 and died[s + n - 1]:
                    term_tr[s + n - 2] = 1.0
        eps = json.loads((root / "episodes.json").read_text()) if (root / "episodes.json").exists() else None
        succ = [bool(e["success"]) for e in eps] if eps is not None else [bool(b) for b in man.get("bc", [0] * len(ln))]
        if a.game == "crafter":
            succ = [False] * len(ln)
        names = man.get("task_names")
        tid = man.get("tasks", [0] * len(ln))
        task_c = np.zeros(N, np.int64); task_h = np.zeros(N, np.int64)
        val, vm = np.zeros(N, np.float32), np.zeros(N, np.float32)
        first, tin, nep = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64)
        for k_, (s, n) in enumerate(zip(st, ln)):
            name = names[tid[k_]] if names else (eps[k_]["task"] if eps else tasks[0])
            task_c[s:s + n] = tasks.index(name) if name in tasks else 0
            task_h[s:s + n] = head_names.index(name) if name in head_names else 0
            first[s:s + n] = s; tin[s:s + n] = np.arange(n); nep[s:s + n] = n
            ended = bool(term_tr[s + n - 2]) if n >= 2 else False
            if a.game == "wood":   # task return: gamma^(T-1-t) on a success, 0 on a failed end; no target on the last frame
                for t in range(n - 1):
                    val[s + t] = a.gamma ** (n - 2 - t) if succ[k_] else 0.0
                    vm[s + t] = float(ended)
            else:                  # native return (train_slm_reason_act_d4.py frame_split_data)
                G = 0.0
                for t in range(n - 2, -1, -1):
                    G = rew[s + t] + a.gamma * G; val[s + t] = G; vm[s + t] = float(ended or n - 1 - t >= a.value_horizon)
        z = torch.load(zpath) if zpath and Path(zpath).exists() and not str(zpath).endswith(".npy") else encode(root, zpath)
        assert len(z) == N, (root, len(z), N)
        T_ = lambda x, dt=None: torch.as_tensor(x, dtype=dt).to(dev)
        d = dict(z=z.to(dev), act=T_(act), rew=T_(rew), term=T_(term_tr), val=T_(val), vm=T_(vm), tc=T_(task_c), th=T_(task_h),
                 first=T_(first), tin=T_(tin), nep=T_(nep))
        ok = np.flatnonzero(tin < nep - 1)                              # at least one real transition ahead
        near = np.flatnonzero((tin < nep - 1) & (tin >= nep - 1 - H) & (term_tr[np.minimum(np.array(first) + nep - 2, N - 1)] > 0))
        d["starts"], d["near"] = ok, near
        print(f"{root}: {len(ln)} episodes, {N} frames, starts {len(ok)}, near an end {len(near)}, "
              f"terminal transitions {int(term_tr.sum())}, value targets {int(vm.sum())}", flush=True)
        return d

    base = load_dir(a.data, a.latents or str(out / "latents" / "z_data.pt"))
    onl = [load_dir(d_, Path(d_) / "z.pt", True) for d_ in a.online.split(",")] if a.online else []
    val = load_dir(a.val, a.val_latents or str(out / "latents" / "z_val.pt")) if a.val else None

    # ---------------- one batch: dream the real actions, the scorer on every dreamed window ----------------
    def pick(d, n):
        ne = int(round(a.end_frac * n)) if len(d["near"]) else 0
        return np.concatenate([rng.choice(d["near"], ne), rng.choice(d["starts"], n - ne)]).astype(np.int64)

    def batch(d, idx):
        i = torch.as_tensor(idx, device=dev)
        f0, tin, nep = d["first"][i], d["tin"][i], d["nep"][i]
        back = torch.maximum(i[:, None] - torch.arange(3, -1, -1, device=dev)[None], f0[:, None])    # z_{t-3..t}
        Zs = d["z"][back].float()
        prev = torch.stack([torch.where(i - 3 + j >= f0, d["act"][(i - 3 + j).clamp_min(0)], torch.zeros_like(i))
                            for j in range(3)], 1)                                                   # a_{t-3..t-1}
        steps = torch.arange(H, device=dev)[None]
        fut = (i[:, None] + steps).clamp(max=len(d["act"]) - 1)
        last = f0 + nep - 1                                                                          # last frame of the episode
        valid = (i[:, None] + 1 + steps) <= last[:, None]                                           # real frame t+k exists
        pa = torch.where(valid, d["act"][fut], torch.zeros_like(fut))
        tr = (i[:, None] + steps).clamp(max=len(d["rew"]) - 1)                                     # transition t+k-1 -> t+k
        r_t = torch.where(valid, d["rew"][tr], torch.zeros_like(d["rew"][tr]))
        e_t = torch.where(valid, d["term"][tr], torch.zeros_like(d["term"][tr]))
        fk = (i[:, None] + 1 + steps).clamp(max=len(d["val"]) - 1)
        v_t, v_m = d["val"][fk], d["vm"][fk] * valid.float()
        return Zs, prev, pa, valid, r_t, e_t, v_t, v_m, d["th"][i], d["tc"][i]

    def scorer(hd, cr, Zs, prev, pa, th, tc, grad):
        """dream + the heads / critic on the H dreamed windows -> reward logits [B,H,bins], continue logits [B,H], critic out [B,H(,bins)]."""
        B = Zs.shape[0]
        acts = torch.cat([prev[:, 1:], pa], 1)
        zf = torch.cat([Zs[:, 4 - ctx:], Zs[:, -1:].expand(-1, H, -1, -1)], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            d_, _, _ = windowed_rollout(dyn, z_full=zf, actions=F.one_hot(acts, 17).float(), window=window, ctx=ctx, K=a.K,
                                        k_max=k_max, tau_ctx=tau)
        S = torch.cat([Zs, d_.float()], 1)
        into = torch.cat([prev, pa], 1)
        rl, cl, vo = [], [], []
        with torch.set_grad_enabled(grad):
            for k in range(1, H + 1):
                av = torch.zeros(B, 4, 18, device=dev); av[:, 0, 17] = 1.0
                av[:, 1:, :17] = F.one_hot(into[:, k:k + 3], 17).float()
                stp = torch.full((B, 4), _log2_int(k_max), device=dev, dtype=torch.long)
                sg = torch.full((B, 4), sig, device=dev, dtype=torch.long)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ag = hd.tokens((B, 4), task_ids=th.reshape(B, 1).expand(B, 4), device=dev)
                    o = dyn(av, stp, sg, (1 - tau) * S[:, k:k + 4], agent_tokens=ag)
                hh = o[-1][:, -1].float().flatten(1)
                rl.append(hd.reward.at(hh, 0).float()); cl.append(hd.cont(hh).squeeze(-1).float())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb, mask, _ = cr.prefix(S[:, k:k + 4], pre, post_ids[tc], post_mask[tc])
                    hv = cr.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, -1]
                vo.append(cr.value_head.logits(hv) if HL else cr.value_head(hv).squeeze(-1).float())
        return torch.stack(rl, 1), torch.stack(cl, 1), torch.stack(vo, 1)

    def losses(hd, cr, b, grad):
        Zs, prev, pa, valid, r_t, e_t, v_t, v_m, th, tc = b
        rl, cl, vo = scorer(hd, cr, Zs, prev, pa, th, tc, grad)
        m = valid.float()
        l_r = hd.reward.loss(rl, r_t, m)
        l_c = (F.binary_cross_entropy_with_logits(cl, e_t, reduction="none") * m).sum() / m.sum().clamp_min(1)
        if HL:
            l_v = cr.value_head.loss(vo.flatten(0, 1), v_t.flatten(), v_m.flatten())
            v_hat = cr.value_head.mean(vo)
        else:
            l_v = (((vo - v_t) ** 2) * v_m).sum() / v_m.sum().clamp_min(1); v_hat = vo
        stats = dict(r_mae=float(((hd.reward.mean(rl) - r_t).abs() * m).sum() / m.sum().clamp_min(1)),
                     v_mse=float((((v_hat - v_t) ** 2) * v_m).sum() / v_m.sum().clamp_min(1)),
                     p_end_on_end=float((torch.sigmoid(cl) * e_t * m).sum() / (e_t * m).sum().clamp_min(1)),
                     p_end_else=float((torch.sigmoid(cl) * (1 - e_t) * m).sum() / ((1 - e_t) * m).sum().clamp_min(1)))
        return l_r, l_c, l_v, stats

    @torch.no_grad()
    def evaluate(hd, cr):
        d = val if val is not None else base
        r = np.random.default_rng(1234); acc = {}
        hd.eval(); cr.eval()
        for _ in range(a.eval_batches):
            ne = int(round(a.end_frac * a.batch)) if len(d["near"]) else 0
            idx = np.concatenate([r.choice(d["near"], ne), r.choice(d["starts"], a.batch - ne)])
            l_r, l_c, l_v, st = losses(hd, cr, batch(d, idx), False)
            for k_, v_ in dict(st, l_reward=float(l_r), l_cont=float(l_c), l_value=float(l_v)).items():
                acc[k_] = acc.get(k_, 0.0) + v_ / a.eval_batches
        return acc

    before = evaluate(heads0, critic0); print("DREAM-HEADS before", json.dumps(before), flush=True)
    opt = torch.optim.AdamW([{"params": heads.parameters(), "lr": a.lr_heads}, {"params": critic.parameters(), "lr": a.lr_critic}],
                            betas=(0.9, 0.95), weight_decay=0.01)
    base_lr = [g["lr"] for g in opt.param_groups]
    t0 = time.time(); run = {}
    heads.train(); critic.train()
    for step in range(1, a.steps + 1):
        f = min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g_, l0 in zip(opt.param_groups, base_lr):
            g_["lr"] = l0 * f
        n_on = int(round(a.online_frac * a.batch)) if onl else 0
        parts = [(base, pick(base, a.batch - n_on))]
        if n_on:
            d_on = onl[rng.integers(len(onl))]; parts.append((d_on, pick(d_on, n_on)))
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for d, idx in parts:
            l_r, l_c, l_v, st = losses(heads, critic, batch(d, idx), True)
            loss = (l_r + l_c + l_v) * len(idx) / a.batch
            loss.backward(); tot += float(loss)
            for k_, v_ in st.items():
                run[k_] = run.get(k_, 0.0) + v_ * len(idx) / a.batch
        torch.nn.utils.clip_grad_norm_(list(heads.parameters()) + list(critic.parameters()), 1.0)
        opt.step()
        run["loss"] = run.get("loss", 0.0) + tot
        if step % 50 == 0:
            print(f"step {step} " + " ".join(f"{k_} {v_ / 50:.4f}" for k_, v_ in run.items()) + f" {time.time() - t0:.0f}s", flush=True)
            run = {}
    after = evaluate(heads, critic); print("DREAM-HEADS after", json.dumps(after), flush=True)
    torch.save(dict(hck, agent=heads.state_dict(), dream_trained=dict(args=vars(a), before=before, after=after)), out / "heads.pt")
    torch.save(dict(cst, model=critic.state_dict(), dream_trained=dict(args=vars(a), before=before, after=after)), out / "critic.pt")
    (out / "dream_heads.json").write_text(json.dumps(dict(before=before, after=after, args=vars(a), minutes=(time.time() - t0) / 60), indent=1))
    print("DREAM-HEADS-DONE", out, flush=True)


if __name__ == "__main__":
    main()
