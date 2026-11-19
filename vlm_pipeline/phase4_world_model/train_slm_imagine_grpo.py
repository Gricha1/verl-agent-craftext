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
The SLM keeps its reconstruction / value heads untouched (not in the loss). Checkpoints in the train_slm_reason_act_d4.py
format (eval_slm_d4_env.py reads them). Every --eval-every updates: on held-out validation starts, mean score, death share
(sum_k A_k p_k with r_k < 0) and success share of the greedy plan, and the same for the start checkpoint at update 0.

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
    for k in ("--slm", "--dyn", "--heads", "--raw-data", "--reasoning", "--train", "--val", "--latent-cache", "--out"):
        p.add_argument(k, required=True)
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
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-starts", type=int, default=512)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=7950)
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
    max_tokens = 80 + 6 * K
    params = [q for n, q in model.named_parameters() if not n.startswith(("head.", "value_head."))]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---------------- world model ----------------
    hck = torch.load(a.heads, map_location="cpu", weights_only=False)
    world = WorldModel(a.dyn, device=device, n_agent=int(hck["n_agent"]))
    world.load_dynamics_weights(hck["dyn"])
    dyn = world.dyn.eval()
    heads = build_agent_heads(hck["agent_cfg"]).to(device).eval()
    heads.load_state_dict(hck["agent"])
    term_bin = terminal_bins(heads.reward, world.continues_from_reward, device)
    head_task = {g: i for i, g in enumerate(world.dataset.manifest["task_names"])}
    k_max, tau = world.k_max, world.tau_ctx
    window = int(world.window); ctx = window - 1
    sig = min(int(round((1.0 - tau) * k_max)), k_max)
    H = a.horizon

    # ---------------- data ----------------
    episodes = load_episodes(Path(a.raw_data), None, None)
    reasoning = {}
    for split in ("train", "valid"):
        for line in open(Path(a.reasoning) / f"{split}.jsonl"):
            r = json.loads(line)
            m = re.match(r"<reasoning>(.*)</reasoning>\nACTION: (\S+)$", r["cot_target"], re.S)
            if m:
                reasoning[(int(r["source_episode_id"]), int(r["step"]))] = m.group(1).strip()

    def split(root, tag):
        frames, starts, lengths = load_split(root)
        z = torch.load(Path(a.latent_cache) / f"z_{tag}.pt").to(device)
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

    zT, fT, rowsT = split(a.train, "train")
    zV, fV, rowsV = split(a.val, "val")
    near = [i for i, r in enumerate(rowsT) if r[4]]
    print(f"train starts {len(rowsT)} ({len(near)} near death), val {len(rowsV)}", flush=True)
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
    def score(Zs, prev, g, plans):
        """Zs [C,4,64,16], prev [C,3] actions a_{t-3..t-1}, g [C] SLM task ids, plans list of action lists
        -> score [C], death mass [C], success mass [C]."""
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
        sc = (disc * alive[:, :H] * rk * valid).sum(1) + a.gamma ** L.float() * alive.gather(1, L[:, None])[:, 0] * vk.gather(1, last[:, None])[:, 0]
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

    log = open(out / "metrics.jsonl", "a")
    res = evaluate(0); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res), flush=True)
    model.train()
    t0 = time.time(); run = dict(score=0.0, adv=0.0, pg=0.0, bc=0.0, death=0.0, parsed=0.0, n=0)
    for step in range(1, a.updates + 1):
        nd = int(round(a.death_frac * a.starts)) if near else 0
        pick = list(rng.choice(near, nd)) + list(rng.choice(len(rowsT), a.starts - nd))
        Zs, g, prev = batch_of(rowsT, zT, fT, pick)
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, logits, _ = model(ZG, pre, post_ids[gG], post_mask[gG], tg, tm)
        lp = -F.cross_entropy(logits.float().transpose(1, 2), tg, reduction="none")
        seq_lp = (lp * tm).sum(1) / tm.sum(1).clamp_min(1)
        pg = -(adv.detach() * seq_lp).mean()
        loss = pg
        bc_rows = [i for i, j in enumerate(pick) if rowsT[j][3] is not None]
        bc = torch.zeros((), device=device)
        if a.bc_weight > 0 and bc_rows:
            tgts = [rowsT[pick[i]][3] for i in bc_rows]
            Wb = max(map(len, tgts))
            tb = torch.zeros(len(tgts), Wb, dtype=torch.long, device=device); mb = torch.zeros_like(tb)
            for i, t_ in enumerate(tgts):
                tb[i, :len(t_)] = torch.tensor(t_, device=device); mb[i, :len(t_)] = 1
            sel = torch.tensor(bc_rows, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lgb, _ = model(Zs[sel], pre, post_ids[g[sel]], post_mask[g[sel]], tb, mb)
            ce = F.cross_entropy(lgb.float().transpose(1, 2), tb, reduction="none")
            bc = (ce * mb).sum() / mb.sum()
            loss = loss + a.bc_weight * bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        run["score"] += float(sc.mean()); run["adv"] += float(adv.abs().mean()); run["pg"] += float(pg); run["bc"] += float(bc)
        run["death"] += float(dead.mean()); run["parsed"] += float(np.mean(ok)); run["n"] += 1
        if step % 10 == 0:
            n = run.pop("n")
            print(f"update {step} " + " ".join(f"{k} {v / n:.4f}" for k, v in run.items()) + f" {step / (time.time() - t0):.3f} upd/s",
                  flush=True)
            run = dict(score=0.0, adv=0.0, pg=0.0, bc=0.0, death=0.0, parsed=0.0, n=0)
        if step % a.eval_every == 0 or step == a.updates:
            res = evaluate(step); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res), flush=True)
        if step % a.save_every == 0 or step == a.updates:
            save(step, res)
    print("GRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
