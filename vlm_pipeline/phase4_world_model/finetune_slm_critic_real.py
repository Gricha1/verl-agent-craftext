#!/usr/bin/env python3
"""Dyna loop step 3b (user 2026-11-21, the ZeroHero recipe): the HL-Gauss critic (hl_critic.py; stage 1:
train_slm_reason_act_d4.py --critic-from) moves from the teacher's returns to the returns of OUR planner in the real
environment, before the policy update uses it (their actor warmup: the critic adapts first).

  targets  online frames (all --online traj dirs of eval_slm_d4_env.py --save-traj): G^lambda_t of hl_critic.online_targets,
           GAE(gamma 0.99, lambda 0.95) returns from the real rewards with the values V_old of the critic before this step
           (computed once, as PPO does per batch); the end of an episode is worth 0 at a death, 1 at a finished wood task,
           V_old at a timeout. Base frames (the BC dataset, --base-frac of every batch): the teacher's returns of stage 1, an
           anchor against forgetting the states the planner has not reached yet.
  loss     HL-Gauss cross-entropy, AdamW lr --lr (their critic lr ~1e-4), cosine to 0, grad clip 1.
  output   <out>/critic.pt (the input checkpoint with the new weights and a "finetune" record) and critic_ft.json: CE / MSE /
           correlation on held-out online episodes (--val-frac) and on the base val split, before and after.

  python finetune_slm_critic_real.py --critic <critic .pt> --online <traj>[,<traj>] --game wood|crafter --train <base train>
     --val <base val> --latent-cache <dir with z_train.pt, z_val.pt> [--raw-data <wood raw>] --out <dir>
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic", required=True)
    p.add_argument("--online", required=True)
    p.add_argument("--game", choices=("wood", "crafter"), required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--latent-cache", required=True)
    p.add_argument("--raw-data", default="", help="wood: raw episodes (the teacher's task returns)")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--base-frac", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--value-horizon", type=int, default=300)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from hl_critic import attach_for, online_targets
    from train_slm_reason_act_d4 import SYSTEM_PLAN, SLMReasonAct
    from train_slm_recon_d4 import windows
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cst = torch.load(a.critic, map_location="cpu", weights_only=False)
    assert cst.get("critic_hl") is not None, "needs an HL-Gauss critic checkpoint (train_slm_reason_act_d4.py --critic-from)"
    cfg = cst["args"]; tasks = cst["tasks"]
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    model = attach_for(SLMReasonAct(lm, cfg["readout"]).to(dev), cst)
    model.load_state_dict(cst["model"])
    old = copy.deepcopy(model).eval().requires_grad_(False)
    pre = torch.tensor([enc("System:\n" + SYSTEM_PLAN.format(cfg["plan"]) + "\nUser:\n")], device=dev)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=dev); post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1

    def logits_of(net, z, first, idx, g):
        emb, mask, _ = net.prefix(windows(z, first, idx), pre, post_ids[g], post_mask[g])
        return net.value_head.logits(net.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, -1])

    @torch.no_grad()
    def values(net, z, first, g):
        vs = []
        for s in range(0, len(z), 512):
            idx = torch.arange(s, min(s + 512, len(z)), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vs.append(net.value_head.mean(logits_of(net, z, first, idx, g[idx])))
        return torch.cat(vs)

    def first_of(starts, lengths):
        return torch.cat([torch.full((n,), int(s), dtype=torch.long) for s, n in zip(starts, lengths)]).to(dev)

    # ---- base (teacher) frames with their stage-1 targets ----
    base_eps = None
    if a.game == "wood":
        from train_slm_world_model import add_value_targets, load_episodes
        base_eps = load_episodes(Path(a.raw_data), None, None); add_value_targets(base_eps, a.gamma)

    def base_split(root, tag):
        root = Path(root)
        man = json.loads(((root / "manifest.json") if (root / "manifest.json").exists() else root / "wood_manifest.json").read_text())
        z = torch.load(Path(a.latent_cache) / f"z_{tag}.pt").to(dev)
        lengths = man["lengths"]
        starts = man["starts"] if "starts" in man else np.concatenate(([0], np.cumsum(lengths)[:-1])).tolist()
        N = sum(lengths); val = np.zeros(N, np.float32); m = np.zeros(N, np.float32); g = np.zeros(N, np.int64)
        if a.game == "crafter":   # train_slm_reason_act_d4.py frame_split_data
            rew = np.load(root / "rewards.npy").astype(np.float64); died = np.load(root / "terminal_in.npy")
            for s, n in zip(starts, lengths):
                Gr, ended = 0.0, bool(died[s + n - 1])
                for t in range(n - 2, -1, -1):
                    Gr = rew[s + t] + a.gamma * Gr; val[s + t] = Gr; m[s + t] = float(ended or n - 1 - t >= a.value_horizon)
        else:                     # train_slm_reason_act_d4.py split_data / add_value_targets
            for key, s, n in zip(man["episode_ids"], starts, lengths):
                ep = base_eps[int(key)]
                assert len(ep) == n
                for t, it in enumerate(ep):
                    val[s + t] = it["value"]; m[s + t] = it["value_mask"]; g[s + t] = tasks.index(ep[0]["instruction"])
        assert len(z) == N, (len(z), N)
        return dict(z=z, first=first_of(starts, lengths), G=torch.from_numpy(val).to(dev), g=torch.from_numpy(g).to(dev),
                    rows=torch.from_numpy(np.flatnonzero(m)).to(dev))
    bT, bV = base_split(a.train, "train"), base_split(a.val, "val")
    bV["rows"] = bV["rows"][torch.randperm(len(bV["rows"]), device=dev)[:4096]]

    # ---- online frames: lambda-returns of the real episodes under V_old ----
    zs, firsts, Gs, gs, tr_rows, va_rows, f0 = [], [], [], [], [], [], 0
    for d in a.online.split(","):
        d = Path(d)
        man = json.loads((d / "wood_manifest.json").read_text()); eps = json.loads((d / "episodes.json").read_text())
        z = torch.load(d / "z.pt").to(dev); rew = np.load(d / "rewards.npy")
        first = first_of(man["starts"], man["lengths"])
        g = torch.cat([torch.full((n,), tasks.index(e["task"]), dtype=torch.long) for n, e in zip(man["lengths"], eps)]).to(dev)
        v_old = values(old, z, first, g)
        G = online_targets(v_old, rew, man["starts"], man["lengths"], eps, a.game, a.gamma, a.lam).to(dev)
        order = rng.permutation(len(eps)); nv = max(1, int(round(a.val_frac * len(eps))))
        for j, k in enumerate(order):
            s, n = man["starts"][k], man["lengths"][k]
            (va_rows if j < nv else tr_rows).append(torch.arange(f0 + s, f0 + s + n))
        zs.append(z); firsts.append(first + f0); Gs.append(G); gs.append(g); f0 += len(z)
    oz, of, oG, og = torch.cat(zs), torch.cat(firsts), torch.cat(Gs), torch.cat(gs)
    oT, oV = torch.cat(tr_rows).to(dev), torch.cat(va_rows).to(dev)
    print(f"online frames {len(oz)} (train {len(oT)}, val {len(oV)}), target mean {float(oG.mean()):.3f}; base rows "
          f"{len(bT['rows'])}, val {len(bV['rows'])}", flush=True)

    @torch.no_grad()
    def evaluate(net):
        net.eval(); res = {}
        sets = dict(online=(oz, of, oG, og, oV), base=(bV["z"], bV["first"], bV["G"], bV["g"], bV["rows"]))
        for name, (z, first, G, g, rows) in sets.items():
            ce = se = 0.0; ps = []
            for s in range(0, len(rows), 512):
                idx = rows[s:s + 512]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = logits_of(net, z, first, idx, g[idx])
                ce += float(net.value_head.loss(lg, G[idx])) * len(idx); v = net.value_head.mean(lg)
                se += float(((v - G[idx]) ** 2).sum()); ps.append(v)
            pr = torch.cat(ps); y = G[rows]
            res[name] = dict(ce=ce / len(rows), mse=se / len(rows), mse_const=float(((y - y.mean()) ** 2).mean()),
                             corr=float(torch.corrcoef(torch.stack([y.float(), pr.float()]))[0, 1]), mean_pred=float(pr.mean()),
                             mean_target=float(y.mean()))
        net.train()
        return res

    before = evaluate(model); print("CRITIC-FT before", json.dumps(before), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    nb = int(round(a.base_frac * a.batch)); no = a.batch - nb
    t0 = time.time(); run = 0.0
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        io = oT[torch.randint(0, len(oT), (no,), device=dev)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = logits_of(model, oz, of, io, og[io])
        loss = model.value_head.loss(lo, oG[io]) * no / a.batch
        if nb:
            ib = bT["rows"][torch.randint(0, len(bT["rows"]), (nb,), device=dev)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lb = logits_of(model, bT["z"], bT["first"], ib, bT["g"][ib])
            loss = loss + model.value_head.loss(lb, bT["G"][ib]) * nb / a.batch
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); run += float(loss)
        if step % 100 == 0:
            print(f"step {step} ce {run / 100:.4f} lr {lr:.1e} {time.time() - t0:.0f}s", flush=True); run = 0.0
    after = evaluate(model); print("CRITIC-FT after", json.dumps(after), flush=True)
    st = dict(cst, model=model.state_dict(), finetune=dict(args=vars(a), before=before, after=after, from_critic=a.critic))
    torch.save(st, out / "critic.pt")
    (out / "critic_ft.json").write_text(json.dumps(dict(before=before, after=after, args=vars(a), minutes=(time.time() - t0) / 60), indent=1))
    print("CRITIC-FT-DONE", out, flush=True)


if __name__ == "__main__":
    main()
