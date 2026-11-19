#!/usr/bin/env python3
"""Multi-horizon value heads on frozen Dreamer 4 and SLM features, and their stacked ensemble (user, 2026-11-19).

Both models stay frozen (the planner's dream, reward and plans are unchanged); new heads read their features:
  Dreamer  h_t: phase-2 agent tokens of the real window z_{t-3..t} with the actions leading into it and the task
           (flattened, n_agent * d), as eval_slm_d4_env.py reads them;
  SLM      s_t: last hidden state at "Assistant:\n" of SLM(System + Z_t + Task), the input of its own value_head.
Targets, one per discount gamma in --gammas (task return, the same definition as add_value_targets):
  G^gamma_t = gamma^(n-2-t) on a successful episode, 0 on a failed one, frame n-1 and episodes without done excluded.
Heads: per model LayerNorm -> Linear(d, 256) -> GELU -> Linear(256, len(gammas)), MSE, AdamW, early stopping on half A of
the validation episodes (--val-episodes: collector dirs the models never saw, with failures).
Ensemble: ridge regression on half B of the validation episodes, target G^0.99 (the SLM value scale the planner uses), over
the 2 x len(gammas) new heads plus the two original value heads (SLM value, Dreamer value^k, k = log .99 / log gamma_D);
alpha by 5-fold CV over B's episodes. Reported on A, B and --test-episodes (e.g. the teacher on the evaluation seeds; never
used for fitting): R^2 against G^0.99 and AUROC of frames of successful vs failed episodes.

  python train_value_heads_d4_slm.py --slm <step.pt> --dyn <phase-1 dyn> --heads <phase-2 latest.pt> --raw-data <wood2200 raw>
     --train <wood2200_v3> --latent-cache <dir with z_train.pt> --val-episodes d1,d2,.. --test-episodes d --out <dir>
Writes <out>/ensemble.pt (head weights, ridge weights, feature order) for eval_slm_d4_env.py --value-ensemble, and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHeads(nn.Module):
    def __init__(self, d, n_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x.float())


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    for k in ("--slm", "--dyn", "--heads", "--raw-data", "--train", "--latent-cache", "--val-episodes", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--test-episodes", default="")
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--gammas", default="0.95,0.99,0.997")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=7950)
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from train_slm_reason_act_d4 import SYSTEM, SYSTEM_PLAN, SLMReasonAct
    from train_slm_recon_d4 import windows
    from train_slm_world_model import load_episodes, read_image
    from dreamer4.models.agent_heads import build_agent_heads, flatten_agent_out
    from dreamer4.models.dynamics import _log2_int
    from dreamer4.models.tokenizer import FrozenTokenizer
    from dreamer4.models.world_model import WorldModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gammas = [float(g) for g in a.gammas.split(",")]
    G99 = gammas.index(0.99)

    # ---------------- frozen models ----------------
    base = torch.load(a.slm, map_location="cpu", weights_only=False)
    cfg, tasks = base["args"], base["tasks"]
    K = cfg.get("plan") or 0
    text = PreTrainedTokenizerFast(tokenizer_file=str(Path(cfg["text_tokenizer"]) / "tokenizer.json"))
    enc = lambda s: text.encode(s, add_special_tokens=False)
    lm = AutoModelForCausalLM.from_pretrained(cfg["init_checkpoint"], dtype=torch.float32)
    lm.resize_token_embeddings(len(text)); lm.tie_weights()
    model = SLMReasonAct(lm, cfg["readout"]).to(dev)
    model.load_state_dict(base["model"]); model.eval()
    pre = torch.tensor([enc("System:\n" + (SYSTEM_PLAN.format(K) if K else SYSTEM) + "\nUser:\n")], device=dev)
    posts = [enc(f"\nTask: {t}\nAssistant:\n") for t in tasks]
    Lp = max(map(len, posts))
    post_ids = torch.zeros(len(tasks), Lp, dtype=torch.long, device=dev); post_mask = torch.zeros_like(post_ids)
    for i, ids in enumerate(posts):
        post_ids[i, Lp - len(ids):] = torch.tensor(ids); post_mask[i, Lp - len(ids):] = 1
    hck = torch.load(a.heads, map_location="cpu", weights_only=False)
    world = WorldModel(a.dyn, device=dev, n_agent=int(hck["n_agent"]))
    world.load_dynamics_weights(hck["dyn"])
    dyn = world.dyn.eval()
    heads = build_agent_heads(hck["agent_cfg"]).to(dev).eval()
    heads.load_state_dict(hck["agent"])
    head_task = {g: i for i, g in enumerate(world.dataset.manifest["task_names"])}
    k_max, tau = world.k_max, world.tau_ctx
    sig = min(int(round((1.0 - tau) * k_max)), k_max)
    gamma_d = float(hck["train_cfg"]["gamma"])
    kpow = math.log(0.99) / math.log(gamma_d)
    tok = FrozenTokenizer(cfg["tok"], history=4, device=dev)

    # ---------------- data: episodes -> z, rows ----------------
    def encode_episodes(eps, keys):
        zs = []
        with torch.no_grad():
            for k in keys:
                root = eps[k][0]["_root"]
                x = torch.from_numpy(np.stack([np.round(np.moveaxis(read_image(root, it["image"]), 0, -1)[:63, :63] * 255)
                                               .astype(np.uint8) for it in eps[k]]))
                zs.append(torch.cat([tok.encode_frames(x[None, max(0, s - 3):s + 256].to(dev))[0, min(s, 3):]
                                     for s in range(0, len(x), 256)]).half())
        return torch.cat(zs)

    def load_dirs(dirs):
        eps = {}
        for d in dirs:
            e = load_episodes(Path(d), None, None)
            for k, v in e.items():
                for it in v:
                    it["_root"] = Path(d)
                assert k not in eps, k
                eps[k] = v
        return eps

    def make_split(eps, keys, z):
        rows = dict(first=[], g=[], gd=[], prev=[], T=[], m=[], succ=[], died=[], ep=[])
        first = 0
        for e, k in enumerate(keys):
            ep = eps[k]; n = len(ep)
            ok = bool(ep[-1]["success"]); ended = bool(ep[-1]["done"])
            died = (not ok) and ended and n < 1000
            moves = [int(it["action"]) for it in ep]
            g = tasks.index(ep[0]["instruction"])
            for t in range(n):
                rows["first"].append(first); rows["g"].append(g); rows["gd"].append(head_task[ep[0]["instruction"]])
                rows["prev"].append([moves[t - 3 + i] if t - 3 + i >= 0 else -1 for i in range(3)])
                rows["T"].append([float(ok) * gm ** (n - 2 - t) if t < n - 1 else 0.0 for gm in gammas])
                rows["m"].append(float(ended and t < n - 1)); rows["succ"].append(ok); rows["died"].append(died); rows["ep"].append(e)
            first += n
        assert z.shape[0] == first, (z.shape, first)
        return {k: np.asarray(v) for k, v in rows.items()}, z

    @torch.no_grad()
    def features(rows, z):
        """-> SLM hidden [N, dS] fp16, Dreamer h [N, dD] fp16, original SLM value, original Dreamer value (CPU)."""
        N = z.shape[0]
        ep_first = torch.tensor(rows["first"], device=dev); prev = torch.tensor(rows["prev"], device=dev)
        fs, fd, vs, vd = [], [], [], []
        for s in range(0, N, a.chunk):
            idx = torch.arange(s, min(N, s + a.chunk), device=dev); B = idx.shape[0]
            Z = windows(z, ep_first, idx)
            g = torch.tensor(rows["g"][s:s + B], device=dev); gd = torch.tensor(rows["gd"][s:s + B], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb, mask, _ = model.prefix(Z, pre, post_ids[g], post_mask[g])
                hs = model.backbone(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, -1]
                vs.append(model.value_head(hs).squeeze(-1).float().cpu())
            fs.append(hs.half().cpu())
            av = torch.zeros(B, 4, 18, device=dev); av[:, 0, 17] = 1.0
            for j in range(1, 4):
                act = prev[s:s + B, j - 1]; real = act >= 0
                av[:, j, :17] = F.one_hot(act.clamp_min(0), 17).float() * real[:, None]
                av[:, j, 17] = (~real).float()
            st = torch.full((B, 4), _log2_int(k_max), device=dev, dtype=torch.long)
            sg = torch.full((B, 4), sig, device=dev, dtype=torch.long)
            ag = heads.tokens((B, 4), task_ids=gd.reshape(B, 1).expand(B, 4), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = dyn(av, st, sg, (1 - tau) * Z, agent_tokens=ag)
            h = flatten_agent_out(o[-1][:, -1].float())
            vd.append(heads.value.mean(heads.value.at(h, 0).float()).float().cpu())
            fd.append(h.half().cpu())
        return torch.cat(fs), torch.cat(fd), torch.cat(vs).numpy(), torch.cat(vd).numpy()

    t0 = time.time()
    tr_eps = load_episodes(Path(a.raw_data), None, None)
    man = json.loads((Path(a.train) / "manifest.json").read_text())
    tr_keys = [int(k) for k in man["episode_ids"]]
    tr_rows, z_tr = make_split(tr_eps, tr_keys, torch.load(Path(a.latent_cache) / "z_train.pt").to(dev))
    fS_tr, fD_tr, _, _ = features(tr_rows, z_tr); del z_tr
    print(f"train {len(tr_keys)} episodes {len(tr_rows['m'])} frames, features {tuple(fS_tr.shape)} {tuple(fD_tr.shape)} "
          f"{time.time() - t0:.0f}s", flush=True)
    va_eps = load_dirs(a.val_episodes.split(","))
    va_keys = sorted(va_eps)
    va_rows, z_va = make_split(va_eps, va_keys, encode_episodes(va_eps, va_keys).to(dev))
    fS_va, fD_va, vS0_va, vD0_va = features(va_rows, z_va); del z_va
    n_va = len(va_keys)
    va_died = sorted(set(va_rows["ep"][va_rows["died"]].tolist()))
    print(f"val {n_va} episodes ({len(va_died)} with death) {len(va_rows['m'])} frames {time.time() - t0:.0f}s", flush=True)
    perm = rng.permutation(n_va)
    in_A = np.isin(va_rows["ep"], perm[: n_va // 2])        # early stopping
    in_B = ~in_A                                            # ensemble weights
    te = None
    if a.test_episodes:
        te_eps = load_dirs(a.test_episodes.split(","))
        te_keys = sorted(te_eps)
        te_rows, z_te = make_split(te_eps, te_keys, encode_episodes(te_eps, te_keys).to(dev))
        te = (te_rows, *features(te_rows, z_te)); del z_te
        print(f"test {len(te_keys)} episodes {len(te_rows['m'])} frames", flush=True)

    # ---------------- heads ----------------
    def fit_head(Xtr, Ytr, Mtr, Xva, Yva, Mva, name):
        head = ValueHeads(Xtr.shape[1], len(gammas)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
        Ytr_t, Mtr_t = torch.tensor(Ytr, dtype=torch.float32), torch.tensor(Mtr, dtype=torch.float32)
        idx_tr = np.flatnonzero(Mtr > 0)
        best, best_state, bad, hist = float("inf"), None, 0, []
        for ep in range(a.epochs):
            head.train(); rng.shuffle(idx_tr); tl = 0.0
            for s in range(0, len(idx_tr), a.batch):
                b = torch.from_numpy(idx_tr[s:s + a.batch])
                loss = F.mse_loss(head(Xtr[b].to(dev)), Ytr_t[b].to(dev))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += float(loss) * len(b)
            head.eval()
            with torch.no_grad():
                P = torch.cat([head(Xva[s:s + 8192].to(dev)).cpu() for s in range(0, len(Xva), 8192)]).numpy()
            mk = Mva > 0
            vl = float(np.mean((P[mk] - Yva[mk]) ** 2))
            hist.append(dict(epoch=ep + 1, train_mse=tl / len(idx_tr), val_mse=vl,
                             val_mse_per_gamma=np.mean((P[mk] - Yva[mk]) ** 2, 0).round(5).tolist()))
            print(f"{name} epoch {ep + 1} train {tl / len(idx_tr):.5f} valA {vl:.5f} per gamma {hist[-1]['val_mse_per_gamma']}", flush=True)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        head.load_state_dict(best_state); head.eval()
        return head, hist

    T_tr, M_tr = tr_rows["T"], tr_rows["m"]
    hS, histS = fit_head(fS_tr, T_tr, M_tr, fS_va[in_A], va_rows["T"][in_A], va_rows["m"][in_A], "slm")
    hD, histD = fit_head(fD_tr, T_tr, M_tr, fD_va[in_A], va_rows["T"][in_A], va_rows["m"][in_A], "dreamer")
    del fS_tr, fD_tr

    @torch.no_grad()
    def predictors(fS, fD, vS0, vD0):
        pS = torch.cat([hS(fS[s:s + 8192].to(dev)).cpu() for s in range(0, len(fS), 8192)]).numpy()
        pD = torch.cat([hD(fD[s:s + 8192].to(dev)).cpu() for s in range(0, len(fD), 8192)]).numpy()
        return np.concatenate([pS, pD, vS0[:, None], np.clip(vD0, 0, 1)[:, None] ** kpow], 1)
    names = [f"slm_g{g}" for g in gammas] + [f"dreamer_g{g}" for g in gammas] + ["slm_value_orig", f"dreamer_value_orig^{kpow:.2f}"]

    # ---------------- ridge ensemble on half B ----------------
    X_va = predictors(fS_va, fD_va, vS0_va, vD0_va)
    y_va = va_rows["T"][:, G99]
    mB = in_B & (va_rows["m"] > 0)
    XB, yB, eB = X_va[mB], y_va[mB], va_rows["ep"][mB]

    def ridge(X, y, alpha):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - mu) / sd
        w = np.linalg.solve(Xs.T @ Xs + alpha * len(y) * np.eye(X.shape[1]), Xs.T @ (y - y.mean()))
        return w / sd, y.mean() - (mu / sd) @ w
    folds = np.array_split(rng.permutation(np.unique(eB)), 5)
    cv = {}
    for alpha in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        err = 0.0
        for f in folds:
            te_m = np.isin(eB, f)
            w, b = ridge(XB[~te_m], yB[~te_m], alpha)
            err += float(np.sum((XB[te_m] @ w + b - yB[te_m]) ** 2))
        cv[alpha] = err / len(yB)
    alpha = min(cv, key=cv.get)
    w, b = ridge(XB, yB, alpha)
    print("ridge cv mse", {k: round(v, 5) for k, v in cv.items()}, "alpha", alpha, flush=True)
    print("weights", dict(zip(names, np.round(w, 4).tolist())), "bias", round(float(b), 4), flush=True)

    # ---------------- report ----------------
    def report(X, rows, tag):
        m = rows["m"] > 0
        y = rows["T"][m, G99]
        succ, died = rows["succ"][m], rows["died"][m]
        res = {}
        cols = {n: X[m, i] for i, n in enumerate(names)}
        cols["ensemble"] = X[m] @ w + b
        for n, v in cols.items():
            res[n] = dict(r2=float(1 - np.mean((v - y) ** 2) / np.var(y)), mse=float(np.mean((v - y) ** 2)),
                          auroc_succ_vs_died=auroc(v[succ], v[died]))
        print(tag, {n: (round(r["r2"], 3), round(r["auroc_succ_vs_died"], 3)) for n, r in res.items()}, flush=True)
        return res
    metrics = dict(gammas=gammas, names=names, weights=w.tolist(), bias=float(b), alpha=alpha, ridge_cv=cv, kpow=kpow,
                   hist_slm=histS, hist_dreamer=histD, val_episodes=n_va, val_died=len(va_died),
                   valA=report(X_va[in_A], {k: v[in_A] for k, v in va_rows.items()}, "valA"),
                   valB=report(X_va[in_B], {k: v[in_B] for k, v in va_rows.items()}, "valB"))
    if te is not None:
        te_rows, fS_te, fD_te, vS0_te, vD0_te = te
        X_te = predictors(fS_te, fD_te, vS0_te, vD0_te)
        metrics["test"] = report(X_te, te_rows, "test")
        np.savez(out / "test_predictors.npz", X=X_te, **{k: v for k, v in te_rows.items() if k != "prev"})
    np.savez(out / "val_predictors.npz", X=X_va, inA=in_A, **{k: v for k, v in va_rows.items() if k != "prev"})
    torch.save(dict(slm_head=hS.state_dict(), dreamer_head=hD.state_dict(), d_slm=int(fS_va.shape[1]), d_dreamer=int(fD_va.shape[1]),
                    gammas=gammas, names=names, weights=w.tolist(), bias=float(b), kpow=kpow, slm=a.slm, heads=a.heads),
               out / "ensemble.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("VALUE_HEADS_DONE", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
