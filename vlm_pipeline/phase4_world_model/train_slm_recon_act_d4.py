#!/usr/bin/env python3
"""SLM over frozen Dreamer 4 latents with three heads: reconstruction of x_t (as train_slm_recon_d4.py), the teacher's action
and the value of the task (phase 4, the SLM state used for acting).

Sequence fed to the SLM (inputs_embeds, width d = 256 for slm5m):

  [ E("Task: g") ; adapter(z_{t-3..t}) + slot + index (256 tokens) ; r_1..r_R (readout) ; a (one action/value token) ]

  reconstruction  Zhat = tanh(MLP(u_{r_1..r_R}))  -> frozen Dreamer 4 decoder -> xhat_t,   L_rec = MSE + 0.2 LPIPS
  action          pi(a | g, Z) = softmax(MLP(u_a))                                       L_act = CE(pi, teacher action)
  value           V(g, Z) = MLP(u_a)                                                     L_val = (V - G_t)^2 on ended episodes
                  G_t = gamma^(T-1-t) if the episode completes the task on arriving at its last state T, else 0
                  (add_value_targets of train_slm_world_model.py, gamma 0.99; last state and cut episodes masked)

The three losses differ by orders of magnitude (L_rec ~1e-4 after training, CE ~1), so each is divided by a running mean of
itself (EMA 0.99, as the RMS normalisation of Dreamer 4's loss combiner): L = sum_i w_i L_i / sg(EMA_i). Every term then pulls
with a comparable force and reconstruction is not drowned by the action loss.

--from-run <dir>/latest.pt fine-tunes a train_slm_recon_d4.py run (backbone, adapter, readout, reconstruction head are loaded;
the task embedding uses the SLM's own token embeddings; the action token and both heads are new). Without it every part starts
from initial_random / random init.

  python train_slm_recon_act_d4.py --tok <tok ckpt> --init-checkpoint <initial_random> --raw-data <wood2200 raw collection> \
     --train <wood2200_v3> --val <wood2200_d4/val> --latent-cache <dir> --readout 64 --out <dir> [--from-run <run dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_slm_recon_d4 import OBJECTS, SLMRecon, decode, encode_split, load_split, windows

N_ACTIONS = 17


class SLMReconAct(SLMRecon):
    def __init__(self, lm, readout, **kw):
        super().__init__(lm, readout, **kw)
        d = lm.config.hidden_size
        self.embed = lm.get_input_embeddings()
        self.act_token = nn.Parameter(torch.randn(1, d) * .02)
        self.action_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_ACTIONS))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, Z, task_ids, task_mask):
        B = Z.shape[0]
        e = self.adapter(Z) + self.slot + self.index
        e = torch.cat([self.embed(task_ids).to(e.dtype), e.reshape(B, -1, e.shape[-1]),
                       self.readout.expand(B, -1, -1), self.act_token.expand(B, -1, -1)], 1)
        mask = torch.cat([task_mask, torch.ones(B, e.shape[1] - task_mask.shape[1], dtype=task_mask.dtype, device=e.device)], 1)
        h = self.backbone(inputs_embeds=e, attention_mask=mask).last_hidden_state
        u = h[:, -self.R - 1:-1]
        zhat = torch.tanh(self.head(u))
        if self.R == 1:
            zhat = zhat.view(B, self.w, self.n_lat, self.d_lat)
        else:
            zhat = zhat.view(B, self.n_lat, self.w, self.d_lat).transpose(1, 2)
        a = h[:, -1]
        return zhat, self.action_head(a).float(), self.value_head(a).squeeze(-1).float()


def frame_labels(root, episodes, tasks):
    """Per frame of a split (manifest order): task index, teacher action, value target and mask, from the raw collection."""
    root = Path(root)
    man_path = root / "manifest.json" if (root / "manifest.json").exists() else root / "wood_manifest.json"
    man = json.loads(man_path.read_text())
    task, action, value, vmask = [], [], [], []
    for key, n in zip(man["episode_ids"], man["lengths"]):
        ep = episodes[int(key)]
        assert len(ep) == n, (key, len(ep), n)
        task += [tasks.index(ep[0]["instruction"])] * n
        action += [it["action"] for it in ep]
        value += [it["value"] for it in ep]
        vmask += [it["value_mask"] for it in ep]
    return (torch.tensor(task), torch.tensor(action), torch.tensor(value, dtype=torch.float32),
            torch.tensor(vmask, dtype=torch.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--from-run", default="", help="train_slm_recon_d4.py run dir: fine-tune its latest.pt")
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--readout", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--val-samples", type=int, default=2048)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--rec-weight", type=float, default=1.0)
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--value-gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=7950)
    p.add_argument("--latent-cache", default="")
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from imagine_eval import classify_tiles, tile_prototypes
    from train_slm_world_model import add_value_targets, load_episodes
    from dreamer4.models.tokenizer import FrozenTokenizer
    from dreamer4.train.perceptual import LpipsPerceptual
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    tok = FrozenTokenizer(a.tok, history=4, device=device)
    lpips = LpipsPerceptual(net="alex", up=128).to(device)
    protos, index, names = tile_prototypes()
    protos, index, names = torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device), list(names)

    episodes = load_episodes(Path(a.raw_data), None, None)
    add_value_targets(episodes, a.value_gamma)
    tasks = sorted({ep[0]["instruction"] for ep in episodes.values()})
    text_tok = AutoTokenizer.from_pretrained(a.init_checkpoint)
    ids = [text_tok(f"Task: {t}", add_special_tokens=False)["input_ids"] for t in tasks]
    L = max(map(len, ids))
    task_ids = torch.zeros(len(tasks), L, dtype=torch.long, device=device)
    task_mask = torch.zeros(len(tasks), L, dtype=torch.long, device=device)
    for i, t in enumerate(ids):
        task_ids[i, :len(t)] = torch.tensor(t); task_mask[i, :len(t)] = 1
    print("tasks", tasks, "task tokens", [len(t) for t in ids], flush=True)

    data = {}
    for tag, root in (("train", a.train), ("val", a.val)):
        frames, starts, lengths = load_split(root)
        cache = Path(a.latent_cache) / f"z_{tag}.pt" if a.latent_cache else None
        z, x, first = encode_split(tok, frames, starts, lengths, device, cache)
        labels = [t.to(device) for t in frame_labels(root, episodes, tasks)]
        data[tag] = (z, x, first, *labels)
        print(tag, "frames", len(z), "value-masked", int(labels[3].sum()), "positive", int((labels[2] > 0).sum()), flush=True)
    zt, xt, ft, gt, at, vt, mt = data["train"]
    zv, xv, fv, gv, av, vv, mv = data["val"]
    val_idx = torch.randperm(len(zv), generator=torch.Generator().manual_seed(a.seed))[:a.val_samples].to(device)
    vmean = float((vt * mt).sum() / mt.sum())

    lm = AutoModelForCausalLM.from_pretrained(a.init_checkpoint, dtype=torch.float32)
    model = SLMReconAct(lm, a.readout).to(device)
    if a.from_run:
        state = torch.load(Path(a.from_run) / "latest.pt", map_location="cpu", weights_only=False)
        assert state["args"]["readout"] == a.readout
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print("fine-tune from", a.from_run, "step", state["step"], "new parameters:", sorted({k.split(".")[0] for k in missing}),
              "unexpected:", unexpected, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))

    def to_img(x):
        return x.permute(0, 3, 1, 2).float() / 255

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        sums = dict(mse=0.0, lpips=0.0, action_ce=0.0, action_acc=0.0)
        vse = vn = vconst = 0.0
        hit = {o: 0 for o in OBJECTS}; real_n = {o: 0 for o in OBJECTS}
        n = 0
        for s in range(0, len(val_idx), 256):
            idx = val_idx[s:s + 256]
            Z = windows(zv, fv, idx)
            x = to_img(xv[idx.cpu()].to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zhat, logits, value = model(Z, task_ids[gv[idx]], task_mask[gv[idx]])
                xh = decode(tok, zhat.float()).float()
            b = len(idx); n += b
            sums["mse"] += F.mse_loss(xh, x).item() * b
            sums["lpips"] += lpips(xh, x).item() * b
            sums["action_ce"] += F.cross_entropy(logits, av[idx]).item() * b
            sums["action_acc"] += (logits.argmax(-1) == av[idx]).float().sum().item()
            m = mv[idx]
            vse += float(((value - vv[idx]) ** 2 * m).sum()); vconst += float(((vmean - vv[idx]) ** 2 * m).sum()); vn += float(m.sum())
            real, pred = classify_tiles(x, protos, index)[0], classify_tiles(xh, protos, index)[0]
            for o in OBJECTS:
                c = names.index(o); r = real == c
                real_n[o] += int(r.sum()); hit[o] += int(((pred == c) & r).sum())
        res = {k: v / n for k, v in sums.items()}
        res["psnr"] = float(10 * np.log10(1 / max(res["mse"], 1e-10)))
        res["value_mse"] = vse / max(vn, 1); res["value_mse_const"] = vconst / max(vn, 1)
        for o in OBJECTS:
            res[f"slm_recall_{o}"] = hit[o] / max(real_n[o], 1)
        res["real_tiles"] = real_n; res["step"] = step
        model.train()
        return res

    log = open(out / "metrics.jsonl", "a")
    if a.from_run:   # the starting point: reconstruction of the loaded run, untrained action and value heads
        res = evaluate(0); log.write(json.dumps(res) + "\n"); log.flush(); print("[val]", json.dumps(res), flush=True)
    ema = {}
    weights = dict(rec=a.rec_weight, act=a.action_weight, val=a.value_weight)
    step, t0 = 0, time.time()
    run = {k: 0.0 for k in ("rec", "act", "val", "acc")}
    model.train()
    best = None
    while step < a.steps:
        idx = torch.randint(0, len(zt), (a.batch,), device=device)
        Z = windows(zt, ft, idx)
        x = to_img(xt[idx.cpu()].to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zhat, logits, value = model(Z, task_ids[gt[idx]], task_mask[gt[idx]])
            xh = decode(tok, zhat).float()
        terms = dict(rec=F.mse_loss(xh, x) + a.lpips_weight * lpips(xh, x),
                     act=F.cross_entropy(logits, at[idx]),
                     val=((value - vt[idx]) ** 2 * mt[idx]).sum() / mt[idx].sum().clamp_min(1))
        loss = 0.0
        for k, v in terms.items():
            ema[k] = v.item() if k not in ema else .99 * ema[k] + .01 * v.item()
            loss = loss + weights[k] * v / max(ema[k], 1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        for k in terms:
            run[k] += terms[k].item()
        run["acc"] += (logits.argmax(-1) == at[idx]).float().mean().item()
        if step % 100 == 0:
            print(f"step {step} rec {run['rec']/100:.5f} act_ce {run['act']/100:.4f} acc {run['acc']/100:.3f} "
                  f"val_mse {run['val']/100:.5f} gnorm {gn.item():.2f} {step/(time.time()-t0):.2f} it/s", flush=True)
            run = {k: 0.0 for k in run}
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate(step)
            log.write(json.dumps(res) + "\n"); log.flush()
            print("[val]", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
            state = dict(model=model.state_dict(), step=step, args=vars(a), val=res, tasks=tasks)
            torch.save(state, out / "latest.pt")
            if best is None or res["action_ce"] < best:
                best = res["action_ce"]; torch.save(state, out / "best.pt")
    print("TRAIN_DONE", step, flush=True)


if __name__ == "__main__":
    main()
