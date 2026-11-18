#!/usr/bin/env python3
"""The SLM learns to read the latents of a frozen Dreamer 4 tokenizer: it must reconstruct the frame x_t through the frozen
Dreamer 4 decoder (phase 4, grounding step before the SLM predicts the next state).

Dreamer 4 (4ku/dreamer4, wood2200 run of 2026-11-07) encodes frame t with a history of 4 frames, z_t = E(x_{t-3..t}),
64 latent tokens x 16 numbers, and decodes x_t from the window z_{t-3..t}. Here:

  input   Z = (z_{t-3}, z_{t-2}, z_{t-1}, z_t)   4 x 64 x 16, frozen encoder, replicate-padded at the episode start
  adapter e_{s,i} = MLP(Z_{s,i}) + slot_s + index_i                      256 input embeddings of width d (SLM width)
  SLM     u = SLM([e; r_1..r_R])[last R positions]                       whole SLM trained, from initial_random
  head    Zhat = tanh(MLP(u))                                             R = 1: one readout token -> 4 x 64 x 16
                                                                          R = 64: token i -> latent i over the 4 slots
  decoder xhat_t = D(Zhat)[last frame]                                    frozen Dreamer 4 decoder, gradients pass through
  loss    L = ||xhat_t - x_t||^2 + 0.2 LPIPS(xhat_t, x_t)                 Dreamer 4 tokenizer terms (AlexNet LPIPS, 128 px)

R = 1 is the bottleneck variant: the whole frame must pass through one hidden state of the SLM (u_t, later read by the
policy and value heads). Reported on validation episodes: pixel MSE, PSNR, latent MSE ||Zhat - Z||^2, LPIPS, and the recall
of zombie / cow / tree / table tiles (tile classifier of imagine_eval) for the SLM and for the tokenizer itself.

  python train_slm_recon_d4.py --tok <dreamer4 tok>/checkpoints/latest.pt --init-checkpoint <phase-2 run>/initial_random \
     --train <dir with frames.npy + manifest> --val <dir> --out <dir> --readout 1
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

OBJECTS = ("zombie", "cow", "tree", "table")


def load_split(root):
    root = Path(root)
    man_path = root / "manifest.json" if (root / "manifest.json").exists() else root / "wood_manifest.json"
    man = json.loads(man_path.read_text())
    frames = np.load(root / "frames.npy", mmap_mode="r")
    return frames, man["starts"], man["lengths"]


@torch.no_grad()
def encode_split(tok, frames, starts, lengths, device, cache=None, chunk=256):
    """Per-frame latents z_t (history window inside the frozen encoder) and the frames, both on the GPU.
    The latents are kept in `cache` (a .pt file) and read from it on the next run."""
    xs = [torch.from_numpy(np.asarray(frames[s:s + n, :63, :63])) for s, n in zip(starts, lengths)]
    if cache is not None and Path(cache).exists():
        z = torch.load(cache).to(device)
    else:
        zs = []
        for x, n in zip(xs, lengths):
            zs.append(torch.cat([tok.encode_frames(x[None, max(0, a - 3):a + chunk].to(device))[0, min(a, 3):]
                                 for a in range(0, n, chunk)]).half())
        z = torch.cat(zs)
        if cache is not None:
            torch.save(z.cpu(), cache)
    x = torch.cat(xs).pin_memory()          # frames stay in host memory, batches are copied over
    assert len(z) == len(x), (len(z), len(x))
    ep_first = torch.cat([torch.full((n,), int(o), dtype=torch.long) for n, o in zip(lengths, np.concatenate(([0], np.cumsum(lengths)[:-1])))])
    return z, x, ep_first.to(device)


def windows(z, ep_first, idx, w=4):
    """(B,) frame indices -> (B, w, 64, 16) latents z_{t-w+1..t}, the first latent repeated before the episode start."""
    back = idx[:, None] - torch.arange(w - 1, -1, -1, device=idx.device)[None]
    back = torch.maximum(back, ep_first[idx][:, None])
    return z[back].float()


class SLMRecon(nn.Module):
    def __init__(self, lm, readout, n_lat=64, d_lat=16, window=4, head_hidden=1024):
        super().__init__()
        d = lm.config.hidden_size
        self.backbone = lm.model
        self.R, self.n_lat, self.d_lat, self.w = readout, n_lat, d_lat, window
        self.adapter = nn.Sequential(nn.Linear(d_lat, d), nn.GELU(), nn.Linear(d, d))
        self.slot = nn.Parameter(torch.randn(window, 1, d) * .02)
        self.index = nn.Parameter(torch.randn(1, n_lat, d) * .02)
        self.readout = nn.Parameter(torch.randn(readout, d) * .02)
        out = window * n_lat * d_lat // readout
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(), nn.Linear(head_hidden, out))

    def forward(self, Z):
        B = Z.shape[0]
        e = self.adapter(Z) + self.slot + self.index                         # (B, w, 64, d)
        e = torch.cat([e.reshape(B, -1, e.shape[-1]), self.readout.expand(B, -1, -1)], 1)
        u = self.backbone(inputs_embeds=e).last_hidden_state[:, -self.R:]   # (B, R, d)
        zhat = torch.tanh(self.head(u))
        if self.R == 1:
            zhat = zhat.view(B, self.w, self.n_lat, self.d_lat)
        else:
            zhat = zhat.view(B, self.n_lat, self.w, self.d_lat).transpose(1, 2)
        return zhat, u


def decode(tok, zhat):
    from dreamer4.models.tokenizer import unpatchify
    patches = tok.tok.decoder(zhat)[:, -1]                                    # last frame of the window
    return unpatchify(patches[:, None], tok.H, tok.W, C=tok.C, patch_size=tok.patch_size)[:, 0].clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="/w/repo")
    p.add_argument("--readout", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--val-samples", type=int, default=2048)
    p.add_argument("--lpips-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7950)
    p.add_argument("--latent-cache", default="", help="directory for z_train.pt / z_val.pt of this tokenizer")
    a = p.parse_args()
    sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
    from imagine_eval import classify_tiles, tile_prototypes
    from dreamer4.models.tokenizer import FrozenTokenizer
    from dreamer4.train.perceptual import LpipsPerceptual
    from transformers import AutoModelForCausalLM

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda")
    tok = FrozenTokenizer(a.tok, history=4, device=device)
    assert tok.n_latents == 64 and tok.d_bottleneck == 16, (tok.n_latents, tok.d_bottleneck)
    lpips = LpipsPerceptual(net="alex", up=128).to(device)
    protos, index, names = tile_prototypes()
    protos, index, names = torch.from_numpy(protos).to(device), torch.from_numpy(index).to(device), list(names)

    t0 = time.time()
    data = {}
    for tag, root in (("train", a.train), ("val", a.val)):
        frames, starts, lengths = load_split(root)
        cache = Path(a.latent_cache) / f"z_{tag}.pt" if a.latent_cache else None
        data[tag] = encode_split(tok, frames, starts, lengths, device, cache)
        print(tag, "episodes", len(lengths), "frames", len(data[tag][0]), "encoded in", round(time.time() - t0), "s", flush=True)
    zt, xt, ft = data["train"]
    zv, xv, fv = data["val"]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    val_idx = torch.randperm(len(zv), generator=g)[:a.val_samples].to(device)

    lm = AutoModelForCausalLM.from_pretrained(a.init_checkpoint, dtype=torch.float32)
    model = SLMRecon(lm, a.readout).to(device)
    n_slm = sum(v.numel() for v in model.backbone.parameters())
    n_all = sum(v.numel() for v in model.parameters())
    print(f"SLM backbone {n_slm/1e6:.2f}M, adapter+readout+head {(n_all-n_slm)/1e6:.2f}M, readout tokens {a.readout}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))

    def to_img(x):
        return x.permute(0, 3, 1, 2).float() / 255

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        sums = dict(mse=0.0, lpips=0.0, latent_mse=0.0, tok_mse=0.0, tok_lpips=0.0)
        hit = {k: {o: 0 for o in OBJECTS} for k in ("slm", "tok")}
        real_n = {o: 0 for o in OBJECTS}
        n = 0
        for s in range(0, len(val_idx), 256):
            idx = val_idx[s:s + 256]
            Z = windows(zv, fv, idx)
            x = to_img(xv[idx.cpu()].to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zhat, _ = model(Z)
                xh = decode(tok, zhat.float()).float()
                xr = decode(tok, Z).float()
            b = len(idx); n += b
            sums["mse"] += F.mse_loss(xh, x).item() * b
            sums["tok_mse"] += F.mse_loss(xr, x).item() * b
            sums["lpips"] += lpips(xh, x).item() * b
            sums["tok_lpips"] += lpips(xr, x).item() * b
            sums["latent_mse"] += F.mse_loss(zhat.float(), Z).item() * b
            real = classify_tiles(x, protos, index)[0]
            pred = {"slm": classify_tiles(xh, protos, index)[0], "tok": classify_tiles(xr, protos, index)[0]}
            for o in OBJECTS:
                c = names.index(o)
                r = real == c
                real_n[o] += int(r.sum())
                for k in pred:
                    hit[k][o] += int(((pred[k] == c) & r).sum())
        res = {k: v / n for k, v in sums.items()}
        res["psnr"] = float(10 * np.log10(1 / max(res["mse"], 1e-10)))
        res["tok_psnr"] = float(10 * np.log10(1 / max(res["tok_mse"], 1e-10)))
        for k in hit:
            for o in OBJECTS:
                res[f"{k}_recall_{o}"] = hit[k][o] / max(real_n[o], 1)
        res["real_tiles"] = real_n
        res["step"] = step
        if step % (a.eval_every * 5) == 0 or step == a.steps:
            sel = idx[:8]
            np.savez_compressed(out / f"examples_{step:06d}.npz", real=xv[sel.cpu()].numpy(),
                                tok=(xr[:8].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8),
                                slm=(xh[:8].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8))
        model.train()
        return res

    log = open(out / "metrics.jsonl", "a")
    best = None
    step, t0 = 0, time.time()
    run_loss = run_mse = run_lp = 0.0
    model.train()
    while step < a.steps:
        idx = torch.randint(0, len(zt), (a.batch,), device=device)
        Z = windows(zt, ft, idx)
        x = to_img(xt[idx.cpu()].to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zhat, _ = model(Z)
            xh = decode(tok, zhat).float()
        mse = F.mse_loss(xh, x)
        lp = lpips(xh, x)
        loss = mse + a.lpips_weight * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        run_loss += loss.item(); run_mse += mse.item(); run_lp += lp.item()
        if step % 100 == 0:
            print(f"step {step} loss {run_loss/100:.5f} mse {run_mse/100:.5f} lpips {run_lp/100:.4f} gnorm {gn.item():.2f} "
                  f"{step/(time.time()-t0):.2f} it/s", flush=True)
            run_loss = run_mse = run_lp = 0.0
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate(step)
            log.write(json.dumps(res) + "\n"); log.flush()
            print("[val]", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
            state = dict(model=model.state_dict(), step=step, args=vars(a), val=res)
            torch.save(state, out / "latest.pt")
            score = res["mse"] + a.lpips_weight * res["lpips"]
            if best is None or score < best:
                best = score
                torch.save(state, out / "best.pt")
    print("TRAIN_DONE", step, flush=True)


if __name__ == "__main__":
    main()
