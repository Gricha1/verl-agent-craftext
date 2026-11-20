#!/usr/bin/env python3
"""Dyna loop step 2b (user, 2026-11-20: "размораживаем энкодер декодер у дримера"): finetune the Dreamer 4 tokenizer
(encoder AND decoder) on the frames the planner met in the environment, so that states the 2200 demonstration episodes
never showed get their own latents instead of being squeezed into the dataset's latent space.

Batches are half online clips (all --online traj dirs, the last --val-frac of each dir's episodes held out) and half
base clips (the BC dataset, --base). Loss = the tokenizer's own phase-1 objective (patch MSE + 0.2 LPIPS, RMS-normalised
by dreamer4's LossCombiner, same as train_tokenizer.py on wood2200) + --anchor x relative latent drift on the BASE clips:
    anchor = || z_new(x) - z_ref(x) ||^2 / Var(z_ref),   x = base clip, z_ref = the frozen --ref tokenizer (the phase-1 one)
The anchor keeps the dataset latents where the dynamics, the SLM (it reads z) and the frozen SLM critic were trained, while
the new states are free to move. Clips are 4 frames (the history the dynamics' FrozenTokenizer encodes with), 63 x 63.

Outputs in --out:
  latest.pt     the tokenizer checkpoint (train_tokenizer.py format, the new weights in "model", no EMA / optimiser)
  dyn_tok.pt    a copy of --dyn whose config points at latest.pt: pass it as --dyn to train_heads.py, eval_slm_d4_env.py and
                train_slm_imagine_grpo.py, and they all encode with the finetuned tokenizer (dynamics weights unchanged;
                train_heads refits them on the new latents)
  tok_ft.json   held-out PSNR / inventory-cell errors / latent drift before and after, online and base

  python finetune_d4_tokenizer.py --tok <init ckpt> --ref <phase-1 ckpt> --dyn <dyn ckpt> --base <dataset dir>
         --online <traj dir>[,<traj dir>..] --out <dir> [--steps 3000]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def episodes(d):
    d = Path(d)
    mf = d / "wood_manifest.json" if (d / "wood_manifest.json").exists() else d / "manifest.json"
    m = json.loads(mf.read_text())
    return np.load(d / "frames.npy", mmap_mode="r"), m["starts"], m["lengths"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tok", default="", help="tokenizer checkpoint to start from (the previous iteration's; default: the one of --dyn)")
    p.add_argument("--ref", default="", help="anchor tokenizer (default: the one of --dyn, i.e. phase 1, so drift does not add up)")
    p.add_argument("--dyn", required=True, help="phase-1 dynamics checkpoint; dyn_tok.pt = this one pointed at the new tokenizer")
    p.add_argument("--base", required=True)
    p.add_argument("--online", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--online-frac", type=float, default=0.5)
    p.add_argument("--anchor", type=float, default=10.0)
    p.add_argument("--perceptual", type=float, default=0.2, help="LPIPS weight (0: MSE only)")
    p.add_argument("--base-frames", type=int, default=400000, help="random base episodes up to this many frames")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-clips", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from dreamer4.train.common import LossCombiner
    from dreamer4.train.perceptual import build_perceptual
    from dreamer4.train.train_tokenizer import load_tokenizer_checkpoint
    from dreamer4.models.transformer.transformer import patchify, unpatchify

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok0 = torch.load(a.dyn, map_location="cpu", weights_only=False)["config"]["tokenizer"]["ckpt"]   # phase-1 tokenizer
    a.tok = a.tok or tok0; a.ref = a.ref or tok0
    tok, ck = load_tokenizer_checkpoint(a.tok, device=dev)
    ref, _ = load_tokenizer_checkpoint(a.ref, device=dev)
    ref.requires_grad_(False)
    ps = ck["config"]["model"]["patch_size"]; Hh, Ww, C = ck["frame_meta"]["H"], ck["frame_meta"]["W"], ck["frame_meta"]["C"]
    W = 4

    # ---- data: whole episodes in RAM (uint8, cropped to the tokenizer's view) ----
    def load(dirs, max_frames=0):
        tr, va = [], []
        for d in dirs:
            f, st, ln = episodes(d)
            idx = [i for i in range(len(ln)) if ln[i] >= 1]
            rng.shuffle(idx)
            keep = idx
            if max_frames:
                keep, tot = [], 0
                for i in idx:
                    if tot >= max_frames:
                        break
                    keep.append(i); tot += ln[i]
            nv = max(1, int(round(a.val_frac * len(keep))))
            for j, i in enumerate(keep):
                x = np.ascontiguousarray(f[st[i]:st[i] + ln[i], :Hh, :Ww])
                (va if j < nv else tr).append(x)
        return tr, va
    on_tr, on_va = load(a.online.split(","))
    ba_tr, ba_va = load([a.base], a.base_frames)
    print(f"online episodes {len(on_tr)} train / {len(on_va)} val, {sum(map(len, on_tr))} frames; "
          f"base {len(ba_tr)} / {len(ba_va)}, {sum(map(len, ba_tr))} frames", flush=True)

    def clips(eps, n, r):
        """n random 4-frame clips [t-3..t] (the first frame repeated before the episode start) -> (n, 4, C, H, W) float."""
        lens = np.array([len(e) for e in eps], dtype=np.float64)
        ei = r.choice(len(eps), n, p=lens / lens.sum())
        out_ = np.empty((n, W, Hh, Ww, C), np.uint8)
        for k, e in enumerate(ei):
            t = int(r.integers(len(eps[e])))
            out_[k] = eps[e][np.maximum(np.arange(t - W + 1, t + 1), 0)]
        return torch.from_numpy(out_).to(dev).float().div_(255).permute(0, 1, 4, 2, 3)

    vr = np.random.default_rng(12345)
    val = {"online": clips(on_va, a.val_clips, vr), "base": clips(ba_va, a.val_clips, vr)}

    @torch.no_grad()
    def evaluate(model):
        model.eval(); res = {}
        for name, x in val.items():
            psnr, bad, drift, num, den = [], [], [], 0.0, 0.0
            for s in range(0, len(x), 256):
                pt = patchify(x[s:s + 256], ps)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model.encode(pt)
                    zr = ref.encode(pt)
                    y = model.decoder(z)[:, -1]
                img = unpatchify(y.float()[:, None], Hh, Ww, C=C, patch_size=ps)[:, 0].clamp(0, 1)   # (B, C, H, W)
                t = x[s:s + 256, -1]
                e2 = ((img - t) ** 2).mean((1, 2, 3))
                psnr += (10 * torch.log10(1 / e2.clamp_min(1e-10))).tolist()
                inv = (img[:, :, 49:63] - t[:, :, 49:63]).abs().mean(1)            # status bar: 2 x 9 cells of 7 x 7
                bad += (inv.reshape(-1, 2, 7, 9, 7).mean((2, 4)).amax((1, 2)) > 0.12).tolist()
                num += float(((z[:, -1].float() - zr[:, -1].float()) ** 2).sum()); den += float(zr[:, -1].float().var() * zr[:, -1].numel())
            pp = np.array(psnr)
            res[name] = dict(psnr=float(pp.mean()), psnr_p1=float(np.percentile(pp, 1)), below40=float((pp < 40).mean()),
                             inv_bad=float(np.mean(bad)), drift=num / max(den, 1e-12))
        model.train()
        return res

    before = evaluate(tok)
    print("TOK-FT before", json.dumps(before), flush=True)

    perceptual = build_perceptual(backbone="lpips" if a.perceptual > 0 else "none", weight=a.perceptual, up=128,
                                  lpips_net="alex", dino_weight=1.0, device=dev)
    weights = {"recon": 1.0}; weights.update({n: w for n, _, w in perceptual})
    comb = LossCombiner(weights, normalize=True, decay=0.99, floor_frac=0.2, device=dev)
    opt = torch.optim.AdamW(tok.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    with torch.no_grad():   # Var(z_ref) on base clips, the anchor's scale
        pt = patchify(clips(ba_tr, 512, rng), ps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zvar = float(ref.encode(pt)[:, -1].float().var())
    n_on = int(round(a.online_frac * a.batch)); n_ba = a.batch - n_on
    tok.train(); t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.cat([clips(on_tr, n_on, rng), clips(ba_tr, n_ba, rng)])
        pt = patchify(x, ps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = tok.encoder(pt)
            pred = tok.decoder(z)
            with torch.no_grad():
                zr = ref.encode(pt[n_on:])
            loss_recon = F.mse_loss(pred.float(), pt.float())
        terms = {"recon": loss_recon}
        if perceptual:
            pr = unpatchify(pred.float(), Hh, Ww, C=C, patch_size=ps).reshape(-1, C, Hh, Ww)
            gt = unpatchify(pt.float(), Hh, Ww, C=C, patch_size=ps).reshape(-1, C, Hh, Ww)
            for n, mod, _ in perceptual:
                terms[n] = mod(pr, gt)
        loss, raw = comb(terms)
        anchor = F.mse_loss(z[n_on:].float(), zr.float()) / zvar
        loss = loss + a.anchor * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step} loss {float(loss.detach()):.4f} recon {float(loss_recon):.2e} anchor {float(anchor):.4f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in raw.items() if k != "recon")
                  + f" gn {float(gn):.2f} lr {lr:.1e} {time.time() - t0:.0f}s", flush=True)
    after = evaluate(tok)
    print("TOK-FT after", json.dumps(after), flush=True)

    ck = dict(ck, model=tok.state_dict(), ema=None, step=int(ck.get("step", 0)),
              finetune=dict(args=vars(a), before=before, after=after))
    ck.pop("opt", None); ck.pop("loss_norm", None)
    torch.save(ck, out / "latest.pt")
    dk = torch.load(a.dyn, map_location="cpu", weights_only=False)
    dk["config"]["tokenizer"]["ckpt"] = str(out / "latest.pt"); dk["config"]["tokenizer"]["decoder_ckpt"] = ""
    dk["tokenizer_ckpt"] = str(out / "latest.pt")
    torch.save(dk, out / "dyn_tok.pt")
    (out / "tok_ft.json").write_text(json.dumps(dict(before=before, after=after, args=vars(a), minutes=(time.time() - t0) / 60), indent=1))
    print("TOK-FT-DONE", out, flush=True)


if __name__ == "__main__":
    main()
