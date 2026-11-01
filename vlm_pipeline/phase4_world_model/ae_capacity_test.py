#!/usr/bin/env python3
"""Diagnostic only (not a pipeline model): can the world model's image nets reconstruct a frame at all?

The SLM world model (train_slm_world_model.py --z-kind categorical --image-net dreamer) does not draw trees even
when z comes from the real frame. This trains the same DreamerV3 encoder / decoder as plain autoencoders on the
same frames, without the SLM and without dynamics, in three variants:

  cat_nokl : frame -> encoder -> 32x32 categorical z (straight-through) -> decoder        (no KL)
  cat_kl   : the same + the world model's KL on z_0: 0.5 max(1, KL(sg q || p)) + 0.1 max(1, KL(q || sg p))
             against a learned prior (p(z | h = 0) of the world model)
  noz      : frame -> encoder (4096 features) -> decoder                                   (no z: decoder bound)

Loss = 1/2 sum over pixels (x^ - x)^2, as in the world model. Train frames: the 165 train episodes; validation:
the 35 validation episodes of the same split.

  python ae_capacity_test.py --raw-data <collection> --split-from <CE data dir> --out <dir> [--updates 20000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from train_slm_world_model import (PIXELS, categorical, categorical_kl_dims, dreamer_decoder, dreamer_encoder,
                                   load_episodes, read_image, save_json)

GROUPS, CLASSES = 32, 32
SHOW = [(9, 72 + d) for d in range(1, 9)] + [(93, 48 + d) for d in range(1, 9)]  # frames of the imagination figures


class AE:
    def __init__(self, variant, device):
        import torch
        import torch.nn as nn
        self.variant, self.torch = variant, torch
        modules = {"encoder": dreamer_encoder(nn)}
        if variant == "noz":
            modules["decoder"] = dreamer_decoder(nn, 256 * 4 * 4)
        else:
            modules["posterior"] = nn.Sequential(nn.Linear(256 * 4 * 4, 512), nn.GELU(), nn.Linear(512, GROUPS * CLASSES))
            modules["decoder"] = dreamer_decoder(nn, GROUPS * CLASSES)
            if variant == "cat_kl":
                modules["prior"] = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, GROUPS * CLASSES))
        self.net = nn.ModuleDict(modules).to(device)
        self.device = device

    def forward(self, images, sample=True):
        """Returns (reconstruction, KL loss, raw KL per frame)."""
        torch = self.torch
        embed = self.net["encoder"](images)
        zero = images.new_zeros(())
        if self.variant == "noz":
            return self.net["decoder"](embed), zero, zero
        q = categorical(self.net["posterior"](embed), GROUPS, CLASSES)
        probs = q[0]
        if sample:
            index = torch.multinomial(probs.flatten(0, 1), 1).view(probs.shape[:-1])
            onehot = torch.nn.functional.one_hot(index, CLASSES).to(probs.dtype)
            z = (onehot + probs - probs.detach()).flatten(1)
        else:
            z = torch.nn.functional.one_hot(probs.argmax(-1), CLASSES).to(probs.dtype).flatten(1)
        recon = self.net["decoder"](z)
        if self.variant != "cat_kl":
            return recon, zero, zero
        p = categorical(self.net["prior"](images.new_zeros(len(images), 512)), GROUPS, CLASSES)  # p(z | h = 0)
        dyn = categorical_kl_dims(probs.detach(), p[0]).sum(-1)
        rep = categorical_kl_dims(probs, p[0].detach()).sum(-1)
        kl = 0.5 * dyn.clamp_min(1.0).mean() + 0.1 * rep.clamp_min(1.0).mean()
        return recon, kl, dyn.detach().mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--split-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default="cat_nokl,cat_kl,noz")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="the world model's learning rate")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7970)
    args = parser.parse_args()
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = load_episodes(args.raw_data, None)
    valid_ids = {json.loads(line)["source_episode_id"] for line in (args.split_from / "valid.jsonl").open()}
    started = time.time()
    to_u8 = lambda name: (read_image(args.raw_data, name) * 255).round().astype(np.uint8)
    train = np.stack([to_u8(i["image"]) for k, e in episodes.items() if k not in valid_ids for i in e])
    valid = np.stack([to_u8(i["image"]) for k, e in episodes.items() if k in valid_ids for i in e])
    show = np.stack([to_u8(episodes[k][t]["image"]) for k, t in SHOW])
    print(json.dumps(dict(train_frames=len(train), validation_frames=len(valid), load_seconds=time.time() - started)), flush=True)
    as_float = lambda a: torch.from_numpy(a).to(device).float() / 255
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | dict(train_frames=len(train), validation_frames=len(valid), show=SHOW))
    for variant in args.variants.split(","):
        model = AE(variant, device)
        optimizer = torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=.01)
        log = open(args.out / f"{variant}_metrics.jsonl", "w")
        snapshots = {}
        t0 = time.time()
        for update in range(1, args.updates + 1):
            model.net.train()
            images = as_float(train[rng.integers(0, len(train), args.batch_size)])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                recon, kl, _ = model.forward(images)
            loss = .5 * (recon.float() - images).square().flatten(1).sum(-1).mean() + kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            optimizer.step()
            if update % args.eval_every == 0 or update == args.updates:
                model.net.eval()
                sq, kls = 0.0, []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    for start in range(0, len(valid), 256):
                        batch = as_float(valid[start:start + 256])
                        out, _, kl_raw = model.forward(batch, sample=False)
                        sq += float((out.float() - batch).square().mean(dim=(1, 2, 3)).sum())
                        kls.append(float(kl_raw))
                    shown = model.forward(as_float(show), sample=False)[0].float().clamp(0, 1).cpu().numpy()
                row = dict(variant=variant, update=update, frames_seen=update * args.batch_size,
                           validation_mse_per_pixel=sq / len(valid), validation_kl_raw=float(np.mean(kls)),
                           train_loss_per_pixel=float(loss.detach()) * 2 / PIXELS, seconds=time.time() - t0)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(json.dumps(row), flush=True)
                if update in (1000, 5000, args.updates):
                    snapshots[update] = (shown * 255).round().astype(np.uint8)
        log.close()
        np.savez_compressed(args.out / f"{variant}_recon.npz", real=show, **{f"update_{k}": v for k, v in snapshots.items()})
    save_json(args.out / "complete.json", dict(seconds=time.time() - started))


if __name__ == "__main__":
    main()
