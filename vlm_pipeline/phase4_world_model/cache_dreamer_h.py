#!/usr/bin/env python3
"""Agent-token outputs h_t of a Dreamer 4 phase-2 model (train_heads.py, patched with the task as text) for every frame of the
SLM train / val splits, the input of train_slm_reason_act_d4.py --dreamer-h.

For frame t of an episode the window is the one phase 2 trains on (world_model.window_frames / sample_clips): slots
t-3..t clamped at the episode start, slot 0 and repeated slots carry the start flag, a fresh slot j the action that led into
it; latents read at the context signal level (1 - tau_ctx) z without noise; agent tokens from the episode's task. Saved
fp16 [N, n_agent, d_model] in the frame order of the latent cache (z_train.pt / z_val.pt of the same tokenizer).

  python cache_dreamer_h.py --dyn <phase-1 dyn> --heads <phase-2>/checkpoints/latest.pt --raw-data <wood2200 raw> \
     --train <wood2200_v3> --val <wood2200_d4/val> --latent-cache <dir with z_*.pt> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
for k in ("--dyn", "--heads", "--raw-data", "--train", "--val", "--latent-cache", "--out"):
    p.add_argument(k, required=True)
p.add_argument("--repo", default="/w/repo")
p.add_argument("--batch", type=int, default=1024)
a = p.parse_args()
sys.path.insert(0, f"{a.repo}/vlm_pipeline/phase4_world_model")
from train_slm_world_model import load_episodes  # noqa: E402
from dreamer4.models.agent_heads import build_agent_heads, flatten_agent_out  # noqa: E402
from dreamer4.models.dynamics import _log2_int  # noqa: E402
from dreamer4.models.world_model import WorldModel  # noqa: E402

device = torch.device("cuda")
hck = torch.load(a.heads, map_location="cpu", weights_only=False)
world = WorldModel(a.dyn, device=device, n_agent=int(hck["n_agent"]))
world.load_dynamics_weights(hck["dyn"])
dyn = world.dyn.eval()
heads = build_agent_heads(hck["agent_cfg"]).to(device).eval()
heads.load_state_dict(hck["agent"])
names = world.dataset.manifest["task_names"]
k_max, tau = world.k_max, world.tau_ctx
sig = min(int(round((1.0 - tau) * k_max)), k_max)
episodes = load_episodes(Path(a.raw_data), None, None)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
info = {}
for tag, root in (("train", a.train), ("val", a.val)):
    root = Path(root)
    man = json.loads(((root / "manifest.json") if (root / "manifest.json").exists() else (root / "wood_manifest.json")).read_text())
    z = torch.load(Path(a.latent_cache) / f"z_{tag}.pt").to(device)
    gidx, acts, fresh, task = [], [], [], []
    first = 0
    for key, n in zip(man["episode_ids"], man["lengths"]):
        ep = episodes[int(key)]
        assert len(ep) == n
        g = names.index(ep[0]["instruction"])
        moves = [it["action"] for it in ep]
        for t in range(n):
            s = [max(0, t - 3 + j) for j in range(4)]
            gidx.append([first + x for x in s])
            fr = [False] + [s[j] != s[j - 1] for j in range(1, 4)]
            fresh.append(fr)
            acts.append([moves[s[j] - 1] if fr[j] else 17 for j in range(4)])   # 17 = start flag
            task.append(g)
        first += n
    assert first == len(z), (first, len(z))
    gidx, acts, task = (torch.tensor(x, device=device) for x in (gidx, acts, task))
    hs, exact = [], []
    with torch.no_grad():
        for s0 in range(0, len(gidx), a.batch):
            sl = slice(s0, s0 + a.batch)
            B = gidx[sl].shape[0]
            zw = z[gidx[sl]].float()
            av = F.one_hot(acts[sl], 18).float()
            st = torch.full((B, 4), _log2_int(k_max), device=device, dtype=torch.long)
            sg = torch.full((B, 4), sig, device=device, dtype=torch.long)
            ag = heads.tokens((B, 4), task_ids=task[sl].reshape(B, 1).expand(B, 4), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = dyn(av, st, sg, (1 - tau) * zw, agent_tokens=ag)
            h = o[-1][:, -1].float()                                              # (B, n_agent, d)
            hs.append(h.half())
            if hasattr(heads, "instr"):
                from dreamer4.models.agent_heads import instr_exact
                exact.append(instr_exact(heads, flatten_agent_out(h[:, None]), task[sl]) * B)
    H = torch.cat(hs).cpu()
    torch.save(H, out / f"h_{tag}.pt")
    info[tag] = dict(frames=len(H), shape=list(H.shape), instr_exact=sum(exact) / len(H) if exact else None)
    print(tag, info[tag], flush=True)
(out / "info.json").write_text(json.dumps(dict(dyn=a.dyn, heads=a.heads, step=hck.get("step"), **info), indent=1))
print("CACHE_DONE", flush=True)
