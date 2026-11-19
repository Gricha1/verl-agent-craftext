#!/usr/bin/env python3
"""Replay of the online Dyna loop (dyna_loop.sh) for Dreamer 4 phase 2: the episodes the planner played (eval_slm_d4_env.py
--save-traj dirs) in front of the wood2200 native dataset (build_wood_d4_heads.py native), one WoodEpisodeDataset.

The online episodes are repeated r times, r = max(1, round(share / (1 - share) * wood2200 frames / online frames)), so that
about --share of the sampled windows come from the policy's own states (128 episodes of ~100 steps are 4% of wood2200's
290k frames). They go FIRST: train_heads.py holds out the LAST --val_frac of the episodes, which stay wood2200 episodes,
and its 50 % demonstration half clones only bc = 1 episodes (successful ones, teacher or planner).

  python build_online_d4.py <wood2200 native dir> <out dir> <traj dir> [<traj dir> ...] [--share 0.3]

Qwen-teacher Crafter data (crafter_dyna_loop.sh): the base is a build_qwen_crafter64.py dir (manifest without task names,
terminal rule, tasks or bc: one task, deaths in terminals.npy); --base-episodes N takes N random base episodes (3M frames
would cost phase 2 an hour of encoding per iteration). Base episodes then get task 0 and bc 1 (the teacher), and the
merged manifest keeps the online parts' task names and the "native" rule (terminals as recorded).
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("base")
p.add_argument("out")
p.add_argument("traj", nargs="+")
p.add_argument("--share", type=float, default=0.3)
p.add_argument("--max-repeat", type=int, default=20)
p.add_argument("--base-episodes", type=int, default=0, help="N > 0: a random subset of N base episodes")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
base, out = Path(a.base), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
bm = json.loads((base / "wood_manifest.json").read_text())
parts = [(Path(d), json.loads((Path(d) / "wood_manifest.json").read_text())) for d in a.traj]
qwen = "task_names" not in bm
if qwen:   # build_qwen_crafter64.py base: one task, the teacher's episodes are all demonstrations
    bm = dict(bm, task_names=parts[0][1]["task_names"], terminal_rule="native", tasks=[0] * len(bm["lengths"]),
              bc=[1] * len(bm["lengths"]))
for d, m in parts:
    assert m["task_names"] == bm["task_names"], (d, m["task_names"], bm["task_names"])
    assert m["terminal_rule"] == bm["terminal_rule"] == "native", (d, m["terminal_rule"], bm["terminal_rule"])
base_pick = list(range(len(bm["lengths"])))
if a.base_episodes:
    base_pick = sorted(np.random.default_rng(a.seed).choice(len(base_pick), min(a.base_episodes, len(base_pick)), replace=False).tolist())
n_base = sum(bm["lengths"][i] for i in base_pick); n_on = sum(sum(m["lengths"]) for _, m in parts)
r = min(a.max_repeat, max(1, round(a.share / (1 - a.share) * n_base / n_on)))
order = [(d, m) for _ in range(r) for d, m in parts] + [(base, bm)]
N = n_on * r + n_base
frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8, shape=(N, 64, 64, 3))
acts, rews, terms = np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, bool)
starts, lengths, ids, tasks, bc = [], [], [], [], []
pos = 0
for d, m in order:
    if d == base and len(base_pick) < len(bm["lengths"]):   # base subset: copy its episodes one by one
        f = np.load(d / "frames.npy", mmap_mode="r")
        A, Rw, T = np.load(d / "actions.npy", mmap_mode="r"), np.load(d / "rewards.npy", mmap_mode="r"), np.load(d / "terminals.npy", mmap_mode="r")
        for i in base_pick:
            s0, n = bm["starts"][i], bm["lengths"][i]
            frames[pos:pos + n] = f[s0:s0 + n]; acts[pos:pos + n] = A[s0:s0 + n]; rews[pos:pos + n] = Rw[s0:s0 + n]
            terms[pos:pos + n] = T[s0:s0 + n]
            starts.append(pos); lengths.append(n); ids.append(bm["episode_ids"][i]); tasks.append(bm["tasks"][i]); bc.append(bm["bc"][i])
            pos += n
        continue
    f = np.load(d / "frames.npy", mmap_mode="r")
    n = len(f)
    assert n == sum(m["lengths"]) and f.shape[1:] == (64, 64, 3), (d, f.shape)
    for s in range(0, n, 20000):
        frames[pos + s:pos + min(n, s + 20000)] = f[s:s + 20000]
    acts[pos:pos + n] = np.load(d / "actions.npy"); rews[pos:pos + n] = np.load(d / "rewards.npy")
    terms[pos:pos + n] = np.load(d / "terminals.npy")
    starts += [pos + s for s in m["starts"]]; lengths += m["lengths"]; ids += m["episode_ids"]; tasks += m["tasks"]; bc += m["bc"]
    pos += n
frames.flush()
np.save(out / "actions.npy", acts); np.save(out / "rewards.npy", rews); np.save(out / "terminals.npy", terms)
(out / "wood_manifest.json").write_text(json.dumps(dict(
    starts=starts, lengths=lengths, episode_ids=ids, n_actions=17, tasks=tasks, bc=bc, task_names=bm["task_names"],
    terminal_rule="native", reward_rule=bm.get("reward_rule", ""),
    online=dict(parts=[str(d) for d, _ in parts], repeat=r, online_frames=n_on, base_frames=n_base, share=n_on * r / N,
                base_episodes=len(base_pick)))))
print(f"online {len(parts)} dirs {n_on} frames x{r} + base {n_base} -> {N} frames, {len(lengths)} episodes, "
      f"online share {n_on * r / N:.3f}, successes {sum(bc)}", flush=True)
