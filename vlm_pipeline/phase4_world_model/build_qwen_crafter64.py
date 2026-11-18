#!/usr/bin/env python3
"""Qwen-teacher Crafter trajectories (RLMM/zerohero-crafter-memory-20261106) as 64 x 64 episodes for Dreamer 4 and DreamerV3.

The dataset stores each step as a record (action, reward, done) plus lossless tile codes of the screen. Shards are time
slices: one shard holds ~11 consecutive steps of ~400 worlds, so episodes are joined across the shards of a run by
(run, world_seed, episode) and sorted by step. Screens are re-rendered with ZeroHero's codec at 64 x 64, which is what
stock Crafter gives an agent with size=(64, 64): a 63 x 63 view (9 x 9 tiles of 7 px), last row and column black.

Pass 1 reads the records of every shard and lays the episodes out; pass 2 decodes the screens of every shard straight into
the final memmap. Only episodes that start at step 0 and have no gaps are kept.

Output per split (train / val; val = world_seed % 50 == 0, about 2%):
  frames.npy      uint8 [N, 64, 64, 3]
  actions.npy     int32 [N]   action taken in frame t (0 = noop for the terminal frame)
  rewards.npy     float [N]   reward of the action in frame t (Dreamer 4 adapter convention)
  terminals.npy   bool  [N]   the frame after t is a death (Dreamer 4 convention)
  rewards_in.npy  float [N]   reward that led to frame t (DreamerV3 convention)
  terminal_in.npy bool  [N]   frame t is a death (DreamerV3 convention)
  manifest.json + wood_manifest.json: starts, lengths, episode ids (keys), n_actions 17

  python build_qwen_crafter64.py --root <extracted archive> --code <ZeroHero code> --out <dir> --procs 48
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import lzma
import os
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ARGS = None


def read_records(shard):
    with tarfile.open(shard) as tar:
        names = {m.name.rsplit("/", 1)[-1]: m for m in tar.getmembers()}
        text = lzma.decompress(tar.extractfile(names["records.jsonl.xz"]).read()).decode()
        blob = tar.extractfile(names["screens.npz"]).read()
    return [json.loads(line) for line in text.splitlines()], blob


def run_of(shard):
    return shard.split("/workers/")[0].rsplit("/", 1)[-1]


def meta(shard):
    records, _ = read_records(shard)
    run = run_of(shard)
    return [(run, r["world_seed"], r["episode"], r["step"], r.get("action", -1), float(r.get("reward", 0.0)),
             bool(r.get("terminal", False)), bool(r.get("done", False))) for r in records]


def decode(job):
    shard, places = job              # places: list of (record index, split, global row)
    sys.path.insert(0, ARGS.code)
    from rezero_world import crafter_tile_codec as codec
    codec.SIZE = (64, 64)            # stock Crafter observation size; unit 64 // 9 = 7 px
    records, blob = read_records(shard)
    dec = codec.Decoder(blob)
    out = {s: np.load(Path(ARGS.out) / s / "frames.npy", mmap_mode="r+") for s in {p[1] for p in places}}
    for i, split, row in places:
        out[split][row] = dec.frame(records[i]["screen"])
    for m in out.values():
        m.flush()
    return len(places)


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=48)
    p.add_argument("--limit-shards", type=int, default=0, help="smoke: only the first N shards of each worker")
    ARGS = p.parse_args()
    shards = sorted(glob.glob(f"{ARGS.root}/data/memory_medium_20261105_clean/*/workers/*/shards/*.tar"))
    if ARGS.limit_shards:
        keep, seen = [], {}
        for s in shards:
            w = s.rsplit("/shards/", 1)[0]
            seen[w] = seen.get(w, 0) + 1
            if seen[w] <= ARGS.limit_shards:
                keep.append(s)
        shards = keep
    t0 = time.time()
    with Pool(ARGS.procs) as pool:
        metas = pool.map(meta, shards, chunksize=1)
    print("pass 1:", len(shards), "shards,", sum(map(len, metas)), "records,", round(time.time() - t0), "s", flush=True)

    steps = {}                       # key -> {step: (shard index, record index, action, reward, terminal, done)}
    for si, rows in enumerate(metas):
        for ri, (run, world, ep, step, action, reward, terminal, done) in enumerate(rows):
            steps.setdefault((run, world, ep), {})[step] = (si, ri, action, reward, terminal, done)
    kept, dropped = {"train": [], "val": []}, 0
    for key in sorted(steps):
        st = steps[key]
        if min(st) != 0 or max(st) != len(st) - 1 or len(st) < 2:
            dropped += 1
            continue
        kept["val" if key[1] % 50 == 0 else "train"].append(key)
    print("episodes kept", {k: len(v) for k, v in kept.items()}, "dropped (gap or not from step 0)", dropped, flush=True)

    jobs = {}
    for split, keys in kept.items():
        root = Path(ARGS.out) / split
        root.mkdir(parents=True, exist_ok=True)
        lengths = [len(steps[k]) for k in keys]
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(int).tolist()
        N = int(sum(lengths))
        np.lib.format.open_memmap(root / "frames.npy", "w+", np.uint8, (N, 64, 64, 3)).flush()
        actions, rewards, terminals = np.zeros(N, np.int32), np.zeros(N, np.float32), np.zeros(N, bool)
        rewards_in, terminal_in = np.zeros(N, np.float32), np.zeros(N, bool)
        complete = 0
        for key, first in zip(keys, starts):
            st = steps[key]
            L = len(st)
            for t in range(L):
                si, ri, action, reward, terminal, done = st[t]
                row = first + t
                actions[row] = max(action, 0)
                terminal_in[row] = terminal
                if t + 1 < L:
                    rewards[row] = reward
                    rewards_in[row + 1] = reward
                    terminals[row] = st[t + 1][4]
                jobs.setdefault(si, []).append((ri, split, row))
            complete += st[L - 1][4] or st[L - 1][5]
        for name, arr in (("actions", actions), ("rewards", rewards), ("terminals", terminals),
                          ("rewards_in", rewards_in), ("terminal_in", terminal_in)):
            np.save(root / f"{name}.npy", arr)
        ids = ["/".join(map(str, k)) for k in keys]
        man = dict(starts=starts, lengths=lengths, episode_ids=ids, n_actions=17)
        (root / "manifest.json").write_text(json.dumps(man))
        (root / "wood_manifest.json").write_text(json.dumps(man))
        print(split, "episodes", len(keys), "complete (death or done)", int(complete), "frames", N,
              "reward sum", round(float(rewards.sum()), 1), "deaths", int(terminal_in.sum()), flush=True)

    t0 = time.time()
    done = 0
    with Pool(ARGS.procs) as pool:
        for n in pool.imap_unordered(decode, [(shards[si], places) for si, places in sorted(jobs.items())], chunksize=1):
            done += n
            if done % 200000 < n:
                print("decoded", done, round(time.time() - t0), "s", flush=True)
    print("pass 2 decoded", done, "frames in", round(time.time() - t0), "s", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
