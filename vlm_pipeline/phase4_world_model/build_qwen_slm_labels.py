#!/usr/bin/env python3
"""Reasoning labels of the Qwen-teacher Crafter episodes (RLMM/zerohero-crafter-memory-20261106) for the SLM over the Dreamer 4
latents (train_slm_reason_act_d4.py --frame-labels), row-aligned with build_qwen_crafter64.py's frames.

Each record with a response holds the teacher's text "<reasoning>R</reasoning>\n<action>A</action>\n<memory>M</memory>".
A frame gets the label R when the parsed action A is the action the record executed and the record is not an invalid-action
fallback; the memory is dropped (the SLM has none). Rows follow the episode layout of <crafter64>/{train,val}/manifest.json
(episode id "run/world_seed/episode", row = start + step).

Output <out>/{train,val}.jsonl: {"row": int, "reasoning": str}, one line per labelled frame, and <out>/stats.json.

  python build_qwen_slm_labels.py --root <extracted archive> --crafter64 <build_qwen_crafter64 out> --out <dir> --procs 64
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import re
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

NAMES = ("Noop", "Move West", "Move East", "Move North", "Move South", "Do", "Sleep", "Place Stone", "Place Table",
         "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
         "Make Wood Sword", "Make Stone Sword", "Make Iron Sword")   # stock Crafter action order (= ZeroHero ACTIONS)
RE = re.compile(r"<reasoning>(.*?)</reasoning>\s*<action>(.*?)</action>", re.S)


def labels(shard):
    run = shard.split("/workers/")[0].rsplit("/", 1)[-1]
    with tarfile.open(shard) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("records.jsonl.xz"))
        text = lzma.decompress(tar.extractfile(member).read()).decode()
    out, why = [], dict(no_response=0, unparsed=0, other_action=0, fallback=0)
    for line in text.splitlines():
        r = json.loads(line)
        if not r.get("response"):
            why["no_response"] += 1; continue
        m = RE.search(r["response"])
        name = m.group(2).strip() if m else None
        if name not in NAMES:
            why["unparsed"] += 1; continue
        if r.get("invalid_fallback"):
            why["fallback"] += 1; continue
        if NAMES.index(name) != r.get("action"):
            why["other_action"] += 1; continue
        out.append((f"{run}/{r['world_seed']}/{r['episode']}", int(r["step"]), m.group(1).strip()))
    return out, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--crafter64", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=64)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    where = {}
    for split in ("train", "val"):
        man = json.loads((Path(a.crafter64) / split / "manifest.json").read_text())
        for key, s, n in zip(man["episode_ids"], man["starts"], man["lengths"]):
            where[key] = (split, s, n)
    shards = sorted(glob.glob(f"{a.root}/data/memory_medium_20261105_clean/*/workers/*/shards/*.tar"))
    t0 = time.time()
    rows = {"train": [], "val": []}
    why = dict(no_response=0, unparsed=0, other_action=0, fallback=0, no_episode=0)
    with Pool(a.procs) as pool:
        for got, w in pool.imap_unordered(labels, shards, chunksize=1):
            for k, v in w.items():
                why[k] += v
            for key, step, reasoning in got:
                if key not in where or step >= where[key][2]:
                    why["no_episode"] += 1; continue
                split, s, _ = where[key]
                rows[split].append((s + step, reasoning))
    stats = dict(shards=len(shards), seconds=round(time.time() - t0), skipped=why)
    for split, rs in rows.items():
        rs.sort()
        assert len({r for r, _ in rs}) == len(rs), (split, "duplicate rows")
        with open(out / f"{split}.jsonl", "w") as f:
            for row, reasoning in rs:
                f.write(json.dumps(dict(row=row, reasoning=reasoning)) + "\n")
        chars = sorted(len(r) for _, r in rs)
        stats[split] = dict(labelled=len(rs), reasoning_chars_mean=round(sum(chars) / max(len(chars), 1), 1),
                            reasoning_chars_p99=chars[int(.99 * len(chars))] if chars else 0, reasoning_chars_max=chars[-1] if chars else 0)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)
    print("LABELS_DONE", flush=True)


if __name__ == "__main__":
    main()
