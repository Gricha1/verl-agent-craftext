#!/usr/bin/env python3
"""Merge teacher collections (collect_flax_teacher_trajectories.py outputs) into one collection directory.

One task set, one teacher: every source must have the same craftext_settings and teacher_checkpoint. Episode ids
must not overlap (collect with --episode-id-offset). Frames are hard-linked into <out>/images (same filesystem,
nothing is copied); train.parquet / val.parquet are the concatenated rows of the sources.

  python merge_collections.py --sources <collection> <collection> ... --out <merged collection>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifests = [json.loads((source / "manifest.json").read_text()) for source in args.sources]
    for key in ("craftext_settings", "teacher_checkpoint", "max_steps", "collection_policy"):
        values = {str(m.get(key)) for m in manifests}
        if len(values) != 1:
            raise ValueError(f"sources differ in {key}: {values}")
    episodes = [t["episode_id"] for m in manifests for t in m["trajectories"]]
    if len(episodes) != len(set(episodes)):
        raise ValueError("episode ids overlap between sources")
    if args.out.exists():
        raise FileExistsError(args.out)
    images = args.out / "images"
    images.mkdir(parents=True)
    tables = {"train": [], "val": []}
    for source in args.sources:
        for name in os.listdir(source / "images"):
            os.link(source / "images" / name, images / name)
        for split in tables:
            tables[split].append(pq.read_table(source / f"{split}.parquet"))
    for split, parts in tables.items():
        pq.write_table(pa.concat_tables(parts), args.out / f"{split}.parquet")
    first = manifests[0]
    manifest = dict(first, sources=[str(s) for s in args.sources], seeds=[s for m in manifests for s in m["seeds"]],
                    number_of_states=sum(m["number_of_states"] for m in manifests),
                    train_states=sum(m["train_states"] for m in manifests), val_states=sum(m["val_states"] for m in manifests),
                    trajectories=[t for m in manifests for t in m["trajectories"]])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("number_of_states", "train_states", "val_states")}
                     | dict(episodes=len(episodes), images=len(os.listdir(images)))), flush=True)


if __name__ == "__main__":
    main()
