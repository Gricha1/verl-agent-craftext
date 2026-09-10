#!/usr/bin/env python3
"""Import ALFWorld Comet runs into experiments/registry.yaml (curated).

Usage on aicenteritl:
  export COMET_API_KEY=...
  python3 experiment_dashboard/import_alfworld_comet.py --apply

Default is dry-run.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import list_alfworld_from_comet  # noqa: E402
from registry_store import get_experiment, upsert_experiment  # noqa: E402

KEYWORDS = [
    (r"dual", "dual_llm"),
    (r"actor.?value|actor_value", "actor_value"),
    (r"entrop", "entropy"),
    (r"lora", "lora"),
    (r"baseline|final", "baseline"),
]


def classify(name: str) -> tuple[str, list[str]]:
    n = (name or "").lower()
    tags = ["alfworld", "imported"]
    arch = "unknown"
    for pat, tag in KEYWORDS:
        if re.search(pat, n, re.I):
            tags.append(tag)
            if tag in ("dual_llm", "actor_value"):
                arch = tag
    return arch, tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    data = list_alfworld_from_comet(limit=args.limit)
    if data.get("error"):
        print("ERROR:", data["error"])
        return 1
    rows = data.get("experiments") or []
    print(f"fetched {len(rows)} alfworld runs")
    # Prefer names that look meaningful
    scored = []
    for r in rows:
        name = r.get("name") or ""
        score = 0
        for pat, _ in KEYWORDS:
            if re.search(pat, name, re.I):
                score += 1
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("name") or "")))
    chosen = [r for s, r in scored if s > 0][:15] or [r for _, r in scored[:10]]
    print(f"curating {len(chosen)} runs")
    for r in chosen:
        key = r.get("comet_key")
        if not key:
            continue
        eid = f"alf_{str(key)[:8]}"
        arch, tags = classify(r.get("name") or "")
        exp = {
            "id": eid,
            "name": r.get("name") or eid,
            "comet_project": "gregory-gorbov/verl-agent-alfworld",
            "comet_key": key,
            "comet_url": r.get("url"),
            "environment": "alfworld",
            "architecture": arch,
            "actor_value": arch == "actor_value",
            "reward_mode": "sparse_R",
            "gamma": 1.0,
            "lambda": 1.0,
            "status": "inconclusive",
            "notes": "Auto-imported from Comet; curate status/metrics manually.",
            "hypothesis": ["H3"],
            "parent_experiment": None,
            "tags": tags,
        }
        print("-", eid, r.get("name"), arch)
        if args.apply:
            if get_experiment(eid):
                print("  exists, updating")
            upsert_experiment(exp)
    if args.apply:
        # drop placeholder if real imports exist
        ph = get_experiment("alfworld_import_pending")
        if ph and chosen:
            ph["status"] = "aborted"
            ph["notes"] = (ph.get("notes") or "") + "\nReplaced by imported alf_* entries."
            upsert_experiment(ph)
            print("marked alfworld_import_pending aborted")
    else:
        print("dry-run only; pass --apply to write registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
