#!/usr/bin/env python3
"""Write the immutable fixed-seed contract used by CrafText validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    artifact = {
        "format": "craftext-jax-vector-validation-v1",
        "base_seed": args.base_seed,
        "episodes": args.episodes,
        "lane_ids": list(range(args.episodes)),
        "seed_policy": "A fresh JAX vector environment is reset from PRNGKey(base_seed) for every validation. "
        "Lane identities are the deterministic jax.random.split order 0..episodes-1.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
