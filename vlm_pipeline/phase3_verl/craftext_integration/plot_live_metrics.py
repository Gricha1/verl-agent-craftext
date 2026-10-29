#!/usr/bin/env python3
"""Render compact training curves from a veRL console launcher log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"


def metric(line: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}:{NUMBER}", line)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: dict[int, dict[str, float]] = {}
    for raw_line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        step_match = re.search(r"\bstep:(\d+)\b", line)
        if not step_match or "critic/rewards/mean:" not in line:
            continue
        step = int(step_match.group(1))
        values = {
            "step": float(step),
            "reward_mean": metric(line, "critic/rewards/mean") or float("nan"),
            "reward_max": metric(line, "critic/rewards/max") or float("nan"),
            "reward_min": metric(line, "critic/rewards/min") or float("nan"),
            "entropy": metric(line, "actor/entropy") or float("nan"),
            "response_length": metric(line, "response_length/mean") or float("nan"),
            "response_clip_ratio": metric(line, "response_length/clip_ratio") or float("nan"),
        }
        rows[step] = values

    ordered = [rows[step] for step in sorted(rows)]
    if not ordered:
        raise SystemExit("No PPO metric rows found in log")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered[0].keys())
        writer.writeheader()
        writer.writerows(ordered)

    steps = [int(row["step"]) for row in ordered]
    rewards = [row["reward_mean"] for row in ordered]
    window = min(5, len(rewards))
    smooth = [sum(rewards[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(rewards))]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(steps, rewards, color="#8bb9e8", marker="o", ms=3, label="mean episodic reward")
    axes[0].plot(steps, smooth, color="#08519c", lw=2.2, label=f"moving mean ({window})")
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].set_ylabel("reward")
    axes[0].set_title("CrafText 8×8 PPO — reward during training")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(steps, [row["entropy"] for row in ordered], color="#7b3294", label="policy entropy")
    axes[1].plot(steps, [row["response_length"] for row in ordered], color="#d95f0e", label="mean response tokens")
    axes[1].plot(steps, [320 * row["response_clip_ratio"] for row in ordered], color="#1b9e77", label="320 × response clip ratio")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
