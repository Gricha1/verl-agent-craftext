#!/usr/bin/env python3
"""Create one exhaustive scalar-metric dashboard and a wide CSV from veRL logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([A-Za-z][A-Za-z0-9_./-]*):([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # A dict makes a later duplicate log line for the same update overwrite an
    # incomplete earlier rendering of tqdm output.
    rows: dict[int, dict[str, float]] = {}
    for raw_line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        step_match = re.search(r"\bstep:(\d+)\b", line)
        if not step_match or "training/global_step:" not in line:
            continue
        step = int(step_match.group(1))
        values = {name: float(value) for name, value in PAIR.findall(line)}
        values["step"] = float(step)
        rows[step] = values

    if not rows:
        raise SystemExit("No completed PPO metric rows found")

    ordered_steps = sorted(rows)
    metric_names = sorted({name for row in rows.values() for name in row if name != "step"})
    table = [rows[step] for step in ordered_steps]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *metric_names])
        writer.writeheader()
        for row in table:
            writer.writerow({name: row.get(name, "") for name in writer.fieldnames})

    columns = 5
    panels = len(metric_names)
    rows_count = math.ceil(panels / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(20, max(5, rows_count * 2.45)), squeeze=False)
    fig.suptitle(f"CrafText PPO — all veRL scalar metrics ({len(ordered_steps)} completed updates)", fontsize=17)

    for axis, name in zip(axes.flat, metric_names, strict=False):
        y_values = [rows[step].get(name, float("nan")) for step in ordered_steps]
        axis.plot(ordered_steps, y_values, color="#1769aa", linewidth=1.25, marker="o", markersize=1.8)
        axis.set_title(name, fontsize=8)
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.tick_params(labelsize=6)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))

    for axis in list(axes.flat)[panels:]:
        axis.set_visible(False)

    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("PPO update", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.992), h_pad=1.25, w_pad=0.8)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {panels} metrics x {len(ordered_steps)} updates to {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
