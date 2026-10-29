"""Convert the frozen teacher-action parquet split to plain CE/SFT examples.

The collector contains state prompts and the CNN+BERT teacher's greedy action,
but it deliberately has no fabricated chain-of-thought.  We therefore train a
minimal, fixed non-empty reasoning wrapper and supervise the teacher action
with next-token cross entropy.  This is an action-imitation baseline, not a
claim that the fixed sentence is a faithful teacher rationale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def target(action: str) -> str:
    return (
        "<reasoning>I inspect the current state and select the next action.</reasoning>\n"
        f"<action>{action}</action>"
    )


def convert(source: Path, destination: Path) -> int:
    rows = pq.read_table(source).to_pylist()
    converted = []
    for row in rows:
        action = row["extra_info"]["teacher_action"]
        converted.append(
            {
                "messages": list(row["prompt"])
                + [{"role": "assistant", "content": target(str(action))}],
                "teacher_action": str(action),
                "source_extra_info": row["extra_info"],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), destination)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    train = convert(args.input_dir / "train.parquet", args.output_dir / "train.parquet")
    val = convert(args.input_dir / "val.parquet", args.output_dir / "val.parquet")
    print(f"Converted CE/SFT dataset: train={train} val={val}", flush=True)


if __name__ == "__main__":
    main()
