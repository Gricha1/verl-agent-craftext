"""Create the tiny prompt dataset used by the isolated CrafText PPO smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path


def make_rows(count: int) -> list[dict]:
    return [
        {
            # The custom agent loop obtains the actual task/observation from
            # CrafText.  This seed prompt only supplies the regular veRL data
            # contract (uid/index/data_source) to the trainer.
            "prompt": [{"role": "user", "content": "Start one CrafText episode."}],
            "data_source": "craftext_debug_square_8x8",
            "extra_info": {"index": index},
        }
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", default=4, type=int)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")

    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(make_rows(args.rows)), args.output)
    print(f"wrote {args.rows} CrafText smoke prompts to {args.output}")


if __name__ == "__main__":
    main()
