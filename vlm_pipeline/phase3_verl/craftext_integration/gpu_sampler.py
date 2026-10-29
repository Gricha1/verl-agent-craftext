#!/usr/bin/env python3
"""Low-overhead whole-run GPU sampler for an isolated benchmark.

Run this on the Docker host before the training container starts.  It writes a
portable CSV once per interval and does not attach to, pause, or profile any
CUDA process.  One row is emitted per GPU, with all compute PIDs using it.
"""
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


RUNNING = True


def _stop(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def _query(fields: str, *, applications: bool = False) -> list[list[str]]:
    command = ["nvidia-smi", f"--query-{'compute-apps' if applications else 'gpu'}={fields}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU ids")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    selected = {item.strip() for item in args.gpu_ids.split(",") if item.strip()}
    if not selected:
        raise SystemExit("--gpu-ids must not be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    columns = [
        "timestamp_utc",
        "gpu_id",
        "gpu_uuid",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "pstate",
        "compute_pids",
        "process_names",
        "process_memory_mib",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        while RUNNING:
            started = time.monotonic()
            timestamp = datetime.now(UTC).isoformat()
            try:
                gpu_rows = _query("index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,pstate")
                app_rows = _query("gpu_uuid,pid,process_name,used_memory", applications=True)
                apps: dict[str, list[list[str]]] = {}
                for row in app_rows:
                    if len(row) == 4:
                        apps.setdefault(row[0].strip(), []).append(row)
                for row in gpu_rows:
                    if len(row) != 8 or row[0].strip() not in selected:
                        continue
                    processes = apps.get(row[1].strip(), [])
                    sample = dict(zip(columns[:9], [timestamp, *[value.strip() for value in row]], strict=True))
                    sample.update(
                        compute_pids=";".join(process[1].strip() for process in processes),
                        process_names=";".join(process[2].strip() for process in processes),
                        process_memory_mib=";".join(process[3].strip() for process in processes),
                    )
                    writer.writerow(sample)
                file.flush()
            except (OSError, subprocess.CalledProcessError) as error:
                writer.writerow({"timestamp_utc": timestamp, "gpu_id": "sampler_error", "process_names": str(error)})
                file.flush()
            time.sleep(max(0.0, args.interval_seconds - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
