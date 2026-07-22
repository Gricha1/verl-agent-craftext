#!/usr/bin/env python3
"""Poll nvidia-smi and log per-GPU utilization and memory to console and Comet ML."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional


def query_gpus() -> List[Dict[str, float | int | str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpus: List[Dict[str, float | int | str]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, util, mem_used, mem_total = parts[:5]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "utilization_pct": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
            }
        )
    return gpus


def format_console_line(gpus: List[Dict[str, float | int | str]], elapsed_sec: float) -> str:
    parts = []
    for gpu in gpus:
        used_gb = float(gpu["memory_used_mb"]) / 1024.0
        total_gb = float(gpu["memory_total_mb"]) / 1024.0
        parts.append(
            f"GPU{gpu['index']}: util={float(gpu['utilization_pct']):.0f}% "
            f"mem={used_gb:.1f}/{total_gb:.1f} GB"
        )
    return f"[gpu_profiler t={elapsed_sec:.0f}s] " + " | ".join(parts)


def metrics_from_gpus(gpus: List[Dict[str, float | int | str]], elapsed_sec: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {"profiler/elapsed_sec": elapsed_sec}
    total_used_gb = 0.0
    util_sum = 0.0
    for gpu in gpus:
        idx = int(gpu["index"])
        used_gb = float(gpu["memory_used_mb"]) / 1024.0
        total_gb = float(gpu["memory_total_mb"]) / 1024.0
        util = float(gpu["utilization_pct"])
        metrics[f"gpu/{idx}/utilization_pct"] = util
        metrics[f"gpu/{idx}/memory_used_gb"] = used_gb
        metrics[f"gpu/{idx}/memory_total_gb"] = total_gb
        if total_gb > 0:
            metrics[f"gpu/{idx}/memory_used_pct"] = 100.0 * used_gb / total_gb
        total_used_gb += used_gb
        util_sum += util
    if gpus:
        metrics["gpu/mean_utilization_pct"] = util_sum / len(gpus)
        metrics["gpu/total_memory_used_gb"] = total_used_gb
    return metrics


def read_experiment_key(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        key = f.read().strip()
    return key or None


def attach_comet_experiment(
    api_key: Optional[str],
    workspace: Optional[str],
    experiment_key: str,
):
    from comet_ml import ExistingExperiment

    kwargs = {"previous_experiment": experiment_key}
    if api_key:
        kwargs["api_key"] = api_key
    if workspace:
        kwargs["workspace"] = workspace
    return ExistingExperiment(**kwargs)


def create_comet_experiment(
    api_key: Optional[str],
    workspace: Optional[str],
    project_name: str,
    experiment_name: str,
):
    from comet_ml import Experiment

    return Experiment(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        experiment_name=experiment_name,
        auto_param_logging=False,
        auto_metric_logging=False,
        display_summary_level=0,
    )


def wait_for_comet_experiment(
    key_file: str,
    wait_sec: float,
    api_key: Optional[str],
    workspace: Optional[str],
    project_name: str,
    experiment_name: str,
    poll_sec: float = 2.0,
):
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        key = read_experiment_key(key_file)
        if key:
            try:
                experiment = attach_comet_experiment(api_key, workspace, key)
                print(
                    f"[gpu_profiler] attached to Comet experiment via key file: {key_file}",
                    flush=True,
                )
                return experiment
            except Exception as exc:
                print(f"[gpu_profiler] failed to attach to Comet experiment: {exc}", flush=True)
        time.sleep(poll_sec)

    if experiment_name:
        try:
            experiment = create_comet_experiment(
                api_key=api_key,
                workspace=workspace,
                project_name=project_name,
                experiment_name=f"{experiment_name}_gpu_profiler",
            )
            print(
                "[gpu_profiler] created standalone Comet experiment "
                f"{experiment_name}_gpu_profiler",
                flush=True,
            )
            return experiment
        except Exception as exc:
            print(f"[gpu_profiler] failed to create Comet experiment: {exc}", flush=True)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU utilization profiler via nvidia-smi")
    parser.add_argument("--interval", type=float, default=float(os.environ.get("GPU_PROFILER_INTERVAL_SEC", "5")))
    parser.add_argument("--comet-project", default=os.environ.get("COMET_PROJECT_NAME", "verl_agent_alfworld"))
    parser.add_argument("--comet-experiment", default=os.environ.get("RUN_NAME", ""))
    parser.add_argument(
        "--comet-key-file",
        default=os.environ.get(
            "COMET_EXPERIMENT_KEY_FILE",
            os.path.join(os.environ.get("RAY_TEMP_DIR", "/tmp/ray_temp"), "comet_experiment_key.txt"),
        ),
    )
    parser.add_argument("--comet-wait-sec", type=float, default=float(os.environ.get("GPU_PROFILER_COMET_WAIT_SEC", "600")))
    parser.add_argument("--no-comet", action="store_true")
    args = parser.parse_args()

    if args.interval <= 0:
        print("[gpu_profiler] interval must be > 0", file=sys.stderr)
        return 1

    try:
        subprocess.run(["nvidia-smi", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[gpu_profiler] nvidia-smi not available", file=sys.stderr)
        return 1

    api_key = os.environ.get("COMET_API_KEY")
    workspace = os.environ.get("COMET_WORKSPACE")
    experiment = None
    if not args.no_comet and api_key:
        experiment = wait_for_comet_experiment(
            key_file=args.comet_key_file,
            wait_sec=args.comet_wait_sec,
            api_key=api_key,
            workspace=workspace,
            project_name=args.comet_project,
            experiment_name=args.comet_experiment,
        )
    elif not args.no_comet:
        print("[gpu_profiler] COMET_API_KEY not set, console-only mode", flush=True)

    start = time.time()
    sample_idx = 0
    print(
        f"[gpu_profiler] started (interval={args.interval}s, comet={'yes' if experiment else 'no'})",
        flush=True,
    )

    try:
        while True:
            elapsed = time.time() - start
            gpus = query_gpus()
            print(format_console_line(gpus, elapsed), flush=True)

            if experiment is not None:
                metrics = metrics_from_gpus(gpus, elapsed)
                for key, value in metrics.items():
                    experiment.log_metric(key, value, step=sample_idx)
                sample_idx += 1
            elif not args.no_comet and api_key and experiment is None:
                key = read_experiment_key(args.comet_key_file)
                if key:
                    try:
                        experiment = attach_comet_experiment(api_key, workspace, key)
                        print(
                            f"[gpu_profiler] attached to Comet experiment late via {args.comet_key_file}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[gpu_profiler] late Comet attach failed: {exc}", flush=True)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[gpu_profiler] stopped", flush=True)
    finally:
        if experiment is not None and hasattr(experiment, "end"):
            experiment.end()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
