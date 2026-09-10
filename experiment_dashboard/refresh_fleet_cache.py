"""Probe GPU fleet from a machine that has SSH aliases (e.g. Windows + NetBird).

Writes experiments/runtime/fleet_cache.json for the dashboard on aicenteritl.
Usage (Windows):
  python experiment_dashboard/refresh_fleet_cache.py
  # or after SSH tunnel / with remote write:
  python experiment_dashboard/refresh_fleet_cache.py --scp aicenteritl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "runtime" / "fleet_cache.json"

HOSTS = [
    {"id": "aicenteritl", "ssh": "aicenteritl", "label": "aicenteritl"},
    {"id": "aicenter1", "ssh": "aicenter1", "label": "aicenter1"},
    {"id": "aicenter2", "ssh": "aicenter2", "label": "aicenter2"},
    {"id": "aicenter3", "ssh": "aicenter3", "label": "aicenter3"},
    {"id": "ml3", "ssh": "ml3", "label": "ml3"},
    {"id": "ml4", "ssh": "ml4", "label": "ml4"},
    {"id": "h200", "ssh": "h200", "label": "h200"},
]

GPU_CMD = (
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu "
    "--format=csv,noheader,nounits; echo '---PROCS---'; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader"
)


def _parse_gpus(text: str) -> list[dict]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            used, total = float(parts[2]), float(parts[3])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                    "memory_free_mb": max(0.0, total - used),
                    "util_pct": float(parts[4]),
                    "temp_c": float(parts[5]) if parts[5] else None,
                    "busy": used > 1024 or float(parts[4]) > 5,
                }
            )
        except ValueError:
            continue
    return gpus


def _parse_procs(text: str) -> list[dict]:
    procs = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            name = parts[2]
            procs.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "name": name,
                    "used_memory": parts[3],
                    "foreign": "gorbov" not in name.lower(),
                }
            )
    return procs


def probe(ssh: str) -> dict:
    try:
        p = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=6",
                "-o",
                "StrictHostKeyChecking=accept-new",
                ssh,
                GPU_CMD,
            ],
            capture_output=True,
            text=True,
            timeout=14,
        )
        if p.returncode != 0:
            return {
                "ok": False,
                "error": ((p.stderr or p.stdout) or f"rc={p.returncode}")[:240],
                "gpus": [],
                "processes": [],
                "any_busy": False,
                "foreign_jobs_suspected": False,
            }
        gpu_part, _, proc_part = p.stdout.partition("---PROCS---")
        gpus = _parse_gpus(gpu_part)
        procs = _parse_procs(proc_part)
        return {
            "ok": True,
            "error": None,
            "gpus": gpus,
            "processes": procs,
            "any_busy": any(g.get("busy") for g in gpus),
            "foreign_jobs_suspected": any(x.get("foreign") for x in procs),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(e)[:240],
            "gpus": [],
            "processes": [],
            "any_busy": False,
            "foreign_jobs_suspected": False,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scp", default="", help="scp cache to host: e.g. aicenteritl")
    args = ap.parse_args()
    servers = []

    def one(h):
        r = probe(h["ssh"])
        return {"id": h["id"], "label": h["label"], "ssh": h["ssh"], **r}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        servers = list(pool.map(one, HOSTS))
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "refresh_fleet_cache.py",
        "elapsed_sec": round(time.time() - t0, 2),
        "servers": servers,
        "any_free_host": any(s.get("ok") and s.get("gpus") and not s.get("any_busy") for s in servers),
        "host_count": len(servers),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)
    for s in servers:
        st = "OK" if s.get("ok") else "FAIL"
        mem = ""
        if s.get("gpus"):
            g0 = s["gpus"][0]
            mem = f" gpu0={int(g0['memory_used_mb'])}/{int(g0['memory_total_mb'])} util={g0['util_pct']}%"
        print(f"  {s['id']}: {st}{mem} {(s.get('error') or '')[:60]}")
    if args.scp:
        remote = f"{args.scp}:/home/gorbov_gv/safe_rl_nlp/experiments/runtime/fleet_cache.json"
        subprocess.run(["scp", "-o", "BatchMode=yes", str(OUT), remote], check=False)
        print("scp ->", remote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
