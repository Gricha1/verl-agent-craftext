"""Probe GPU fleet from a machine that has SSH aliases (e.g. Windows + NetBird).

Writes experiments/runtime/fleet_cache.json for the dashboard on aicenteritl.
Usage (Windows):
  python experiment_dashboard/refresh_fleet_cache.py
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
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader; "
    "echo '---RAM---'; free -b"
)

# Host NVML broken on aicenter2 (driver/library mismatch); docker still sees GPUs.
DOCKER_GPU_CMD = (
    "docker exec openpangu-sync nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
    "utilization.gpu,temperature.gpu --format=csv,noheader,nounits; echo '---PROCS---'; "
    "docker exec openpangu-sync nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,"
    "used_gpu_memory --format=csv,noheader; echo '---RAM---'; free -b"
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


def _parse_ram(text: str) -> dict | None:
    for line in (text or "").splitlines():
        if not line.startswith("Mem:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            total = int(parts[1])
            used = int(parts[2])
            avail = int(parts[6]) if len(parts) > 6 else max(0, total - used)
            return {
                "total_bytes": total,
                "used_bytes": used,
                "available_bytes": avail,
                "used_pct": round(100.0 * used / total, 1) if total else 0.0,
            }
        except ValueError:
            return None
    return None


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


def _parse_payload(stdout: str) -> dict:
    gpu_part, _, rest = stdout.partition("---PROCS---")
    proc_part, _, ram_part = rest.partition("---RAM---")
    gpus = _parse_gpus(gpu_part)
    procs = _parse_procs(proc_part)
    ram = _parse_ram(ram_part)
    return {
        "ok": True,
        "error": None,
        "gpus": gpus,
        "processes": procs,
        "ram": ram,
        "any_busy": any(g.get("busy") for g in gpus),
        "foreign_jobs_suspected": any(x.get("foreign") for x in procs),
    }


def _ssh(ssh: str, remote: str, timeout: float = 18.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=6",
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh,
            remote,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def probe(ssh: str) -> dict:
    try:
        p = _ssh(ssh, GPU_CMD)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "Failed to initialize NVML" not in out:
            return _parse_payload(p.stdout)

        # aicenter2: host nvidia-smi broken after driver update; docker path works.
        if ssh == "aicenter2" or "Failed to initialize NVML" in out:
            p2 = _ssh(ssh, DOCKER_GPU_CMD, timeout=25.0)
            if p2.returncode == 0 and "Failed to initialize NVML" not in ((p2.stdout or "") + (p2.stderr or "")):
                parsed = _parse_payload(p2.stdout)
                parsed["probe_via"] = "docker"
                parsed["host_nvml_error"] = ((p.stderr or p.stdout) or "")[:160]
                return parsed

        return {
            "ok": False,
            "error": ((p.stderr or p.stdout) or f"rc={p.returncode}")[:240],
            "gpus": [],
            "processes": [],
            "ram": None,
            "any_busy": False,
            "foreign_jobs_suspected": False,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(e)[:240],
            "gpus": [],
            "processes": [],
            "ram": None,
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
        "elapsed_s": round(time.time() - t0, 2),
        "servers": servers,
        "any_free_host": any(s.get("ok") and s.get("gpus") and not s.get("any_busy") for s in servers),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for s in servers:
        if s.get("ok") and s.get("gpus"):
            g0 = s["gpus"][0]
            via = f" via={s['probe_via']}" if s.get("probe_via") else ""
            ram = s.get("ram") or {}
            ram_s = f" ram={ram.get('used_pct')}%" if ram else ""
            print(
                f"  {s['id']}: OK gpu0={int(g0['memory_used_mb'])}/{int(g0['memory_total_mb'])} "
                f"util={g0['util_pct']}%{ram_s}{via}"
            )
        else:
            print(f"  {s['id']}: FAIL {(s.get('error') or '')[:60]}")

    if args.scp:
        dest = f"{args.scp}:/home/gorbov_gv/safe_rl_nlp/experiments/runtime/fleet_cache.json"
        subprocess.run(["scp", "-o", "BatchMode=yes", str(OUT), dest], check=False)
        print(f"scp -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
