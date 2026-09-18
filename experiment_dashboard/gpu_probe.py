"""GPU / docker status probes (local + SSH fleet). Never launches training."""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
HOSTS_CFG = Path(__file__).resolve().parent / "hosts.yaml"

DEFAULT_HOSTS = [
    {"id": "aicenteritl", "ssh": None, "label": "aicenteritl"},
    {"id": "aicenter1", "ssh": "aicenter1", "label": "aicenter1"},
    {"id": "aicenter2", "ssh": "aicenter2", "label": "aicenter2"},
    {"id": "aicenter3", "ssh": "aicenter3", "label": "aicenter3"},
    {"id": "ml3", "ssh": "ml3", "label": "ml3"},
    {"id": "ml4", "ssh": "ml4", "label": "ml4"},
    {"id": "h200", "ssh": "h200", "label": "h200"},
]


def _run(cmd: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def _load_hosts() -> list[dict[str, Any]]:
    if HOSTS_CFG.is_file():
        data = yaml.safe_load(HOSTS_CFG.read_text(encoding="utf-8")) or {}
        hosts = data.get("hosts") or []
        if hosts:
            return hosts
    return list(DEFAULT_HOSTS)


def _parse_gpu_csv(out: str) -> list[dict[str, Any]]:
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            used = float(parts[2])
            total = float(parts[3])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                    "memory_free_mb": max(0.0, total - used),
                    "util_pct": float(parts[4]),
                    "temp_c": float(parts[5]) if parts[5] != "" else None,
                    "busy": used > 1024 or float(parts[4]) > 5,
                }
            )
        except ValueError:
            continue
    return gpus


def _parse_procs_csv(out: str) -> list[dict[str, Any]]:
    procs = []
    for line in (out or "").strip().splitlines():
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


_GPU_Q = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]
_PROC_Q = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader",
]
_RAM_Q = "free -b"


def _parse_ram(out: str) -> dict[str, Any] | None:
    for line in (out or "").splitlines():
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


def nvidia_smi(ssh_host: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
    if ssh_host:
        remote = (
            " ".join(_GPU_Q)
            + "; echo '---PROCS---'; "
            + " ".join(_PROC_Q)
            + "; echo '---RAM---'; "
            + _RAM_Q
        )
        code, out, err = _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=4",
                "-o",
                "StrictHostKeyChecking=accept-new",
                ssh_host,
                remote,
            ],
            timeout=timeout,
        )
        if code != 0:
            return {
                "ok": False,
                "error": (err or out or f"ssh rc={code}").strip()[:240],
                "gpus": [],
                "processes": [],
                "ram": None,
                "any_busy": False,
                "foreign_jobs_suspected": False,
                "ssh_host": ssh_host,
            }
        gpu_part, _, rest = out.partition("---PROCS---")
        proc_part, _, ram_part = rest.partition("---RAM---")
        gpus = _parse_gpu_csv(gpu_part)
        procs = _parse_procs_csv(proc_part)
        ram = _parse_ram(ram_part)
    else:
        if not shutil.which("nvidia-smi"):
            return {"ok": False, "error": "nvidia-smi not found", "gpus": [], "processes": []}
        code, out, err = _run(_GPU_Q, timeout=timeout)
        gpus = _parse_gpu_csv(out) if code == 0 else []
        code2, out2, _ = _run(_PROC_Q, timeout=timeout)
        procs = _parse_procs_csv(out2) if code2 == 0 else []
        code3, out3, _ = _run(["bash", "-lc", _RAM_Q], timeout=timeout)
        ram = _parse_ram(out3) if code3 == 0 else None
        if code != 0:
            return {
                "ok": False,
                "error": err.strip() or "nvidia-smi failed",
                "gpus": [],
                "processes": [],
                "ram": None,
                "any_busy": False,
                "foreign_jobs_suspected": False,
            }

    foreign = any(p.get("foreign") for p in procs) or any(
        ("koksharov" in (p.get("name") or "").lower()) or ("lerobot" in (p.get("name") or "").lower())
        for p in procs
    )
    if ssh_host is None and "ram" not in locals():
        ram = None
    return {
        "ok": True,
        "error": None,
        "gpus": gpus,
        "processes": procs,
        "ram": ram if "ram" in locals() else None,
        "any_busy": any(g.get("busy") for g in gpus),
        "foreign_jobs_suspected": foreign,
        "ssh_host": ssh_host,
    }


def docker_status() -> dict[str, Any]:
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker not found", "containers": []}
    code, out, err = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"])
    containers = []
    if code == 0:
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                containers.append({"name": parts[0], "status": parts[1], "image": parts[2]})
    safe = next((c for c in containers if c["name"].startswith("safe_llm")), None)
    return {
        "ok": code == 0,
        "error": None if code == 0 else (err.strip() or "docker ps failed"),
        "containers": containers,
        "safe_llm_running": bool(safe),
        "safe_llm": safe,
    }


def fleet_status(max_workers: int = 6) -> dict[str, Any]:
    """Prefer Windows-refreshed cache (fleet_cache.json); else SSH from this host."""
    cache_path = ROOT / "experiments" / "runtime" / "fleet_cache.json"
    cached = None
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cached = None

    hosts = _load_hosts()
    servers: list[dict[str, Any]] = []

    def _one(h: dict[str, Any]) -> dict[str, Any]:
        hid = h.get("id") or h.get("label") or "host"
        ssh = h.get("ssh")
        probe = nvidia_smi(ssh_host=ssh if ssh else None)
        return {
            "id": hid,
            "label": h.get("label") or hid,
            "ssh": ssh,
            **probe,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, h) for h in hosts]
        for fut in concurrent.futures.as_completed(futs):
            try:
                servers.append(fut.result())
            except Exception as e:  # noqa: BLE001
                servers.append({"id": "?", "ok": False, "error": str(e), "gpus": [], "processes": []})

    order = {h.get("id"): i for i, h in enumerate(hosts)}
    servers.sort(key=lambda s: order.get(s.get("id"), 999))

    # Merge: if remote SSH failed but cache has that host OK, use cache entry.
    if cached and isinstance(cached.get("servers"), list):
        by_id = {s.get("id"): s for s in cached["servers"] if s.get("id")}
        merged = []
        for s in servers:
            c = by_id.get(s.get("id"))
            if (not s.get("ok")) and c and c.get("ok"):
                merged.append({**c, "from_cache": True})
            elif s.get("ok"):
                merged.append({**s, "from_cache": False})
            elif c:
                merged.append({**c, "from_cache": True, "live_error": s.get("error")})
            else:
                merged.append(s)
        # include cache-only hosts
        seen = {m.get("id") for m in merged}
        for hid, c in by_id.items():
            if hid not in seen:
                merged.append({**c, "from_cache": True})
        servers = merged
        servers.sort(key=lambda s: order.get(s.get("id"), 999))

    any_free = any(s.get("ok") and s.get("gpus") and not s.get("any_busy") for s in servers)
    return {
        "servers": servers,
        "any_free_host": any_free,
        "host_count": len(servers),
        "cache_updated_at": (cached or {}).get("updated_at"),
        "cache_path": str(cache_path) if cache_path.is_file() else None,
    }


def status_bundle() -> dict[str, Any]:
    local = nvidia_smi()
    return {
        "gpu": local,
        "docker": docker_status(),
        "fleet": fleet_status(),
    }
