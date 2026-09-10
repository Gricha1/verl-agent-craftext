"""Launch workflow with mandatory confirmation (never auto-starts training)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_probe import status_bundle
from registry_store import (
    ROOT,
    RUNNING_JSON,
    SNAPSHOTS,
    ensure_dirs,
    get_experiment,
    upsert_experiment,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_meta() -> dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            p = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            return (p.stdout or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": _git("status", "-sb"),
        "diff_stat": _git("diff", "--stat", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def prepare_launch(exp_id: str) -> dict[str, Any]:
    """Build preview: GPU status + command + config. Does NOT start."""
    ensure_dirs()
    exp = get_experiment(exp_id)
    if not exp:
        return {"ok": False, "error": f"unknown experiment {exp_id}"}
    if exp.get("status") == "running":
        return {"ok": False, "error": "already marked running", "experiment": exp}

    gpu = status_bundle()
    script = exp.get("launch_script")
    cfg = exp.get("config_path")
    cmd = None
    if script:
        cmd = f"bash {script}"
        if cfg and Path(cfg).name.endswith(".yaml"):
            # scripts usually embed yaml; still show path
            pass
    warnings = []
    if gpu["gpu"].get("foreign_jobs_suspected") or gpu["gpu"].get("any_busy"):
        warnings.append("GPU busy / foreign processes detected — do not launch without user OK")
    if not gpu["docker"].get("safe_llm_running"):
        warnings.append("Docker container safe_llm_ is not running")
    if not script:
        warnings.append("No launch_script in registry — fill before launching")

    preview = {
        "ok": True,
        "requires_confirmation": True,
        "experiment": exp,
        "command": cmd,
        "config_path": cfg,
        "cwd": str(ROOT),
        "gpu": gpu["gpu"],
        "docker": gpu["docker"],
        "git": _git_meta(),
        "warnings": warnings,
        "confirm_token_hint": "POST /api/launch/confirm with confirm=true and exp_id",
    }
    return preview


def write_snapshot(exp_id: str, extra: dict[str, Any] | None = None) -> Path:
    ensure_dirs()
    exp = get_experiment(exp_id) or {"id": exp_id}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOTS / f"{exp_id}_{stamp}.json"
    payload = {
        "created_at": _now(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "experiment": exp,
        "git": _git_meta(),
        "gpu": status_bundle()["gpu"],
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_running() -> dict[str, Any] | None:
    if not RUNNING_JSON.is_file():
        return None
    try:
        return json.loads(RUNNING_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_running() -> None:
    if RUNNING_JSON.is_file():
        RUNNING_JSON.unlink()


def confirm_launch(exp_id: str, confirm: bool, confirm_stop_others: bool = False) -> dict[str, Any]:
    """Start only if confirm=True. Still refuses if GPU foreign unless confirm_stop_others."""
    preview = prepare_launch(exp_id)
    if not preview.get("ok"):
        return preview
    if not confirm:
        return {**preview, "started": False, "message": "confirmation required; not started"}

    gpu = preview["gpu"]
    if gpu.get("foreign_jobs_suspected") and not confirm_stop_others:
        return {
            "ok": False,
            "started": False,
            "error": "Foreign GPU jobs suspected. Re-confirm with confirm_stop_others=true ONLY if you accept sharing risk — still will not kill foreign jobs.",
            "gpu": gpu,
        }

    cmd = preview.get("command")
    if not cmd:
        return {"ok": False, "started": False, "error": "no launch command"}

    # Safety: refuse automatic start in this scaffold unless EXPERIMENT_DASHBOARD_ALLOW_LAUNCH=1
    if os.environ.get("EXPERIMENT_DASHBOARD_ALLOW_LAUNCH") != "1":
        snap = write_snapshot(exp_id, {"dry_run": True, "command": cmd})
        return {
            "ok": True,
            "started": False,
            "dry_run": True,
            "message": (
                "Launch gated: set EXPERIMENT_DASHBOARD_ALLOW_LAUNCH=1 to actually start. "
                "Snapshot written. Prefer agent/user-driven docker+tmux launch after GPU OK."
            ),
            "snapshot": str(snap),
            "command": cmd,
            "preview": preview,
        }

    ensure_dirs()
    logdir = ROOT / "logdir"
    logdir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logdir / f"{exp_id}_{stamp}.log"
    snap = write_snapshot(exp_id, {"command": cmd, "log_path": str(log_path)})

    # Run detached under nohup (host). Prefer docker/tmux in real workflow.
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# launch {_now()} cmd={cmd}\n")
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    running = {
        "exp_id": exp_id,
        "pid": proc.pid,
        "command": cmd,
        "log_path": str(log_path.relative_to(ROOT)),
        "snapshot": str(snap.relative_to(ROOT)),
        "started_at": _now(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }
    RUNNING_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUNNING_JSON.write_text(json.dumps(running, indent=2), encoding="utf-8")

    exp = get_experiment(exp_id) or {"id": exp_id}
    exp["status"] = "running"
    exp["local_log_path"] = running["log_path"]
    upsert_experiment(exp)

    return {"ok": True, "started": True, "running": running, "snapshot": str(snap)}


def stop_running(confirm: bool) -> dict[str, Any]:
    if not confirm:
        return {"ok": False, "error": "confirm=true required"}
    running = load_running()
    if not running:
        return {"ok": False, "error": "no running record"}
    pid = int(running["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    except PermissionError as e:
        return {"ok": False, "error": f"cannot signal pid {pid}: {e}", "running": running}

    exp = get_experiment(running["exp_id"])
    if exp:
        exp["status"] = "aborted"
        upsert_experiment(exp)
    clear_running()
    return {"ok": True, "stopped_pid": pid, "exp_id": running["exp_id"]}
