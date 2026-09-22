"""Experiment dashboard API + UI (FastAPI), gpu_monitor-style."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gpu_probe import fleet_status, status_bundle
from launch_ops import confirm_launch, load_running, prepare_launch, stop_running
from metrics import compare_experiments, list_alfworld_from_comet, metrics_for_experiment
from registry_store import (
    ROOT,
    ensure_dirs,
    get_experiment,
    list_experiments,
    load_hypotheses,
    load_meta,
    overview,
    planned_experiments,
    upsert_experiment,
)

ROOT_DASH = Path(__file__).resolve().parent
STATIC = ROOT_DASH / "static"


def _load_dotenv() -> None:
    env = ROOT_DASH / ".env"
    if not env.is_file():
        env = ROOT / ".env"
    if not env.is_file():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
ensure_dirs()

app = FastAPI(title="Safe RL Experiment Dashboard", version="0.1.0")


class UpsertBody(BaseModel):
    experiment: dict[str, Any]


class LaunchConfirmBody(BaseModel):
    exp_id: str
    confirm: bool = False
    confirm_stop_others: bool = False


class StopBody(BaseModel):
    confirm: bool = False


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "root": str(ROOT), "dashboard": str(ROOT_DASH)}


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    data = overview()
    bundle = status_bundle()
    data["gpu"] = bundle["gpu"]
    data["docker"] = bundle["docker"]
    data["fleet"] = bundle.get("fleet") or {}
    data["running_runtime"] = load_running()
    return data


@app.get("/api/gpu/fleet")
def api_gpu_fleet() -> dict[str, Any]:
    return fleet_status()


@app.get("/api/experiments")
def api_experiments(
    status: str | None = None,
    environment: str | None = None,
    reward_mode: str | None = None,
    architecture: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    items = list_experiments(status=status)
    if environment:
        items = [e for e in items if environment in str(e.get("environment") or "")]
    if reward_mode:
        items = [e for e in items if reward_mode in str(e.get("reward_mode") or "")]
    if architecture:
        items = [e for e in items if architecture in str(e.get("architecture") or "")]
    if q:
        ql = q.lower()
        items = [
            e
            for e in items
            if ql in str(e.get("id", "")).lower()
            or ql in str(e.get("name", "")).lower()
            or ql in str(e.get("comet_key", "")).lower()
            or ql in str(e.get("notes", "")).lower()
        ]
    return {"count": len(items), "experiments": items}


@app.get("/api/experiments/{exp_id}")
def api_experiment(exp_id: str) -> dict[str, Any]:
    exp = get_experiment(exp_id)
    if not exp:
        raise HTTPException(404, f"unknown experiment {exp_id}")
    return {"experiment": exp}


@app.post("/api/experiments/upsert")
def api_upsert(body: UpsertBody) -> dict[str, Any]:
    return {"experiment": upsert_experiment(body.experiment)}


@app.get("/api/hypotheses")
def api_hypotheses() -> dict[str, Any]:
    return load_hypotheses()


@app.get("/api/planned")
def api_planned() -> dict[str, Any]:
    return {"experiments": planned_experiments()}


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    return load_meta()


@app.get("/api/gpu")
def api_gpu() -> dict[str, Any]:
    return status_bundle()


@app.get("/api/compare")
def api_compare(
    ids: str = Query(..., description="comma-separated experiment ids"),
    prefer_comet: bool = False,
) -> dict[str, Any]:
    exp_ids = [x.strip() for x in ids.split(",") if x.strip()]
    return compare_experiments(exp_ids, prefer_comet=prefer_comet)


@app.get("/api/compare/preset/{name}")
def api_compare_preset(name: str, prefer_comet: bool = False) -> dict[str, Any]:
    presets = load_meta().get("compare_presets") or {}
    preset = presets.get(name)
    if not preset:
        raise HTTPException(404, f"unknown preset {name}")
    ids = list(preset.get("experiment_ids") or [])
    data = compare_experiments(ids, prefer_comet=prefer_comet)
    data["preset"] = preset
    data["preset_name"] = name
    return data


@app.get("/api/metrics/{exp_id}")
def api_metrics(exp_id: str, prefer_comet: bool = False) -> dict[str, Any]:
    if not get_experiment(exp_id):
        raise HTTPException(404, f"unknown experiment {exp_id}")
    return metrics_for_experiment(exp_id, prefer_comet=prefer_comet)


@app.get("/api/launch/preview/{exp_id}")
def api_launch_preview(exp_id: str) -> dict[str, Any]:
    return prepare_launch(exp_id)


@app.post("/api/launch/confirm")
def api_launch_confirm(body: LaunchConfirmBody) -> dict[str, Any]:
    return confirm_launch(body.exp_id, body.confirm, body.confirm_stop_others)


@app.get("/api/launch/running")
def api_launch_running() -> dict[str, Any]:
    return {"running": load_running()}


@app.post("/api/launch/stop")
def api_launch_stop(body: StopBody) -> dict[str, Any]:
    return stop_running(body.confirm)


@app.get("/api/alfworld/comet")
def api_alfworld_comet(limit: int = 40) -> dict[str, Any]:
    return list_alfworld_from_comet(limit=limit)


@app.get("/api/value-diagnostics")
def api_value_diagnostics() -> dict[str, Any]:
    """GSM8K / CrafText shared-vs-dual value diagnostics summary (cached JSON)."""
    import json

    path = ROOT / "experiments" / "metrics_cache" / "value_diagnostics_summary.json"
    if not path.is_file():
        raise HTTPException(404, f"missing {path}; run experiment_dashboard/_build_value_diagnostics.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/compare-presets")
def api_compare_presets() -> dict[str, Any]:
    """GSM8K dual-vs-MAE and CrafText 8x8 reward-target Comet compare presets."""
    import json

    path = ROOT / "experiments" / "metrics_cache" / "compare_presets.json"
    if not path.is_file():
        raise HTTPException(404, f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("EXPERIMENT_DASHBOARD_PORT", "8770"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
