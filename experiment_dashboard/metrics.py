"""Metrics helpers: local log scrape + optional Comet API."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from registry_store import METRICS_CACHE, ROOT, get_experiment

METRIC_KEYS = [
    "episode/success_rate",
    "episode/reward/mean",
    "critic/score/mean",
    "critic/returns/mean",
    "critic/vf_explained_var",
    "actor/entropy_loss",
    "actor/pg_loss",
    "actor/kl_loss",
    "training/global_step",
]

_STEP_RE = re.compile(
    r"(?:step|global_step|training/global_step)[=:\s]+(\d+(?:\.\d+)?)",
    re.I,
)
_METRIC_LINE_RE = re.compile(
    r"(episode/success_rate|episode/reward/mean|critic/score/mean|critic/returns/mean|"
    r"critic/vf_explained_var|actor/entropy_loss|actor/pg_loss|actor/kl_loss)"
    r"\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def _cache_path(exp_id: str) -> Path:
    return METRICS_CACHE / f"{exp_id}.json"


def load_cached_metrics(exp_id: str) -> dict[str, Any] | None:
    p = _cache_path(exp_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_metrics(exp_id: str, payload: dict[str, Any]) -> None:
    METRICS_CACHE.mkdir(parents=True, exist_ok=True)
    _cache_path(exp_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_log_header(log_path: Path, max_lines: int = 80) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(log_path), "lines": []}
    if not log_path.is_file():
        out["error"] = "missing"
        return out
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[:max_lines]
    except OSError as e:
        out["error"] = str(e)
        return out
    out["lines"] = lines
    text = "\n".join(lines)
    if "remaining G_t" in text:
        out["token_reward"] = "G_t"
    elif "per-step r_t" in text:
        out["token_reward"] = "r_t"
    m = re.search(r"HISTORY_LENGTH=(\d+)", text)
    if m:
        out["history_length"] = int(m.group(1))
    m = re.search(r"gae_by_trajectory=(\w+)", text)
    if m:
        out["gae_by_trajectory"] = m.group(1).lower() in ("true", "1", "yes")
    return out


def scrape_log_metrics(log_path: Path, max_bytes: int = 2_000_000) -> dict[str, Any]:
    """Best-effort metric scrape from training stdout log."""
    result: dict[str, Any] = {
        "source": "local_log",
        "path": str(log_path),
        "series": {k: [] for k in METRIC_KEYS},
        "summary": {},
    }
    if not log_path.is_file():
        result["error"] = "missing_log"
        return result
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()
            raw = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        result["error"] = str(e)
        return result

    step = None
    for line in raw.splitlines():
        sm = _STEP_RE.search(line)
        if sm:
            try:
                step = int(float(sm.group(1)))
            except ValueError:
                pass
        for mm in _METRIC_LINE_RE.finditer(line):
            key, val_s = mm.group(1), mm.group(2)
            try:
                val = float(val_s)
            except ValueError:
                continue
            result["series"][key].append({"step": step, "value": val})

    for key, pts in result["series"].items():
        if not pts:
            continue
        vals = [p["value"] for p in pts]
        result["summary"][key] = {
            "n": len(vals),
            "last": vals[-1],
            "max": max(vals),
            "min": min(vals),
        }
    return result


def fetch_comet_metrics(comet_project: str, comet_key: str, metric_names: list[str] | None = None) -> dict[str, Any]:
    metric_names = metric_names or METRIC_KEYS
    out: dict[str, Any] = {
        "source": "comet",
        "project": comet_project,
        "key": comet_key,
        "series": {},
        "summary": {},
        "params": {},
    }
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        out["error"] = "COMET_API_KEY not set"
        return out
    try:
        from comet_ml.api import API  # type: ignore
    except Exception as e:  # noqa: BLE001
        out["error"] = f"comet_ml import failed: {e}"
        return out

    try:
        api = API(api_key=api_key)
        if "/" in comet_project:
            ws, proj = comet_project.split("/", 1)
        else:
            ws, proj = "gregory-gorbov", comet_project
        exp = api.get(f"{ws}/{proj}/{comet_key}")
        try:
            out["params"] = dict(exp.get_parameters_summary() or {})
        except Exception:  # noqa: BLE001
            out["params"] = {}
        for name in metric_names:
            try:
                raw = exp.get_metrics(name)
            except Exception:  # noqa: BLE001
                raw = []
            pts = []
            for row in raw or []:
                # comet rows vary: dict with metricValue / step
                if isinstance(row, dict):
                    val = row.get("metricValue", row.get("value"))
                    step = row.get("step", row.get("epoch"))
                    try:
                        pts.append({"step": int(step) if step is not None else None, "value": float(val)})
                    except (TypeError, ValueError):
                        continue
            out["series"][name] = pts
            if pts:
                vals = [p["value"] for p in pts]
                out["summary"][name] = {
                    "n": len(vals),
                    "last": vals[-1],
                    "max": max(vals),
                    "min": min(vals),
                }
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def metrics_for_experiment(exp_id: str, prefer_comet: bool = True, use_cache: bool = True) -> dict[str, Any]:
    exp = get_experiment(exp_id)
    if not exp:
        return {"error": f"unknown experiment {exp_id}"}
    if use_cache:
        cached = load_cached_metrics(exp_id)
        if cached and not cached.get("error"):
            cached["cached"] = True
            return cached

    payload: dict[str, Any] = {
        "experiment_id": exp_id,
        "comet_key": exp.get("comet_key"),
        "header": None,
        "metrics": None,
    }
    log_rel = exp.get("local_log_path")
    log_path = (ROOT / log_rel) if log_rel else None
    if log_path:
        payload["header"] = parse_log_header(log_path)

    metrics = None
    if prefer_comet and exp.get("comet_key") and exp.get("comet_project"):
        metrics = fetch_comet_metrics(exp["comet_project"], exp["comet_key"])
        if metrics.get("error"):
            payload["comet_error"] = metrics["error"]
            metrics = None

    if metrics is None and log_path:
        metrics = scrape_log_metrics(log_path)
        if metrics.get("error"):
            payload["log_error"] = metrics["error"]

    if metrics is None:
        # Fall back to registry summary fields
        metrics = {
            "source": "registry",
            "series": {},
            "summary": {
                "episode/success_rate": {
                    "max": exp.get("success_max"),
                    "last": exp.get("success_last"),
                    "n": 0,
                }
            },
        }

    payload["metrics"] = metrics
    if not metrics.get("error"):
        save_cached_metrics(exp_id, payload)
    return payload


def compare_experiments(exp_ids: list[str], prefer_comet: bool = False) -> dict[str, Any]:
    rows = []
    for eid in exp_ids:
        exp = get_experiment(eid) or {"id": eid, "error": "missing"}
        m = metrics_for_experiment(eid, prefer_comet=prefer_comet, use_cache=True)
        summary = (m.get("metrics") or {}).get("summary") or {}
        rows.append(
            {
                "id": eid,
                "name": exp.get("name"),
                "status": exp.get("status"),
                "reward_mode": exp.get("reward_mode"),
                "history_length": exp.get("history_length"),
                "gae_by_trajectory": exp.get("gae_by_trajectory"),
                "success_max_registry": exp.get("success_max"),
                "success_last_registry": exp.get("success_last"),
                "metric_source": (m.get("metrics") or {}).get("source"),
                "comet_error": m.get("comet_error"),
                "log_error": m.get("log_error"),
                "summary": summary,
                "header_token_reward": (m.get("header") or {}).get("token_reward"),
            }
        )
    return {"experiments": rows, "metric_keys": METRIC_KEYS}


def list_alfworld_from_comet(limit: int = 50) -> dict[str, Any]:
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        return {"error": "COMET_API_KEY not set", "experiments": []}
    try:
        from comet_ml.api import API  # type: ignore

        api = API(api_key=api_key)
        exps = api.get_experiments("gregory-gorbov", "verl-agent-alfworld")
        rows = []
        for e in (exps or [])[:limit]:
            key = getattr(e, "id", None) or getattr(e, "key", None)
            name = getattr(e, "name", None) or getattr(e, "experiment_name", None)
            url = f"https://www.comet.com/gregory-gorbov/verl-agent-alfworld/{key}" if key else None
            rows.append({"comet_key": key, "name": name, "url": url})
        return {"source": "comet", "count": len(rows), "experiments": rows}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "experiments": []}
