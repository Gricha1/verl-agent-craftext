"""Load / save experiment registry YAML files."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
REGISTRY = EXP_DIR / "registry.yaml"
HYPOTHESES = EXP_DIR / "hypotheses.yaml"
META = EXP_DIR / "meta.yaml"
SNAPSHOTS = EXP_DIR / "snapshots"
METRICS_CACHE = EXP_DIR / "metrics_cache"
RUNTIME = EXP_DIR / "runtime"
RUNNING_JSON = RUNTIME / "running.json"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(text, encoding="utf-8")


def load_registry() -> dict[str, Any]:
    return _read_yaml(REGISTRY)


def load_hypotheses() -> dict[str, Any]:
    return _read_yaml(HYPOTHESES)


def load_meta() -> dict[str, Any]:
    return _read_yaml(META)


def save_registry(data: dict[str, Any]) -> None:
    data = deepcopy(data)
    data["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _write_yaml(REGISTRY, data)


def list_experiments(status: str | None = None) -> list[dict[str, Any]]:
    exps = list(load_registry().get("experiments") or [])
    if status:
        exps = [e for e in exps if str(e.get("status")) == status]
    return exps


def get_experiment(exp_id: str) -> dict[str, Any] | None:
    for e in list_experiments():
        if e.get("id") == exp_id:
            return e
    return None


def upsert_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    if not exp.get("id"):
        raise ValueError("experiment.id required")
    reg = load_registry()
    items = list(reg.get("experiments") or [])
    found = False
    for i, e in enumerate(items):
        if e.get("id") == exp["id"]:
            merged = {**e, **exp}
            items[i] = merged
            found = True
            exp = merged
            break
    if not found:
        items.append(exp)
    reg["experiments"] = items
    save_registry(reg)
    return exp


def planned_experiments() -> list[dict[str, Any]]:
    items = [e for e in list_experiments() if e.get("status") == "planned"]
    items.sort(key=lambda e: (e.get("planned") or {}).get("priority", 99))
    return items


def overview() -> dict[str, Any]:
    meta = load_meta()
    hyps = load_hypotheses().get("hypotheses") or []
    exps = list_experiments()
    by_id = {e["id"]: e for e in exps if e.get("id")}
    running = [e for e in exps if e.get("status") == "running"]
    planned = planned_experiments()
    active_h = [h for h in hyps if h.get("id") in set(meta.get("active_hypothesis_ids") or [])]
    return {
        "research_goal": meta.get("research_goal"),
        "current_baseline": by_id.get(meta.get("current_baseline_id")),
        "current_best": by_id.get(meta.get("current_best_id")),
        "latest_non_planned": _latest(exps),
        "running": running,
        "next_planned": by_id.get(meta.get("next_planned_id")) or (planned[0] if planned else None),
        "active_hypotheses": active_h,
        "counts": _counts(exps),
        "compare_presets": meta.get("compare_presets") or {},
        "ladder": meta.get("ladder") or [],
        "rules": meta.get("rules") or [],
        "host": meta.get("host"),
        "repo_path": meta.get("repo_path"),
        "updated": meta.get("updated"),
        "registry_updated": load_registry().get("updated"),
    }


def _latest(exps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer newest start_date; among ties prefer long/failed over early oom/aborted."""
    dated = [e for e in exps if e.get("status") != "planned" and e.get("start_date")]
    if not dated:
        return None
    rank = {"successful": 3, "failed": 2, "inconclusive": 2, "oom": 0, "aborted": 0, "running": 4}

    def key(e: dict[str, Any]) -> tuple:
        return (str(e.get("start_date")), rank.get(str(e.get("status")), 1), str(e.get("id")))

    dated.sort(key=key, reverse=True)
    return dated[0]


def _counts(exps: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in exps:
        st = str(e.get("status") or "unknown")
        out[st] = out.get(st, 0) + 1
    out["total"] = len(exps)
    return out


def ensure_dirs() -> None:
    for p in (EXP_DIR, SNAPSHOTS, METRICS_CACHE, RUNTIME):
        p.mkdir(parents=True, exist_ok=True)
