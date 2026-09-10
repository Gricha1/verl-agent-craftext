# Experiment Dashboard

Central UI + registry for `/home/gorbov_gv/safe_rl_nlp` experiments (debug_square / ALFWorld / Comet).

Style aligned with lab `gpu_monitor`: FastAPI + dark static UI (DM Sans / JetBrains Mono / teal).

## Source of truth

| Path | Role |
|------|------|
| `experiments/registry.yaml` | All experiments (historical + planned + running) |
| `experiments/hypotheses.yaml` | H1–H9 |
| `experiments/meta.yaml` | Research goal, baseline ids, compare presets, rules |
| `experiments/snapshots/` | Per-launch reproducibility JSON |
| `experiments/metrics_cache/` | Cached Comet/log scrapes |
| `experiments/runtime/` | Dashboard PID + running.json |

## Start (on aicenteritl)

```bash
cd /home/gorbov_gv/safe_rl_nlp/experiment_dashboard
bash scripts/start.sh
# open http://127.0.0.1:8770/
# from laptop: ssh -L 8770:127.0.0.1:8770 aicenteritl
```

Stop:

```bash
bash scripts/stop.sh
```

Optional env:

- `COMET_API_KEY` — live Comet metrics / ALFWorld import
- `EXPERIMENT_DASHBOARD_PORT` — default `8770`
- `EXPERIMENT_DASHBOARD_ALLOW_LAUNCH=1` — required for real process start (otherwise dry-run + snapshot only)
- `EXPERIMENT_DASHBOARD_PYTHON` — python binary

## Safety

- Training never auto-starts from Overview.
- Launch tab: Preview → `nvidia-smi` + warnings → Confirm.
- Confirm without `ALLOW_LAUNCH` writes snapshot only.
- Foreign GPU jobs are reported; dashboard does not kill them.
- User "стоп" / no confirm ⇒ do not restart.

## Tabs

1. **Overview** — goal, baseline, best, latest, planned, GPU, docker  
2. **Experiments** — filterable registry table + detail  
3. **Compare** — preset `8x8 r_t vs G_t` + custom ids  
4. **Hypotheses** — H1–H9  
5. **Planned** — sparse R/G_t, gamma 0.99/0.95, n-step 3/5  
6. **Launch** — preview/confirm/stop workflow  

## API (selected)

- `GET /api/overview`
- `GET /api/experiments`
- `GET /api/hypotheses`
- `GET /api/compare/preset/rt_vs_gt_8x8`
- `GET /api/gpu`
- `GET /api/launch/preview/{exp_id}`
- `POST /api/launch/confirm`
- `POST /api/launch/stop`
- `GET /api/alfworld/comet`
