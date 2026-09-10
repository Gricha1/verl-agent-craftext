#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
PORT="${EXPERIMENT_DASHBOARD_PORT:-8770}"
mkdir -p "$REPO/experiments/runtime" "$REPO/experiment_dashboard/logs"
PIDFILE="$REPO/experiments/runtime/dashboard.pid"
LOG="$ROOT/logs/uvicorn.out.log"
ERR="$ROOT/logs/uvicorn.err.log"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE") http://127.0.0.1:${PORT}/"
  exit 0
fi

PYTHON="${EXPERIMENT_DASHBOARD_PYTHON:-python3}"
if ! "$PYTHON" -c "import fastapi,uvicorn,yaml" 2>/dev/null; then
  echo "Installing dashboard deps into user/site for $PYTHON ..."
  "$PYTHON" -m pip install -q -r requirements.txt
fi

nohup "$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>"$ERR" &
echo $! >"$PIDFILE"
sleep 1
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "started pid=$(cat "$PIDFILE") http://127.0.0.1:${PORT}/"
else
  echo "failed to start; see $ERR" >&2
  exit 1
fi
