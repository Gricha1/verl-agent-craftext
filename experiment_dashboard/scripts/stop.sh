#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PIDFILE="$REPO/experiments/runtime/dashboard.pid"
if [[ ! -f "$PIDFILE" ]]; then
  echo "not running"
  exit 0
fi
pid="$(cat "$PIDFILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid" || true
  sleep 0.5
  kill -9 "$pid" 2>/dev/null || true
  echo "stopped $pid"
else
  echo "stale pidfile $pid"
fi
rm -f "$PIDFILE"
