#!/bin/bash
# Start GPU profiler in background (nvidia-smi -> console + Comet ML).
#
# Usage:
#   bash scripts/run_gpu_profiler.sh
#   GPU_PROFILER_INTERVAL_SEC=3 bash scripts/run_gpu_profiler.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp}"
mkdir -p "$RAY_TEMP_DIR"

export GPU_PROFILER_INTERVAL_SEC="${GPU_PROFILER_INTERVAL_SEC:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl_agent_alfworld}"
export COMET_EXPERIMENT_KEY_FILE="${COMET_EXPERIMENT_KEY_FILE:-$RAY_TEMP_DIR/comet_experiment_key.txt}"
export GPU_PROFILER_LOG="${GPU_PROFILER_LOG:-$RAY_TEMP_DIR/gpu_profiler.log}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu_profiler] nvidia-smi not found, skipping"
  exit 0
fi

python3 -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

python3 "$PROJECT_ROOT/scripts/gpu_profiler.py" \
  --interval "$GPU_PROFILER_INTERVAL_SEC" \
  --comet-project "$COMET_PROJECT_NAME" \
  --comet-experiment "${RUN_NAME:-gpu_profiler}" \
  --comet-key-file "$COMET_EXPERIMENT_KEY_FILE" \
  >>"$GPU_PROFILER_LOG" 2>&1 &

echo "$!" > "$RAY_TEMP_DIR/gpu_profiler.pid"
echo "[gpu_profiler] started pid=$(cat "$RAY_TEMP_DIR/gpu_profiler.pid"), log=$GPU_PROFILER_LOG"
