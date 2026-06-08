#!/bin/bash
# Quick test: load debug_square_8x8 and walk to wooden block (no LLM, no PPO).
#
# Usage:
#   bash examples/ppo_trainer/debug_square_go_to_wood.sh
#   TASK=wood GIF=1 bash examples/ppo_trainer/debug_square_go_to_wood.sh
#   TASK=stone bash examples/ppo_trainer/debug_square_go_to_wood.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "verl-agent-311" ]; then
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate verl-agent-311
  fi
fi

# Leave CRAFTAX_RELOAD_TEXTURES unset to use cached textures (fast). Set to True to rebuild.
# export CRAFTAX_RELOAD_TEXTURES=True
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export CRAFTAX_PATH="${CRAFTAX_PATH:-$CAGED_CRAFTEXT_PATH/Craftax}"
export PYTHONPATH="${CRAFTAX_PATH}:${CAGED_CRAFTEXT_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"

TASK="${TASK:-wood}"
SEED="${SEED:-0}"
GIF="${GIF:-0}"

ARGS=(--task "$TASK" --seed "$SEED")
if [ "$GIF" = "1" ]; then
  ARGS+=(--gif "gif/debug_square_go_to_${TASK}.gif")
fi

echo "=========================================="
echo "debug_square go-to-goal test (no LLM)"
echo "=========================================="
echo "[INFO] TASK=$TASK SEED=$SEED"
echo "[INFO] CAGED_CRAFTEXT_PATH=$CAGED_CRAFTEXT_PATH"
echo "[INFO] wood path: UP UP UP RIGHT -> adjacent to w@(1,6)"
echo "[INFO] expect: done=True reward~+1 on last step"
echo "=========================================="

python3 scripts/debug_square_go_to_goal.py "${ARGS[@]}"
