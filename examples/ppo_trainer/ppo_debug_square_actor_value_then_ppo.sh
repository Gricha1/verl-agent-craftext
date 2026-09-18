#!/bin/bash
# Sequential debug_square PPO: actor-value (100 epochs) -> cooldown -> standard PPO (100 epochs).
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_actor_value_then_ppo.sh
#   TOTAL_EPOCHS=50 RAY_COOLDOWN_SECONDS=180 bash examples/ppo_trainer/ppo_debug_square_actor_value_then_ppo.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square_actor_value_then_ppo.sh
#
# Extra Hydra overrides are forwarded to BOTH runs, e.g.:
#   bash .../ppo_debug_square_actor_value_then_ppo.sh trainer.save_freq=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
RAY_COOLDOWN_SECONDS="${RAY_COOLDOWN_SECONDS:-120}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

cleanup_ray_between_runs() {
  echo "=========================================="
  echo "[INFO] Phase 1 finished. Cleaning up Ray before phase 2..."
  echo "[INFO] ray stop --force (ignore errors if Ray already down)"
  ray stop --force 2>/dev/null || true
  echo "[INFO] Waiting ${RAY_COOLDOWN_SECONDS}s for GPU/Ray workers to release resources..."
  sleep "$RAY_COOLDOWN_SECONDS"
  echo "[INFO] Cooldown done."
  echo "=========================================="
}

echo "=========================================="
echo "debug_square PPO sequence (${TOTAL_EPOCHS} epochs each)"
echo "=========================================="
echo "[INFO] Phase 1: ppo_debug_square_actor_value.sh"
echo "[INFO] Phase 2: ppo_debug_square.sh (after Ray cooldown)"
echo "[INFO] TOTAL_EPOCHS=$TOTAL_EPOCHS"
echo "[INFO] RAY_COOLDOWN_SECONDS=$RAY_COOLDOWN_SECONDS"
echo "[INFO] RUN_TAG=$RUN_TAG"

export RUN_NAME="${RUN_NAME_PHASE1:-PPO Debug Square actor-value ${TOTAL_EPOCHS}ep ${RUN_TAG}}"
echo "[INFO] RUN_NAME (phase 1)=$RUN_NAME"

bash examples/ppo_trainer/ppo_debug_square_actor_value.sh \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"

cleanup_ray_between_runs

export RUN_NAME="${RUN_NAME_PHASE2:-PPO Debug Square standard ${TOTAL_EPOCHS}ep ${RUN_TAG}}"
echo "[INFO] RUN_NAME (phase 2)=$RUN_NAME"

bash examples/ppo_trainer/ppo_debug_square.sh \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"

echo "=========================================="
echo "[INFO] Both PPO runs completed."
echo "=========================================="
