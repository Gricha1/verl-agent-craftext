#!/bin/bash
# Random single-token policy on debug_square_8x8 — no vLLM, no checkpoint.
# Prints full prompt + saves GIF under ./gif/
#
# Usage:
#   bash examples/ppo_trainer/debug_square_random_policy_video.sh
#   STEPS=30 SEED=1 bash examples/ppo_trainer/debug_square_random_policy_video.sh

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

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export RUN_NAME="${RUN_NAME:-random_policy_debug_square_$(date +%Y%m%d-%H%M%S)}"

export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

STEPS="${STEPS:-50}"
SEED="${SEED:-0}"
GIF_PATH="${GIF_PATH:-gif/random_policy_debug_square.gif}"

echo "=========================================="
echo "debug_square random policy video (no vLLM)"
echo "=========================================="
echo "[INFO] STEPS=$STEPS SEED=$SEED"
echo "[INFO] RUN_NAME=$RUN_NAME"
echo "[INFO] Comet project=verl_agent_caged_craftext"
echo "[INFO] prompt_template=single_token_action"
echo "[INFO] output: $GIF_PATH and ${GIF_PATH%.gif}_prompt.txt"

python3 scripts/debug_square_random_policy_video.py \
  --steps "$STEPS" \
  --seed "$SEED" \
  --gif-path "$GIF_PATH" \
  --prompt-template single_token_action
