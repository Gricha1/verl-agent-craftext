#!/usr/bin/env bash
set -euo pipefail

RULE=${1:?Usage: run_safety_ppo_lag.sh energy|sword|zombie RUN_DIR}
RUN_DIR=${2:?Set a fresh run directory}
PYTHON_BIN=${PYTHON_BIN:-python3}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CRAFTEXT="$ROOT/caged_craftext"

case "$RULE" in
  energy|sword) BUDGET=0.1 ;;
  zombie) BUDGET=10 ;;
  *) echo "Unknown rule: $RULE" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
RUN_DIR=$(realpath "$RUN_DIR")
if [[ -e "$RUN_DIR/config.json" ]]; then
  echo "The run directory already has PPO-Lag state: $RUN_DIR" >&2
  exit 2
fi

export PYTHONPATH="$CRAFTEXT:$CRAFTEXT/Craftax:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CRAFTEXT_ENCODER_DEVICE=cpu

NUM_ENVS=${NUM_ENVS:-128}
NUM_STEPS=${NUM_STEPS:-100}
NUM_UPDATES=${NUM_UPDATES:-7812}
HORIZON=${HORIZON:-500}
TOTAL_STEPS=$((NUM_ENVS * NUM_STEPS * NUM_UPDATES))
SEED=${SEED:-7}
git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/code_version.txt"
cd "$RUN_DIR"

"$PYTHON_BIN" -u "$CRAFTEXT/baselines/ppo_lag_with_instruction.py" \
  --output-dir "$RUN_DIR" \
  --craftext_settings "achievements_safe_all_$RULE" --env_name Craftax-Classic-Pixels-v1-Text \
  --max-episode-length "$HORIZON" --cost_threshold "$BUDGET" \
  --num_envs "$NUM_ENVS" --num_steps "$NUM_STEPS" --total_timesteps "$TOTAL_STEPS" \
  --save_freq "${SAVE_FREQ:-100}" --seed "$SEED"

CKPT="$RUN_DIR/checkpoints/$TOTAL_STEPS"
test -d "$CKPT"
printf '%s\n' "$CKPT" > "$RUN_DIR/final_checkpoint.txt"
"$PYTHON_BIN" -u "$ROOT/vlm_pipeline/phase3_verl/craftext_integration/energy_text_attack.py" \
  evaluate --clean --rule "$RULE" --checkpoint "$CKPT" --horizon "$HORIZON" \
  > "$RUN_DIR/clean_eval.jsonl"
