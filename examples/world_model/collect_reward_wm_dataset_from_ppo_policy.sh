#!/bin/bash
# Collect reward-WM parquet from PPO (or any HF LoRA) rollouts on debug_square_8x8.
#
# Usage:
#   CKPT_PATH=/path/to/ppo_lora_hf bash examples/world_model/collect_reward_wm_dataset_from_ppo_policy.sh
#   OUT_DIR=training_checkpoints/.../ppo_rollout_dataset_h6 CKPT_PATH=... bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to HF LoRA dir (e.g. .../01_ppo_lora_hf)}"
OUT_DIR="${OUT_DIR:-data/reward_wm_ppo_rollout_h${REWARD_HORIZON:-6}}"
REWARD_HORIZON="${REWARD_HORIZON:-6}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-200}"
MAX_STEPS="${MAX_STEPS:-50}"
SEED="${SEED:-0}"
VAL_FRAC="${VAL_FRAC:-0.1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DEVICE="${DEVICE:-cuda:0}"

export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEVICE="cuda:0"
fi

if [ ! -d "$CKPT_PATH" ]; then
  echo "[ERROR] CKPT_PATH not found: $CKPT_PATH" >&2
  exit 1
fi

echo "[INFO] Collecting WM dataset from PPO/LLM rollouts"
echo "[INFO] CKPT_PATH=$CKPT_PATH"
echo "[INFO] OUT_DIR=$OUT_DIR  REWARD_HORIZON=$REWARD_HORIZON"
echo "[INFO] EPISODES_PER_TASK=$EPISODES_PER_TASK  TEMPERATURE=$TEMPERATURE"

python3 scripts/collect_reward_wm_dataset_debug_square.py \
  --out-dir "$OUT_DIR" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --val-frac "$VAL_FRAC" \
  --reward-horizon "$REWARD_HORIZON" \
  --policy llm \
  --checkpoint "$CKPT_PATH" \
  --device "$DEVICE" \
  --temperature "$TEMPERATURE" \
  --keep-extra-columns

echo "[OK] PPO rollout dataset:"
echo "  $OUT_DIR/train.parquet"
echo "  $OUT_DIR/val.parquet"
