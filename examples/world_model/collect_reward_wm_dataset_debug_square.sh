#!/bin/bash
set -e

# Collect a fixed offline dataset for reward WM (debug_square_8x8).
#
# Usage:
#   bash examples/world_model/collect_reward_wm_dataset_debug_square.sh
#   REWARD_HORIZON=6 bash examples/world_model/collect_reward_wm_dataset_debug_square.sh
#     -> data/reward_wm_debug_square_8x8_h6/{train,val}.parquet
#   EPISODES_PER_TASK=500 bash examples/world_model/collect_reward_wm_dataset_debug_square.sh

REWARD_HORIZON="${REWARD_HORIZON:-3}"
OUT_DIR="${OUT_DIR:-data/reward_wm_debug_square_8x8_h${REWARD_HORIZON}}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-200}"
MAX_STEPS="${MAX_STEPS:-50}"
SEED="${SEED:-0}"
VAL_FRAC="${VAL_FRAC:-0.1}"
POLICY="${POLICY:-semi}"
EPSILON="${EPSILON:-0.15}"

export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

python3 scripts/collect_reward_wm_dataset_debug_square.py \
  --out-dir "$OUT_DIR" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --val-frac "$VAL_FRAC" \
  --policy "$POLICY" \
  --epsilon "$EPSILON" \
  --reward-horizon "$REWARD_HORIZON" \
  --keep-extra-columns

echo "[OK] Dataset ready (REWARD_HORIZON=$REWARD_HORIZON):"
echo "  - $OUT_DIR/train.parquet"
echo "  - $OUT_DIR/val.parquet"
echo "  (columns state, state_after, action_token enable INVERSE_ACTION_WM from same batch)"

