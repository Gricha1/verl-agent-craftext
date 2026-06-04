#!/bin/bash
set -e

# Offline SFT training for reward world model on a fixed parquet dataset.
#
# Expects parquet files with columns:
#   - prompt (string)
#   - response (string, reward token(s) i/j/k/l; concatenated for multi-step horizons)
#
# For reward_horizon>1, collect dataset with matching REWARD_HORIZON first:
#   REWARD_HORIZON=6 bash examples/world_model/collect_reward_wm_dataset_debug_square.sh
# Usage:
#   REWARD_HORIZON=6 bash examples/world_model/train_reward_wm_sft.sh
#
# Joint inverse-action WM (s_t, s_{t+1}) -> a_t from same batch (needs state_after column):
#   INVERSE_ACTION_WM=true REWARD_HORIZON=6 bash examples/world_model/train_reward_wm_sft.sh
#
# Return-conditioned planning (DT-style): (s_t, R̂=+4) -> 6 action tokens from dataset trajectory:
#   PLANNING_WM=true REWARD_HORIZON=6 bash examples/world_model/train_reward_wm_sft.sh
#
# Max-return planner with advantage loss (Ĝ vs G_data baseline, w(A)=σ(A/T)):
#   PLANNING_ADVANTAGE_WM=true REWARD_HORIZON=6 bash examples/world_model/train_reward_wm_sft.sh
# Separate optimizer steps (default): reward step then planner step; set false for joint step:
#   PLANNING_ADV_SEPARATE_STEP=false PLANNING_ADVANTAGE_WM=true ...
#
# Reward + inverse + planning (recommended for H=6):
#   CUDA_VISIBLE_DEVICES=1 INVERSE_ACTION_WM=true PLANNING_WM=true REWARD_HORIZON=6 \
#     bash examples/world_model/train_reward_wm_sft.sh
#
# Common overrides:
#   DATA_DIR=data/reward_wm_debug_square_8x8_h6 MODEL=Qwen/Qwen2.5-1.5B-Instruct bash ...
#   EXP_NAME=rm_sft_test TOTAL_EPOCHS=1 LR=1e-5 bash examples/world_model/train_reward_wm_sft.sh

# Reward lookahead: 1 = single token (legacy); >1 = multi-step rows in dataset.
REWARD_HORIZON="${REWARD_HORIZON:-6}"

# Dataset (auto: data/reward_wm_debug_square_8x8_h{REWARD_HORIZON})
DATA_DIR="${DATA_DIR:-data/reward_wm_debug_square_8x8_h${REWARD_HORIZON}}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$DATA_DIR/val.parquet}"

# Model / output
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DEV_TAG_RAW="${CUDA_VISIBLE_DEVICES:-gpu0}"
DEV_TAG="${DEV_TAG_RAW//,/__}"
RUN_TAG="$(date +%Y%m%d-%H%M%S)_${DEV_TAG}"
OUT_DIR="${OUT_DIR:-/tmp/reward_wm_sft_${RUN_TAG}}"

# Training
# total_training_steps caps the run; total_epochs must be large enough to reach it.
# With REWARD_HORIZON=3, batch must divide by lcm(3, MICRO_BSZ); 128 -> effective 120.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
TRAIN_BSZ="${TRAIN_BSZ:-720}"
MICRO_BSZ="${MICRO_BSZ:-4}"
LR="${LR:-1e-4}"
MAX_LEN="${MAX_LEN:-2048}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Logging
PROJECT_NAME="${PROJECT_NAME:-verl_agent_caged_craftext}"
EXP_NAME="${EXP_NAME:-reward_wm_sft_h${REWARD_HORIZON}_${RUN_TAG}}"
LOG_VAL_TABLES="${LOG_VAL_TABLES:-true}"
# Stratified val PNG table: VAL_TABLE_PER_H rows per horizon (H=3 -> 60, H=8 -> 160).
VAL_TABLE_PER_H="${VAL_TABLE_PER_H:-40}"
VAL_TABLE_N="${VAL_TABLE_N:-$(( VAL_TABLE_PER_H * REWARD_HORIZON ))}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-20}"
# Val metrics: cap batches (micro_bsz=4 -> 64 batches ~ 256 samples). 0 = full val parquet (slow).
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-64}"

# true = equal h1..hH per train batch; false = random shuffle (h mix varies per step).
REWARD_WM_BALANCE_BATCHES="${REWARD_WM_BALANCE_BATCHES:-true}"
# Joint inverse-action WM SFT: (s_t, s_{t+1}) -> a_t from same batch as reward (after reward backward).
INVERSE_ACTION_WM="${INVERSE_ACTION_WM:-false}"
INVERSE_ACTION_LOSS_COEF="${INVERSE_ACTION_LOSS_COEF:-1.0}"
INVERSE_VAL_TABLE_N="${INVERSE_VAL_TABLE_N:-40}"
# Decision Transformer-style planning: target cumulative return -> H action tokens.
PLANNING_WM="${PLANNING_WM:-false}"
PLANNING_LOSS_COEF="${PLANNING_LOSS_COEF:-1.0}"
PLANNING_VAL_TABLE_N="${PLANNING_VAL_TABLE_N:-40}"
# Advantage-loss planner: sample plan, score via reward WM, baseline G_data from dataset.
PLANNING_ADVANTAGE_WM="${PLANNING_ADVANTAGE_WM:-false}"
PLANNING_ADV_SIGMOID_T="${PLANNING_ADV_SIGMOID_T:-1.0}"
# Smaller micro-batch for planning advantage (sample + reward WM scoring + 1 grad forward).
PLANNING_ADV_MICRO_BSZ="${PLANNING_ADV_MICRO_BSZ:-4}"
# true = reward update then planner update (two optimizer steps); false = joint grad accum.
PLANNING_ADV_SEPARATE_STEP="${PLANNING_ADV_SEPARATE_STEP:-true}"
# DataLoader workers: 0 avoids JAX+fork deadlock after caged_craftext import.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# Ensure Comet uses the same run name.
export RUN_NAME="${RUN_NAME:-$EXP_NAME}"

# Comet ML (match PPO job scripts default)
export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"

# Ensure comet_ml exists when comet is enabled
python3 -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

if [ ! -f "$TRAIN_PARQUET" ] || [ ! -f "$VAL_PARQUET" ]; then
  echo "[ERROR] Missing parquet for REWARD_HORIZON=$REWARD_HORIZON:" >&2
  echo "  train: $TRAIN_PARQUET" >&2
  echo "  val:   $VAL_PARQUET" >&2
  echo "Collect first:" >&2
  echo "  REWARD_HORIZON=$REWARD_HORIZON bash examples/world_model/collect_reward_wm_dataset_debug_square.sh" >&2
  exit 1
fi

echo "[INFO] REWARD_HORIZON=$REWARD_HORIZON  DATA_DIR=$DATA_DIR"

if [ "$INVERSE_ACTION_WM" = "true" ] || [ "$INVERSE_ACTION_WM" = "1" ]; then
  echo "[INFO] INVERSE_ACTION_WM=true  loss_coef=$INVERSE_ACTION_LOSS_COEF (same batch as reward)"
  INVERSE_HYDRA_ARGS=(
    +trainer.inverse_action_wm.enable=true
    +trainer.inverse_action_wm.loss_coef="$INVERSE_ACTION_LOSS_COEF"
    +trainer.inverse_action_wm.val_enable="$LOG_VAL_TABLES"
    +trainer.inverse_action_wm.val_table_n="$INVERSE_VAL_TABLE_N"
  )
else
  INVERSE_HYDRA_ARGS=(+trainer.inverse_action_wm.enable=false)
fi

if [ "$PLANNING_ADVANTAGE_WM" = "true" ] || [ "$PLANNING_ADVANTAGE_WM" = "1" ]; then
  echo "[INFO] PLANNING_ADVANTAGE_WM=true  loss_coef=$PLANNING_LOSS_COEF  sigmoid_T=$PLANNING_ADV_SIGMOID_T  adv_micro_bsz=$PLANNING_ADV_MICRO_BSZ  separate_step=$PLANNING_ADV_SEPARATE_STEP  horizon=$REWARD_HORIZON"
  PLANNING_HYDRA_ARGS=(
    +trainer.planning_wm.enable=false
    +trainer.planning_wm.advantage_enable=true
    +trainer.planning_wm.advantage_sigmoid_T="$PLANNING_ADV_SIGMOID_T"
    +trainer.planning_wm.advantage_micro_batch_size="$PLANNING_ADV_MICRO_BSZ"
    +trainer.planning_wm.advantage_separate_optimizer_step="$PLANNING_ADV_SEPARATE_STEP"
    +trainer.planning_wm.loss_coef="$PLANNING_LOSS_COEF"
    +trainer.planning_wm.val_enable="$LOG_VAL_TABLES"
    +trainer.planning_wm.val_table_n="$PLANNING_VAL_TABLE_N"
    +data.planning_wm_enable=true
    +data.planning_advantage_wm_enable=true
    +data.planning_wm_horizon="$REWARD_HORIZON"
  )
elif [ "$PLANNING_WM" = "true" ] || [ "$PLANNING_WM" = "1" ]; then
  echo "[INFO] PLANNING_WM=true  loss_coef=$PLANNING_LOSS_COEF  horizon=$REWARD_HORIZON"
  PLANNING_HYDRA_ARGS=(
    +trainer.planning_wm.enable=true
    +trainer.planning_wm.advantage_enable=false
    +trainer.planning_wm.loss_coef="$PLANNING_LOSS_COEF"
    +trainer.planning_wm.val_enable="$LOG_VAL_TABLES"
    +trainer.planning_wm.val_table_n="$PLANNING_VAL_TABLE_N"
    +data.planning_wm_enable=true
    +data.planning_advantage_wm_enable=false
    +data.planning_wm_horizon="$REWARD_HORIZON"
  )
else
  PLANNING_HYDRA_ARGS=(
    +trainer.planning_wm.enable=false
    +trainer.planning_wm.advantage_enable=false
    +data.planning_wm_enable=false
    +data.planning_advantage_wm_enable=false
  )
fi

python3 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$VAL_PARQUET" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length="$MAX_LEN" \
  data.truncation=error \
  data.train_batch_size="$TRAIN_BSZ" \
  data.micro_batch_size_per_gpu="$MICRO_BSZ" \
  +data.num_workers="$DATALOADER_NUM_WORKERS" \
  model.partial_pretrain="$MODEL" \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.logger='[console,comet]' \
  +trainer.val_every_steps="$VAL_EVERY_STEPS" \
  +trainer.val_max_batches="$VAL_MAX_BATCHES" \
  +trainer.reward_wm_val.enable="$LOG_VAL_TABLES" \
  +trainer.reward_wm_val.table_n="$VAL_TABLE_N" \
  +trainer.reward_horizon="$REWARD_HORIZON" \
  +trainer.reward_wm_balance_horizon_batches="$REWARD_WM_BALANCE_BATCHES" \
  "${INVERSE_HYDRA_ARGS[@]}" \
  "${PLANNING_HYDRA_ARGS[@]}" \
  optim.lr="$LR" \
  optim.lr_scheduler=constant \
  optim.warmup_steps_ratio=0 \
  "$@"

