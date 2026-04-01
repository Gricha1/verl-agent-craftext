#!/bin/bash
# PPO pretrained achievements_wood with optimistic-parallel envs (NO ray env copies).
#
# This uses a single env object that batches N envs internally via
# OptimisticResetVecEnvWrapper (like caged_craftext/exps/ppo_lag*).
#
# Usage:
#   NUM_OPTIMISTIC_ENVS=8 bash examples/ppo_trainer/ppo_pretrained_achievements_wood_optimistic_parallel.sh

set -e

CHECKPOINT_PATH="pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000"

# Number of envs inside optimistic wrapper
NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-32}"
# reset_ratio inside OptimisticResetVecEnvWrapper (must divide NUM_OPTIMISTIC_ENVS)
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"

# Conversion params (must match SFT params)
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_RANK=64
LORA_ALPHA=64
TARGET_MODULES="all-linear"
NUM_GPUS=2

echo "=========================================="
echo "Checkpoint conversion check (if needed)"
echo "=========================================="

bash examples/ppo_trainer/convert_lora_to_fsdp_if_needed.sh \
  "$CHECKPOINT_PATH" \
  "$BASE_MODEL" \
  "$LORA_RANK" \
  "$LORA_ALPHA" \
  "$TARGET_MODULES" \
  "$NUM_GPUS"

echo ""
echo "=========================================="
echo "Starting PPO training (optimistic-parallel envs)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"

# Positional args for run_caged_craftext_lora_job.sh:
# 1 ENGINE
# 2 LOG_PROB_ACTION_ONLY
# 3 NO_REASONING
# 4 TRAIN_DATA_SIZE (== number of parallel envs)
# 5 MAX_RESPONSE_LENGTH
# 6 AUTO_RESET
# 7 USE_ACTION_HEAD
# 8 total_epochs
# 9 USE_ACTOR_LORA
# 10 PROMPT_TEMPLATE_TYPE
# 11 CRITIC_WARMUP
# 12 OBSERVATION_TYPE
bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  "$NUM_OPTIMISTIC_ENVS" \
  32 \
  false \
  false \
  8000 \
  true \
  default_template \
  10 \
  ascii \
  ++env.craftext_settings='achievements_wood' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=True \
  +env.text_render_ray_num_cpus=0.25 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT_PATH" \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_wood_optimistic_parallel

