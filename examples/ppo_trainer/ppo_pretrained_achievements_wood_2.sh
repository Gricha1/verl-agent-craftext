#!/bin/bash
# PPO pretrained achievements_wood, action-only (no reasoning).
#
# Key settings:
# - NO_REASONING=true          -> env.enable_reasoning=False
# - PROMPT_TEMPLATE=default_template (uses <action>...</action>, does NOT request reasoning)
#
# Usage:
#   bash examples/ppo_trainer/ppo_pretrained_achievements_wood_2.sh

set -e

# Path to checkpoint (edit if needed)
CHECKPOINT_PATH="pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000"

# Conversion params (must match SFT params)
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_RANK=64
LORA_ALPHA=64
TARGET_MODULES="all-linear"
NUM_GPUS=2  # Must match trainer.n_gpus_per_node

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
echo "Starting PPO training (action-only output)"
echo "=========================================="

# Args mapping for run_caged_craftext_lora_job.sh (positional):
# 1 ENGINE
# 2 LOG_PROB_ACTION_ONLY
# 3 NO_REASONING
# 4 NUM_GPUS
# 5 NUM_ENVS
# 6 USE_LORA
# 7 USE_WB
# 8 VLLM_PORT
# 9 USE_COMET
# 10 PROMPT_TEMPLATE_TYPE
# 11 MAX_STEPS
# 12 OBSERVATION_TYPE
bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  32 \
  32 \
  false \
  false \
  8000 \
  true \
  default_template \
  10 \
  ascii \
  ++env.craftext_settings='achievements_wood' \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT_PATH" \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_wood_action_only

