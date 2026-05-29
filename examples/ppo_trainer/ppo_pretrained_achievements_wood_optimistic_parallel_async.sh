#!/bin/bash
# Same as ppo_pretrained_achievements_wood_optimistic_parallel.sh, plus bounded stale rollouts:
# every sync_every train steps perform full FSDP→vLLM sync; in between, skip weight copy (stale vLLM).
#
# Usage:
#   NUM_OPTIMISTIC_ENVS=8 bash examples/ppo_trainer/ppo_pretrained_achievements_wood_optimistic_parallel_async.sh

set -e

CHECKPOINT_PATH="pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000"

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-32}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"

# How often to fully sync weights to vLLM (default 2: full sync on steps 1,3,5,…)
STALE_SYNC_EVERY="${STALE_SYNC_EVERY:-2}"

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
echo "Starting PPO (optimistic-parallel + stale_rollout)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] trainer.stale_rollout: enabled=true sync_every=$STALE_SYNC_EVERY"

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
  trainer.stale_rollout.enabled=true \
  trainer.stale_rollout.sync_every="$STALE_SYNC_EVERY" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT_PATH" \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_wood_optimistic_parallel_async
