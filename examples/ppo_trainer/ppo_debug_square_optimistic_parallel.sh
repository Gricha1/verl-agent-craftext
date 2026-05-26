#!/bin/bash
# PPO on fixed 8x8 debug square map (tree border, corner blocks, center spawn).
# Optimistic-parallel envs, training from scratch (no SFT / no resume).
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_optimistic_parallel.sh
#   NUM_OPTIMISTIC_ENVS=8 bash examples/ppo_trainer/ppo_debug_square_optimistic_parallel.sh

set -e

# 32 envs + 32 Ray text-render actors + 2x vLLM ≈ 120GB host RAM → OOM on 128GB nodes.
NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-128}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"

echo "=========================================="
echo "PPO debug_square_8x8 (from scratch, optimistic-parallel)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] AUTO_RESET=false (one episode per rollout slot, up to env.max_steps)"
echo "[INFO] Map: 8x8, tree border, corners: stone / wood / water (3 nav tasks)"
echo "[INFO] Validation + val GIF in Comet every 100k env steps (trainer.env_val_video_freq)"
echo "[INFO] critic_warmup=0 (actor updates from 1st PPO step; CRITIC_WARMUP>0 delays actor/* in Comet)"
echo "[INFO] vLLM gpu_memory_utilization=0.55 (headroom after actor update; NUM_OPTIMISTIC_ENVS still 128)"
echo "[INFO] actor entropy_coeff=0.01, entropy_over_valid_actions=True (H over 17 <action>X</action>)"

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
  0 \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  ++env.use_jax_gpu=True \
  ++env.jax_gpu_fraction=0.15 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  actor_rollout_ref.actor.entropy_action_length_normalize=True \
  trainer.resume_mode=disable \
  trainer.env_val_video_freq=100000 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_optimistic_parallel
