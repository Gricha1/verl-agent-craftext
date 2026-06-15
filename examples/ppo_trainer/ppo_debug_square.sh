#!/bin/bash
# PPO on debug_square_8x8: 3 nav tasks, single-token actions, standard vocab entropy.
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square.sh

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"

echo "=========================================="
echo "PPO debug_square_8x8 (vocab entropy)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] Map: 8x8 — stone / wood / water (adjacent = success)"
echo "[INFO] prompt: single_token_action, max_response_length=1"
echo "[INFO] entropy: full vocabulary (entropy_over_valid_actions=False)"
echo "[INFO] validation: every 20 PPO steps"
echo "[INFO] checkpoints: every 20 PPO steps, keep last 1 -> training_checkpoints/verl_agent_caged_craftext_debug_square"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8}"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  "$NUM_OPTIMISTIC_ENVS" \
  1 \
  true \
  false \
  8000 \
  true \
  single_token_action \
  0 \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=False \
  trainer.resume_mode=disable \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square
