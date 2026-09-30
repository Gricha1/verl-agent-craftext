#!/bin/bash
# PPO debug_square_8x8 — classic PPO: separate Qwen-VL actor + Qwen-VL critic (vocab entropy).
# Analog of ppo_debug_square.sh with pixel observations.
#
# For one VL model (actor-value dual prompt) use ppo_debug_square_actor_value_vl.sh
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_vl.sh
#   NUM_OPTIMISTIC_ENVS=64 bash examples/ppo_trainer/ppo_debug_square_vl.sh

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-64}"
CRITIC_LORA_ALPHA="${CRITIC_LORA_ALPHA:-64}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

echo "=========================================="
echo "PPO debug_square_8x8 (Qwen VL dual-model)"
echo "=========================================="
echo "[INFO] QWEN_VL_MODEL=$QWEN_VL_MODEL"
echo "[INFO] Actor VL: action token + image"
echo "[INFO] Critic VL: value head + image (separate weights)"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] entropy: full vocabulary (entropy_over_valid_actions=False)"
echo "[INFO] validation: every 20 PPO steps"
echo "[INFO] checkpoints -> training_checkpoints/verl_agent_caged_craftext_debug_square_vl"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 VL}"

bash examples/ppo_trainer/run_caged_craftext_vl_dual_lora_job.sh \
  vllm \
  false \
  true \
  "$NUM_OPTIMISTIC_ENVS" \
  1 \
  false \
  false \
  8000 \
  "$USE_ACTOR_LORA" \
  single_token_action \
  0 \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  +env.value_return_min=-5 \
  +env.value_return_max=6 \
  +env.value_return_bin_step=0.4 \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=False \
  critic.model.lora_rank="$CRITIC_LORA_RANK" \
  critic.model.lora_alpha="$CRITIC_LORA_ALPHA" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_vl
