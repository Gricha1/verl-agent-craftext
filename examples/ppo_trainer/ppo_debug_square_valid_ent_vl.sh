#!/bin/bash
# PPO debug_square_8x8 — Qwen-VL dual-model (actor + critic), valid-action entropy.
# Analog of ppo_debug_square_valid_ent.sh with pixel observations.
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_valid_ent_vl.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square_valid_ent_vl.sh
#   ENTROPY_SINGLE_TOKEN_FASTPATH=false bash examples/ppo_trainer/ppo_debug_square_valid_ent_vl.sh

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
ENTROPY_SINGLE_TOKEN_FASTPATH="${ENTROPY_SINGLE_TOKEN_FASTPATH:-true}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-64}"
CRITIC_LORA_ALPHA="${CRITIC_LORA_ALPHA:-64}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

echo "=========================================="
echo "PPO debug_square_8x8 (Qwen VL valid-action entropy)"
echo "=========================================="
echo "[INFO] QWEN_VL_MODEL=$QWEN_VL_MODEL"
echo "[INFO] Actor VL + Critic VL (separate weights)"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] entropy: H over 17 action tokens"
echo "[INFO] entropy_action_single_token_fastpath: $ENTROPY_SINGLE_TOKEN_FASTPATH"
echo "[INFO] validation: every 20 PPO steps"
echo "[INFO] checkpoints -> training_checkpoints/verl_agent_caged_craftext_debug_square_vl_valid_ent"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 VL valid-action entropy}"

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
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  actor_rollout_ref.actor.entropy_action_single_token_fastpath="$ENTROPY_SINGLE_TOKEN_FASTPATH" \
  actor_rollout_ref.actor.entropy_action_batched_forward=False \
  actor_rollout_ref.actor.entropy_action_length_normalize=True \
  critic.model.lora_rank="$CRITIC_LORA_RANK" \
  critic.model.lora_alpha="$CRITIC_LORA_ALPHA" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_vl_valid_ent
