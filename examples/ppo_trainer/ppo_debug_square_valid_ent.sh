#!/bin/bash
# PPO on debug_square_8x8: dual actor+critic, entropy H over 17 valid action tokens.
# Resources shuffled each reset; GAE along env trajectories (to episode end).
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_valid_ent.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square_valid_ent.sh
#   ENTROPY_SINGLE_TOKEN_FASTPATH=false bash ...  # slow B×17 teacher-forcing path

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-128}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
ENTROPY_SINGLE_TOKEN_FASTPATH="${ENTROPY_SINGLE_TOKEN_FASTPATH:-true}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-64}"
CRITIC_LORA_ALPHA="${CRITIC_LORA_ALPHA:-64}"

echo "=========================================="
echo "PPO debug_square_8x8 (dual + valid-action entropy)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] Map: 8x8 — stone / wood / water shuffled corners (adjacent = success)"
echo "[INFO] prompt: single_token_action, max_response_length=1"
echo "[INFO] entropy: H over 17 action tokens (entropy_over_valid_actions=True)"
echo "[INFO] GAE: by trajectory (returns to episode end)"
echo "[INFO] entropy_action_single_token_fastpath: $ENTROPY_SINGLE_TOKEN_FASTPATH"
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA (actor lora_rank=64 when true)"
echo "[INFO] CRITIC_LORA_RANK=$CRITIC_LORA_RANK CRITIC_LORA_ALPHA=$CRITIC_LORA_ALPHA"
echo "[INFO] auto_reset: false (one episode per env slot, max 50 steps)"
echo "[INFO] checkpoints -> training_checkpoints/verl_agent_caged_craftext_debug_square_valid_ent"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 dual valid-action entropy}"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
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
  algorithm.gae_by_trajectory=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
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
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_valid_ent
