#!/bin/bash
# PPO on debug_square_8x8: same LLM for actor (action token) and critic (return token).
#
# Dual-prompt, one token per forward:
#   actor prompt (single_token_action)  -> 1 action token
#   critic prompt (single_token_return) -> 1 return bin token (m..u = 0..8)
#
# GAE uses V(s) from critic forward; value loss = CE on return bin vs MC remaining return.
# critic_warmup: first N PPO steps train only return-token CE (policy frozen).
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
#   CRITIC_WARMUP=50 bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
#   ACTOR_VALUE_SEPARATE_STEPS=true bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
#   ACTOR_VALUE_TARGET_ENCODING=two_hot bash examples/ppo_trainer/ppo_debug_square_actor_value.sh

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ENTROPY_SINGLE_TOKEN_FASTPATH="${ENTROPY_SINGLE_TOKEN_FASTPATH:-true}"
ENTROPY_BAND_ENABLE="${ENTROPY_BAND_ENABLE:-false}"
ENTROPY_BAND_LOW="${ENTROPY_BAND_LOW:-0.7}"
ENTROPY_BAND_HIGH="${ENTROPY_BAND_HIGH:-1.4}"
ENTROPY_BAND_COEF_LR="${ENTROPY_BAND_COEF_LR:-0.05}"
ENTROPY_BAND_COEF_LOW="${ENTROPY_BAND_COEF_LOW:-0.0}"
ENTROPY_BAND_COEF_HIGH="${ENTROPY_BAND_COEF_HIGH:-1.0}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"

echo "=========================================="
echo "PPO debug_square_8x8 (dual-prompt actor-value)"
echo "=========================================="
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] Map: 8x8 — stone / wood / water (adjacent = success)"
echo "[INFO] actor prompt: single_token_action, max_response_length=1"
echo "[INFO] critic prompt: single_token_return -> return m..u (0..8 remaining MC return)"
echo "[INFO] critic: token head on same LLM (no separate critic network)"
echo "[INFO] value warmup (critic_warmup): $CRITIC_WARMUP PPO steps (return-token CE only)"
echo "[INFO] actor_value_separate_optimizer_steps: $ACTOR_VALUE_SEPARATE_STEPS"
echo "[INFO] actor_value_target_encoding: $ACTOR_VALUE_TARGET_ENCODING"
echo "[INFO] entropy: H over 17 action tokens (1 forward + mask, entropy_over_valid_actions=True)"
echo "[INFO] entropy_action_single_token_fastpath: $ENTROPY_SINGLE_TOKEN_FASTPATH"
echo "[INFO] entropy_band (AEnt): enable=$ENTROPY_BAND_ENABLE range=[$ENTROPY_BAND_LOW, $ENTROPY_BAND_HIGH]"
echo "[INFO] validation: every 20 PPO steps, 2 GIFs (actor prompt + critic prompt)"
echo "[INFO] checkpoints: every 20 PPO steps, keep last 1 -> training_checkpoints/verl_agent_caged_craftext_debug_square_actor_value"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 dual-prompt actor value}"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  "$NUM_OPTIMISTIC_ENVS" \
  1 \
  false \
  false \
  8000 \
  true \
  single_token_action \
  "$CRITIC_WARMUP" \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  +env.value_prompt_template_type=single_token_return \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps="$ACTOR_VALUE_SEPARATE_STEPS" \
  actor_rollout_ref.actor.actor_value_target_encoding="$ACTOR_VALUE_TARGET_ENCODING" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_coeff_schedule.enable=False \
  actor_rollout_ref.actor.entropy_coeff_schedule.schedule=log \
  actor_rollout_ref.actor.entropy_band.enable="$ENTROPY_BAND_ENABLE" \
  actor_rollout_ref.actor.entropy_band.low="$ENTROPY_BAND_LOW" \
  actor_rollout_ref.actor.entropy_band.high="$ENTROPY_BAND_HIGH" \
  actor_rollout_ref.actor.entropy_band.coef_lr="$ENTROPY_BAND_COEF_LR" \
  actor_rollout_ref.actor.entropy_band.coef_low="$ENTROPY_BAND_COEF_LOW" \
  actor_rollout_ref.actor.entropy_band.coef_high="$ENTROPY_BAND_COEF_HIGH" \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  actor_rollout_ref.actor.entropy_action_single_token_fastpath="$ENTROPY_SINGLE_TOKEN_FASTPATH" \
  actor_rollout_ref.actor.entropy_action_batched_forward=False \
  actor_rollout_ref.actor.entropy_action_length_normalize=True \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.resume_mode=disable \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_actor_value
