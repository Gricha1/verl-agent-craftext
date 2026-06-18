#!/bin/bash
# PPO on GSM8K: actor-value critic bins [-1,3] step 0.2 (21 levels), GAE-return value targets.
#
# Dual-prompt:
#   actor prompt -> full math solution (max_response_length tokens)
#   critic prompt (single_token_return) -> 1 return bin token
#
# GAE uses V(s) from critic forward on the same LLM; value loss = CE on return bin.
#
# Usage:
#   bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
#   CRITIC_WARMUP=20 TRAIN_BATCH_SIZE=32 bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh

set -e

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--1}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-3}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.2}"

echo "=========================================="
echo "PPO GSM8K (dual-prompt actor-value)"
echo "=========================================="
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[INFO] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[INFO] actor: full solution generation"
echo "[INFO] critic: return bins [$RETURN_BIN_MIN,$RETURN_BIN_MAX] step=$RETURN_BIN_STEP"
echo "[INFO] value targets: per-token GAE returns (actor_value_target_from_returns=True)"
echo "[INFO] critic_warmup: $CRITIC_WARMUP PPO steps"
echo "[INFO] actor_value_loss_coef: $ACTOR_VALUE_LOSS_COEF"
echo "[INFO] entropy: full vocabulary (entropy_over_valid_actions=False)"
echo "[INFO] checkpoints: disabled (trainer.save_freq=-1)"

export RUN_NAME="${RUN_NAME:-PPO GSM8K dual-prompt actor value}"

bash examples/ppo_trainer/run_gsm8k_lora_job.sh \
  vllm \
  false \
  "$TRAIN_BATCH_SIZE" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_EPOCHS" \
  true \
  "$CRITIC_WARMUP" \
  +env.value_prompt_template_type=single_token_return \
  +env.value_return_min="$RETURN_BIN_MIN" \
  +env.value_return_max="$RETURN_BIN_MAX" \
  +env.value_return_bin_step="$RETURN_BIN_STEP" \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps="$ACTOR_VALUE_SEPARATE_STEPS" \
  actor_rollout_ref.actor.actor_value_target_encoding="$ACTOR_VALUE_TARGET_ENCODING" \
  actor_rollout_ref.actor.actor_value_entropy_coef="$ACTOR_VALUE_ENTROPY_COEF" \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min="$RETURN_BIN_MIN" \
  actor_rollout_ref.actor.actor_value_return_max="$RETURN_BIN_MAX" \
  actor_rollout_ref.actor.actor_value_return_bin_step="$RETURN_BIN_STEP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=False \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_gsm8k_actor_value
