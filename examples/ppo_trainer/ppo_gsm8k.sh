#!/bin/bash
# PPO on GSM8K: full-text solutions, separate critic, standard vocabulary entropy.
#
# Usage:
#   bash examples/ppo_trainer/ppo_gsm8k.sh
#   TRAIN_BATCH_SIZE=32 bash examples/ppo_trainer/ppo_gsm8k.sh

set -e

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-128}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-128}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-128}"
CRITIC_LORA_ALPHA="${CRITIC_LORA_ALPHA:-128}"

echo "=========================================="
echo "PPO GSM8K (standard critic + vocab entropy)"
echo "=========================================="
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[INFO] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[INFO] single-turn env (max_steps=1), rule-based reward"
echo "[INFO] entropy: full vocabulary (entropy_over_valid_actions=False)"
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA (actor lora_rank=$ACTOR_LORA_RANK when true)"
echo "[INFO] CRITIC_LORA_RANK=$CRITIC_LORA_RANK CRITIC_LORA_ALPHA=$CRITIC_LORA_ALPHA"
echo "[INFO] checkpoints: disabled (trainer.save_freq=-1)"

export RUN_NAME="${RUN_NAME:-PPO GSM8K}"

export ACTOR_LORA_RANK
export ACTOR_LORA_ALPHA

bash examples/ppo_trainer/run_gsm8k_lora_job.sh \
  vllm \
  false \
  "$TRAIN_BATCH_SIZE" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_EPOCHS" \
  "$USE_ACTOR_LORA" \
  0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=False \
  critic.model.lora_rank="$CRITIC_LORA_RANK" \
  critic.model.lora_alpha="$CRITIC_LORA_ALPHA" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_gsm8k_ppo \
  "$@"
