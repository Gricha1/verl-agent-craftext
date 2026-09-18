#!/usr/bin/env bash
# GSM8K shared actor-value + LoRA64 + MAE value loss (no WM / plan-Q).
# Matched as closely as possible to dual LoRA64 baseline b7573e3e.
#
# Usage (on aicenter3, AFTER confirming free GPUs):
#   CUDA_VISIBLE_DEVICES=1,6 N_GPUS=2 bash examples/ppo_trainer/ppo_gsm8k_actor_value_mae_lora64.sh
#
# Prefer user-requested 2,3 when free:
#   CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 bash examples/ppo_trainer/ppo_gsm8k_actor_value_mae_lora64.sh

set -euo pipefail

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"
ACTOR_VALUE_LOSS_TYPE="${ACTOR_VALUE_LOSS_TYPE:-mae}"
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0.1}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--0.2}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-1.2}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.2}"
USE_ACTOR_LORA=true
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-64}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-64}"

export RUN_NAME="${RUN_NAME:-gsm8k_shared_av_lora64_mae}"
export ACTOR_LORA_RANK ACTOR_LORA_ALPHA
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl-agent-caged-craftext}"
# Comet credentials are never committed: export COMET_API_KEY, or keep it in the
# git-ignored ~/.config/verl_comet.env (see experiment_dashboard/.env.example).
if [ -f "$HOME/.config/verl_comet.env" ]; then . "$HOME/.config/verl_comet.env"; fi
export COMET_API_KEY="${COMET_API_KEY:?COMET_API_KEY is unset (see ~/.config/verl_comet.env)}"

echo "=========================================="
echo "PPO GSM8K shared actor-value + LoRA + MAE"
echo "=========================================="
echo "[INFO] value_loss_type=$ACTOR_VALUE_LOSS_TYPE (must be mae)"
echo "[INFO] encoding=$ACTOR_VALUE_TARGET_ENCODING"
echo "[INFO] LoRA rank=$ACTOR_LORA_RANK"
echo "[INFO] NO reward-WM / NO MC-Q / NO plan-Q"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} N_GPUS=${N_GPUS:-unset}"

bash examples/ppo_trainer/run_gsm8k_lora_job.sh \
  vllm \
  false \
  "$TRAIN_BATCH_SIZE" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_EPOCHS" \
  "$USE_ACTOR_LORA" \
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
  actor_rollout_ref.actor.actor_value_loss_type="$ACTOR_VALUE_LOSS_TYPE" \
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
  trainer.default_local_dir=training_checkpoints/verl_agent_gsm8k_actor_value_mae_lora64 \
  trainer.actor_value_online_reward_wm.enable=false \
  trainer.actor_value_online_plan_q_wm.enable=false \
  trainer.n_gpus_per_node="${N_GPUS:-2}" \
  trainer.nnodes=1 \
  trainer.experiment_name="$RUN_NAME" \
  "$@"
