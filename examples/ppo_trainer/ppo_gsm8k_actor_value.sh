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
#   ACTOR_VALUE_ONLINE_REWARD_WM=true bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
#   ACTOR_VALUE_MC_Q_WM=true bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
#   bash examples/ppo_trainer/ppo_gsm8k_actor_value_reward_wm.sh

set -e

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0.1}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--0.2}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-1.2}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.2}"
ACTOR_VALUE_ONLINE_REWARD_WM="${ACTOR_VALUE_ONLINE_REWARD_WM:-false}"
ACTOR_VALUE_MC_Q_WM="${ACTOR_VALUE_MC_Q_WM:-false}"
ACTOR_VALUE_PLAN_Q_WM="${ACTOR_VALUE_PLAN_Q_WM:-false}"
ACTOR_VALUE_REWARD_WM_LOSS_COEF="${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}"
ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}}"
ACTOR_VALUE_REWARD_WM_PROMPT_STYLE="${ACTOR_VALUE_REWARD_WM_PROMPT_STYLE:-gsm8k}"
ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH="${ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH:-2048}"
GSM8K_MC_Q_STEP_SPLIT="${GSM8K_MC_Q_STEP_SPLIT:-newline}"
GSM8K_MC_Q_MAX_STEPS="${GSM8K_MC_Q_MAX_STEPS:-32}"
GSM8K_PLAN_Q_HORIZON="${GSM8K_PLAN_Q_HORIZON:-100}"
GSM8K_PLAN_Q_GAMMA="${GSM8K_PLAN_Q_GAMMA:-1.0}"
GSM8K_PLAN_Q_STEP_SPLIT="${GSM8K_PLAN_Q_STEP_SPLIT:-newline}"
GSM8K_PLAN_Q_MAX_PROMPT_LENGTH="${GSM8K_PLAN_Q_MAX_PROMPT_LENGTH:-32768}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-128}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-128}"

if [ "$ACTOR_VALUE_PLAN_Q_WM" = true ]; then
  ACTOR_VALUE_ONLINE_REWARD_WM=false
  ACTOR_VALUE_MC_Q_WM=false
fi

if [ "$ACTOR_VALUE_MC_Q_WM" = true ]; then
  ACTOR_VALUE_ONLINE_REWARD_WM=true
  ACTOR_VALUE_REWARD_WM_PROMPT_STYLE=gsm8k_mc_q
fi

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
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA (rank=$ACTOR_LORA_RANK alpha=$ACTOR_LORA_ALPHA, actor=value same weights)"
echo "[INFO] actor_value_online_reward_wm: $ACTOR_VALUE_ONLINE_REWARD_WM (style=$ACTOR_VALUE_REWARD_WM_PROMPT_STYLE, loss_coef=$ACTOR_VALUE_REWARD_WM_LOSS_COEF)"
echo "[INFO] actor_value_online_plan_q_wm: $ACTOR_VALUE_PLAN_Q_WM (horizon=$GSM8K_PLAN_Q_HORIZON fixed_only=true style=gsm8k)"
if [ "$ACTOR_VALUE_MC_Q_WM" = true ]; then
  echo "[INFO] MC-Q WM: step_split=$GSM8K_MC_Q_STEP_SPLIT max_steps_per_traj=$GSM8K_MC_Q_MAX_STEPS (target=final episode success j/k)"
fi

export RUN_NAME="${RUN_NAME:-PPO GSM8K dual-prompt actor value}"

export ACTOR_LORA_RANK
export ACTOR_LORA_ALPHA

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
  trainer.default_local_dir=training_checkpoints/verl_agent_gsm8k_actor_value \
  trainer.actor_value_online_reward_wm.enable="$ACTOR_VALUE_ONLINE_REWARD_WM" \
  trainer.actor_value_online_reward_wm.loss_coef="$ACTOR_VALUE_REWARD_WM_LOSS_COEF" \
  trainer.actor_value_online_reward_wm.prompt_style="$ACTOR_VALUE_REWARD_WM_PROMPT_STYLE" \
  trainer.actor_value_online_reward_wm.max_prompt_length="$ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH" \
  trainer.actor_value_online_reward_wm.gsm8k_mc_q_step_split="$GSM8K_MC_Q_STEP_SPLIT" \
  trainer.actor_value_online_reward_wm.gsm8k_mc_q_max_steps_per_traj="$GSM8K_MC_Q_MAX_STEPS" \
  trainer.actor_value_online_plan_q_wm.enable="$ACTOR_VALUE_PLAN_Q_WM" \
  trainer.actor_value_online_plan_q_wm.loss_coef="$ACTOR_VALUE_PLAN_Q_LOSS_COEF" \
  trainer.actor_value_online_plan_q_wm.prompt_style=gsm8k \
  trainer.actor_value_online_plan_q_wm.plan_horizon="$GSM8K_PLAN_Q_HORIZON" \
  trainer.actor_value_online_plan_q_wm.fixed_horizon_only=True \
  trainer.actor_value_online_plan_q_wm.gamma="$GSM8K_PLAN_Q_GAMMA" \
  trainer.actor_value_online_plan_q_wm.gsm8k_step_split="$GSM8K_PLAN_Q_STEP_SPLIT" \
  trainer.actor_value_online_plan_q_wm.max_prompt_length="$GSM8K_PLAN_Q_MAX_PROMPT_LENGTH" \
  trainer.validation_plan_q.max_prompt_length="$GSM8K_PLAN_Q_MAX_PROMPT_LENGTH" \
  "$@"
