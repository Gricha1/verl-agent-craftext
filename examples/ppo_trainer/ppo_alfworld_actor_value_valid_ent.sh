#!/bin/bash
# PPO AlfWorld: single LLM actor+value (return bin head), entropy over admissible actions.
#
# Env reward is normalized to [0,1] (won -> 1.0). Default return bins [-0.2, 1.2] step 0.2
# map MC targets: fail G=0 -> bin B(0), win G=1 -> bin G(1.0).
#
# Dual-prompt:
#   actor prompt -> 1 encoded admissible-action token
#   critic prompt (single_token_return) -> 1 return bin token
#
# Usage:
#   bash examples/ppo_trainer/ppo_alfworld_actor_value_valid_ent.sh
#   CRITIC_WARMUP=10 bash examples/ppo_trainer/ppo_alfworld_actor_value_valid_ent.sh

set -e

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0.1}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--0.2}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-1.2}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.2}"
ACTOR_VALUE_ONLINE_REWARD_WM="${ACTOR_VALUE_ONLINE_REWARD_WM:-false}"
ACTOR_VALUE_PLAN_Q_WM="${ACTOR_VALUE_PLAN_Q_WM:-false}"
ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-0.1}"
ALFWORLD_PLAN_Q_HORIZON="${ALFWORLD_PLAN_Q_HORIZON:-6}"
ALFWORLD_PLAN_Q_GAMMA="${ALFWORLD_PLAN_Q_GAMMA:-1.0}"
ALFWORLD_PLAN_Q_MAX_STEPS="${ALFWORLD_PLAN_Q_MAX_STEPS:-50}"
ALFWORLD_PLAN_Q_MAX_PROMPT_LENGTH="${ALFWORLD_PLAN_Q_MAX_PROMPT_LENGTH:-4096}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-64}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-64}"

if [ "$ACTOR_VALUE_PLAN_Q_WM" = true ]; then
  ACTOR_VALUE_ONLINE_REWARD_WM=false
fi

echo "=========================================="
echo "PPO AlfWorld (dual-prompt actor-value + admissible entropy)"
echo "=========================================="
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[INFO] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[INFO] critic: return bins [$RETURN_BIN_MIN,$RETURN_BIN_MAX] step=$RETURN_BIN_STEP"
echo "[INFO] entropy: H over dynamic admissible tokens (entropy_over_valid_actions=True)"

export RUN_NAME="${RUN_NAME:-PPO AlfWorld actor-value admissible entropy}"
export ACTOR_LORA_RANK
export ACTOR_LORA_ALPHA
export VAL_DATA_SIZE

bash examples/ppo_trainer/run_alfworld_lora_job.sh \
  vllm \
  false \
  "$TRAIN_BATCH_SIZE" \
  "$MAX_RESPONSE_LENGTH" \
  "$TOTAL_EPOCHS" \
  "$USE_ACTOR_LORA" \
  "$CRITIC_WARMUP" \
  +env.prompt_template_type=single_token_action \
  +env.value_prompt_template_type=single_token_return \
  +env.value_return_min="$RETURN_BIN_MIN" \
  +env.value_return_max="$RETURN_BIN_MAX" \
  +env.value_return_bin_step="$RETURN_BIN_STEP" \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.single_token_actions=True \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps="$ACTOR_VALUE_SEPARATE_STEPS" \
  actor_rollout_ref.actor.actor_value_target_encoding="$ACTOR_VALUE_TARGET_ENCODING" \
  actor_rollout_ref.actor.actor_value_entropy_coef="$ACTOR_VALUE_ENTROPY_COEF" \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min="$RETURN_BIN_MIN" \
  actor_rollout_ref.actor.actor_value_return_max="$RETURN_BIN_MAX" \
  actor_rollout_ref.actor.actor_value_return_bin_step="$RETURN_BIN_STEP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_alfworld_actor_value_valid_ent \
  trainer.actor_value_online_reward_wm.enable="$ACTOR_VALUE_ONLINE_REWARD_WM" \
  trainer.actor_value_online_plan_q_wm.enable="$ACTOR_VALUE_PLAN_Q_WM" \
  trainer.actor_value_online_plan_q_wm.loss_coef="$ACTOR_VALUE_PLAN_Q_LOSS_COEF" \
  trainer.actor_value_online_plan_q_wm.prompt_style=alfworld \
  trainer.actor_value_online_plan_q_wm.plan_horizon="$ALFWORLD_PLAN_Q_HORIZON" \
  trainer.actor_value_online_plan_q_wm.gamma="$ALFWORLD_PLAN_Q_GAMMA" \
  trainer.actor_value_online_plan_q_wm.max_steps_per_traj="$ALFWORLD_PLAN_Q_MAX_STEPS" \
  trainer.actor_value_online_plan_q_wm.max_prompt_length="$ALFWORLD_PLAN_Q_MAX_PROMPT_LENGTH" \
  "$@"
