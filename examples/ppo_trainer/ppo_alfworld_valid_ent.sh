#!/bin/bash
# PPO dual on AlfWorld: single-token actions + entropy over admissible commands.
#
# Usage:
#   bash examples/ppo_trainer/ppo_alfworld_valid_ent.sh
#   TRAIN_BATCH_SIZE=64 bash examples/ppo_trainer/ppo_alfworld_valid_ent.sh

set -e

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-64}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-64}"
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-0}"
CRITIC_LORA_ALPHA="${CRITIC_LORA_ALPHA:-16}"

echo "=========================================="
echo "PPO AlfWorld dual (admissible-action entropy)"
echo "=========================================="
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[INFO] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[INFO] entropy: H over dynamic admissible tokens (entropy_over_valid_actions=True)"

export RUN_NAME="${RUN_NAME:-PPO AlfWorld dual admissible entropy}"
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
  0 \
  +env.prompt_template_type=single_token_action \
  actor_rollout_ref.actor.single_token_actions=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  critic.model.lora_rank="$CRITIC_LORA_RANK" \
  critic.model.lora_alpha="$CRITIC_LORA_ALPHA" \
  critic.ppo_micro_batch_size_per_gpu=128 \
  critic.ppo_mini_batch_size=256 \
  critic.model.fsdp_config.param_offload=False \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_alfworld_ppo_dual_valid_ent \
  "$@"
