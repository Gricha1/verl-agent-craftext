#!/usr/bin/env bash
# 16x16 shared actor/value PPO with action-name reasoning.
# Set ACTOR_VALUE_LOSS_TYPE=clipped_mae for CMAE (rho via CLIPPED_MAE_RHO).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-320}"
N_GPUS="${N_GPUS:-1}"
ACTOR_VALUE_LOSS_TYPE="${ACTOR_VALUE_LOSS_TYPE:-ce}"
CLIPPED_MAE_RHO="${CLIPPED_MAE_RHO:-0.2}"
RUN_NAME="${RUN_NAME:-ppo_caged_craftext_16x16_shared_av_reasoning_${ACTOR_VALUE_LOSS_TYPE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-training_checkpoints/$RUN_NAME}"

if [[ "$ACTOR_VALUE_LOSS_TYPE" != "ce" && "$ACTOR_VALUE_LOSS_TYPE" != "clipped_mae" ]]; then
  echo "[FATAL] ACTOR_VALUE_LOSS_TYPE must be ce or clipped_mae"
  exit 2
fi

echo "[INFO] 16x16 shared actor/value with reasoning"
echo "[INFO] response=<think>...</think><action>ACTION_NAME</action>"
echo "[INFO] history actions=50, reasoning=3; value loss=$ACTOR_VALUE_LOSS_TYPE rho=$CLIPPED_MAE_RHO"
echo "[INFO] n_gpus=$N_GPUS checkpoint=$CHECKPOINT_DIR"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm false false "$NUM_OPTIMISTIC_ENVS" "$MAX_RESPONSE_LENGTH" false false 8000 true \
  single_token_action_reasoning 0 ascii \
  ++env.craftext_settings=debug_square_16x16 \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio=8 \
  +env.use_ray_text_render_workers=False \
  +env.value_prompt_template_type=single_token_return \
  +env.value_return_min=-26 \
  +env.value_return_max=16 \
  +env.value_return_bin_step=1.5 \
  +env.reasoning_history_length=3 \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  env.history_length=50 \
  data.max_prompt_length=3072 \
  algorithm.use_actor_value_token=True \
  algorithm.gae_by_trajectory=False \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_target_encoding=two_hot \
  actor_rollout_ref.actor.actor_value_loss_type="$ACTOR_VALUE_LOSS_TYPE" \
  +actor_rollout_ref.actor.clipped_mae_rho="$CLIPPED_MAE_RHO" \
  actor_rollout_ref.actor.actor_value_loss_coef=1.0 \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps=true \
  actor_rollout_ref.actor.actor_value_entropy_coef=0.1 \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min=-26 \
  actor_rollout_ref.actor.actor_value_return_max=16 \
  actor_rollout_ref.actor.actor_value_return_bin_step=1.5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  critic.ppo_micro_batch_size_per_gpu=16 \
  critic.ppo_mini_batch_size=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=False \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.critic_warmup=0 \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.experiment_name="$RUN_NAME"
