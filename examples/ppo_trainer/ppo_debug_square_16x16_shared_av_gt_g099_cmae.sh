#!/usr/bin/env bash
# Shared actor-value PPO: debug_square_16x16, history=50, G_t^0.99, two-hot CMAE.
# Matched to ds8_shared_av_gt_g099_h50 except map + 16x16 bins [-26,16]/1.5.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS="${N_GPUS:-2}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export FORCE_NEW_RAY_CLUSTER="${FORCE_NEW_RAY_CLUSTER:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_YAML="${HYPER_YAML:-$SCRIPT_DIR/config/ppo_debug_square_16x16_shared_av_gt_g099_cmae.yaml}"

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_VALUE_SEPARATE_STEPS="${ACTOR_VALUE_SEPARATE_STEPS:-true}"
ACTOR_VALUE_TARGET_ENCODING="${ACTOR_VALUE_TARGET_ENCODING:-two_hot}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0.1}"
# 16x16-appropriate bins (NOT 8x8 [-5,6]/0.4 — would clip).
RETURN_BIN_MIN="${RETURN_BIN_MIN:--26}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-16}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-1.5}"
HISTORY_LENGTH="${HISTORY_LENGTH:-50}"
export RUN_NAME="${RUN_NAME:-ds16_shared_av_gt_g099_cmae_h50}"
CLIPPED_MAE_RHO="${CLIPPED_MAE_RHO:-0.2}"

echo "=========================================="
echo "PPO debug_square_16x16 SHARED actor-value G_t^0.99 clipped MAE"
echo "=========================================="
echo "[INFO] HYPER_YAML=$HYPER_YAML"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_GPUS=$N_GPUS"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS HISTORY_LENGTH=$HISTORY_LENGTH"
echo "[INFO] remaining_return_gamma=0.99 use_remaining_return_as_token_reward=True"
echo "[INFO] use_actor_value_token=True separate_critic=False value_loss=two_hot_clipped_mae rho=$CLIPPED_MAE_RHO"
echo "[INFO] bins=[$RETURN_BIN_MIN,$RETURN_BIN_MAX] step=$RETURN_BIN_STEP (16x16 range)"
echo "[INFO] gae_by_trajectory=False (no double-count)"
echo "[INFO] actor_lora stage checkpoints at global_step 40/110 (early/late)"

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
  "$CRITIC_WARMUP" \
  ascii \
  ++env.craftext_settings='debug_square_16x16' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  +env.value_prompt_template_type=single_token_return \
  +env.value_return_min="$RETURN_BIN_MIN" \
  +env.value_return_max="$RETURN_BIN_MAX" \
  +env.value_return_bin_step="$RETURN_BIN_STEP" \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  env.history_length="$HISTORY_LENGTH" \
  reward_model.use_episode_return_as_token_reward=False \
  reward_model.use_remaining_return_as_token_reward=True \
  reward_model.remaining_return_gamma=0.99 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.gae_by_trajectory=False \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps="$ACTOR_VALUE_SEPARATE_STEPS" \
  actor_rollout_ref.actor.actor_value_target_encoding="$ACTOR_VALUE_TARGET_ENCODING" \
  actor_rollout_ref.actor.actor_value_loss_type=clipped_mae \
  actor_rollout_ref.actor.actor_value_clipped_mae_rho="$CLIPPED_MAE_RHO" \
  actor_rollout_ref.actor.actor_value_entropy_coef="$ACTOR_VALUE_ENTROPY_COEF" \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min="$RETURN_BIN_MIN" \
  actor_rollout_ref.actor.actor_value_return_max="$RETURN_BIN_MAX" \
  actor_rollout_ref.actor.actor_value_return_bin_step="$RETURN_BIN_STEP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_over_valid_actions=True \
  actor_rollout_ref.actor.entropy_action_single_token_fastpath=True \
  actor_rollout_ref.actor.entropy_action_batched_forward=False \
  actor_rollout_ref.actor.entropy_action_length_normalize=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  ray_init.local_fs_capacity_threshold=0.999 \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.actor_lora_only_checkpoint=True \
  'trainer.actor_lora_stage_steps=[40,110]' \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_16x16_shared_av_gt_g099_cmae \
  trainer.actor_value_online_reward_wm.enable=False \
  trainer.actor_value_online_plan_q_wm.enable=False \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  "$@"
