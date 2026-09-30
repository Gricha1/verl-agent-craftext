#!/bin/bash
# PPO debug_square_8x8 — one Qwen-VL: actor prompt (action token) + value prompt (return token).
# Image observation via CagedCraftextVLEnv; dual-prompt actor-value on a single VL backbone.
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh
#   NUM_OPTIMISTIC_ENVS=16 bash examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh
#   CRITIC_WARMUP=50 bash examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh
#   QWEN_VL_MODEL=Qwen/Qwen2-VL-2B-Instruct bash examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh

set -e

NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-32}"
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
ACTOR_VALUE_ENTROPY_COEF="${ACTOR_VALUE_ENTROPY_COEF:-0.1}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--5}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-6}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.4}"
ACTOR_VALUE_ONLINE_REWARD_WM="${ACTOR_VALUE_ONLINE_REWARD_WM:-false}"
ACTOR_VALUE_REWARD_WM_LOSS_COEF="${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}"
ACTOR_VALUE_PLAN_Q_WM="${ACTOR_VALUE_PLAN_Q_WM:-false}"
ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}}"
CRAFTEXT_PLAN_Q_HORIZON="${CRAFTEXT_PLAN_Q_HORIZON:-6}"
CRAFTEXT_PLAN_Q_GAMMA="${CRAFTEXT_PLAN_Q_GAMMA:-1.0}"
CRAFTEXT_MC_Q_MAX_STEPS="${CRAFTEXT_MC_Q_MAX_STEPS:-50}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

echo "=========================================="
echo "PPO debug_square_8x8 (Qwen VL actor-value)"
echo "=========================================="
echo "[INFO] QWEN_VL_MODEL=$QWEN_VL_MODEL"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] Map: 8x8 — stone / wood / water (adjacent = success)"
echo "[INFO] obs: pixel frame + <image> in prompt (CagedCraftextVLEnv)"
echo "[INFO] actor prompt: single_token_action (VL)"
echo "[INFO] value prompt: single_token_return_vl"
echo "[INFO] value warmup (critic_warmup): $CRITIC_WARMUP PPO steps"
echo "[INFO] actor_value_separate_optimizer_steps: $ACTOR_VALUE_SEPARATE_STEPS"
echo "[INFO] entropy: H over 17 action tokens (entropy_over_valid_actions=True)"
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA"
echo "[INFO] validation: every 20 PPO steps"
echo "[INFO] checkpoints -> training_checkpoints/verl_agent_caged_craftext_debug_square_vl_actor_value"

export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 VL actor-value}"

bash examples/ppo_trainer/run_caged_craftext_vl_lora_job.sh \
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
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
  +env.use_ray_text_render_workers=False \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  actor_rollout_ref.actor.actor_value_separate_optimizer_steps="$ACTOR_VALUE_SEPARATE_STEPS" \
  actor_rollout_ref.actor.actor_value_target_encoding="$ACTOR_VALUE_TARGET_ENCODING" \
  actor_rollout_ref.actor.actor_value_entropy_coef="$ACTOR_VALUE_ENTROPY_COEF" \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.actor.entropy_coeff_schedule.enable=False \
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
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_debug_square_vl_actor_value \
  trainer.actor_value_online_reward_wm.enable="$ACTOR_VALUE_ONLINE_REWARD_WM" \
  trainer.actor_value_online_reward_wm.loss_coef="$ACTOR_VALUE_REWARD_WM_LOSS_COEF" \
  trainer.actor_value_online_plan_q_wm.enable=$(if [ "$ACTOR_VALUE_PLAN_Q_WM" = "true" ]; then echo "True"; else echo "False"; fi) \
  trainer.actor_value_online_plan_q_wm.loss_coef="$ACTOR_VALUE_PLAN_Q_LOSS_COEF" \
  trainer.actor_value_online_plan_q_wm.plan_horizon="$CRAFTEXT_PLAN_Q_HORIZON" \
  trainer.actor_value_online_plan_q_wm.gamma="$CRAFTEXT_PLAN_Q_GAMMA" \
  trainer.actor_value_online_plan_q_wm.max_steps_per_traj="$CRAFTEXT_MC_Q_MAX_STEPS"
