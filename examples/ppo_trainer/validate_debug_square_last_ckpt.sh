#!/bin/bash
# Validate the latest checkpoint for debug_square_8x8.
#
# Produces:
# - GIF trajectory in ./gif/ (same as training validation video)
# - action histogram PNG in ./gif/ (same as training)
#
# Default checkpoint root: ppo_debug_square.sh (override for act_entropy run).
#
# GPU: stop any running training on the same node first (see preflight below).
# Usage:
#   bash examples/ppo_trainer/validate_debug_square_last_ckpt.sh
#   ALLOW_BUSY_GPU=1 bash ...   # skip GPU preflight (not recommended)

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-training_checkpoints/verl_agent_caged_craftext_debug_square}"
if [ -n "${1:-}" ] && [ -d "$1" ]; then
  CKPT_ROOT="$1"
  shift
fi

if [ ! -d "$CKPT_ROOT" ]; then
  echo "[ERROR] CKPT_ROOT not found: $CKPT_ROOT" >&2
  exit 1
fi

LATEST_FILE="$CKPT_ROOT/latest_checkpointed_iteration.txt"
if [ ! -f "$LATEST_FILE" ]; then
  echo "[ERROR] Missing $LATEST_FILE (no checkpoints yet?)" >&2
  exit 1
fi

STEP="$(cat "$LATEST_FILE" | tr -d ' \n\r\t')"
if [ -z "$STEP" ]; then
  echo "[ERROR] Empty latest_checkpointed_iteration.txt" >&2
  exit 1
fi

CKPT_PATH="$CKPT_ROOT/global_step_${STEP}"
if [ ! -d "$CKPT_PATH" ]; then
  echo "[ERROR] Checkpoint folder not found: $CKPT_PATH" >&2
  exit 1
fi

# --- GPU preflight: val loads actor FSDP + critic + vLLM (TP=2); needs ~all VRAM on 2 GPUs ---
if [ "${ALLOW_BUSY_GPU:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t _gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
  if [ "${#_gpu_pids[@]}" -gt 0 ]; then
    echo "[ERROR] GPU already in use by other process(es) (e.g. training still running):" >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv 2>/dev/null || nvidia-smi
    echo "" >&2
    echo "Stop training first, then re-run validation. Example:" >&2
    echo "  kill ${_gpu_pids[*]}   # or: pkill -f 'verl.trainer.main_ppo'" >&2
    echo "  bash examples/ppo_trainer/validate_debug_square_last_ckpt.sh" >&2
    echo "To override: ALLOW_BUSY_GPU=1 bash examples/ppo_trainer/validate_debug_square_last_ckpt.sh" >&2
    exit 1
  fi
fi

echo "=========================================="
echo "Validate debug_square_8x8 latest checkpoint"
echo "=========================================="
echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] latest_step=$STEP"
echo "[INFO] CKPT_PATH=$CKPT_PATH"
echo "[INFO] val do_sample=True, val temperature=1.0"
echo "[INFO] val-only footprint: train_batch=8, val_batch=8, JAX on CPU, vLLM gpu_mem=0.78"
echo "[INFO] outputs: ./gif/val_trajectory_step0.gif and ./gif/validation_action_hist_step0.png"

# Reuse the standard launcher. val_only=True exits after validation.
# Args to run_caged_craftext_lora_job.sh: ENGINE, log_prob_only, no_reasoning, train_size, max_resp_len, ...
bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  8 \
  1 \
  false \
  false \
  1 \
  true \
  single_token_action \
  0 \
  ascii \
  ++env.craftext_settings='debug_square_8x8' \
  +env.use_optimistic_parallel=True \
  +env.optimistic_reset_ratio=8 \
  +env.use_ray_text_render_workers=False \
  ++env.use_jax_gpu=False \
  data.val_batch_size=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.78 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  critic.model.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.env_val_video_freq=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_PATH" \
  trainer.default_local_dir="$CKPT_ROOT" \
  "$@"
