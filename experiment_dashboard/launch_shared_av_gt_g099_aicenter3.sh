#!/usr/bin/env bash
# Launch shared AV G_t^0.99 on aicenter3 GPUs 4,5 (freed after dual 90deefed).
set -euo pipefail
cd /home/gorbov_gv/safe_rl_nlp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export N_GPUS="${N_GPUS:-2}"
export PYTHONPATH="/home/gorbov_gv/safe_rl_nlp:/home/gorbov_gv/safe_rl_nlp/caged_craftext:${PYTHONPATH:-}"
export CAGED_CRAFTEXT_PATH="/home/gorbov_gv/safe_rl_nlp/caged_craftext"
export CRAFTAX_PATH="/home/gorbov_gv/safe_rl_nlp/caged_craftext/Craftax"
# Comet credentials are never committed: export COMET_API_KEY, or keep it in the
# git-ignored ~/.config/verl_comet.env (see experiment_dashboard/.env.example).
if [ -f "$HOME/.config/verl_comet.env" ]; then . "$HOME/.config/verl_comet.env"; fi
export COMET_API_KEY="${COMET_API_KEY:?COMET_API_KEY is unset (see ~/.config/verl_comet.env)}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl-agent-caged-craftext}"
export COMET_TIMEOUT="${COMET_TIMEOUT:-120}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp_gorbov_shared_av_gt_g099}"
export FORCE_NEW_RAY_CLUSTER="${FORCE_NEW_RAY_CLUSTER:-1}"
export PATH="/home/gorbov_gv/quantization/async/.venv/bin:$PATH"
export CONDA_DEFAULT_ENV=verl-agent-311
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT:-}"
export RUN_NAME="${RUN_NAME:-ds8_shared_av_gt_g099_h50}"
export NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-50}"

mkdir -p logdir "$RAY_TEMP_DIR" "$HOME/data/verl-agent/text" \
  training_checkpoints/verl_agent_caged_craftext_debug_square_shared_av_gt_g099

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logdir/ppo_debug_square_shared_av_gt_g099_${STAMP}.log"
PIDFILE="logdir/ppo_debug_square_shared_av_gt_g099_${STAMP}.pid"

{
  echo "[INFO] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
  echo "[INFO] SOURCE_GIT_COMMIT=${SOURCE_GIT_COMMIT:-UNSET}"
  echo "[INFO] RUN_NAME=$RUN_NAME"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
} | tee "$LOG"

chmod +x examples/ppo_trainer/ppo_debug_square_shared_av_gt_g099.sh

nohup bash examples/ppo_trainer/ppo_debug_square_shared_av_gt_g099.sh >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "[INFO] started pid=$PID" | tee -a "$LOG"
echo "[INFO] log=$LOG" | tee -a "$LOG"
echo "[INFO] pidfile=$PIDFILE" | tee -a "$LOG"
sleep 12
tail -n 80 "$LOG" || true
