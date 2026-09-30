#!/usr/bin/env bash
# Bare-metal launcher for aicenter3 (no docker group).
# Uses quantization/async venv; skips conda activate via CONDA_DEFAULT_ENV.
set -euo pipefail
cd /home/gorbov_gv/safe_rl_nlp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS="${N_GPUS:-2}"
export PYTHONPATH="/home/gorbov_gv/safe_rl_nlp:/home/gorbov_gv/safe_rl_nlp/caged_craftext:${PYTHONPATH:-}"
export CAGED_CRAFTEXT_PATH="/home/gorbov_gv/safe_rl_nlp/caged_craftext"
export CRAFTAX_PATH="/home/gorbov_gv/safe_rl_nlp/caged_craftext/Craftax"
# Comet credentials are never committed: export COMET_API_KEY, or keep it in the
# git-ignored ~/.config/verl_comet.env (see experiment_dashboard/.env.example).
if [ -f "$HOME/.config/verl_comet.env" ]; then . "$HOME/.config/verl_comet.env"; fi
export COMET_API_KEY="${COMET_API_KEY:?COMET_API_KEY is unset (see ~/.config/verl_comet.env)}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}"
# UI project uses hyphen; job default is underscore.
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl-agent-caged-craftext}"
export COMET_TIMEOUT="${COMET_TIMEOUT:-120}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp_gorbov}"
export PATH="/home/gorbov_gv/quantization/async/.venv/bin:$PATH"
# Skip conda activate branch in run_caged_craftext_lora_job.sh
export CONDA_DEFAULT_ENV=verl-agent-311
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Match ++env.use_jax_gpu=False — prevent jax from grabbing H100s during env build.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"

mkdir -p logdir "$RAY_TEMP_DIR" "$HOME/data/verl-agent/text"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logdir/ppo_debug_square_gt_g099_${STAMP}.log"
PIDFILE="logdir/ppo_debug_square_gt_g099_${STAMP}.pid"

echo "[INFO] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse HEAD 2>/dev/null || echo NOGIT)" | tee "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$LOG"

# Ensure text parquet exists (idempotent). Prefer already-copied files.
if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
  python -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size 64 \
    --val_data_size 128 \
    --local_dir "$HOME/data/verl-agent/" >>"$LOG" 2>&1 || {
    echo "[WARN] prepare failed; job script will retry" | tee -a "$LOG"
  }
fi
# Banner in ppo_debug_square.sh still says r_t; real token score comes from HYPER overrides.
echo "[INFO] expect remaining G_t^0.99 via reward_model.use_remaining_return_as_token_reward=True" | tee -a "$LOG"

chmod +x examples/ppo_trainer/ppo_debug_square_gt_g099.sh examples/ppo_trainer/ppo_debug_square.sh

nohup bash examples/ppo_trainer/ppo_debug_square_gt_g099.sh >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "[INFO] started pid=$PID" | tee -a "$LOG"
echo "[INFO] log=$LOG" | tee -a "$LOG"
echo "[INFO] pidfile=$PIDFILE" | tee -a "$LOG"
sleep 8
tail -n 60 "$LOG" || true
