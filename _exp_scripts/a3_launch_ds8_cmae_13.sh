#!/usr/bin/env bash
# aicenter3: 8x8 shared AV CMAE on physical GPUs 1+3 (non-adjacent -> disable P2P).
# Matched to ds8_shared_av_gt_g099_h50 (CE baseline 535900d8...) except value_loss.
set -euo pipefail
REPO=/home/gorbov_gv/safe_rl_nlp
cd "$REPO"

export CUDA_VISIBLE_DEVICES=1,3
export N_GPUS=2
export RUN_NAME=ds8_shared_av_gt_g099_cmae_h50
export SOURCE_GIT_COMMIT="$(cat SOURCE_GIT_COMMIT 2>/dev/null || echo ad08adef12fee9f6e1b852a0a6b9276ebee2b2da)"
# Comet credentials are never committed: export COMET_API_KEY, or keep it in the
# git-ignored ~/.config/verl_comet.env (see experiment_dashboard/.env.example).
if [ -f "$HOME/.config/verl_comet.env" ]; then . "$HOME/.config/verl_comet.env"; fi
export COMET_API_KEY="${COMET_API_KEY:?COMET_API_KEY is unset (see ~/.config/verl_comet.env)}"
export COMET_WORKSPACE=gregory-gorbov
export COMET_PROJECT_NAME=verl-agent-caged-craftext
export COMET_TIMEOUT=120
export FORCE_NEW_RAY_CLUSTER=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export JAX_PLATFORMS=cpu
export CRAFTAX_RELOAD_TEXTURES=True
# Disk-safe Ray temp on NFS (avoid /tmp/ray overflow lesson).
export RAY_TEMP_DIR=/mnt/aicenter1-datasets/gorbov_gv/ray_ds8_cmae
# Bin settings for 8x8 (not 16x16).
export RETURN_BIN_MIN=-5
export RETURN_BIN_MAX=6
export RETURN_BIN_STEP=0.4
export CLIPPED_MAE_RHO=0.2
export HISTORY_LENGTH=50

source /home/gorbov_gv/quantization/async/.venv/bin/activate
mkdir -p logdir training_checkpoints "$RAY_TEMP_DIR"

echo "===== preflight 8x8 CMAE ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i 1,3,4,5
for i in 1 3; do
  u=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${u:-999}" -gt 500 ]; then echo "[FATAL] GPU$i busy $u"; exit 2; fi
done

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_${STAMP}.log"
PIDFILE="logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_${STAMP}.pid"

{
  echo "[INFO] host=$(hostname) phys=1,3 CVD=$CUDA_VISIBLE_DEVICES RUN=$RUN_NAME rho=$CLIPPED_MAE_RHO"
  echo "[INFO] RAY_TEMP_DIR=$RAY_TEMP_DIR"
  echo "[INFO] bins=[$RETURN_BIN_MIN,$RETURN_BIN_MAX] step=$RETURN_BIN_STEP (8x8)"
  free -h | head -2
  df -h / | head -2
  df -h /mnt/aicenter1-datasets | head -2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i 1,3,4,5
} | tee "$LOG"

chmod +x examples/ppo_trainer/ppo_debug_square_8x8_shared_av_gt_g099_cmae.sh
nohup bash examples/ppo_trainer/ppo_debug_square_8x8_shared_av_gt_g099_cmae.sh \
  >"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[INFO] pid=$(cat "$PIDFILE") log=$REPO/$LOG"
sleep 20
tail -n 30 "$LOG" || true
ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd || true
nvidia-smi -i 1,3 --query-gpu=index,memory.used,utilization.gpu --format=csv
