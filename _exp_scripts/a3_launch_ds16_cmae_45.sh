#!/usr/bin/env bash
# aicenter3: 16x16 shared AV clipped-MAE on physical GPUs 4+5 (adjacent)
set -euo pipefail
REPO=/home/gorbov_gv/safe_rl_nlp
cd "$REPO"

# Phys GPUs 4+5. CUDA enumeration SKIPS broken phys GPU2, so CUDA indices
# 3,4 map to phys 4,5. (Index CVD=4,5 would land on phys 5+6; UUID-based CVD
# breaks verl int(gpu_id) parsing — both observed 2026-09-14.)
export CUDA_VISIBLE_DEVICES=3,4
export N_GPUS=2
export RUN_NAME=ds16_shared_av_gt_g099_cmae_h50
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
# Keep this path short: Ray creates AF_UNIX sockets below session_*/sockets.
export RAY_TEMP_DIR=/mnt/aicenter1-datasets/gorbov_gv/ray
export NUM_OPTIMISTIC_ENVS=64
export HISTORY_LENGTH=50
# physical GPUs 4+5
export CLIPPED_MAE_RHO=0.2

source /home/gorbov_gv/quantization/async/.venv/bin/activate
mkdir -p logdir training_checkpoints "$RAY_TEMP_DIR"

echo "===== preflight 16x16 CMAE ====="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
for i in 4 5; do
  u=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${u:-999}" -gt 500 ]; then echo "[FATAL] GPU$i busy $u"; exit 2; fi
done

python3 - << "PYEOF"
import os, sys, ctypes
try:
    import torch
except ImportError:
    sys.exit("[FATAL] torch unavailable for CVD check")
bus = [int(torch.cuda.get_device_properties(i).pci_bus_id) for i in range(torch.cuda.device_count())]
if bus != [0x81, 0xA1]:
    sys.exit(f"[FATAL] CVD maps to pci {bus}, expected phys 4,5 (0x81,0xA1). Broken-GPU enumeration changed?")
print(f"[INFO] CVD check OK: cuda0->0x81 (phys4), cuda1->0xA1 (phys5)")
PYEOF

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logdir/ppo_debug_square_16x16_shared_av_gt_g099_cmae_${STAMP}.log"
PIDFILE="logdir/ppo_debug_square_16x16_shared_av_gt_g099_cmae_${STAMP}.pid"

{
  echo "[INFO] host=$(hostname) phys=4,5 CVD=$CUDA_VISIBLE_DEVICES RUN=$RUN_NAME rho=$CLIPPED_MAE_RHO"
  echo "[INFO] RAY_TEMP_DIR=$RAY_TEMP_DIR"
  free -h | head -2
  df -h / | head -2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
} | tee "$LOG"

chmod +x examples/ppo_trainer/ppo_debug_square_16x16_shared_av_gt_g099_cmae.sh
# lower vLLM util to leave room after FSDP (past OOM at 0.75)
nohup bash examples/ppo_trainer/ppo_debug_square_16x16_shared_av_gt_g099_cmae.sh \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  trainer.n_gpus_per_node=2 \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[INFO] pid=$(cat "$PIDFILE") log=$REPO/$LOG"
sleep 25
tail -n 40 "$LOG" || true
ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd || true
nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv
