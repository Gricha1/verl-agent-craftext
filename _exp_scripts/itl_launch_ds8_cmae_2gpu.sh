#!/usr/bin/env bash
# aicenteritl launcher: 8x8 shared actor-value PPO, G_t^0.99, two-hot clipped MAE
# (rho=0.2), 2 GPUs.
#
# Canonical experiment: ds8_shared_av_gt_g099_cmae_h50_no_reasoning
#   docs: safe_rl_nlp_experiments.md §1.1 (canonical) + §2 (matched 8x8 vs 16x16)
#         safe_rl_nlp.md §9 (current state), §12.1 (roadmap #1), §14 (agent rules)
#   Matched to CE baseline ds8_shared_av_gt_g099_h50_a3_535900d8 except
#   value_loss: two_hot CE -> two_hot clipped_mae, rho=0.2.
#
# Usage:  bash _exp_scripts/itl_launch_ds8_cmae_2gpu.sh
# Env overrides: CUDA_VISIBLE_DEVICES (default 0,1), RAY_TEMP_DIR, RUN_NAME,
#   NUM_OPTIMISTIC_ENVS, HISTORY_LENGTH, RETURN_BIN_MIN/MAX/STEP, CLIPPED_MAE_RHO.
# N_GPUS is intentionally fixed at 2 (confirmed launch: 8x8 environment, 2 GPUs).
set -euo pipefail

REPO="${REPO:-/home/gorbov_gv/safe_rl_nlp}"
cd "$REPO"
mkdir -p logdir training_checkpoints

echo "=========================================================="
echo "aicenteritl | 8x8 shared AV PPO | G_t^0.99 | two-hot CMAE"
echo "canonical: ds8_shared_av_gt_g099_cmae_h50_no_reasoning | 2 GPUs"
echo "=========================================================="

# ---------------------------------------------------------------------------
# 1) MANDATORY preflight: nvidia-smi before ANY training launch
#    (safe_rl_nlp.md §14.1). Foreign jobs are reported, never touched (§14.2).
# ---------------------------------------------------------------------------
echo "[preflight] nvidia-smi:"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
echo "[preflight] compute apps (do NOT touch foreign processes):"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${N_GPUS:-2}"
if [ "$N_GPUS" != "2" ]; then
  echo "[FATAL] this launcher is fixed at 2 GPUs (got N_GPUS=$N_GPUS). Stop and confirm."
  exit 2
fi
export N_GPUS

# NOTE: on hosts with a broken GPU the CUDA index -> physical GPU mapping shifts
# (aicenter3 GPU2 lesson, safe_rl_nlp.md §7.5). Verify the mapping by uuid below
# before trusting the index.
for i in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
  echo "[preflight] selected GPU: $(nvidia-smi -i "$i" --query-gpu=index,uuid,memory.used --format=csv,noheader | tr -d ' ')"
  used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${used:-999}" -gt 500 ]; then
    echo "[FATAL] GPU$i is busy (${used} MiB used). Pick another pair via CUDA_VISIBLE_DEVICES."
    echo "[FATAL] Not touching foreign jobs (safe_rl_nlp.md §14.2)."
    exit 2
  fi
done

echo "[preflight] disk / memory (disk-full lesson 2026-09-13):"
df -h / /tmp 2>/dev/null || true
free -h | head -2

# ---------------------------------------------------------------------------
# 2) Ray temp: MUST be a SHORT path (AF_UNIX 107-byte socket path limit).
#    Deliberately NOT under /mnt/aicenter1-datasets: that failed as
#    validate_socket_filename "AF_UNIX path length >107" (Comet 1f29addc).
# ---------------------------------------------------------------------------
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp_gorbov_ds8_cmae}"
mkdir -p "$RAY_TEMP_DIR"
echo "[INFO] RAY_TEMP_DIR=$RAY_TEMP_DIR (len=${#RAY_TEMP_DIR})"

# ---------------------------------------------------------------------------
# 3) Ray / Comet / backend env
# ---------------------------------------------------------------------------
export FORCE_NEW_RAY_CLUSTER="${FORCE_NEW_RAY_CLUSTER:-1}"
export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl-agent-caged-craftext}"
export COMET_TIMEOUT="${COMET_TIMEOUT:-120}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
# NOTE: run_caged_craftext_lora_job.sh line 92 exports VLLM_ATTENTION_BACKEND=XFORMERS
# unconditionally, so it wins over this value for the actual vLLM engine.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"

# ---------------------------------------------------------------------------
# 4) Canonical matched config (safe_rl_nlp_experiments.md §1.1 / §2).
#    Nothing deviates from the canonical run: reasoning OFF (the loader passes
#    NO_REASONING=true -> +env.enable_reasoning=False), G_t^0.99, batch 64
#    (validated shared-AV baseline), bins [-5,6]/0.4, CMAE rho=0.2, hist 50,
#    single-token action, rolling actor-only checkpoints (inside the 8x8 loader).
# ---------------------------------------------------------------------------
export RUN_NAME="${RUN_NAME:-ds8_shared_av_gt_g099_cmae_h50_itl}"
export NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"   # -> data.train_batch_size=64
export HISTORY_LENGTH="${HISTORY_LENGTH:-50}"
export RETURN_BIN_MIN="${RETURN_BIN_MIN:--5}"
export RETURN_BIN_MAX="${RETURN_BIN_MAX:-6}"
export RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.4}"
export CLIPPED_MAE_RHO="${CLIPPED_MAE_RHO:-0.2}"
# data.max_prompt_length: canonical value is still TBD in the docs (§10.2); the
# loader uses MAX_PROMPT_LENGTH (default 1024). Export an explicitly measured
# value here instead of relying on the old v9/v10 (reasoning=ON) numbers.
echo "[INFO] RUN_NAME=$RUN_NAME batch=$NUM_OPTIMISTIC_ENVS hist=$HISTORY_LENGTH max_prompt_length=${MAX_PROMPT_LENGTH:-1024 (loader default, NOT yet measured for NO-reasoning)}"
echo "[INFO] bins=[$RETURN_BIN_MIN,$RETURN_BIN_MAX]/$RETURN_BIN_STEP rho=$CLIPPED_MAE_RHO reasoning=OFF"

# ---------------------------------------------------------------------------
# 5) python environment (host venv used by the a3 launchers)
# ---------------------------------------------------------------------------
if [ -f /home/gorbov_gv/safe_rl_nlp/.venv-itl/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/gorbov_gv/safe_rl_nlp/.venv-itl/bin/activate
  echo "[INFO] venv: /home/gorbov_gv/safe_rl_nlp/.venv-itl"
else
  echo "[WARN] host venv not found; run_caged_craftext_lora_job.sh will try conda verl-agent-311"
fi

# ---------------------------------------------------------------------------
# 6) launch
# ---------------------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logdir/ppo_debug_square_8x8_shared_av_gt_g099_cmae_itl_${STAMP}.log"
PIDFILE="${LOG%.log}.pid"
SRC_COMMIT="$(cat SOURCE_GIT_COMMIT 2>/dev/null || echo dirty-tree)"

{
  echo "[INFO] host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES N_GPUS=$N_GPUS"
  echo "[INFO] RUN_NAME=$RUN_NAME src_git_commit=$SRC_COMMIT"
  echo "[INFO] RAY_TEMP_DIR=$RAY_TEMP_DIR"
  echo "[INFO] token_reward=G_t^0.99 gae_by_trajectory=False value_loss=two_hot_clipped_mae"
  echo "[INFO] ckpt: rolling actor_lora_only save_freq=5 max_actor_ckpt_to_keep=2 test_freq=20"
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
} | tee "$LOG"

chmod +x examples/ppo_trainer/ppo_debug_square_8x8_shared_av_gt_g099_cmae.sh

nohup bash examples/ppo_trainer/ppo_debug_square_8x8_shared_av_gt_g099_cmae.sh \
  trainer.n_gpus_per_node=2 \
  >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[INFO] pid=$(cat "$PIDFILE")"
echo "[INFO] log=$REPO/$LOG"

sleep 25
tail -n 40 "$LOG" || true
ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd || true
echo "[INFO] comet key file: $RAY_TEMP_DIR/comet_experiment_key.txt"
cat "$RAY_TEMP_DIR/comet_experiment_key.txt" 2>/dev/null || echo "[INFO] comet key not written yet"
nvidia-smi -i $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ') \
  --query-gpu=index,memory.used,utilization.gpu --format=csv

cat <<EOT
[INFO] After the first PPO update, verify (safe_rl_nlp_experiments.md §8.2):
  tail -f $REPO/$LOG
    - episode/success_rate logged and non-negative
    - value_token_loss, value_token_accuracy logged
    - actor_value/p_target_mass_le_0.2, actor_value/clipped_mae, actor_value/unclipped_mae
    - value_grad_norm finite, no OOM, /tmp not filling up (disk-full lesson)
    - checkpoint at global_step 5 really exists (rolling save_freq=5 keep=2)
[INFO] Then register Comet key + this log path in experiments/registry.yaml
      (entry ds8_shared_av_gt_g099_cmae_h50_itl, flip status planned -> running).
EOT
