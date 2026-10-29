#!/usr/bin/env bash
# Launch one isolated modern-veRL reproduction of Comet run 06c4… .
# This is an operational launcher, not a hyperparameter source: compat.env
# contains the archived experiment's effective settings.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/mnt/aicenter1-datasets/gorbov_gv/modern_verl_runs/modern_verl_craftext16_06c4_compat_tb_20261015_r7}"
GPU_ID="${GPU_ID:-0,1}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0,1}"
SOURCE_ROOT=/mnt/aicenter1-datasets/gorbov_gv/modern_verl_craftext_benchmark_v1
LEGACY_ROOT=/mnt/aicenter1-datasets/gorbov_gv/modern_verl_craftext_legacy_env
MODEL_PATH=/home/gorbov_gv/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

[[ -f "${RUN_ROOT}/compat.env" ]]
[[ ! -e "${RUN_ROOT}/launcher.pid" ]]
COMET_API_KEY="$(sed -n -E 's/^[[:space:]]*api_key[[:space:]]*=[[:space:]]*//p' /home/gorbov_gv/.comet.config | head -n 1)"
[[ -n "${COMET_API_KEY}" ]]

nohup env \
  MODERN_VERL_IMAGE=modern-verl-craftext:v091-benchmark-contention-probe-v1 \
  MODEL_PATH="${MODEL_PATH}" \
  CRAFTEXT_LEGACY_ROOT="${LEGACY_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  GPU_ID="${GPU_ID}" \
  CONTAINER_GPU_ID="${CONTAINER_GPU_ID}" \
  N_GPUS=2 \
  NUM_ENVS=32 \
  TOTAL_EPOCHS=8000 \
  MAX_ENV_STEPS=50 \
  PROJECT_NAME=verl-agent-caged-craftext \
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-modern_verl_craftext16_06c4_compat_tb_20261015_r7}" \
  COMET_API_KEY="${COMET_API_KEY}" \
  bash "${SOURCE_ROOT}/craftext_integration/run_ppo_8x8_benchmark_docker.sh" \
  >"${RUN_ROOT}/launcher.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${RUN_ROOT}/launcher.pid"
printf 'LAUNCHER_PID=%s\n' "${pid}"
