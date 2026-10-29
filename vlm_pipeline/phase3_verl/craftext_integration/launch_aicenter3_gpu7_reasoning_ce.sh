#!/usr/bin/env bash
# Replaces only the preceding, unsuccessful PPO experiment on GPU 7.
set -euo pipefail

ROOT="/mnt/aicenter1-datasets/gorbov_gv/modern_verl_reasoning_rl_a3_20261016"
RUN_NAME="qwen_offline_teacher_reasoning_ce_16x16_h100gpu7_20261017_r1"
RUN_ROOT_HOST="${ROOT}/run/${RUN_NAME}"
RUN_ROOT_CONTAINER="/mnt/run/run/${RUN_NAME}"
CONTAINER="${RUN_NAME}"
IMAGE="modern-verl-craftext:v091-smoke-craftext"
PHYSICAL_GPU_ID=7
CUDA_GPU_ID=0
MODEL_HOST="/home/gorbov_gv/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_CONTAINER="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SOURCE_DATASET_CONTAINER="/mnt/run/run/qwen_offline_teacher_reasoning_required_16x16_h100gpu7_20261017_r1/dataset"

mkdir -p "${RUN_ROOT_HOST}"; chmod 777 "${RUN_ROOT_HOST}"
rm -f "${RUN_ROOT_HOST}/launcher.log"
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
  echo "Container ${CONTAINER} already exists; refusing to replace it." >&2; exit 1
fi

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ID}" docker run -d --name "${CONTAINER}" --user 0:0 --entrypoint=/bin/bash --shm-size=32g \
  --volume="${ROOT}:/mnt/run" \
  --volume="${ROOT}/repo/craftext_integration:/workspace/verl/craftext_integration" \
  --volume="${ROOT}/repo/patches/model.py:/workspace/verl/verl/utils/model.py:ro" \
  --volume="${ROOT}/repo/patches/tracking.py:/workspace/verl/verl/utils/tracking.py:ro" \
  --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/group:/etc/group:ro" \
  --volume="/home/gorbov_gv/.cache/huggingface:/root/.cache/huggingface" \
  --volume="/home/gorbov_gv/.comet.config:/root/.comet.config:ro" \
  --volume="/home/gorbov_gv/.comet.config:/tmp/.comet.config:ro" \
  "${IMAGE}" -lc "export HOME=/tmp USER=root LOGNAME=root USERNAME=root XDG_CACHE_HOME=/tmp/cache XDG_CONFIG_HOME=/tmp/config MPLCONFIGDIR=/tmp/mpl TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor VERL_ATTENTION_IMPLEMENTATION=sdpa CUDA_VISIBLE_DEVICES=${CUDA_GPU_ID} PYTHONUNBUFFERED=1 PYTHON_BIN=python3 MODEL_PATH=${MODEL_CONTAINER} RUN_ROOT=${RUN_ROOT_CONTAINER} SOURCE_DATASET_DIR=${SOURCE_DATASET_CONTAINER} EXPERIMENT_NAME=${RUN_NAME} TOTAL_EPOCHS=30 TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE=8 MAX_LENGTH=1024 LOGGER_BACKENDS='[console,comet,tensorboard]'; mkdir -p /tmp/cache /tmp/config /tmp/mpl /tmp/torchinductor; python3 -m pip install --no-cache-dir --no-deps 'trl==0.27.0'; python3 -m pip install --no-cache-dir 'comet_ml>=3.50.0'; cd /workspace/verl && bash craftext_integration/run_offline_teacher_action_sft.sh > ${RUN_ROOT_CONTAINER}/launcher.log 2>&1"

echo "RUNNING container=${CONTAINER} physical_gpu=${PHYSICAL_GPU_ID} run_root=${RUN_ROOT_HOST}"
