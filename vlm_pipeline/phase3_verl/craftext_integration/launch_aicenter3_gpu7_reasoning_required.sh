#!/usr/bin/env bash
# One isolated offline RL run.  It never joins a pre-existing Ray cluster.
set -euo pipefail

ROOT="/mnt/aicenter1-datasets/gorbov_gv/modern_verl_reasoning_rl_a3_20261016"
RUN_NAME="qwen_offline_teacher_reasoning_required_16x16_h100gpu7_20261017_r1"
RUN_ROOT_HOST="${ROOT}/run/${RUN_NAME}"
RUN_ROOT_CONTAINER="/mnt/run/run/${RUN_NAME}"
RAY_SHM_HOST="/dev/shm/gorbov_gv_verl_reasoning_gpu7_r1"
CONTAINER="${RUN_NAME}"
IMAGE="modern-verl-craftext:v091-smoke-craftext"
PHYSICAL_GPU_ID=7
# Docker's CDI mapping exposes the selected physical GPU as local CUDA:0.
CUDA_GPU_ID=0
MODEL_HOST="/home/gorbov_gv/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_CONTAINER="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

mkdir -p "${RUN_ROOT_HOST}"
chmod 777 "${RUN_ROOT_HOST}"
# Ray's default spill directory is under /tmp, a shared, nearly full system
# filesystem on aicenter3.  Keep this run's IPC sockets, logs and spill files
# on the host shared-memory filesystem instead.  This is isolated per run.
mkdir -p "${RAY_SHM_HOST}"
chmod 700 "${RAY_SHM_HOST}"
# A prior root-squashed container can leave this regular file non-writable to
# the next launch.  The directory is private to this run, so replace only its
# launcher log before starting.
rm -f "${RUN_ROOT_HOST}/launcher.log"
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER}"; then
  echo "Container ${CONTAINER} already exists; refusing to replace it." >&2
  exit 1
fi

# aicenter3 uses rootless Docker and maps the caller to UID 1007.  The vLLM
# image lacks that account, while PyTorch calls getpwuid(), so expose the host
# account database read-only.
# The aicenter3 Docker audit wrapper enforces GPU access from the host-side
# CUDA_VISIBLE_DEVICES value (and strips --gpus/--device supplied by callers).
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ID}" docker run -d --name "${CONTAINER}" --user 0:0 --entrypoint=/bin/bash --shm-size=32g \
  --volume="${ROOT}:/mnt/run" \
  --volume="${ROOT}/repo/craftext_integration:/workspace/verl/craftext_integration" \
  --volume="${ROOT}/repo/patches/model.py:/workspace/verl/verl/utils/model.py:ro" \
  --volume="${ROOT}/repo/patches/tracking.py:/workspace/verl/verl/utils/tracking.py:ro" \
  --volume="${RAY_SHM_HOST}:/raytmp" \
  --volume="/etc/passwd:/etc/passwd:ro" \
  --volume="/etc/group:/etc/group:ro" \
  --volume="/home/gorbov_gv/.cache/huggingface:/root/.cache/huggingface" \
  --volume="/home/gorbov_gv/.comet.config:/root/.comet.config:ro" \
  --volume="/home/gorbov_gv/.comet.config:/tmp/.comet.config:ro" \
  "${IMAGE}" -lc "export TMPDIR=/raytmp TMP=/raytmp TEMP=/raytmp HOME=/tmp USER=root LOGNAME=root USERNAME=root XDG_CACHE_HOME=/tmp/cache XDG_CONFIG_HOME=/tmp/config MPLCONFIGDIR=/tmp/mpl TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor VERL_ATTENTION_IMPLEMENTATION=sdpa CRAFTEXT_RUNTIME_CACHE_DIR=${RUN_ROOT_CONTAINER}/runtime CUDA_VISIBLE_DEVICES=${CUDA_GPU_ID} PYTHONUNBUFFERED=1 PYTHON_BIN=python3 MODEL_PATH=${MODEL_CONTAINER} CRAFTEXT_LEGACY_ROOT=/mnt/run/legacy TEACHER_CHECKPOINT=/mnt/run/teacher_latest.pt RUN_ROOT=${RUN_ROOT_CONTAINER} GPU_ID=${CUDA_GPU_ID} RAY_SESSION_DIR=/raytmp EXPERIMENT_NAME=${RUN_NAME} CRAFTEXT_SETTINGS=debug_square_16x16 MAP_SEEDS=20261016,20261017 TASK_INDICES=0 MAX_STEPS=50 TOTAL_EPOCHS=8000 TRAIN_BATCH_SIZE=32 VAL_BATCH_SIZE=16 MAX_PROMPT_LENGTH=3072 MAX_RESPONSE_LENGTH=320 LOGGER_BACKENDS='[console,comet,file,tensorboard]'; mkdir -p /raytmp /tmp/cache /tmp/config /tmp/mpl /tmp/torchinductor; python3 -m pip install --no-cache-dir --no-deps 'trl==0.27.0'; python3 -m pip install --no-cache-dir 'comet_ml>=3.50.0'; cd /workspace/verl && bash craftext_integration/run_offline_teacher_action_rl.sh > ${RUN_ROOT_CONTAINER}/launcher.log 2>&1"

echo "RUNNING container=${CONTAINER} physical_gpu=${PHYSICAL_GPU_ID} run_root=${RUN_ROOT_HOST}"
