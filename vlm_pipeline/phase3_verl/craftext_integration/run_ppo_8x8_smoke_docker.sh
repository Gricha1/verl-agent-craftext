#!/usr/bin/env bash
# Start the smoke test in an isolated container.  It never joins an existing
# Ray session and has no relation to the baseline container/processes.
set -euo pipefail

: "${MODERN_VERL_IMAGE:?Set MODERN_VERL_IMAGE to the separately-built modern veRL smoke image}"
: "${MODEL_PATH:?Set MODEL_PATH to the host model directory}"
: "${CRAFTEXT_LEGACY_ROOT:?Set CRAFTEXT_LEGACY_ROOT to the host CrafText checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to a fresh host output directory}"

# Hugging Face cache snapshots are symlink farms: files in
# snapshots/<revision> point to ../../blobs.  Mount the corresponding blobs
# directory at the path those links resolve to inside the container.
# A HuggingFace cache snapshot needs its sibling blobs directory; a materialised
# CE checkpoint has regular files and deliberately does not.  Support both.
MODEL_BLOBS_PATH="${MODEL_BLOBS_PATH:-$(dirname "$(dirname "${MODEL_PATH}")")/blobs}"
MODEL_BLOBS_MOUNT=()
if [[ -d "${MODEL_BLOBS_PATH}" ]]; then
  MODEL_BLOBS_MOUNT=(-v "${MODEL_BLOBS_PATH}:/blobs:ro")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-modern-verl-craftext-smoke-$(date +%Y%m%d-%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-4}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-1024}"
N_GPUS="${N_GPUS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-320}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${NUM_ENVS}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${NUM_ENVS}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${NUM_ENVS}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
TEST_FREQ="${TEST_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console]}"
PROJECT_NAME="${PROJECT_NAME:-modern_verl_craftext}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_8x8_reasoning_${N_GPUS}gpu}"
VALIDATION_SEED="${VALIDATION_SEED:-20261014}"
VALIDATION_TRAJECTORY_LIMIT="${VALIDATION_TRAJECTORY_LIMIT:-10}"
PROFILE_STEPS="${PROFILE_STEPS:-}"
# This path is evaluated inside the container; /workspace/run is the mounted
# host RUN_ROOT, so the trace remains available on the host after exit.
PROFILE_SAVE_PATH="${PROFILE_SAVE_PATH:-/workspace/run/profiles/torch}"
FILE_LOGGER_PATH="${FILE_LOGGER_PATH:-/workspace/run/logs/metrics.jsonl}"
GPU_SAMPLER="${GPU_SAMPLER:-false}"
GPU_SAMPLER_INTERVAL_SECONDS="${GPU_SAMPLER_INTERVAL_SECONDS:-1}"
# GPU_ID selects physical GPUs for the site Docker wrapper.  Its numbering is
# re-based inside the container, hence this distinct variable.
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# On aicenter3 `/usr/local/bin/docker` is a site wrapper.  It deliberately
# ignores `--gpus` and container-level CUDA visibility variables, selecting the
# DeviceRequest exclusively from the *host-side* CUDA_VISIBLE_DEVICES instead.
# Set it on the docker client invocation so the container gets exactly GPU_ID.
# The vLLM base image intentionally exposes python3 only.  Clearing the Ray
# address is an additional guard against joining the baseline cluster. Craftax
# builds a texture cache alongside its assets, so its independent legacy clone
# is writable; it is never the r8 checkout.
mkdir -p "${RUN_ROOT}"
if [[ "${LOGGER_BACKENDS}" == *comet* ]]; then
  : "${COMET_API_KEY:?COMET_API_KEY is required when LOGGER_BACKENDS includes comet}"
fi

SAMPLER_PID=""
cleanup() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
if [[ "${GPU_SAMPLER}" == true ]]; then
  mkdir -p "${RUN_ROOT}/profiles"
  python3 "${REPO_ROOT}/craftext_integration/gpu_sampler.py" \
    --output "${RUN_ROOT}/profiles/gpu_samples.csv" --gpu-ids "${GPU_ID}" \
    --interval-seconds "${GPU_SAMPLER_INTERVAL_SECONDS}" &
  SAMPLER_PID="$!"
fi

# The aicenter3 wrapper mangles `-e VAR=value` (it drops `-e` and leaves the
# value as a positional argument), so the container environment is passed as an
# env file.  COMET_* are listed by name only: Docker copies them from the client
# environment and their values never reach disk.
CONTAINER_ENV_FILE="${RUN_ROOT}/container.env"
cat > "${CONTAINER_ENV_FILE}" <<ENVEOF
CRAFTEXT_LEGACY_ROOT=/workspace/legacy_craftext
MODEL_PATH=/workspace/model
RUN_ROOT=/workspace/run
GPU_ID=${CONTAINER_GPU_ID}
CRAF_TEXT_ENV_CONFIG=${CRAF_TEXT_ENV_CONFIG:-debug_square_8x8}
CRAF_TEXT_RESET_RATIO=${CRAF_TEXT_RESET_RATIO:-1}
CRAF_TEXT_SEED=${CRAF_TEXT_SEED:-20261013}
CRAF_TEXT_ACTION_HISTORY_LENGTH=${CRAF_TEXT_ACTION_HISTORY_LENGTH:-50}
CRAF_TEXT_REASONING_HISTORY_LENGTH=${CRAF_TEXT_REASONING_HISTORY_LENGTH:-3}
CRAF_TEXT_VISUAL_PROMPT_HISTORY=${CRAF_TEXT_VISUAL_PROMPT_HISTORY:-8}
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}
USE_KL_LOSS=${USE_KL_LOSS:-false}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
CRAF_TEXT_REWARD_MODE=${CRAF_TEXT_REWARD_MODE:-immediate}
CRAF_TEXT_REMAINING_RETURN_GAMMA=${CRAF_TEXT_REMAINING_RETURN_GAMMA:-1.0}
CRAF_TEXT_USE_INVALID_ACTION_PENALTY=${CRAF_TEXT_USE_INVALID_ACTION_PENALTY:-false}
CRAF_TEXT_INVALID_ACTION_PENALTY_COEF=${CRAF_TEXT_INVALID_ACTION_PENALTY_COEF:-0.0}
PER_TURN_MAX_TOKENS=${PER_TURN_MAX_TOKENS:-320}
NUM_ENVS=${NUM_ENVS}
MAX_ENV_STEPS=${MAX_ENV_STEPS}
TOTAL_EPOCHS=${TOTAL_EPOCHS}
VLLM_MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS}
N_GPUS=${N_GPUS}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}
TEST_FREQ=${TEST_FREQ}
SAVE_FREQ=${SAVE_FREQ}
LOGGER_BACKENDS=${LOGGER_BACKENDS}
PROJECT_NAME=${PROJECT_NAME}
EXPERIMENT_NAME=${EXPERIMENT_NAME}
VALIDATION_SEED=${VALIDATION_SEED}
VALIDATION_TRAJECTORY_LIMIT=${VALIDATION_TRAJECTORY_LIMIT}
PROFILE_STEPS=${PROFILE_STEPS}
PROFILE_SAVE_PATH=${PROFILE_SAVE_PATH}
FILE_LOGGER_PATH=${FILE_LOGGER_PATH}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-true}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-true}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1536}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-${NUM_ENVS}}
TENSORBOARD_DIR=/workspace/run/tensorboard
COMET_API_KEY
COMET_WORKSPACE
PYTHON_BIN=python3
RAY_ADDRESS=
HOME=/workspace/run
USER=gorbov_gv
LOGNAME=gorbov_gv
TORCHINDUCTOR_CACHE_DIR=/workspace/run/torchinductor
HF_HOME=/workspace/run/hf
ENVEOF

env CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name="${CONTAINER_NAME}" \
  --network host \
  --shm-size=8g \
  --entrypoint=bash \
  --user="${HOST_UID}:${HOST_GID}" \
  -v "${REPO_ROOT}:/workspace/modern_verl_craftext:ro" \
  -v "${CRAFTEXT_LEGACY_ROOT}:/workspace/legacy_craftext" \
  -v "${MODEL_PATH}:/workspace/model:ro" \
  "${MODEL_BLOBS_MOUNT[@]}" \
  -v "${RUN_ROOT}:/workspace/run" \
  --env-file "${CONTAINER_ENV_FILE}" \
  -w /workspace/modern_verl_craftext \
  "${MODERN_VERL_IMAGE}" \
  craftext_integration/run_ppo_8x8_smoke.sh
