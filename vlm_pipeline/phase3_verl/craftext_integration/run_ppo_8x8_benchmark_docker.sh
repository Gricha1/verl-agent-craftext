#!/usr/bin/env bash
# Controlled, measurable repeat of the modern 8x8 reasoning PPO configuration.
# It is intentionally a wrapper: the underlying smoke launcher keeps its old
# defaults, while this file makes every benchmark override visible and auditable.
set -euo pipefail

: "${COMET_API_KEY:?Set COMET_API_KEY for the benchmark Comet experiment}"
: "${COMET_WORKSPACE:=gregory-gorbov}"
: "${MODERN_VERL_IMAGE:?Set the image rebuilt with craftext_integration/Dockerfile.smoke}"
: "${MODEL_PATH:?Set a local Qwen2.5-1.5B-Instruct checkpoint}"
: "${CRAFTEXT_LEGACY_ROOT:?Set the isolated legacy CrafText checkout}"
: "${RUN_ROOT:?Set a new, empty benchmark output directory}"

GPU_ID="${GPU_ID:-0,1}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0,1}"
N_GPUS="${N_GPUS:-2}"
NUM_ENVS="${NUM_ENVS:-64}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-modern_verl_craftext8_reasoning_profiled_v1}"

# These are deliberately the current r3 policy/optimization values.  The only
# operational additions are telemetry, validation, and profiler collection.
export GPU_ID CONTAINER_GPU_ID N_GPUS NUM_ENVS TOTAL_EPOCHS EXPERIMENT_NAME
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-50}"
export VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-4096}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-320}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
export VAL_BEFORE_TRAIN=true
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export LOGGER_BACKENDS='[console,comet,file]'
export PROJECT_NAME="${PROJECT_NAME:-verl-agent-caged-craftext}"
export VALIDATION_SEED="${VALIDATION_SEED:-20261014}"
export VALIDATION_TRAJECTORY_LIMIT="${VALIDATION_TRAJECTORY_LIMIT:-10}"
export GPU_SAMPLER=true
export GPU_SAMPLER_INTERVAL_SECONDS="${GPU_SAMPLER_INTERVAL_SECONDS:-1}"
# A two-step torch trace is separate from the lightweight whole-run sampler.
export PROFILE_STEPS="${PROFILE_STEPS:-[1,2]}"
export PROFILE_SAVE_PATH="${PROFILE_SAVE_PATH:-/workspace/run/profiles/torch}"
export FILE_LOGGER_PATH="${FILE_LOGGER_PATH:-/workspace/run/logs/metrics.jsonl}"

exec "$(dirname "${BASH_SOURCE[0]}")/run_ppo_8x8_smoke_docker.sh"
