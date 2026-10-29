#!/usr/bin/env bash
# Export the FSDP actor checkpoint as a standard PEFT adapter for inference.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/gorbov_gv/modern_verl_craftext}"
STEP_DIR="${STEP_DIR:-${REPO_ROOT}/runs/qwen_offline_teacher_action_16x16_a100gpu1_20261016_r3/checkpoints/global_step_650}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/qwen_offline_teacher_action_16x16_a100gpu1_20261016_r3/exports/actor_step_650}"
BASE_MODEL="${BASE_MODEL:-/home/gorbov_gv/models/qwen2.5-1.5b-instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/gorbov_gv/modern_verl_venv/bin/python}"

mkdir -p "${OUT_DIR}"
# The FSDP merger conventionally finds tokenizer/config under actor/huggingface.
# Training did not copy base artifacts into every checkpoint, so provide a read-only link.
ln -sfn "${BASE_MODEL}" "${STEP_DIR}/actor/huggingface"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${STEP_DIR}/actor" \
  --target_dir "${OUT_DIR}"
