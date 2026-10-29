#!/usr/bin/env bash
# Validation-only rollout of r3's actor. No optimizer, critic, Ray or vLLM.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/gorbov_gv/modern_verl_craftext}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qwen_offline_teacher_action_16x16_a100gpu1_20261016_r3}"
PYTHON_BIN="${PYTHON_BIN:-/home/gorbov_gv/modern_verl_venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export JAX_PLATFORMS=cpu
export CRAFTEXT_LEGACY_ROOT="${CRAFTEXT_LEGACY_ROOT:-/home/gorbov_gv/safe_rl_nlp}"
export PYTHONUNBUFFERED=1
LOG_FILE="${RUN_ROOT}/craftext_eval_step650.log"
exec >>"${LOG_FILE}" 2>&1

echo "$(date -Is) starting validation-only r3 actor eval on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/evaluate_r3_actor_in_craftext.py" \
  --base-model /home/gorbov_gv/models/qwen2.5-1.5b-instruct \
  --adapter "${RUN_ROOT}/exports/actor_step_650/lora_adapter" \
  --legacy-root "${CRAFTEXT_LEGACY_ROOT}" \
  --output-dir "${RUN_ROOT}/craftext_eval_step650" \
  --seeds 20261016,20261017,20261018,20261019 \
  --task-index 0 --max-steps 50 --max-new-tokens 96
