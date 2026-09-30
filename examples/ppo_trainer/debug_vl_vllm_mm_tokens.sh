#!/bin/bash
# Smoke: PPO rollout vLLM path (preprocess -> generate). Exit 0 = OK for full PPO.
#
#   bash examples/ppo_trainer/debug_vl_vllm_mm_tokens.sh
#   DEBUG_VL_BATCH_SIZE=4 bash examples/ppo_trainer/debug_vl_vllm_mm_tokens.sh

set -e

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "verl-agent-311" ]; then
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate verl-agent-311
  fi
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export VL_MAX_PROMPT_LENGTH="${VL_MAX_PROMPT_LENGTH:-768}"
export VL_MM_MAX_PIXELS="${VL_MM_MAX_PIXELS:-280000}"
export VL_MM_MIN_PIXELS="${VL_MM_MIN_PIXELS:-65536}"
export VL_GPU_MEMORY_UTIL="${VL_GPU_MEMORY_UTIL:-0.50}"
export DEBUG_VL_BATCH_SIZE="${DEBUG_VL_BATCH_SIZE:-17}"
export DEBUG_VL_TENSOR_PARALLEL="${DEBUG_VL_TENSOR_PARALLEL:-1}"
# Batched VL + chunked prefill breaks placeholder count (5508 vs 5295); match single-trajectory smoke.
export VL_ENABLE_CHUNKED_PREFILL="${VL_ENABLE_CHUNKED_PREFILL:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export CRAFTAX_PATH="${CAGED_CRAFTEXT_PATH}/Craftax"
export PYTHONPATH="${PROJECT_ROOT}:${CRAFTAX_PATH}:${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}"

echo "=========================================="
echo "PPO rollout vLLM smoke (production path)"
echo "=========================================="
echo "[INFO] exit 0 = generate OK; exit 1 = same bug as PPO training"

cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/debug_qwen_vl_vllm_mm_tokens.py
