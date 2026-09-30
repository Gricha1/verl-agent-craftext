#!/bin/bash
# One debug_square task, few env steps, vLLM generate each step (fastest VL rollout smoke).
#
#   bash examples/ppo_trainer/debug_vl_single_trajectory.sh
#   STEPS=5 TASK=wood bash examples/ppo_trainer/debug_vl_single_trajectory.sh

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
export DEBUG_VL_TENSOR_PARALLEL="${DEBUG_VL_TENSOR_PARALLEL:-1}"
export TASK="${TASK:-stone}"
export STEPS="${STEPS:-3}"
export SEED="${SEED:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export CRAFTAX_PATH="${CAGED_CRAFTEXT_PATH}/Craftax"
export PYTHONPATH="${PROJECT_ROOT}:${CRAFTAX_PATH}:${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}"

echo "=========================================="
echo "VL single trajectory (1 task, ${STEPS} steps)"
echo "=========================================="
echo "[INFO] task=${TASK} — exit 0 = rollout OK; exit 1 = vLLM merge bug"

cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/debug_vl_single_trajectory.py --task "$TASK" --steps "$STEPS" --seed "$SEED"
