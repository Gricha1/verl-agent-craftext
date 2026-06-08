#!/bin/bash
# Validate reward-WM SFT checkpoint as receding-horizon LLM planner on debug_square_8x8.
#
# Each env step (default): greedy plan H=6 actions, execute only the 1st token, re-plan next step.
# PPO_ACTOR_PROMPT=true: same prompt as PPO actor (single_token_action), one greedy action per step.
# BASE_MODEL_ONLY=true: load Qwen base weights only (no SFT/LoRA checkpoint).
# Runs stone / wood / water; reports success rate + reward; saves one GIF per task.
#
# Usage:
#   bash examples/world_model/validate_debug_square_llm_planning.sh
#   CKPT_PATH=/path/to/latest EPISODES_PER_TASK=50 bash examples/world_model/validate_debug_square_llm_planning.sh
#   CUDA_VISIBLE_DEVICES=1 bash examples/world_model/validate_debug_square_llm_planning.sh
#   PPO_ACTOR_PROMPT=true bash examples/world_model/validate_debug_square_llm_planning.sh
#   BASE_MODEL_ONLY=true bash examples/world_model/validate_debug_square_llm_planning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "verl-agent-311" ]; then
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate verl-agent-311
  fi
fi

CKPT_PATH="${CKPT_PATH:-/usr/home/workspace/training_checkpoints/reward_wm_sft_h6_20260606-085007_1/latest}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
BASE_MODEL_ONLY="${BASE_MODEL_ONLY:-false}"
REWARD_HORIZON="${REWARD_HORIZON:-6}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-20}"
MAX_STEPS="${MAX_STEPS:-50}"
SEED="${SEED:-0}"
GIF_PATH="${GIF_PATH:-gif/llm_planning_debug_square_h${REWARD_HORIZON}.gif}"
PLANNING_MODE="${PLANNING_MODE:-max_return}"
PPO_ACTOR_PROMPT="${PPO_ACTOR_PROMPT:-false}"
DEVICE="${DEVICE:-cuda:0}"

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export RUN_NAME="${RUN_NAME:-llm_planning_debug_square_h${REWARD_HORIZON}_$(date +%Y%m%d-%H%M%S)}"

export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEVICE="cuda:0"
fi

python3 -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

if [ "$BASE_MODEL_ONLY" != "true" ] && [ "$BASE_MODEL_ONLY" != "1" ]; then
  if [ ! -d "$CKPT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT_PATH" >&2
    echo "Set CKPT_PATH=.../latest or train first:" >&2
    echo "  REWARD_HORIZON=6 bash examples/world_model/train_reward_wm_sft.sh" >&2
    echo "Or use BASE_MODEL_ONLY=true to validate the untrained base model." >&2
    exit 1
  fi
fi

echo "=========================================="
echo "LLM planning validation (debug_square_8x8)"
echo "=========================================="
echo "[INFO] CKPT_PATH=${CKPT_PATH:-(skipped)}"
echo "[INFO] BASE_MODEL=$BASE_MODEL  BASE_MODEL_ONLY=$BASE_MODEL_ONLY"
echo "[INFO] REWARD_HORIZON=$REWARD_HORIZON  PLANNING_MODE=$PLANNING_MODE  PPO_ACTOR_PROMPT=$PPO_ACTOR_PROMPT"
echo "[INFO] EPISODES_PER_TASK=$EPISODES_PER_TASK  MAX_STEPS=$MAX_STEPS  SEED=$SEED"
echo "[INFO] DEVICE=$DEVICE  RUN_NAME=$RUN_NAME"
echo "[INFO] Comet project=verl_agent_caged_craftext (run appears right after script start)"
if [ "$BASE_MODEL_ONLY" = "true" ] || [ "$BASE_MODEL_ONLY" = "1" ]; then
  echo "[INFO] weights: base model only ($BASE_MODEL) — no SFT/LoRA"
fi
if [ "$PPO_ACTOR_PROMPT" = "true" ] || [ "$PPO_ACTOR_PROMPT" = "1" ]; then
  echo "[INFO] strategy: PPO actor prompt (single_token_action) -> greedy 1 action token per step"
else
  echo "[INFO] strategy: plan H actions -> execute 1st -> re-plan each step"
fi
echo "[INFO] tasks: stone (0), wood (1), water (2)"
echo "[INFO] GIF outputs:"
echo "       ${GIF_PATH%.gif}_stone.gif"
echo "       ${GIF_PATH%.gif}_wood.gif"
echo "       ${GIF_PATH%.gif}_water.gif"
echo "[INFO] Action histograms (same episode as each GIF):"
echo "       ${GIF_PATH%.gif}_stone_action_hist.png"
echo "       ${GIF_PATH%.gif}_wood_action_hist.png"
echo "       ${GIF_PATH%.gif}_water_action_hist.png"

EXTRA_ARGS=()
if [ -n "$BASE_MODEL" ]; then
  EXTRA_ARGS+=(--base-model "$BASE_MODEL")
fi
if [ "$BASE_MODEL_ONLY" = "true" ] || [ "$BASE_MODEL_ONLY" = "1" ]; then
  EXTRA_ARGS+=(--base-model-only)
  export RUN_NAME="${RUN_NAME:-llm_planning_debug_square_base_only_$(date +%Y%m%d-%H%M%S)}"
else
  EXTRA_ARGS+=(--checkpoint "$CKPT_PATH")
fi
if [ "$PPO_ACTOR_PROMPT" = "true" ] || [ "$PPO_ACTOR_PROMPT" = "1" ]; then
  EXTRA_ARGS+=(--ppo-actor-prompt)
fi

python3 scripts/validate_debug_square_llm_planning.py \
  --horizon "$REWARD_HORIZON" \
  --planning-mode "$PLANNING_MODE" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --gif-path "$GIF_PATH" \
  "${EXTRA_ARGS[@]}" \
  "$@"
