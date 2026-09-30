#!/bin/bash
# AlfWorld three-phase PPO benchmark (40 global steps each):
#   Phase 1: single LLM actor+value + admissible-action entropy
#   Phase 2: dual actor+critic, no admissible-action entropy
#   Phase 3: dual actor+critic + admissible-action entropy
#
# Usage:
#   bash examples/ppo_trainer/ppo_alfworld_actor_value_then_dual_30steps.sh
#   STEPS_PER_PHASE=40 bash examples/ppo_trainer/ppo_alfworld_actor_value_then_dual_30steps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

STEPS_PER_PHASE="${STEPS_PER_PHASE:-40}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1}"
USE_ACTOR_LORA="${USE_ACTOR_LORA:-true}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-64}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-64}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

export TRAIN_BATCH_SIZE
export VAL_DATA_SIZE
export ACTOR_LORA_RANK
export ACTOR_LORA_ALPHA
export USE_ACTOR_LORA

COMMON_TRAINER_OVERRIDES=(
  "trainer.total_training_steps=${STEPS_PER_PHASE}"
  "trainer.total_epochs=${STEPS_PER_PHASE}"
  "trainer.val_before_train=False"
  "trainer.test_freq=-1"
  "trainer.save_freq=-1"
  "trainer.resume_mode=disable"
)

echo "=========================================="
echo "AlfWorld PPO sequential run (3 x ${STEPS_PER_PHASE} global steps)"
echo "=========================================="
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "[INFO] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[INFO] RUN_TAG=$RUN_TAG"

echo ""
echo ">>> Phase 1/3: actor-value + valid entropy (${STEPS_PER_PHASE} steps)"
export RUN_NAME="${RUN_NAME_PHASE1:-PPO AlfWorld actor-value valid-ent ${STEPS_PER_PHASE}step ${RUN_TAG}}"
export GPU_PROFILER_LOG="${GPU_PROFILER_LOG:-${RAY_TEMP_DIR:-/tmp/ray_temp}/gpu_profiler_phase1.log}"

bash examples/ppo_trainer/ppo_alfworld_actor_value_valid_ent.sh \
  "${COMMON_TRAINER_OVERRIDES[@]}" \
  trainer.default_local_dir="training_checkpoints/verl_agent_alfworld_actor_value_valid_ent_${STEPS_PER_PHASE}step_${RUN_TAG}" \
  "$@"

echo ""
echo ">>> Phase 2/3: dual actor+critic, no valid entropy (${STEPS_PER_PHASE} steps)"
export RUN_NAME="${RUN_NAME_PHASE2:-PPO AlfWorld dual no-valid-ent ${STEPS_PER_PHASE}step ${RUN_TAG}}"
export GPU_PROFILER_LOG="${GPU_PROFILER_LOG_PHASE2:-${RAY_TEMP_DIR:-/tmp/ray_temp}/gpu_profiler_phase2.log}"

bash examples/ppo_trainer/ppo_alfworld.sh \
  "${COMMON_TRAINER_OVERRIDES[@]}" \
  trainer.default_local_dir="training_checkpoints/verl_agent_alfworld_ppo_dual_${STEPS_PER_PHASE}step_${RUN_TAG}" \
  "$@"

echo ""
echo ">>> Phase 3/3: dual actor+critic + valid entropy (${STEPS_PER_PHASE} steps)"
export RUN_NAME="${RUN_NAME_PHASE3:-PPO AlfWorld dual valid-ent ${STEPS_PER_PHASE}step ${RUN_TAG}}"
export GPU_PROFILER_LOG="${GPU_PROFILER_LOG_PHASE3:-${RAY_TEMP_DIR:-/tmp/ray_temp}/gpu_profiler_phase3.log}"

bash examples/ppo_trainer/ppo_alfworld_valid_ent.sh \
  "${COMMON_TRAINER_OVERRIDES[@]}" \
  trainer.default_local_dir="training_checkpoints/verl_agent_alfworld_ppo_dual_valid_ent_${STEPS_PER_PHASE}step_${RUN_TAG}" \
  "$@"

echo ""
echo "[INFO] Sequential AlfWorld PPO finished: 3 phases x ${STEPS_PER_PHASE} steps."
