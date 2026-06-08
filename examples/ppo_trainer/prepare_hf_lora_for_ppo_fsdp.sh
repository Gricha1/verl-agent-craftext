#!/bin/bash
# Convert HuggingFace LoRA dir (adapter at root, e.g. reward_wm .../latest) to PPO FSDP resume path.
#
# Usage:
#   bash examples/ppo_trainer/prepare_hf_lora_for_ppo_fsdp.sh \
#     /path/to/latest /path/to/output_fsdp_ckpt
#   bash examples/ppo_trainer/prepare_hf_lora_for_ppo_fsdp.sh \
#     /path/to/latest /path/to/output_fsdp_ckpt Qwen/Qwen2.5-1.5B-Instruct 2

set -euo pipefail

ADAPTER_PATH="${1:?adapter path (HF LoRA dir, e.g. .../latest)}"
OUTPUT_PATH="${2:?output FSDP checkpoint root (will contain global_step_0/actor/)}"
BASE_MODEL="${3:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_GPUS="${4:-2}"
LORA_RANK="${5:-64}"
LORA_ALPHA="${6:-64}"
TARGET_MODULES="${7:-all-linear}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! "$ADAPTER_PATH" = /* ]]; then
  ADAPTER_PATH="$PROJECT_ROOT/$ADAPTER_PATH"
fi
if [[ ! "$OUTPUT_PATH" = /* ]]; then
  OUTPUT_PATH="$PROJECT_ROOT/$OUTPUT_PATH"
fi

if [ ! -d "$ADAPTER_PATH" ]; then
  echo "[ERROR] Adapter path not found: $ADAPTER_PATH" >&2
  exit 1
fi

if [ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$ADAPTER_PATH/adapter_model.bin" ]; then
  echo "[ERROR] No adapter_model.safetensors/bin in $ADAPTER_PATH" >&2
  exit 1
fi

GLOBAL_STEP_DIR="$OUTPUT_PATH/global_step_0"
ACTOR_PATH="$GLOBAL_STEP_DIR/actor"

# Legacy layout from older prepare script: OUTPUT_PATH/actor -> global_step_0/actor
if [ -d "$OUTPUT_PATH/actor" ] && [ ! -d "$ACTOR_PATH" ]; then
  mkdir -p "$GLOBAL_STEP_DIR"
  mv "$OUTPUT_PATH/actor" "$ACTOR_PATH"
  echo "[INFO] Migrated legacy FSDP layout to $GLOBAL_STEP_DIR"
fi

if [ -f "$ACTOR_PATH/model_world_size_${NUM_GPUS}_rank_0.pt" ]; then
  echo "[OK] FSDP checkpoint already exists: $GLOBAL_STEP_DIR (world_size=$NUM_GPUS)"
  exit 0
fi

if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD="torchrun"
else
  TORCHRUN_CMD="python3 -m torch.distributed.run"
fi

echo "[INFO] Converting HF LoRA -> PPO FSDP"
echo "  adapter=$ADAPTER_PATH"
echo "  resume_path=$GLOBAL_STEP_DIR"
echo "  base_model=$BASE_MODEL"
echo "  num_gpus=$NUM_GPUS"

$TORCHRUN_CMD --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  scripts/convert_sft_to_fsdp.py \
  --base_model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --output_path "$GLOBAL_STEP_DIR" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules "$TARGET_MODULES" \
  --trust_remote_code

echo "[OK] PPO resume_from_path: $GLOBAL_STEP_DIR"
