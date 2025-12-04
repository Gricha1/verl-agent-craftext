#!/bin/bash
set -x

# Скрипт для извлечения LoRA адаптера из FSDP чекпоинта для использования в vLLM инференсе
#
# Использование:
#   bash extract_lora.sh [CHECKPOINT_PATH] [OUTPUT_PATH] [BASE_MODEL] [LORA_RANK] [LORA_ALPHA] [TARGET_MODULES]
#
# Пример:
#   bash extract_lora.sh \
#       /path/to/fsdp_checkpoint/global_step_0 \
#       /path/to/output/lora_adapter \
#       Qwen/Qwen2.5-1.5B-Instruct \
#       64 64 all-linear

CHECKPOINT_PATH=${1:-/home/n.sorokin/verl-agent/checkpoints/verl_agent_craftext/run_sft_qwen2.5_1.5b_fsdp_achievements_wood_20251203-194330/global_step_0}
OUTPUT_PATH=${2:-/home/n.sorokin/verl-agent/checkpoints/verl_agent_craftext/run_sft_qwen2.5_1.5b_fsdp_achievements_wood_20251203-194330/global_step_0/actor/lora_adapter}
BASE_MODEL=${3:-Qwen/Qwen2.5-1.5B-Instruct}
LORA_RANK=${4:-64}
LORA_ALPHA=${5:-64}
TARGET_MODULES=${6:-all-linear}
NUM_GPUS=${7:-1}

# Проверка существования чекпоинта
if [ ! -d "$CHECKPOINT_PATH/actor" ]; then
    echo "Ошибка: Actor checkpoint не найден: $CHECKPOINT_PATH/actor"
    echo "Укажите путь к FSDP чекпоинту (например, .../global_step_0)"
    exit 1
fi

# Создаем выходную директорию
mkdir -p "$OUTPUT_PATH"

echo "Извлечение LoRA адаптера:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Base model: $BASE_MODEL"
echo "  LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA"
echo "  Target modules: $TARGET_MODULES"
echo "  Number of GPUs: $NUM_GPUS"

# Запускаем извлечение через torchrun
# Используем conda run для активации окружения verl-agent-311
conda run -n verl-agent-311 python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    /home/n.sorokin/verl-agent/scripts/extract_lora_from_fsdp.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --base_model "$BASE_MODEL" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES"

echo "LoRA адаптер извлечен в: $OUTPUT_PATH"

