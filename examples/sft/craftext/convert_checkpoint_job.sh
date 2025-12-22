#!/bin/bash
set -x

# Скрипт для конвертации HuggingFace LoRA чекпоинта (из SFT обучения) в FSDP формат для RL обучения
#
# Использование:
#   bash convert_checkpoint.sh [BASE_MODEL] [ADAPTER_PATH] [OUTPUT_PATH] [LORA_RANK] [LORA_ALPHA] [TARGET_MODULES] [NUM_GPUS]
#
# Пример:
#   bash convert_checkpoint.sh \
#       Qwen/Qwen2.5-1.5B-Instruct \
#       /path/to/sft_checkpoint/global_step_33 \
#       /path/to/output_fsdp_checkpoint_global_step_0 \
#       64 64 all-linear 1

source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

BASE_MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
ADAPTER_PATH=${2:-/home/jovyan/nsorokin/verl-agent-craftext/checkpoints/verl_agent_craftext/run_sft_qwen2.5_1.5b_achievements_wood_20251203-194330/global_step_33}
OUTPUT_PATH=${3:-/home/jovyan/nsorokin/verl-agent-craftext/checkpoints/verl_agent_craftext/run_sft_qwen2.5_1.5b_fsdp_achievements_wood_20251203-194330/global_step_0}
LORA_RANK=${4:-64}
LORA_ALPHA=${5:-64}
TARGET_MODULES=${6:-all-linear}
NUM_GPUS=${7:-1}

# Проверка существования adapter_path
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Ошибка: Adapter path не найден: $ADAPTER_PATH"
    echo "Укажите путь к SFT чекпоинту (например, .../global_step_N)"
    exit 1
fi

# Создаем выходную директорию
mkdir -p "$OUTPUT_PATH"

echo "Конвертация чекпоинта:"
echo "  Base model: $BASE_MODEL"
echo "  Adapter path: $ADAPTER_PATH"
echo "  Output path: $OUTPUT_PATH"
echo "  LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA"
echo "  Target modules: $TARGET_MODULES"
echo "  Number of GPUs: $NUM_GPUS"

# Запускаем конвертацию через torchrun (требуется для FSDP)
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    /home/jovyan/nsorokin/verl-agent-craftext/scripts/convert_sft_to_fsdp.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --output_path "$OUTPUT_PATH" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES" \

echo "Конвертация завершена! FSDP чекпоинт сохранен в: $OUTPUT_PATH/actor"