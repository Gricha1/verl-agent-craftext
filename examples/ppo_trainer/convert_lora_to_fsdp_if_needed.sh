#!/bin/bash
# Скрипт для автоматической конвертации LoRA адаптера в FSDP формат, если FSDP чекпоинт отсутствует
#
# Использование:
#   bash examples/ppo_trainer/convert_lora_to_fsdp_if_needed.sh [CHECKPOINT_PATH] [BASE_MODEL] [LORA_RANK] [LORA_ALPHA] [TARGET_MODULES] [NUM_GPUS]
#
# Пример:
#   bash examples/ppo_trainer/convert_lora_to_fsdp_if_needed.sh \
#       pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000 \
#       Qwen/Qwen2.5-1.5B-Instruct \
#       64 64 all-linear 2

CHECKPOINT_PATH=${1:-pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000}
BASE_MODEL=${2:-Qwen/Qwen2.5-1.5B-Instruct}
LORA_RANK=${3:-64}
LORA_ALPHA=${4:-64}
TARGET_MODULES=${5:-all-linear}
NUM_GPUS=${6:-2}

# Определяем корень проекта для правильной обработки относительных путей
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Преобразуем относительный путь в абсолютный относительно корня проекта
if [[ ! "$CHECKPOINT_PATH" = /* ]]; then
    CHECKPOINT_PATH="$PROJECT_ROOT/$CHECKPOINT_PATH"
fi

# Получаем абсолютный путь к чекпоинту
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Ошибка: Путь к чекпоинту не найден: $CHECKPOINT_PATH"
    exit 1
fi

# Нормализуем путь (убираем лишние слеши и символы)
CHECKPOINT_PATH="$(cd "$CHECKPOINT_PATH" && pwd)"

echo "Проверка чекпоинта: $CHECKPOINT_PATH"

# Проверяем наличие FSDP чекпоинта (файлы model_world_size_*_rank_*.pt в папке actor/)
ACTOR_PATH="$CHECKPOINT_PATH/actor"
FSDP_FILES_EXIST=false

if [ -d "$ACTOR_PATH" ]; then
    # Проверяем наличие FSDP файлов для нужного количества GPU
    # Ищем файлы с нужным world_size (NUM_GPUS)
    EXPECTED_FILES=0
    FOUND_FILES=0
    
    for rank in $(seq 0 $((NUM_GPUS - 1))); do
        if [ -f "$ACTOR_PATH/model_world_size_${NUM_GPUS}_rank_${rank}.pt" ]; then
            FOUND_FILES=$((FOUND_FILES + 1))
        fi
        EXPECTED_FILES=$((EXPECTED_FILES + 1))
    done
    
    # Если найдены все необходимые файлы для нужного количества GPU
    if [ "$FOUND_FILES" -eq "$EXPECTED_FILES" ] && [ "$EXPECTED_FILES" -gt 0 ]; then
        FSDP_FILES_EXIST=true
        echo "✓ FSDP чекпоинт уже существует в $ACTOR_PATH (world_size=$NUM_GPUS, найдено $FOUND_FILES/$EXPECTED_FILES файлов)"
    elif [ "$FOUND_FILES" -gt 0 ]; then
        echo "⚠ Найдены FSDP файлы, но для другого количества GPU (найдено $FOUND_FILES файлов, требуется $EXPECTED_FILES)"
        echo "  Будет выполнена конвертация для правильного количества GPU"
    fi
fi

# Если FSDP чекпоинт уже есть с правильным количеством GPU, пропускаем конвертацию
if [ "$FSDP_FILES_EXIST" = true ]; then
    echo "Конвертация не требуется. Используется существующий FSDP чекпоинт."
    exit 0
fi

# Проверяем наличие LoRA адаптера
LORA_ADAPTER_PATH="$CHECKPOINT_PATH/lora_adapter"
if [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo "Ошибка: LoRA адаптер не найден в $LORA_ADAPTER_PATH"
    echo "И FSDP чекпоинт отсутствует в $ACTOR_PATH"
    echo "Необходимо либо предоставить LoRA адаптер, либо FSDP чекпоинт"
    exit 1
fi

# Проверяем наличие файлов адаптера
if [ ! -f "$LORA_ADAPTER_PATH/adapter_model.safetensors" ] && [ ! -f "$LORA_ADAPTER_PATH/adapter_model.bin" ]; then
    echo "Ошибка: Файлы адаптера не найдены в $LORA_ADAPTER_PATH"
    echo "Ожидаются: adapter_model.safetensors или adapter_model.bin"
    exit 1
fi

echo "LoRA адаптер найден. Начинаем конвертацию в FSDP формат..."
echo "  Base model: $BASE_MODEL"
echo "  LoRA adapter: $LORA_ADAPTER_PATH"
echo "  Output path: $CHECKPOINT_PATH"
echo "  LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA"
echo "  Target modules: $TARGET_MODULES"
echo "  Number of GPUs: $NUM_GPUS"

# Определяем путь к скрипту конвертации (уже определен выше)
CONVERT_SCRIPT="$PROJECT_ROOT/scripts/convert_sft_to_fsdp.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Ошибка: Скрипт конвертации не найден: $CONVERT_SCRIPT"
    exit 1
fi

# Проверяем наличие torchrun или python -m torch.distributed.run
if command -v torchrun &> /dev/null; then
    TORCHRUN_CMD="torchrun"
elif python3 -m torch.distributed.run --help &> /dev/null 2>&1; then
    TORCHRUN_CMD="python3 -m torch.distributed.run"
else
    echo "Ошибка: torchrun не найден. Установите PyTorch с поддержкой distributed."
    exit 1
fi

# Запускаем конвертацию
echo "Запуск конвертации через $TORCHRUN_CMD..."
cd "$PROJECT_ROOT"

if ! $TORCHRUN_CMD --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    "$CONVERT_SCRIPT" \
    --base_model "$BASE_MODEL" \
    --adapter_path "$LORA_ADAPTER_PATH" \
    --output_path "$CHECKPOINT_PATH" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES" \
    --trust_remote_code; then
    echo "Ошибка: Конвертация завершилась с ошибкой"
    exit 1
fi

# Проверяем успешность конвертации
EXPECTED_FILES=0
FOUND_FILES=0

for rank in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -f "$ACTOR_PATH/model_world_size_${NUM_GPUS}_rank_${rank}.pt" ]; then
        FOUND_FILES=$((FOUND_FILES + 1))
    fi
    EXPECTED_FILES=$((EXPECTED_FILES + 1))
done

if [ "$FOUND_FILES" -eq "$EXPECTED_FILES" ] && [ "$EXPECTED_FILES" -gt 0 ]; then
    echo "✓ Конвертация успешно завершена! FSDP чекпоинт сохранен в: $ACTOR_PATH"
    echo "  Создано файлов: $FOUND_FILES/$EXPECTED_FILES (world_size=$NUM_GPUS)"
else
    echo "Ошибка: Конвертация завершилась, но FSDP файлы не найдены в $ACTOR_PATH"
    echo "  Ожидалось файлов: $EXPECTED_FILES, найдено: $FOUND_FILES"
    exit 1
fi
