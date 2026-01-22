#!/bin/bash
# Скрипт для автоматической валидации с последним чекпоинтом обучения
# Использование:
#   ./validate_latest_caged_craftext.sh [RUN_NAME] [OPTIONS...]
#
# Если RUN_NAME не указан, будет использован последний запуск из checkpoints/

set -e

# Настройки по умолчанию
PROJECT_NAME="${PROJECT_NAME:-verl_agent_caged_craftext}"
CRAFTEXT_SETTINGS="${CRAFTEXT_SETTINGS:-achievements_safe_budget_drink}"
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-./checkpoints}"

# Если RUN_NAME не передан, ищем последний запуск
if [ -z "$1" ] || [[ "$1" == --* ]]; then
    # Ищем последний запуск по дате модификации
    if [ -d "${BASE_CHECKPOINT_DIR}/${PROJECT_NAME}" ]; then
        LATEST_RUN=$(ls -t "${BASE_CHECKPOINT_DIR}/${PROJECT_NAME}" 2>/dev/null | head -1)
        if [ -z "$LATEST_RUN" ]; then
            echo "Ошибка: Не найдено ни одного запуска в ${BASE_CHECKPOINT_DIR}/${PROJECT_NAME}"
            exit 1
        fi
        RUN_NAME="$LATEST_RUN"
        echo "Используется последний запуск: $RUN_NAME"
    else
        echo "Ошибка: Директория ${BASE_CHECKPOINT_DIR}/${PROJECT_NAME} не существует"
        exit 1
    fi
else
    RUN_NAME="$1"
    shift  # Убираем RUN_NAME из аргументов
fi

# Путь к директории с чекпоинтами для этого запуска
RUN_DIR="${BASE_CHECKPOINT_DIR}/${PROJECT_NAME}/${RUN_NAME}"

if [ ! -d "$RUN_DIR" ]; then
    echo "Ошибка: Директория $RUN_DIR не существует"
    exit 1
fi

# Ищем последний чекпоинт через latest_checkpointed_iteration.txt
TRACKER_FILE="${RUN_DIR}/latest_checkpointed_iteration.txt"
if [ -f "$TRACKER_FILE" ]; then
    LATEST_STEP=$(cat "$TRACKER_FILE")
    CHECKPOINT_PATH="${RUN_DIR}/global_step_${LATEST_STEP}"
    echo "Найден последний чекпоинт: global_step_${LATEST_STEP}"
else
    # Если трекер не найден, ищем последний global_step_* по дате
    LATEST_STEP_DIR=$(ls -td "${RUN_DIR}"/global_step_* 2>/dev/null | head -1)
    if [ -z "$LATEST_STEP_DIR" ]; then
        echo "Ошибка: Не найдено ни одного чекпоинта в $RUN_DIR"
        exit 1
    fi
    CHECKPOINT_PATH="$LATEST_STEP_DIR"
    LATEST_STEP=$(basename "$CHECKPOINT_PATH" | sed 's/global_step_//')
    echo "Найден последний чекпоинт (по дате): global_step_${LATEST_STEP}"
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Ошибка: Чекпоинт $CHECKPOINT_PATH не существует"
    exit 1
fi

echo "=========================================="
echo "Валидация Caged Craftext"
echo "=========================================="
echo "Проект: $PROJECT_NAME"
echo "Запуск: $RUN_NAME"
echo "Чекпоинт: $CHECKPOINT_PATH"
echo "Настройки: $CRAFTEXT_SETTINGS"
echo "=========================================="

# Запускаем валидацию
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m evaluation.evaluation_caged_craftext \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --project_name "$PROJECT_NAME" \
    --run_name "$RUN_NAME" \
    --craftext_settings "$CRAFTEXT_SETTINGS" \
    --do_sample \
    --temperature 0.4 \
    "$@"  # Передаем остальные аргументы (--num_episodes, --output_dir и т.д.)
