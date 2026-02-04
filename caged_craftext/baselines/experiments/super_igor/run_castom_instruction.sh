#!/bin/bash

# Путь к папке с экспериментами
EXPERIMENTS_DIR="./super_experiments"

# Отображаем список доступных экспериментов
echo "Available experiments:"
EXPERIMENTS=($(ls -d "$EXPERIMENTS_DIR"/*/))
for i in "${!EXPERIMENTS[@]}"; do
    echo "$i) ${EXPERIMENTS[$i]}"
done

# Запрашиваем номер эксперимента
read -p "Enter the experiment number to run: " EXP_NUM
if [[ -z "${EXPERIMENTS[$EXP_NUM]}" ]]; then
    echo "Invalid experiment number."
    exit 1
fi

EXPERIMENT_PATH="${EXPERIMENTS[$EXP_NUM]}"
EXPERIMENT_NAME=$(basename "$EXPERIMENT_PATH")

# Путь к файлу с последним чекпоинтом
LAST_CHECKPOINT_PATH="${EXPERIMENT_PATH}/path_to_last_checkpoint.txt"

# Проверяем, существует ли файл
if [[ ! -f "$LAST_CHECKPOINT_PATH" ]]; then
    echo "Checkpoint file not found at $LAST_CHECKPOINT_PATH"
    exit 1
fi

# Читаем содержимое файла и извлекаем нужную часть пути
FULL_CHECKPOINT_PATH=$(cat "$LAST_CHECKPOINT_PATH")
EXPERIMENT_NAME=$(basename "$(dirname "$(dirname "$FULL_CHECKPOINT_PATH")")")


# Вывод для отладки
echo "Using experiment name: $EXPERIMENT_NAME"

# Запрашиваем команду для запуска
read -p "Enter the custom command (e.g., 'Collect tree'): " CUSTOM_COMMAND

# Генерация путей для датасета и сохранения
DATASET_PATH="${EXPERIMENT_PATH}/temp_dataset/super_dataset0_0.json"
SAVE_DATASET_PATH="${EXPERIMENT_PATH}/temp_dataset/train_0_0_.json"

# Путь к модели LLM
LLM_PATH="./pretrained_plan_llm/3_"

echo policy_inference.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --craftext_settings "SI_simplified_set" \
    --num_envs 1 \
    --plan_with_llm True \
    --inference 0 \
    --llm_path "$LLM_PATH" \
    --augment 0 \
    --dataset_path "$DATASET_PATH" \
    --save_dataset_path "$SAVE_DATASET_PATH" \
    --num_return_sequences 1 \
    --custom_command "$CUSTOM_COMMAND"
# Запуск команды
python policy_inference.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --craftext_settings "SI_simplified_set" \
    --num_envs 1 \
    --plan_with_llm True \
    --inference 0 \
    --llm_path "$LLM_PATH" \
    --augment 0 \
    --dataset_path "$DATASET_PATH" \
    --save_dataset_path "$SAVE_DATASET_PATH" \
    --num_return_sequences 1 \
    --custom_command "$CUSTOM_COMMAND"
