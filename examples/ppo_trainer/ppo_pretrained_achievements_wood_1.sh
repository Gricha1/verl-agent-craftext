#!/bin/bash
# Команда запуска PPO обучения с предобученными весами на achievements_wood
# с новым шаблоном наблюдения ascii_v2.
# Использование: bash examples/ppo_trainer/ppo_pretrained_achievements_wood_1.sh
#
# Скрипт автоматически конвертирует LoRA адаптер в FSDP формат, если необходимо.

set -e

# Путь к чекпоинту (можно изменить при необходимости)
CHECKPOINT_PATH="pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000"

# Параметры для конвертации (соответствуют параметрам обучения)
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LORA_RANK=64
LORA_ALPHA=64
TARGET_MODULES="all-linear"
NUM_GPUS=2  # Должно соответствовать trainer.n_gpus_per_node

echo "=========================================="
echo "Проверка и конвертация чекпоинта (если необходимо)"
echo "=========================================="

bash examples/ppo_trainer/convert_lora_to_fsdp_if_needed.sh \
    "$CHECKPOINT_PATH" \
    "$BASE_MODEL" \
    "$LORA_RANK" \
    "$LORA_ALPHA" \
    "$TARGET_MODULES" \
    "$NUM_GPUS"

echo ""
echo "=========================================="
echo "Запуск PPO на achievements_wood (ascii_v2)"
echo "=========================================="

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
    vllm \
    false \
    false \
    64 \
    128 \
    false \
    false \
    8000 \
    true \
    extended_template \
    10 \
    ascii_v2 \
    ++env.craftext_settings='achievements_wood' \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CHECKPOINT_PATH" \
    trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext_wood_1
