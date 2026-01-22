#!/bin/bash
# Скрипт для запуска валидации обученной модели на Caged Craftext с визуализацией

# Пример использования:
# ./run_evaluation_caged_craftext.sh \
#   --run_name "run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000" \
#   --global_step 100 \
#   --craftext_settings "achievements_safe_budget_drink" \
#   --num_episodes 3 \
#   --output_dir "./runs" \
#   --verbose

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m evaluation.evaluation_caged_craftext \
    "$@"
