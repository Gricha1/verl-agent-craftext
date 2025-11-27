#!/bin/bash
# Скрипт для запуска оценки всех моделей последовательно

set -e  # Остановка при ошибке

export CUDA_VISIBLE_DEVICES=1

bash evaluate_qwen2_vl_2b.sh
bash evaluate_qwen2.5_1.5b.sh
bash evaluate_qwen2.5_7b.sh