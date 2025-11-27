#!/bin/bash
# Скрипт для оценки Qwen 2.5 1.5B модели
export CUDA_VISIBLE_DEVICES=1

python evaluate_model.py \
    --model-name "qwen2.5_1.5b" \
    --base-model-path "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset-path "craftext_perception_dataset.jsonl" \
    --checkpoint-path "" \
    --lora-rank 0 \
    --max-response-length 128 \
    --temperature 0.0 \
    --use-text-observation \
    --batch-size 32 \
    --output-dir "evaluation_results"

