#!/bin/bash
# Скрипт для оценки Qwen2-VL-2B модели (VLM)
export CUDA_VISIBLE_DEVICES=1

python evaluate_model.py \
    --model-name "qwen2_vl_2b" \
    --base-model-path "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset-path "craftext_perception_dataset.jsonl" \
    --checkpoint-path "" \
    --lora-rank 0 \
    --max-response-length 128 \
    --temperature 0.0 \
    --no-text-observation \
    --batch-size 1 \
    --output-dir "evaluation_results"

