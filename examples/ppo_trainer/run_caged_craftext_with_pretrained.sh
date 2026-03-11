#!/bin/bash
# Команда запуска PPO обучения с предобученными весами, extended_template и critic_warmup=10
# Использование: bash examples/ppo_trainer/run_caged_craftext_with_pretrained.sh

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
    vllm \
    false \
    false \
    32 \
    512 \
    false \
    false \
    4000 \
    true \
    extended_template \
    10 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000 \
    trainer.default_local_dir=training_checkpoints/verl_agent_caged_craftext
