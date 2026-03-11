#!/bin/bash
# Команда запуска PPO Saute обучения с предобученными весами, extended_template и critic_warmup=10
# Использование: bash examples/ppo_trainer/run_caged_craftext_saute_with_pretrained.sh

PROMPT_TEMPLATE_TYPE=extended_template \
bash examples/ppo_trainer/run_caged_craftext_saute_ppo_lora_job.sh \
    vllm \
    false \
    10 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/home/gorbov_gv/safe_rl_nlp/pretrained_checkpoints/run_sft_qwen2.5_1.5b_lora_achievements_wood_no_reasoning_filtered_160k_20260305-202020/global_step_4000 \
    trainer.default_local_dir=/home/gorbov_gv/safe_rl_nlp/training_checkpoints/verl_agent_caged_craftext
