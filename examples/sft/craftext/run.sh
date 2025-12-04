# examples/sft/craftext/run.sh
#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export TRAIN_DATA=/home/n.sorokin/verl-agent/data/expert_trajectories_sft/train.parquet
export VAL_DATA=/home/n.sorokin/verl-agent/data/expert_trajectories_sft/val.parquet
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# RUN_NAME должен быть определен ДО использования в SAVE_PATH
export RUN_NAME="run_sft_qwen2.5_1.5b_achievements_wood_$(date +%Y%m%d-%H%M%S)"
export SAVE_PATH=/home/n.sorokin/verl-agent/checkpoints/verl_agent_craftext/${RUN_NAME}

# ПРИМЕЧАНИЕ: После завершения SFT обучения чекпоинты будут сохранены в формате HuggingFace (LoRA адаптер).
# Для использования в RL обучении (run_craftext_from_sft.sh) чекпоинт нужно сконвертировать в FSDP формат.
# Используйте convert_checkpoint.sh для конвертации последнего чекпоинта.

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=null \
    data.response_dict_keys=null \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    data.truncation=error \
    model.partial_pretrain=${BASE_MODEL} \
    model.lora_rank=64 \
    model.lora_alpha=64 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=craftext-expert-sft \
    trainer.experiment_name=${RUN_NAME} \
    trainer.total_epochs=3 \
    trainer.logger=['console','tensorboard','comet'] \
    trainer.default_hdfs_dir=null