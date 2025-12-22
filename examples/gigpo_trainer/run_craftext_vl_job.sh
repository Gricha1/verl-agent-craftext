#!/bin/bash
# set -x

export JAX_PLATFORMS=cpu
export RAY_TEMP_DIR="/home/jovyan/nsorokin/ray_temp" 

source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

export CUDA_VISIBLE_DEVICES=0
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.03
train_data_size=2
val_data_size=4
group_size=8
mode="mean_std_norm"

export RUN_NAME="run_gigpo_qwen2_vl_2b_achievements_wood_$(date +%Y%m%d-%H%M%S)"

# Подготовка данных остается той же, так как мы используем текстовый режим
python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.model.path=Qwen/Qwen2-VL-2B-Instruct \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name='craftext/CraftextVLEnv' \
    +env.craftext_settings='achievements_wood' \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.history_length=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','comet'] \
    trainer.project_name='verl_agent_craftext_vl' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=500 \
    trainer.val_before_train=True $@