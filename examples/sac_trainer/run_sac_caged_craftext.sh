#!/bin/bash
# SAC (Soft Actor-Critic) training for Caged Craftext with action head.
# Usage: bash run_sac_caged_craftext.sh [ENGINE] [train_data_size] [total_epochs]
# Example: bash run_sac_caged_craftext.sh vllm 16 2000

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

ENGINE=${1:-vllm}
train_data_size=${2:-16}
total_epochs=${3:-2000}
shift 3 2>/dev/null || true

# SAC requires action head (discrete actions)
max_response_length=1

export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

DEFAULT_CAGED_PATH="$PROJECT_ROOT/caged_craftext"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$DEFAULT_CAGED_PATH}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}:${PROJECT_ROOT}"

export RUN_NAME="run_sac_qwen2.5_1.5b_caged_craftext_$(date +%Y%m%d-%H%M%S)"
echo "[INFO] SAC training: ENGINE=$ENGINE, train_data_size=$train_data_size, total_epochs=$total_epochs"

val_data_size=8
num_cpus_per_env_worker=0.03

python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python -m verl.trainer.main_sac \
    algorithm.adv_estimator=gae \
    algorithm.gamma=0.99 \
    +algorithm.sac_alpha=0.2 \
    +algorithm.sac_tau=0.005 \
    +algorithm.sac_batch_size=256 \
    +algorithm.replay_capacity=50000 \
    +algorithm.num_updates_per_step=1 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=512 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    +actor_rollout_ref.model.use_action_head=True \
    +actor_rollout_ref.model.num_actions=17 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    critic.optim.lr=1e-5 \
    +critic.sac_q=True \
    +critic.num_actions=17 \
    +critic.sac_tau=0.005 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.lora_rank=0 \
    critic.ppo_mini_batch_size=16 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    env.env_name='caged_craftext/CagedCraftextEnv' \
    +env.craftext_settings='achievements_safe_budget_energy_collect_wood' \
    +env.observation_type='ascii' \
    +env.auto_reset=True \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=0 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','comet'] \
    trainer.project_name='verl_agent_sac_caged_craftext' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=0 \
    trainer.env_val_video_freq=200000 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True \
    "$@"
