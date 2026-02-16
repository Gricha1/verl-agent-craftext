#!/bin/bash
# set -x

source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

export JAX_PLATFORMS=cpu
export RAY_TEMP_DIR="/home/jovyan/nsorokin/ray_temp" 

pip install comet_ml

# НЕ устанавливаем CUDA_VISIBLE_DEVICES="" здесь, так как это мешает Ray видеть GPU
# Encoder'ы в caged_craftext уже исправлены и проверяют torch.cuda.is_available() перед использованием CUDA

#source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
#conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/

ENGINE=${1:-vllm}
# LOG_PROB_ACTION_ONLY: если true, log_prob считается только для токенов внутри <action> тегов (без reasoning)
# NO_REASONING: если true, агент не будет генерировать reasoning, только action
# TRAIN_DATA_SIZE: размер обучающей выборки (по умолчанию 32, можно задать 64)
# Использование: bash run_caged_craftext_lora_job.sh vllm false false 32
LOG_PROB_ACTION_ONLY=${2:-false}
NO_REASONING=${3:-false}
train_data_size=${4:-32}
# Убираем аргументы скрипта, чтобы они не передавались в Hydra
shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true
export VLLM_ATTENTION_BACKEND=XFORMERS

# Путь к пакету caged_craftext (должен содержать модуль craftext.environment)
# Путь к клонированному репозиторию CAGED-CrafText
# Вычисляем путь относительно корня проекта (работает и в Docker, и на хосте)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CAGED_PATH="$PROJECT_ROOT/caged_craftext"

# Используем переменную окружения, если установлена, иначе вычисленный путь
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$DEFAULT_CAGED_PATH}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}"
echo "[INFO] CAGED_CRAFTEXT_PATH установлен: $CAGED_CRAFTEXT_PATH"
echo "[INFO] Корень проекта: $PROJECT_ROOT"
echo "[INFO] ENGINE: $ENGINE"
echo "[INFO] LOG_PROB_ACTION_ONLY: $LOG_PROB_ACTION_ONLY"
echo "[INFO] NO_REASONING: $NO_REASONING"
echo "[INFO] TRAIN_DATA_SIZE: $train_data_size"

num_cpus_per_env_worker=0.03
val_data_size=8

# export RUN_NAME="run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="run_ppo_qwen2.5_1.5b_caged_craftext_energy_collect_wood_$(date +%Y%m%d-%H%M%S)"

python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.lora_rank=0 \
    critic.model.lora_alpha=16 \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=16 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    +algorithm.log_prob_action_only=$LOG_PROB_ACTION_ONLY \
    env.env_name='caged_craftext/CagedCraftextEnv' \
    +env.enable_reasoning=$(if [ "$NO_REASONING" = "true" ]; then echo "False"; else echo "True"; fi) \
    +env.craftext_settings='achievements_safe_budget_energy_collect_wood' \
    +env.observation_type='ascii' \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=0 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','comet'] \
    trainer.project_name='verl_agent_caged_craftext' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=0 \
    trainer.total_epochs=500 \
    trainer.val_before_train=False $@
