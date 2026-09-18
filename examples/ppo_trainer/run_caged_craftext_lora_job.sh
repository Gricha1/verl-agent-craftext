#!/bin/bash
# set -x

# Conda: Docker image (verl-agent-311) or legacy jovyan paths
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "verl-agent-311" ]; then
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate verl-agent-311
  elif [ -f /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/jovyan/nsorokin/miniconda3/etc/profile.d/conda.sh
    conda activate /home/jovyan/nsorokin/verl-agent-craftext/verl-agent-conda-venv-311/
  fi
fi

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}"
# Longer timeouts: intermittent network to comet.com on some nodes
export COMET_TIMEOUT="${COMET_TIMEOUT:-120}"
export COMET_WS_CONNECTION_TIMEOUT="${COMET_WS_CONNECTION_TIMEOUT:-60}"
export COMET_WS_CONNECTION_IDLE_TIMEOUT="${COMET_WS_CONNECTION_IDLE_TIMEOUT:-120}"
# Prefer online logging; offline mode hides runs from the UI
export COMET_OFFLINE_DIRECTORY="${COMET_OFFLINE_DIRECTORY:-}"
unset COMET_OFFLINE_MODE 2>/dev/null || true

# Offline validation/preflight must not wait for an external Comet connection.
if [ "${COMET_DISABLED:-0}" = "1" ]; then
    unset COMET_API_KEY
    export GPU_PROFILER_ENABLED=0
fi

# JAX backend: configured in main_ppo via ++env.use_jax_gpu=True (GPU if jaxlib+cuda works, else CPU fallback).
# Do NOT set JAX_PLATFORMS=cuda here — breaks when jaxlib has no CUDA backend (vLLM can still use GPU).
# Regenerate Craftax texture pickle when JAX version changes (avoids ShapedArray/named_shape pickle errors).
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp}"
export COMET_EXPERIMENT_KEY_FILE="${COMET_EXPERIMENT_KEY_FILE:-$RAY_TEMP_DIR/comet_experiment_key.txt}"
export GPU_PROFILER_ENABLED="${GPU_PROFILER_ENABLED:-1}"
export GPU_PROFILER_INTERVAL_SEC="${GPU_PROFILER_INTERVAL_SEC:-5}"
export COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl_agent_caged_craftext}"
mkdir -p "$RAY_TEMP_DIR"

GPU_PROFILER_PID=""
cleanup_gpu_profiler() {
  if [ -n "$GPU_PROFILER_PID" ] && kill -0 "$GPU_PROFILER_PID" 2>/dev/null; then
    kill "$GPU_PROFILER_PID" 2>/dev/null || true
    wait "$GPU_PROFILER_PID" 2>/dev/null || true
    echo "[INFO] GPU profiler stopped (pid=$GPU_PROFILER_PID)"
  fi
}
trap cleanup_gpu_profiler EXIT INT TERM

# comet_ml is installed in Docker image via setup_caged_craftext_deps.sh
python -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

# НЕ устанавливаем CUDA_VISIBLE_DEVICES="" здесь, так как это мешает Ray видеть GPU
# Encoder'ы в caged_craftext уже исправлены и проверяют torch.cuda.is_available() перед использованием CUDA

ENGINE=${1:-vllm}
# LOG_PROB_ACTION_ONLY: если true, log_prob считается только для токенов внутри <action> тегов (без reasoning)
# NO_REASONING: если true, агент не будет генерировать reasoning, только action
# TRAIN_DATA_SIZE: размер обучающей выборки (по умолчанию 32, можно задать 64)
# MAX_RESPONSE_LENGTH: макс. длина ответа (по умолчанию 512)
# AUTO_RESET: если true, среды автоматически перезапускаются при завершении эпизода, чтобы собрать полный rollout (max_steps шагов)
# USE_ACTION_HEAD: если true, actor использует action head (распределение над действиями) вместо text generation
# Использование: bash run_caged_craftext_lora_job.sh vllm false false 32
# С no_reasoning и ответом 32: bash run_caged_craftext_lora_job.sh vllm true true 32 32
# С auto reset: bash run_caged_craftext_lora_job.sh vllm false false 32 512 true
# С action head: bash run_caged_craftext_lora_job.sh vllm false false 32 512 false true
# С расширенным шаблоном extended_template: bash run_caged_craftext_lora_job.sh vllm false false 32 512 false false 4000 true extended_template
# С critic_warmup=10: bash run_caged_craftext_lora_job.sh vllm false false 8 512 false false 4000 true default_template 10
# С новым шаблоном наблюдения: bash run_caged_craftext_lora_job.sh vllm false false 8 128 false false 4000 true default_template 0 ascii_v2
# total_epochs: количество эпох (по умолчанию 4000)
# USE_ACTOR_LORA: true — actor с LoRA (lora_rank=64, lora_alpha=64), false — обучение без LoRA
# PROMPT_TEMPLATE_TYPE: тип шаблона промпта ('default_template' или 'extended_template', по умолчанию 'default_template')
# CRITIC_WARMUP: количество шагов для разогрева критика перед обновлением актора (по умолчанию 10)
# OBSERVATION_TYPE: тип наблюдения ('ascii', 'ascii_v2', 'text'; по умолчанию 'ascii')
LOG_PROB_ACTION_ONLY=${2:-false}
NO_REASONING=${3:-false}
train_data_size=${4:-8}
max_response_length=${5:-512}
AUTO_RESET=${6:-false}
USE_ACTION_HEAD=${7:-false}
total_epochs=${8:-4000}
USE_ACTOR_LORA=${9:-true}
PROMPT_TEMPLATE_TYPE=${10:-default_template}
CRITIC_WARMUP=${11:-10}
OBSERVATION_TYPE=${12:-ascii}
# Убираем аргументы скрипта, чтобы они не передавались в Hydra
shift 12 2>/dev/null || shift 11 2>/dev/null || shift 10 2>/dev/null || shift 9 2>/dev/null || shift 8 2>/dev/null || shift 7 2>/dev/null || shift 6 2>/dev/null || shift 5 2>/dev/null || shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true

# Если используется action head или single-token actions, max_response_length = 1
# single_token_action: 1 token (action); critic uses separate value_prompt_template_type
if [ "$USE_ACTION_HEAD" = "true" ] || [ "$PROMPT_TEMPLATE_TYPE" = "single_token_action" ]; then
    max_response_length=1
    echo "[INFO] max_response_length=1 (USE_ACTION_HEAD=$USE_ACTION_HEAD, PROMPT_TEMPLATE_TYPE=$PROMPT_TEMPLATE_TYPE)"
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# Путь к пакету caged_craftext (должен содержать модуль craftext.environment)
# Путь к клонированному репозиторию CAGED-CrafText
# Вычисляем путь относительно корня проекта (работает и в Docker, и на хосте)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CAGED_PATH="$PROJECT_ROOT/caged_craftext"

# Используем переменную окружения, если установлена, иначе вычисленный путь
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$DEFAULT_CAGED_PATH}"
export CRAFTAX_PATH="${CAGED_CRAFTEXT_PATH}/Craftax"
export PYTHONPATH="${CRAFTAX_PATH}:${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}"
echo "[INFO] CAGED_CRAFTEXT_PATH установлен: $CAGED_CRAFTEXT_PATH"
echo "[INFO] CRAFTAX_PATH (local, not pip): $CRAFTAX_PATH"
echo "[INFO] Корень проекта: $PROJECT_ROOT"
echo "[INFO] ENGINE: $ENGINE"
echo "[INFO] LOG_PROB_ACTION_ONLY: $LOG_PROB_ACTION_ONLY"
echo "[INFO] NO_REASONING: $NO_REASONING"
echo "[INFO] TRAIN_DATA_SIZE: $train_data_size"
echo "[INFO] max_response_length: $max_response_length"
echo "[INFO] AUTO_RESET: $AUTO_RESET"
echo "[INFO] USE_ACTION_HEAD: $USE_ACTION_HEAD"
echo "[INFO] total_epochs: $total_epochs"
echo "[INFO] USE_ACTOR_LORA: $USE_ACTOR_LORA"
echo "[INFO] PROMPT_TEMPLATE_TYPE: $PROMPT_TEMPLATE_TYPE"
echo "[INFO] CRITIC_WARMUP: $CRITIC_WARMUP"
echo "[INFO] OBSERVATION_TYPE: $OBSERVATION_TYPE"

num_cpus_per_env_worker=0.03
val_data_size=${VAL_DATA_SIZE:-16}

# export RUN_NAME="run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_$(date +%Y%m%d-%H%M%S)"
export RUN_NAME="${RUN_NAME:-run_ppo_qwen2.5_1.5b_caged_craftext_energy_collect_wood_$(date +%Y%m%d-%H%M%S)}"

rm -f "$COMET_EXPERIMENT_KEY_FILE"
# Comet's metadata uploader may remain alive after it has written the key.  Do
# not let that optional early-registration helper block PPO startup forever.
timeout "${COMET_INIT_TIMEOUT:-45}" python3 "$PROJECT_ROOT/scripts/init_comet_experiment.py" \
  --project "$COMET_PROJECT_NAME" \
  --experiment "$RUN_NAME" \
  --key-file "$COMET_EXPERIMENT_KEY_FILE" || true

if [ "$GPU_PROFILER_ENABLED" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  export GPU_PROFILER_LOG="${GPU_PROFILER_LOG:-$RAY_TEMP_DIR/gpu_profiler.log}"
  python3 "$PROJECT_ROOT/scripts/gpu_profiler.py" \
    --interval "$GPU_PROFILER_INTERVAL_SEC" \
    --comet-project "$COMET_PROJECT_NAME" \
    --comet-experiment "$RUN_NAME" \
    --comet-key-file "$COMET_EXPERIMENT_KEY_FILE" \
    >>"$GPU_PROFILER_LOG" 2>&1 &
  GPU_PROFILER_PID=$!
  echo "[INFO] GPU profiler started (pid=$GPU_PROFILER_PID, interval=${GPU_PROFILER_INTERVAL_SEC}s, log=$GPU_PROFILER_LOG)"
  echo "[INFO] GPU metrics in Comet: gpu/0/utilization_pct, gpu/0/memory_used_gb (tail -f $GPU_PROFILER_LOG)"
else
  echo "[INFO] GPU profiler disabled or nvidia-smi unavailable (GPU_PROFILER_ENABLED=$GPU_PROFILER_ENABLED)"
fi

# On offline clusters the ready-made rollout parquet files are mounted into the
# container.  Do not replace them by downloading the unrelated demo dataset.
if [ "${SKIP_DATA_PREP:-0}" = "1" ]; then
  echo "[INFO] SKIP_DATA_PREP=1: using existing $HOME/data/verl-agent/text/*.parquet"
elif [ -f "$HOME/data/verl-agent/text/train.parquet" ] && [ -f "$HOME/data/verl-agent/text/test.parquet" ]; then
  echo "[INFO] Existing parquet dataset found; skipping demo data preparation"
else
  python -m examples.data_preprocess.prepare \
      --mode 'text' \
      --train_data_size $train_data_size \
      --val_data_size $val_data_size
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.lora_rank=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "64"; else echo "0"; fi) \
    actor_rollout_ref.model.lora_alpha=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "64"; else echo "0"; fi) \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.lora_rank=0 \
    critic.model.lora_alpha=16 \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=128 \
    critic.ppo_micro_batch_size_per_gpu=64 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.use_episode_return_as_token_reward=False \
    +algorithm.log_prob_action_only=$LOG_PROB_ACTION_ONLY \
    env.env_name='caged_craftext/CagedCraftextEnv' \
    +env.enable_reasoning=$(if [ "$NO_REASONING" = "true" ]; then echo "False"; else echo "True"; fi) \
    +env.craftext_settings='achievements_safe_budget_energy_collect_wood' \
    +env.observation_type="$OBSERVATION_TYPE" \
    +env.prompt_template_type=$PROMPT_TEMPLATE_TYPE \
    actor_rollout_ref.actor.single_token_actions=$(if [ "$PROMPT_TEMPLATE_TYPE" = "single_token_action" ]; then echo "True"; else echo "False"; fi) \
    actor_rollout_ref.actor.actor_value_token=False \
    +env.auto_reset=$(if [ "$AUTO_RESET" = "true" ]; then echo "True"; else echo "False"; fi) \
    ++env.use_jax_gpu=False \
    +actor_rollout_ref.model.use_action_head=$(if [ "$USE_ACTION_HEAD" = "true" ]; then echo "True"; else echo "False"; fi) \
    +actor_rollout_ref.model.num_actions=17 \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length="${HISTORY_LENGTH:-50}" \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger=['console','comet'] \
    trainer.project_name="$COMET_PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=0 \
    trainer.env_val_video_freq=200000 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True $@
