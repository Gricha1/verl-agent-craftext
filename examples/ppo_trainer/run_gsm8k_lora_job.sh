#!/bin/bash
# PPO / actor-value PPO on GSM8K (single-turn math, rule-based reward).
#
# Usage:
#   bash examples/ppo_trainer/run_gsm8k_lora_job.sh
#   TRAIN_BATCH_SIZE=32 bash examples/ppo_trainer/run_gsm8k_lora_job.sh vllm
#
# Args (optional):
#   $1 ENGINE (default vllm)
#   $2 LOG_PROB_ACTION_ONLY (default false)
#   $3 TRAIN_BATCH_SIZE (default 64)
#   $4 MAX_RESPONSE_LENGTH (default 512)
#   $5 TOTAL_EPOCHS (default 3)
#   $6 USE_ACTOR_LORA (default true)
#   $7 CRITIC_WARMUP (default 0)
# Remaining args are passed to Hydra.

set -e

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
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp}"
mkdir -p "$RAY_TEMP_DIR"

python -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

ENGINE=${1:-vllm}
LOG_PROB_ACTION_ONLY=${2:-false}
train_data_size=${3:-64}
max_response_length=${4:-512}
total_epochs=${5:-6}
USE_ACTOR_LORA=${6:-true}
CRITIC_WARMUP=${7:-0}
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-64}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-64}"
shift 7 2>/dev/null || shift 6 2>/dev/null || shift 5 2>/dev/null || shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true

export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-$HOME/data/gsm8k}"
val_data_size=${VAL_DATA_SIZE:-128}
num_cpus_per_env_worker=0.01

export RUN_NAME="${RUN_NAME:-run_ppo_gsm8k_qwen2.5_1.5b_$(date +%Y%m%d-%H%M%S)}"

echo "[INFO] PROJECT_ROOT: $PROJECT_ROOT"
echo "[INFO] GSM8K_DATA_DIR: $GSM8K_DATA_DIR"
echo "[INFO] ENGINE: $ENGINE"
echo "[INFO] TRAIN_BATCH_SIZE: $train_data_size"
echo "[INFO] VAL_BATCH_SIZE: $val_data_size"
echo "[INFO] max_response_length: $max_response_length"
echo "[INFO] total_epochs: $total_epochs"
echo "[INFO] USE_ACTOR_LORA: $USE_ACTOR_LORA (rank=$ACTOR_LORA_RANK alpha=$ACTOR_LORA_ALPHA when true)"
echo "[INFO] CRITIC_WARMUP: $CRITIC_WARMUP"

if [ ! -f "$GSM8K_DATA_DIR/train.parquet" ] || [ ! -f "$GSM8K_DATA_DIR/test.parquet" ]; then
  echo "[INFO] Preparing GSM8K parquet (with env_kwargs) ..."
  python "$PROJECT_ROOT/examples/data_preprocess/gsm8k.py" --local_dir "$GSM8K_DATA_DIR"
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files="$GSM8K_DATA_DIR/train.parquet" \
  data.val_files="$GSM8K_DATA_DIR/test.parquet" \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=1024 \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=False \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.model.lora_rank=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "$ACTOR_LORA_RANK"; else echo "0"; fi) \
  actor_rollout_ref.model.lora_alpha=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "$ACTOR_LORA_ALPHA"; else echo "0"; fi) \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.actor.single_token_actions=False \
  actor_rollout_ref.actor.actor_value_token=False \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=True \
  critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  critic.model.lora_rank=0 \
  critic.model.lora_alpha=16 \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_mini_batch_size=64 \
  critic.ppo_micro_batch_size_per_gpu=8 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.use_kl_in_reward=False \
  +algorithm.log_prob_action_only=$LOG_PROB_ACTION_ONLY \
  env.env_name='gsm8k/Gsm8kEnv' \
  +env.gsm8k_score_method=strict \
  +env.gsm8k_format_score=0.0 \
  +env.gsm8k_correct_score=1.0 \
  +env.auto_reset=False \
  env.seed=0 \
  env.max_steps=1 \
  env.history_length=0 \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.critic_warmup=$CRITIC_WARMUP \
  trainer.logger=['console','comet'] \
  trainer.project_name='verl_agent_caged_craftext' \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.env_val_video_freq=0 \
  trainer.total_epochs=$total_epochs \
  trainer.val_before_train=True \
  ++env.use_jax_gpu=False \
  "$@"
