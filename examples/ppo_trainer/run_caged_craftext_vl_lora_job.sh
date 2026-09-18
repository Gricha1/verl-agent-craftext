#!/bin/bash
# PPO actor-value (one VL model, dual prompts) on caged_craftext.
# For classic PPO with separate actor + critic VL models use run_caged_craftext_vl_dual_lora_job.sh
#
# Minimal weights (HuggingFace):
#   Qwen/Qwen2.5-VL-3B-Instruct  — smallest Qwen2.5-VL (recommended default)
#   Qwen/Qwen2-VL-2B-Instruct    — older 2B line (even smaller, weaker)
#
# Usage:
#   bash examples/ppo_trainer/run_caged_craftext_vl_lora_job.sh vllm false true 16 1 false false 8000 true single_token_action 0 ascii
#   QWEN_VL_MODEL=Qwen/Qwen2-VL-2B-Instruct bash examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh

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

# Comet credentials are never committed: export COMET_API_KEY, or keep it in the
# git-ignored ~/.config/verl_comet.env (see experiment_dashboard/.env.example).
if [ -f "$HOME/.config/verl_comet.env" ]; then . "$HOME/.config/verl_comet.env"; fi
export COMET_API_KEY="${COMET_API_KEY:?COMET_API_KEY is unset (see ~/.config/verl_comet.env)}"
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_temp}"
mkdir -p "$RAY_TEMP_DIR"
# Host ~125GB RAM: TP=1 shards rollout batch; strip PIL multi_modal_data before update (code fix).
# bf16 model load halves init RAM vs default fp32 actor weights.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
ray stop --force 2>/dev/null || true

python -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

ENGINE=${1:-vllm}
LOG_PROB_ACTION_ONLY=${2:-false}
NO_REASONING=${3:-true}
train_data_size=${4:-16}
max_response_length=${5:-1}
AUTO_RESET=${6:-false}
USE_ACTION_HEAD=${7:-false}
total_epochs=${8:-8000}
USE_ACTOR_LORA=${9:-true}
PROMPT_TEMPLATE_TYPE=${10:-single_token_action}
CRITIC_WARMUP=${11:-0}
OBSERVATION_TYPE=${12:-ascii}
shift 12 2>/dev/null || shift 11 2>/dev/null || shift 10 2>/dev/null || shift 9 2>/dev/null || shift 8 2>/dev/null || shift 7 2>/dev/null || shift 6 2>/dev/null || shift 5 2>/dev/null || shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true

if [ "$USE_ACTION_HEAD" = "true" ] || [ "$PROMPT_TEMPLATE_TYPE" = "single_token_action" ]; then
  max_response_length=1
fi

QWEN_VL_MODEL="${QWEN_VL_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
# ~task text + ~336 vision tokens (Craftax frame); same budget as dual VL / text train hparams
VL_MAX_PROMPT_LENGTH="${VL_MAX_PROMPT_LENGTH:-768}"
VL_MM_MAX_PIXELS="${VL_MM_MAX_PIXELS:-196000}"
VL_MM_MIN_PIXELS="${VL_MM_MIN_PIXELS:-65536}"
VL_GPU_MEMORY_UTIL="${VL_GPU_MEMORY_UTIL:-0.22}"
# update_actor RAM: mini/micro batch and logprob chunk size (125GB host OOM at 64 envs / mini=32 / micro=16).
VL_PPO_MINI_BATCH="${VL_PPO_MINI_BATCH:-16}"
VL_PPO_MICRO_BATCH="${VL_PPO_MICRO_BATCH:-8}"
VL_LOGPROB_MICRO_BATCH="${VL_LOGPROB_MICRO_BATCH:-16}"
# TP=1 => vLLM data-parallel across the 2 GPUs (DP=2), sharding the rollout batch to halve CPU RAM.
VL_TENSOR_PARALLEL="${VL_TENSOR_PARALLEL:-1}"
RETURN_BIN_MIN="${RETURN_BIN_MIN:--5}"
RETURN_BIN_MAX="${RETURN_BIN_MAX:-6}"
RETURN_BIN_STEP="${RETURN_BIN_STEP:-0.4}"
ACTOR_VALUE_LOSS_COEF="${ACTOR_VALUE_LOSS_COEF:-1.0}"

export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_CAGED_PATH="$PROJECT_ROOT/caged_craftext"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$DEFAULT_CAGED_PATH}"
export CRAFTAX_PATH="${CAGED_CRAFTEXT_PATH}/Craftax"
export PYTHONPATH="${PROJECT_ROOT}:${CRAFTAX_PATH}:${CAGED_CRAFTEXT_PATH}:${PYTHONPATH}"

num_cpus_per_env_worker=0.03
val_data_size=${VAL_DATA_SIZE:-16}
export RUN_NAME="${RUN_NAME:-run_ppo_qwen_vl_caged_craftext_$(date +%Y%m%d-%H%M%S)}"

echo "[INFO] VL model: $QWEN_VL_MODEL"
echo "[INFO] env: caged_craftext/CagedCraftextVLEnv (pixel obs + <image> in prompt)"
echo "[INFO] train_data_size=$train_data_size max_response_length=$max_response_length"
echo "[INFO] VL_MAX_PROMPT_LENGTH=$VL_MAX_PROMPT_LENGTH VL_MM_MAX_PIXELS=$VL_MM_MAX_PIXELS VL_GPU_MEMORY_UTIL=$VL_GPU_MEMORY_UTIL VL_TENSOR_PARALLEL=$VL_TENSOR_PARALLEL mini=$VL_PPO_MINI_BATCH micro=$VL_PPO_MICRO_BATCH logprob_micro=$VL_LOGPROB_MICRO_BATCH"
echo "[INFO] actor-value: VL image prompt for actor and value (no ASCII observation in prompts)"

python -m examples.data_preprocess.prepare \
  --mode visual \
  --train_data_size "$train_data_size" \
  --val_data_size "$val_data_size"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files=$HOME/data/verl-agent/visual/train.parquet \
  data.val_files=$HOME/data/verl-agent/visual/test.parquet \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=$VL_MAX_PROMPT_LENGTH \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  +data.dataloader_num_workers=0 \
  data.truncation='error' \
  data.image_key=images \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$QWEN_VL_MODEL" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "64"; else echo "0"; fi) \
  actor_rollout_ref.model.lora_alpha=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "64"; else echo "0"; fi) \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$VL_PPO_MINI_BATCH \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$VL_PPO_MICRO_BATCH \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$VL_LOGPROB_MICRO_BATCH \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$VL_TENSOR_PARALLEL \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=$VL_GPU_MEMORY_UTIL \
  +actor_rollout_ref.rollout.limit_images=1 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_pixels=$VL_MM_MIN_PIXELS \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=$VL_MM_MAX_PIXELS \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$VL_LOGPROB_MICRO_BATCH \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_actor_value_token=True \
  actor_rollout_ref.actor.actor_value_token=True \
  actor_rollout_ref.actor.actor_value_loss_coef="$ACTOR_VALUE_LOSS_COEF" \
  actor_rollout_ref.actor.actor_value_target_from_returns=True \
  actor_rollout_ref.actor.actor_value_return_min="$RETURN_BIN_MIN" \
  actor_rollout_ref.actor.actor_value_return_max="$RETURN_BIN_MAX" \
  actor_rollout_ref.actor.actor_value_return_bin_step="$RETURN_BIN_STEP" \
  actor_rollout_ref.actor.single_token_actions=$(if [ "$PROMPT_TEMPLATE_TYPE" = "single_token_action" ]; then echo "True"; else echo "False"; fi) \
  algorithm.use_kl_in_reward=False \
  reward_model.use_episode_return_as_token_reward=False \
  +algorithm.log_prob_action_only=$LOG_PROB_ACTION_ONLY \
  env.env_name='caged_craftext/CagedCraftextVLEnv' \
  +env.enable_reasoning=$(if [ "$NO_REASONING" = "true" ]; then echo "False"; else echo "True"; fi) \
  +env.craftext_settings='debug_square_8x8' \
  +env.observation_type="$OBSERVATION_TYPE" \
  +env.prompt_template_type=$PROMPT_TEMPLATE_TYPE \
  +env.value_prompt_template_type=single_token_return_vl \
  +env.value_return_min="$RETURN_BIN_MIN" \
  +env.value_return_max="$RETURN_BIN_MAX" \
  +env.value_return_bin_step="$RETURN_BIN_STEP" \
  +env.auto_reset=$(if [ "$AUTO_RESET" = "true" ]; then echo "True"; else echo "False"; fi) \
  ++env.use_jax_gpu=False \
  ++env.jax_gpu_fraction=0.15 \
  +actor_rollout_ref.model.use_action_head=$(if [ "$USE_ACTION_HEAD" = "true" ]; then echo "True"; else echo "False"; fi) \
  +actor_rollout_ref.model.num_actions=17 \
  env.seed=0 \
  env.max_steps=50 \
  env.history_length="${HISTORY_LENGTH:-50}" \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.critic_warmup=$CRITIC_WARMUP \
  trainer.logger=['console','comet'] \
  trainer.project_name='verl_agent_caged_craftext' \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=0 \
  trainer.env_val_video_freq=200000 \
  trainer.total_epochs=$total_epochs \
  trainer.val_before_train=True "$@"
