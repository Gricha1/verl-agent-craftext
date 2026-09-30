#!/bin/bash
# PPO on AlfWorld (AlfredTWEnv): dual actor+critic or actor-value overrides via extra Hydra args.
#
# Usage:
#   bash examples/ppo_trainer/run_alfworld_lora_job.sh
#   TRAIN_BATCH_SIZE=64 bash examples/ppo_trainer/run_alfworld_lora_job.sh vllm
#
# Args (optional):
#   $1 ENGINE (default vllm)
#   $2 LOG_PROB_ACTION_ONLY (default false)
#   $3 TRAIN_BATCH_SIZE (default 64)
#   $4 MAX_RESPONSE_LENGTH (default 512)
#   $5 TOTAL_EPOCHS (default 150)
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
# AlfWorld uses TextWorld workers with fork; avoid inheriting Craftax texture reload setup.
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-False}"
# Game files for AlfredTWEnv (config_tw.yaml uses $ALFWORLD_DATA/...).
export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
# Host RAM ~126GB; avoid Ray OOM-killer during FSDP+vLLM+AlfWorld workers.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
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

python -c "import comet_ml" 2>/dev/null || pip install -q comet_ml

ENGINE=${1:-vllm}
LOG_PROB_ACTION_ONLY=${2:-false}
train_data_size=${3:-64}
max_response_length=${4:-512}
total_epochs=${5:-150}
USE_ACTOR_LORA=${6:-true}
CRITIC_WARMUP=${7:-0}
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-128}"
ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-128}"
shift 7 2>/dev/null || shift 6 2>/dev/null || shift 5 2>/dev/null || shift 4 2>/dev/null || shift 3 2>/dev/null || shift 2 2>/dev/null || shift 1 2>/dev/null || true

export VLLM_ATTENTION_BACKEND=XFORMERS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Vendored package lives at .../env_package/alfworld/alfworld; its imports are `import alfworld`.
ALFWORLD_PKG_ROOT="$PROJECT_ROOT/agent_system/environments/env_package/alfworld"
export PYTHONPATH="${PROJECT_ROOT}:${ALFWORLD_PKG_ROOT}:${PYTHONPATH}"

ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-$HOME/data/verl-agent/text}"
val_data_size=${VAL_DATA_SIZE:-64}
num_cpus_per_env_worker=0.1

export RUN_NAME="${RUN_NAME:-run_ppo_alfworld_qwen2.5_1.5b_$(date +%Y%m%d-%H%M%S)}"

rm -f "$COMET_EXPERIMENT_KEY_FILE"
python3 "$PROJECT_ROOT/scripts/init_comet_experiment.py" \
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

echo "[INFO] PROJECT_ROOT: $PROJECT_ROOT"
echo "[INFO] ALFWORLD_DATA: $ALFWORLD_DATA"
echo "[INFO] ALFWORLD_DATA_DIR: $ALFWORLD_DATA_DIR"
echo "[INFO] ENGINE: $ENGINE"
echo "[INFO] TRAIN_BATCH_SIZE: $train_data_size"
echo "[INFO] VAL_BATCH_SIZE: $val_data_size"
echo "[INFO] max_response_length: $max_response_length"
echo "[INFO] total_epochs: $total_epochs"
echo "[INFO] USE_ACTOR_LORA: $USE_ACTOR_LORA (rank=$ACTOR_LORA_RANK alpha=$ACTOR_LORA_ALPHA when true)"
echo "[INFO] CRITIC_WARMUP: $CRITIC_WARMUP"

if [ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ] || \
   ! find "$ALFWORLD_DATA/json_2.1.1/train" -name 'game.tw-pddl' -print -quit 2>/dev/null | grep -q .; then
  echo "[INFO] AlfWorld games missing under $ALFWORLD_DATA — downloading..."
  bash "$PROJECT_ROOT/scripts/download_alfworld_data.sh"
fi
n_games=$(find "$ALFWORLD_DATA/json_2.1.1/train" -name 'game.tw-pddl' 2>/dev/null | wc -l)
echo "[INFO] AlfWorld train games found: $n_games"
if [ "$n_games" -lt 1 ]; then
  echo "[ERROR] Still 0 AlfWorld games after download. Check network / ALFWORLD_DATA=$ALFWORLD_DATA"
  exit 1
fi

if [ ! -f "$ALFWORLD_DATA_DIR/train.parquet" ] || [ ! -f "$ALFWORLD_DATA_DIR/test.parquet" ]; then
  echo "[INFO] Preparing AlfWorld placeholder parquet (text modality) ..."
  python3 "$PROJECT_ROOT/examples/data_preprocess/prepare.py" \
    --mode text \
    --train_data_size "$train_data_size" \
    --val_data_size "$val_data_size"
else
  # Rebuild if placeholder size no longer matches requested batch sizes.
  cur_train=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$ALFWORLD_DATA_DIR/train.parquet')))" 2>/dev/null || echo 0)
  cur_val=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$ALFWORLD_DATA_DIR/test.parquet')))" 2>/dev/null || echo 0)
  if [ "$cur_train" -lt "$train_data_size" ] || [ "$cur_val" -lt "$val_data_size" ]; then
    echo "[INFO] Regenerating parquet: train=$cur_train->$train_data_size val=$cur_val->$val_data_size"
    python3 "$PROJECT_ROOT/examples/data_preprocess/prepare.py" \
      --mode text \
      --train_data_size "$train_data_size" \
      --val_data_size "$val_data_size"
  fi
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files="$ALFWORLD_DATA_DIR/train.parquet" \
  data.val_files="$ALFWORLD_DATA_DIR/test.parquet" \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=2048 \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=False \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.model.lora_rank=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "$ACTOR_LORA_RANK"; else echo "0"; fi) \
  actor_rollout_ref.model.lora_alpha=$(if [ "$USE_ACTOR_LORA" = "true" ]; then echo "$ACTOR_LORA_ALPHA"; else echo "0"; fi) \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.single_token_actions=False \
  actor_rollout_ref.actor.actor_value_token=False \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=True \
  critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  critic.model.lora_rank=0 \
  critic.model.lora_alpha=16 \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_mini_batch_size=256 \
  critic.ppo_micro_batch_size_per_gpu=16 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.use_kl_in_reward=False \
  +algorithm.log_prob_action_only=$LOG_PROB_ACTION_ONLY \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.history_length=2 \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.critic_warmup=$CRITIC_WARMUP \
  trainer.logger=['console','comet'] \
  trainer.project_name="$COMET_PROJECT_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.env_val_video_freq=0 \
  trainer.total_epochs=$total_epochs \
  trainer.val_before_train=True \
  "$@"
