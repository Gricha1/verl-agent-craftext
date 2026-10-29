#!/usr/bin/env bash
# One-map offline outcome-supervised reasoning PPO.  Qwen never steps CrafText.
set -euo pipefail

: "${MODEL_PATH:?local Qwen model path is required}"
: "${CRAFTEXT_LEGACY_ROOT:?legacy teacher/environment checkout is required}"
: "${TEACHER_CHECKPOINT:?saved CNN+BERT teacher checkpoint is required}"
: "${RUN_ROOT:?new run directory is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
MAP_SEEDS="${MAP_SEEDS:-20261016,20261017}"
TASK_INDICES="${TASK_INDICES:-0}"
CRAFTEXT_SETTINGS="${CRAFTEXT_SETTINGS:-debug_square_16x16}"
MAX_STEPS="${MAX_STEPS:-50}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-320}"
ACTOR_PPO_MICRO_BATCH_SIZE="${ACTOR_PPO_MICRO_BATCH_SIZE:-16}"
CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_reasoning_teacher_action_offline_16x16_onemap_gpu0_r1}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console,comet,file,tensorboard]}"

mkdir -p "${RUN_ROOT}"/{dataset,checkpoints,logs,ray,raw_examples}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_PATH CRAFTEXT_LEGACY_ROOT TEACHER_CHECKPOINT RUN_ROOT GPU_ID MAP_SEEDS TASK_INDICES CRAFTEXT_SETTINGS MAX_STEPS
export TOTAL_EPOCHS TRAIN_BATCH_SIZE VAL_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH ACTOR_PPO_MICRO_BATCH_SIZE CRITIC_PPO_MICRO_BATCH_SIZE EXPERIMENT_NAME LOGGER_BACKENDS
export PYTHONPATH="${REPO_ROOT}:${CRAFTEXT_LEGACY_ROOT}:${CRAFTEXT_LEGACY_ROOT}/caged_craftext:${CRAFTEXT_LEGACY_ROOT}/caged_craftext/Craftax:${CRAFTEXT_LEGACY_ROOT}/caged_craftext/baselines:${PYTHONPATH:-}"
# AF_UNIX socket paths are capped at 107 bytes.  The descriptive run directory
# is deliberately long, so keep this isolated Ray session under a short root.
RAY_SESSION_DIR="${RAY_SESSION_DIR:-/tmp/verl_offline_${GPU_ID}}"
mkdir -p "${RAY_SESSION_DIR}"
export RAY_TMPDIR="${RAY_SESSION_DIR}"
export RAY_TEMP_DIR="${RAY_SESSION_DIR}"
export COMET_DISABLE_AUTO_LOGGING=1
export WANDB_MODE=disabled
export VERL_FILE_LOGGER_PATH="${RUN_ROOT}/logs/metrics.jsonl"
if [[ "${LOGGER_BACKENDS}" == *comet* ]]; then
  unset COMET_DISABLE_AUTO_LOGGING
  # The existing aicenter Comet setup keeps its credential in ~/.comet.config.
  # Do not require an exported secret here: comet_ml discovers that config itself.
fi

# Collection is intentionally CPU-only and complete before Qwen training.
JAX_PLATFORMS=cpu "${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/collect_offline_teacher_action_dataset.py" \
  --legacy-root "${CRAFTEXT_LEGACY_ROOT}" \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/dataset" \
  --craftext-settings "${CRAFTEXT_SETTINGS}" \
  --seeds "${MAP_SEEDS}" --task-indices "${TASK_INDICES}" --max-steps "${MAX_STEPS}"

"${PYTHON_BIN}" - <<'PY' "${RUN_ROOT}/resolved_config.json"
import json, os, sys
path = sys.argv[1]
json.dump({k: os.environ[k] for k in sorted(os.environ) if k in {
    'MODEL_PATH','CRAFTEXT_LEGACY_ROOT','TEACHER_CHECKPOINT','RUN_ROOT','MAP_SEEDS','TASK_INDICES','CRAFTEXT_SETTINGS',
    'MAX_STEPS','TOTAL_EPOCHS','TRAIN_BATCH_SIZE','VAL_BATCH_SIZE','MAX_PROMPT_LENGTH',
    'MAX_RESPONSE_LENGTH','EXPERIMENT_NAME','LOGGER_BACKENDS','GPU_ID'}}, open(path, 'w'), indent=2)
PY

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  algorithm.adv_estimator=gae algorithm.use_kl_in_reward=false \
  data.train_files="${RUN_ROOT}/dataset/train.parquet" data.val_files="${RUN_ROOT}/dataset/val.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=false data.filter_overlong_prompts=true data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" critic.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa +actor_rollout_ref.model.override_config._attn_implementation=sdpa +critic.model.override_config.attn_implementation=sdpa +critic.model.override_config._attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=64 actor_rollout_ref.model.lora_alpha=64 actor_rollout_ref.model.lora.merge=true actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE}" actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.calculate_entropy=true actor_rollout_ref.actor.entropy_coeff=0.01 actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false actor_rollout_ref.actor.checkpoint.save_contents=[model,extra] actor_rollout_ref.actor.checkpoint.load_contents=[model,extra] \
  critic.optim.lr=1e-5 critic.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" critic.ppo_micro_batch_size_per_gpu="${CRITIC_PPO_MICRO_BATCH_SIZE}" critic.fsdp.param_offload=false critic.fsdp.optimizer_offload=false critic.model.lora_rank=64 critic.model.lora_alpha=64 critic.model.use_remove_padding=false critic.fsdp.use_torch_compile=false critic.forward_micro_batch_size_per_gpu="${CRITIC_PPO_MICRO_BATCH_SIZE}" critic.checkpoint.save_contents=[model,extra] critic.checkpoint.load_contents=[model,extra] \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.data_parallel_size=1 actor_rollout_ref.rollout.load_format=dummy actor_rollout_ref.rollout.multi_turn.format=chatml actor_rollout_ref.rollout.val_kwargs.do_sample=true actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.gpu_memory_utilization=0.14 actor_rollout_ref.rollout.max_model_len=null actor_rollout_ref.rollout.max_num_seqs=1024 actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.free_cache_engine=false actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.enable_chunked_prefill=true actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.use_torch_compile=false \
  reward.custom_reward_function.path="${REPO_ROOT}/craftext_integration/offline_teacher_action_reward.py" reward.custom_reward_function.name=compute_score \
  trainer.use_v1=true trainer.v1.trainer_mode=sync trainer.total_epochs="${TOTAL_EPOCHS}" trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.val_before_train=true trainer.test_freq=20 trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=2 trainer.max_critic_ckpt_to_keep=2 trainer.resume_mode=disable "trainer.logger=${LOGGER_BACKENDS}" trainer.project_name=verl-agent-caged-craftext trainer.experiment_name="${EXPERIMENT_NAME}" trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
  +ray_kwargs.ray_init._temp_dir="${RAY_SESSION_DIR}"
