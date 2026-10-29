#!/usr/bin/env bash
# Supervised CE baseline over frozen CrafText CNN+BERT teacher trajectories.
set -euo pipefail

: "${MODEL_PATH:?local Qwen model path is required}"
: "${RUN_ROOT:?new run directory is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:?teacher trajectory dataset is required}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_offline_teacher_reasoning_ce_16x16_r1}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console,comet,tensorboard]}"

mkdir -p "${RUN_ROOT}"/{dataset_sft,checkpoints,logs}
export COMET_DISABLE_AUTO_LOGGING=1
export WANDB_MODE=disabled
export VERL_FILE_LOGGER_PATH="${RUN_ROOT}/logs/metrics.jsonl"
if [[ "${LOGGER_BACKENDS}" == *comet* ]]; then unset COMET_DISABLE_AUTO_LOGGING; fi

"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/prepare_offline_teacher_sft_dataset.py" \
  --input-dir "${SOURCE_DATASET_DIR}" --output-dir "${RUN_ROOT}/dataset_sft"

cd "${REPO_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.sft_trainer \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  data.train_files="${RUN_ROOT}/dataset_sft/train.parquet" \
  data.val_files="${RUN_ROOT}/dataset_sft/val.parquet" \
  data.messages_key=messages data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" data.max_token_len_per_gpu="${MAX_LENGTH}" \
  data.max_length="${MAX_LENGTH}" data.truncation=error data.num_workers=2 \
  model.path="${MODEL_PATH}" model.use_remove_padding=false model.enable_gradient_checkpointing=true \
  model.lora_rank=64 model.lora_alpha=64 model.target_modules=all-linear \
  +model.override_config.attn_implementation=sdpa +model.override_config._attn_implementation=sdpa \
  engine=fsdp engine.dtype=bfloat16 engine.use_torch_compile=false engine.param_offload=false engine.optimizer_offload=false \
  optim.lr=1e-4 \
  checkpoint.save_contents=[model,extra] checkpoint.load_contents=[model,extra] \
  trainer.total_epochs="${TOTAL_EPOCHS}" trainer.test_freq=after_each_epoch trainer.save_freq=20 trainer.max_ckpt_to_keep=2 \
  trainer.resume_mode=disable trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  "trainer.logger=${LOGGER_BACKENDS}" trainer.project_name=verl-agent-caged-craftext \
  trainer.experiment_name="${EXPERIMENT_NAME}" trainer.default_local_dir="${RUN_ROOT}/checkpoints"
