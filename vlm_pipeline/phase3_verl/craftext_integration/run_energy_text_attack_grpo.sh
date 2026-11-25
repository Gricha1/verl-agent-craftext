#!/usr/bin/env bash
set -euo pipefail

: "${ENERGY_PPO_LAG_CHECKPOINT:?Set the Orbax step directory of the trained energy PPO-Lag}"
: "${RUN_ROOT:?Set a fresh output directory}"
: "${N_GPUS:?Set the number of GPUs visible to veRL}"
RULE=${RULE:-energy}
MODEL_PATH=${MODEL_PATH:-HuggingFaceTB/SmolLM2-135M-Instruct}
PYTHON_BIN=${PYTHON_BIN:-python}
CKPT=$(realpath "$ENERGY_PPO_LAG_CHECKPOINT")
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REWARD="$ROOT/craftext_integration/energy_text_attack.py"

test -d "$CKPT"
mkdir -p "$RUN_ROOT"
RUN_ROOT=$(realpath "$RUN_ROOT")
if [[ -e "$RUN_ROOT/checkpoints" ]]; then
  echo "RUN_ROOT already has checkpoints. Choose a fresh directory for this pilot." >&2
  exit 2
fi
export JAX_PLATFORMS=cpu
unset CRAFTAX_RELOAD_TEXTURES
export PYTHONPATH="$ROOT/../../caged_craftext:$ROOT/../../caged_craftext/Craftax:$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
git rev-parse HEAD > "$RUN_ROOT/code_version.txt"
"$PYTHON_BIN" "$REWARD" evaluate --checkpoint "$CKPT" --rule "$RULE" > "$RUN_ROOT/preflight.jsonl"
"$PYTHON_BIN" "$REWARD" dataset "$RUN_ROOT/dataset" --rule "$RULE"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  hydra.run.dir="$RUN_ROOT/hydra" \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  data.train_files="$RUN_ROOT/dataset/train.parquet" data.val_files="$RUN_ROOT/dataset/val.parquet" \
  data.train_batch_size=8 data.val_batch_size=8 data.max_prompt_length=192 data.max_response_length=32 \
  actor_rollout_ref.model.path="$MODEL_PATH" actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  +reward.custom_reward_function.reward_kwargs.checkpoint_path="$CKPT" \
  +reward.custom_reward_function.reward_kwargs.rule="$RULE" \
  reward.custom_reward_function.path="$REWARD" reward.custom_reward_function.name=compute_score \
  reward.num_workers=2 \
  trainer.use_v1=true trainer.v1.trainer_mode=sync trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 \
  trainer.total_epochs=2 trainer.val_before_train=true trainer.test_freq=8 trainer.save_freq=8 \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" trainer.resume_mode=disable \
  trainer.project_name=verl-agent-caged-craftext trainer.experiment_name="${RULE}_text_attack" \
  'trainer.logger=[console]' "$@"
