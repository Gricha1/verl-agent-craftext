#!/bin/bash
# 3-stage pipeline: PPO on debug_square (init from WM SFT) -> WM refresh on same PPO rollouts -> planner refresh.
#
# PPO does NOT use a static parquet — rollouts are online each step.
#   • Reward WM: auxiliary loss on every PPO rollout batch (trainer.world_model.task=reward)
#   • Planner WM: after PPO, parquet built from the same rollout buffer (transitions.jsonl)
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_llm_pretrain.sh
#   SKIP_PPO=1 bash examples/ppo_trainer/ppo_debug_square_llm_pretrain.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash examples/ppo_trainer/ppo_debug_square_llm_pretrain.sh
#
# Optional fallback (NOT the default): re-run env with PPO policy to collect extra data:
#   USE_PPO_REPLAY_COLLECT=1 COLLECT_EPISODES_PER_TASK=100 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "verl-agent-311" ]; then
  if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate verl-agent-311
  fi
fi

WM_SFT_CKPT="${WM_SFT_CKPT:-/usr/home/workspace/training_checkpoints/reward_wm_sft_h6_20260606-085007_1/latest}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
REWARD_HORIZON="${REWARD_HORIZON:-6}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_OPTIMISTIC_ENVS="${NUM_OPTIMISTIC_ENVS:-64}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-8}"
WM_LOSS_COEF="${WM_LOSS_COEF:-0.1}"

DEV_TAG_RAW="${CUDA_VISIBLE_DEVICES:-gpu0}"
DEV_TAG="${DEV_TAG_RAW//,/__}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_${DEV_TAG}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/training_checkpoints/ppo_debug_square_llm_pretrain_${RUN_TAG}}"

PPO_FSdp_INIT="${PPO_FSdp_INIT:-$RUN_ROOT/00_ppo_init_from_wm_sft}"
PPO_RESUME_FROM="${PPO_RESUME_FROM:-$PPO_FSdp_INIT/global_step_0}"
PPO_CKPT_DIR="${PPO_CKPT_DIR:-$RUN_ROOT/01_ppo}"
PPO_LORA_OUT="${PPO_LORA_OUT:-$RUN_ROOT/01_ppo_lora_hf}"
WM_ROLLOUT_BUFFER_DIR="${WM_ROLLOUT_BUFFER_DIR:-$RUN_ROOT/ppo_rollout_buffer}"
PPO_ROLLOUT_DATA_DIR="${PPO_ROLLOUT_DATA_DIR:-$RUN_ROOT/ppo_rollout_parquet_h${REWARD_HORIZON}}"
PLANNER_REFRESH_DIR="${PLANNER_REFRESH_DIR:-$RUN_ROOT/02_planner_refresh}"

REFRESH_PLANNER_STEPS="${REFRESH_PLANNER_STEPS:-400}"
USE_PPO_REPLAY_COLLECT="${USE_PPO_REPLAY_COLLECT:-0}"
COLLECT_EPISODES_PER_TASK="${COLLECT_EPISODES_PER_TASK:-200}"

SKIP_PPO="${SKIP_PPO:-0}"
SKIP_PLANNER_REFRESH="${SKIP_PLANNER_REFRESH:-0}"

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CAGED_CRAFTEXT_PATH="${CAGED_CRAFTEXT_PATH:-$PROJECT_ROOT/caged_craftext}"
export PYTHONPATH="${CAGED_CRAFTEXT_PATH}:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT"

resolve_ppo_latest() {
  local dir="$1"
  local latest_file="$dir/latest_checkpointed_iteration.txt"
  local step
  step="$(tr -d ' \n\r\t' < "$latest_file")"
  local path="$dir/global_step_${step}"
  echo "$path"
}

extract_ppo_lora_hf() {
  local fsdp_ckpt="$1"
  local out_dir="$2"
  if [ -f "$out_dir/adapter_model.safetensors" ] || [ -f "$out_dir/adapter_model.bin" ]; then
    echo "[OK] HF LoRA already extracted: $out_dir"
    return 0
  fi
  mkdir -p "$out_dir"
  if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_CMD="torchrun"
  else
    TORCHRUN_CMD="python3 -m torch.distributed.run"
  fi
  $TORCHRUN_CMD --standalone --nnodes=1 --nproc_per_node=1 \
    scripts/extract_lora_from_fsdp.py \
    --checkpoint_path "$fsdp_ckpt" \
    --output_path "$out_dir" \
    --base_model "$BASE_MODEL" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --trust_remote_code
}

echo "=========================================="
echo "PPO debug_square + WM refresh pipeline"
echo "=========================================="
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] WM_SFT_CKPT=$WM_SFT_CKPT"
echo "[INFO] Reward WM: online during PPO (world_model.task=reward, coef=$WM_LOSS_COEF)"
echo "[INFO] Planner WM: SFT on parquet from PPO rollout buffer after training"
echo "[INFO] Rollout buffer: $WM_ROLLOUT_BUFFER_DIR/transitions.jsonl"

# ---------------------------------------------------------------------------
# Stage 1: PPO (+ online reward WM + rollout buffer)
# ---------------------------------------------------------------------------
if [ "$SKIP_PPO" != "1" ]; then
  if [ ! -d "$WM_SFT_CKPT" ]; then
    echo "[ERROR] WM SFT checkpoint not found: $WM_SFT_CKPT" >&2
    exit 1
  fi

  rm -rf "$WM_ROLLOUT_BUFFER_DIR"
  mkdir -p "$WM_ROLLOUT_BUFFER_DIR"

  echo ""
  echo "=== Stage 1a: HF LoRA -> FSDP for PPO resume ==="
  bash examples/ppo_trainer/prepare_hf_lora_for_ppo_fsdp.sh \
    "$WM_SFT_CKPT" "$PPO_FSdp_INIT" "$BASE_MODEL" "$NUM_GPUS" \
    "$LORA_RANK" "$LORA_ALPHA" all-linear

  echo ""
  echo "=== Stage 1b: PPO (same rollouts -> policy + reward WM + buffer) ==="
  export RUN_NAME="${RUN_NAME:-PPO debug_square llm_pretrain ${RUN_TAG}}"

  bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
    vllm false true "$NUM_OPTIMISTIC_ENVS" 1 false false 8000 true \
    single_token_action 10 ascii \
    ++env.craftext_settings='debug_square_8x8' \
    +env.use_optimistic_parallel=True \
    +env.optimistic_reset_ratio="$OPTIMISTIC_RESET_RATIO" \
    +env.use_ray_text_render_workers=False \
    ++env.use_jax_gpu=False \
    ++env.jax_gpu_fraction=0.15 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.entropy_over_valid_actions=False \
    trainer.world_model.enable=True \
    trainer.world_model.task=reward \
    trainer.world_model.use_latent_in_policy=False \
    trainer.world_model.loss_coef="$WM_LOSS_COEF" \
    trainer.world_model.filter_unchanged_transitions=False \
    +trainer.wm_rollout_buffer_dir="$WM_ROLLOUT_BUFFER_DIR" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$PPO_RESUME_FROM" \
    trainer.env_val_video_freq=100000 \
    trainer.default_local_dir="$PPO_CKPT_DIR" \
    trainer.n_gpus_per_node="$NUM_GPUS"

  PPO_LATEST="$(resolve_ppo_latest "$PPO_CKPT_DIR")"
  echo "[OK] PPO finished. latest=$PPO_LATEST"
  extract_ppo_lora_hf "$PPO_LATEST" "$PPO_LORA_OUT"
  SFT_INIT_PATH="$PPO_LORA_OUT"
else
  echo "[SKIP] Stage 1 PPO"
  if [ -d "$PPO_LORA_OUT" ]; then
    SFT_INIT_PATH="$PPO_LORA_OUT"
  elif [ -d "$WM_SFT_CKPT" ]; then
    SFT_INIT_PATH="$WM_SFT_CKPT"
  else
    echo "[ERROR] SKIP_PPO=1 but no adapter found" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Stage 2: Parquet from PPO rollout buffer (same trajectories PPO trained on)
# ---------------------------------------------------------------------------
if [ "$SKIP_PLANNER_REFRESH" != "1" ]; then
  if [ ! -f "$PPO_ROLLOUT_DATA_DIR/train.parquet" ]; then
    if [ "$USE_PPO_REPLAY_COLLECT" = "1" ]; then
      echo ""
      echo "=== Stage 2 (fallback): re-collect from PPO policy in env ==="
      CKPT_PATH="$SFT_INIT_PATH" OUT_DIR="$PPO_ROLLOUT_DATA_DIR" \
        REWARD_HORIZON="$REWARD_HORIZON" EPISODES_PER_TASK="$COLLECT_EPISODES_PER_TASK" \
        bash examples/world_model/collect_reward_wm_dataset_from_ppo_policy.sh
    else
      echo ""
      echo "=== Stage 2: build WM parquet from PPO rollout buffer ==="
      python3 scripts/build_wm_parquet_from_rollout_buffer.py \
        --buffer-dir "$WM_ROLLOUT_BUFFER_DIR" \
        --out-dir "$PPO_ROLLOUT_DATA_DIR" \
        --reward-horizon "$REWARD_HORIZON"
    fi
  else
    echo "[OK] Reusing parquet: $PPO_ROLLOUT_DATA_DIR"
  fi
  DATA_DIR="$PPO_ROLLOUT_DATA_DIR"
fi

# ---------------------------------------------------------------------------
# Stage 3: Planner refresh (advantage planner; reward CE keeps reward head alive)
# ---------------------------------------------------------------------------
if [ "$SKIP_PLANNER_REFRESH" != "1" ]; then
  echo ""
  echo "=== Stage 3: Planner SFT refresh (steps=$REFRESH_PLANNER_STEPS) ==="
  export RUN_NAME="${RUN_NAME:-planner_refresh_${RUN_TAG}}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" NUM_GPUS=1 \
    DATA_DIR="$DATA_DIR" OUT_DIR="$PLANNER_REFRESH_DIR" \
    REWARD_HORIZON="$REWARD_HORIZON" TOTAL_TRAINING_STEPS="$REFRESH_PLANNER_STEPS" \
    PLANNING_ADVANTAGE_WM=true PLANNING_WM=false INVERSE_ACTION_WM=false \
    LORA_INIT_PATH="$SFT_INIT_PATH" \
    bash examples/world_model/train_reward_wm_sft.sh
fi

echo ""
echo "=========================================="
echo "[OK] Pipeline complete"
echo "  PPO checkpoints:   $PPO_CKPT_DIR"
echo "  Rollout buffer:    $WM_ROLLOUT_BUFFER_DIR/transitions.jsonl"
echo "  WM parquet:        $PPO_ROLLOUT_DATA_DIR"
echo "  Planner refresh:   $PLANNER_REFRESH_DIR/latest"
echo "Validate:"
echo "  CKPT_PATH=$PLANNER_REFRESH_DIR/latest bash examples/world_model/validate_debug_square_llm_planning.sh"
echo "=========================================="
