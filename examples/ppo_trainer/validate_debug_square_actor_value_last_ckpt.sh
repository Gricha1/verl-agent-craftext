#!/bin/bash
# Validate latest actor-value PPO checkpoint on debug_square_8x8 (all 3 tasks).
#
# Produces per task (stone / wood / water):
#   gif/val_actor_value_{task}_actor.gif      — actor prompt rollout
#   gif/val_actor_value_{task}_critic.gif     — critic prompt + V bar on the right
#   gif/val_actor_value_{task}_action_hist.png
#
# Default checkpoint root: ppo_debug_square_actor_value.sh
#
# Usage:
#   bash examples/ppo_trainer/validate_debug_square_actor_value_last_ckpt.sh
#   CKPT_ROOT=/path/to/ckpt_root bash examples/ppo_trainer/validate_debug_square_actor_value_last_ckpt.sh
#   ALLOW_BUSY_GPU=1 bash ...   # skip GPU preflight

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT_ROOT="${CKPT_ROOT:-training_checkpoints/verl_agent_caged_craftext_debug_square_actor_value}"
if [ -n "${1:-}" ] && [ -d "$1" ]; then
  CKPT_ROOT="$1"
  shift
fi
# Extra Hydra overrides from the command line (must not use "$@" inside _run_val).
EXTRA_HYDRA_ARGS=("$@")

if [ ! -d "$CKPT_ROOT" ]; then
  echo "[ERROR] CKPT_ROOT not found: $CKPT_ROOT" >&2
  exit 1
fi

LATEST_FILE="$CKPT_ROOT/latest_checkpointed_iteration.txt"
if [ ! -f "$LATEST_FILE" ]; then
  echo "[ERROR] Missing $LATEST_FILE (no checkpoints yet?)" >&2
  exit 1
fi

STEP="$(cat "$LATEST_FILE" | tr -d ' \n\r\t')"
if [ -z "$STEP" ]; then
  echo "[ERROR] Empty latest_checkpointed_iteration.txt" >&2
  exit 1
fi

CKPT_PATH="$CKPT_ROOT/global_step_${STEP}"
if [ ! -d "$CKPT_PATH" ]; then
  echo "[ERROR] Checkpoint folder not found: $CKPT_PATH" >&2
  exit 1
fi
if [ ! -d "$CKPT_PATH/actor" ]; then
  echo "[ERROR] Missing actor weights: $CKPT_PATH/actor" >&2
  exit 1
fi

if [ "${ALLOW_BUSY_GPU:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t _gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
  if [ "${#_gpu_pids[@]}" -gt 0 ]; then
    echo "[ERROR] GPU already in use — stop training first." >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv 2>/dev/null || nvidia-smi
    exit 1
  fi
fi

mkdir -p gif

echo "=========================================="
echo "Validate debug_square actor-value (3 tasks)"
echo "=========================================="
echo "[INFO] CKPT_ROOT=$CKPT_ROOT"
echo "[INFO] latest_step=$STEP"
echo "[INFO] resume_from=$CKPT_PATH"
echo "[INFO] tasks: stone (0), wood (1), water (2)"
echo "[INFO] critic GIF: value bar on the right of the game frame"

# stone=0, wood=1, water=2
TASK_SPECS=(
  "0:stone"
  "1:wood"
  "2:water"
)

_run_val() {
  local scenario_idx="$1"
  local slug="$2"

  echo ""
  echo "---------- task=$slug (scenario_idx=$scenario_idx) ----------"

  export RUN_NAME="${RUN_NAME:-val_actor_value_${slug}_step${STEP}}"
  # One val parquet row per task (job default is 16 → 16× full rollouts).
  export VAL_DATA_SIZE=1

  bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
    vllm \
    false \
    true \
    8 \
    1 \
    false \
    false \
    1 \
    true \
    single_token_action \
    0 \
    ascii \
    ++env.craftext_settings='debug_square_8x8' \
    +env.use_optimistic_parallel=False \
    +env.use_ray_text_render_workers=False \
    +env.fixed_scenario_idx="$scenario_idx" \
    ++env.use_jax_gpu=False \
    +env.value_prompt_template_type=single_token_return \
    algorithm.use_actor_value_token=True \
    actor_rollout_ref.actor.actor_value_token=True \
    data.val_batch_size=1 \
    env.max_steps=15 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.78 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.env_val_video_freq=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT_PATH" \
    trainer.default_local_dir="$CKPT_ROOT" \
    "${EXTRA_HYDRA_ARGS[@]}"

  local actor_gif="gif/val_trajectory_step0.gif"
  local critic_gif="gif/val_critic_trajectory_step0.gif"
  local hist_png="gif/val_action_hist_step0.png"
  local out_actor="gif/val_actor_value_${slug}.gif"
  local out_critic="gif/val_actor_value_${slug}_critic.gif"
  local out_hist="gif/val_actor_value_${slug}_action_hist.png"

  if [ -f "$actor_gif" ]; then
    mv -f "$actor_gif" "$out_actor"
    echo "[INFO] actor GIF -> $out_actor"
  else
    echo "[WARN] missing $actor_gif" >&2
  fi
  if [ -f "$critic_gif" ]; then
    mv -f "$critic_gif" "$out_critic"
    echo "[INFO] critic GIF (with V bar) -> $out_critic"
  else
    echo "[WARN] missing $critic_gif" >&2
  fi
  if [ -f "$hist_png" ]; then
    mv -f "$hist_png" "$out_hist"
    echo "[INFO] action hist -> $out_hist"
  else
    echo "[WARN] missing $hist_png" >&2
  fi
}

for spec in "${TASK_SPECS[@]}"; do
  idx="${spec%%:*}"
  slug="${spec##*:}"
  _run_val "$idx" "$slug"
done

echo ""
echo "[DONE] Validation GIFs in ./gif/:"
echo "  gif/val_actor_value_stone.gif  gif/val_actor_value_stone_critic.gif"
echo "  gif/val_actor_value_wood.gif   gif/val_actor_value_wood_critic.gif"
echo "  gif/val_actor_value_water.gif  gif/val_actor_value_water_critic.gif"
