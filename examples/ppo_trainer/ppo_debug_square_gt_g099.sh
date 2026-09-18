#!/usr/bin/env bash
# PPO debug_square_8x8 dual: discounted remaining return G_t^0.99
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS="${N_GPUS:-2}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_YAML="${HYPER_YAML:-$SCRIPT_DIR/config/ppo_debug_square_8x8_dual_gt_g099.yaml}"
if [ ! -f "$HYPER_YAML" ]; then
  echo "[ERROR] Hyperparam yaml not found: $HYPER_YAML"
  exit 1
fi
exec env HYPER_YAML="$HYPER_YAML" VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  bash "$SCRIPT_DIR/ppo_debug_square.sh" \
  actor_rollout_ref.rollout.enforce_eager=True \
  "$@"
