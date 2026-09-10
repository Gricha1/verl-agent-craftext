#!/usr/bin/env bash
# PPO debug_square_8x8 dual: discounted remaining return G_t^0.99
#
# Matches failed undiscounted G_t baseline (2a42e153) except remaining_return_gamma=0.99.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash examples/ppo_trainer/ppo_debug_square_gt_g099.sh

set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS="${N_GPUS:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_YAML="${HYPER_YAML:-$SCRIPT_DIR/config/ppo_debug_square_8x8_dual_gt_g099.yaml}"

if [ ! -f "$HYPER_YAML" ]; then
  echo "[ERROR] Hyperparam yaml not found: $HYPER_YAML"
  exit 1
fi

# Reuse dual launcher body by pointing HYPER_YAML.
exec env HYPER_YAML="$HYPER_YAML" bash "$SCRIPT_DIR/ppo_debug_square.sh" "$@"
