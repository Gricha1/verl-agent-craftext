#!/bin/bash
# CPU smoke test: AlfWorld PPO configs + integration (no GPU training).
#
# Usage:
#   bash examples/ppo_trainer/debug_alfworld_smoke_cpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT_ROOT"
python3 scripts/smoke_test_alfworld_ppo_cpu.py
