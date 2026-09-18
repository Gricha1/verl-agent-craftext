#!/bin/bash
# GSM8K: 6 sequential training runs, each capped at 80 global PPO steps.
#
# Usage:
#   bash examples/ppo_trainer/run_gsm8k_sweep_6.sh
#   MAX_GLOBAL_STEPS=80 RAY_COOLDOWN_SECONDS=120 bash examples/ppo_trainer/run_gsm8k_sweep_6.sh
#   START_RUN=3 bash examples/ppo_trainer/run_gsm8k_sweep_6.sh   # resume from run 3

set -euo pipefail

MAX_GLOBAL_STEPS="${MAX_GLOBAL_STEPS:-80}"
RAY_COOLDOWN_SECONDS="${RAY_COOLDOWN_SECONDS:-60}"
START_RUN="${START_RUN:-1}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d-%H%M%S)}"
CKPT_ROOT="${CKPT_ROOT:-training_checkpoints/gsm8k_sweep_${SWEEP_TAG}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

COMMON_HYDRA=(
  "trainer.total_training_steps=${MAX_GLOBAL_STEPS}"
  "trainer.resume_mode=disable"
  "trainer.save_freq=-1"
  "trainer.test_freq=20"
)

run_step() {
  local run_id="$1"
  shift
  if (( run_id < START_RUN )); then
    echo "[SKIP] Run ${run_id} (START_RUN=${START_RUN})"
    return 0
  fi
  echo ""
  echo "=========================================="
  echo "GSM8K sweep run ${run_id}/6  (global_step cap=${MAX_GLOBAL_STEPS})"
  echo "=========================================="
  "$@"
  echo "[INFO] Run ${run_id} finished."
  if (( run_id < 6 )); then
    echo "[INFO] Cooldown ${RAY_COOLDOWN_SECONDS}s before next run ..."
    sleep "${RAY_COOLDOWN_SECONDS}"
  fi
}

# 1) actor-value (current) + online reward WM
run_step 1 env \
  RUN_NAME="GSM8K sweep1 av+reward_wm steps${MAX_GLOBAL_STEPS}" \
  ACTOR_VALUE_ONLINE_REWARD_WM=true \
  bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh \
  "${COMMON_HYDRA[@]}" \
  "trainer.default_local_dir=${CKPT_ROOT}/run1_av_reward_wm"

# 2) standard PPO: actor lr=1e-6 (default), critic lr=1e-5
run_step 2 env \
  RUN_NAME="GSM8K sweep2 PPO critic1e-5 steps${MAX_GLOBAL_STEPS}" \
  bash examples/ppo_trainer/ppo_gsm8k.sh \
  "${COMMON_HYDRA[@]}" \
  "actor_rollout_ref.actor.optim.lr=1e-6" \
  "critic.optim.lr=1e-5" \
  "trainer.default_local_dir=${CKPT_ROOT}/run2_ppo_critic1e-5"

# 3) actor-value: single optimizer lr=1e-5 (actor + value)
run_step 3 env \
  RUN_NAME="GSM8K sweep3 av lr1e-5 steps${MAX_GLOBAL_STEPS}" \
  bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh \
  "${COMMON_HYDRA[@]}" \
  "actor_rollout_ref.actor.optim.lr=1e-5" \
  "trainer.default_local_dir=${CKPT_ROOT}/run3_av_lr1e-5"

# 4) actor-value: lr=5e-7
run_step 4 env \
  RUN_NAME="GSM8K sweep4 av lr5e-7 steps${MAX_GLOBAL_STEPS}" \
  bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh \
  "${COMMON_HYDRA[@]}" \
  "actor_rollout_ref.actor.optim.lr=5e-7" \
  "trainer.default_local_dir=${CKPT_ROOT}/run4_av_lr5e-7"

# 5) standard PPO: actor lr=1e-6, critic lr=1e-6
run_step 5 env \
  RUN_NAME="GSM8K sweep5 PPO lr1e-6 steps${MAX_GLOBAL_STEPS}" \
  bash examples/ppo_trainer/ppo_gsm8k.sh \
  "${COMMON_HYDRA[@]}" \
  "actor_rollout_ref.actor.optim.lr=1e-6" \
  "critic.optim.lr=1e-6" \
  "trainer.default_local_dir=${CKPT_ROOT}/run5_ppo_lr1e-6"

# 6) standard PPO: actor lr=5e-7, critic lr=5e-7
run_step 6 env \
  RUN_NAME="GSM8K sweep6 PPO lr5e-7 steps${MAX_GLOBAL_STEPS}" \
  bash examples/ppo_trainer/ppo_gsm8k.sh \
  "${COMMON_HYDRA[@]}" \
  "actor_rollout_ref.actor.optim.lr=5e-7" \
  "critic.optim.lr=5e-7" \
  "trainer.default_local_dir=${CKPT_ROOT}/run6_ppo_lr5e-7"

echo ""
echo "[DONE] All sweep runs completed. Checkpoints under: ${CKPT_ROOT}"
