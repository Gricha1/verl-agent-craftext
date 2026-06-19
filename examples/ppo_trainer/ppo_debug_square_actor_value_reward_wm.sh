#!/bin/bash
# PPO debug_square actor-value + online reward WM (same target as train_reward_wm_sft.sh).
#
# After each PPO step: actor update -> value CE -> reward WM SFT on rollout (s_t, a_t) -> i/j/k/l.
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_actor_value_reward_wm.sh
#   ACTOR_VALUE_REWARD_WM_LOSS_COEF=0.2 bash examples/ppo_trainer/ppo_debug_square_actor_value_reward_wm.sh

set -e

export ACTOR_VALUE_ONLINE_REWARD_WM=true
export ACTOR_VALUE_REWARD_WM_LOSS_COEF="${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}"
export RUN_NAME="${RUN_NAME:-PPO Debug Square actor-value + online reward WM}"

exec bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
