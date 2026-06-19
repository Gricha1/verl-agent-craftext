#!/bin/bash
# PPO GSM8K actor-value + online reward WM on PPO rollout experience.
#
# After each step: actor PPO -> value CE -> reward WM SFT (question + solution -> j/k).
#
# Usage:
#   bash examples/ppo_trainer/ppo_gsm8k_actor_value_reward_wm.sh
#   TRAIN_BATCH_SIZE=32 bash examples/ppo_trainer/ppo_gsm8k_actor_value_reward_wm.sh

set -e

export ACTOR_VALUE_ONLINE_REWARD_WM=true
export ACTOR_VALUE_REWARD_WM_LOSS_COEF="${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}"
export ACTOR_VALUE_REWARD_WM_PROMPT_STYLE="${ACTOR_VALUE_REWARD_WM_PROMPT_STYLE:-gsm8k}"
export ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH="${ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH:-2048}"
export RUN_NAME="${RUN_NAME:-PPO GSM8K actor-value + online reward WM}"

exec bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
