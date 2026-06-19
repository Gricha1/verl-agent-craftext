#!/bin/bash
# PPO GSM8K actor-value + online MC-Q WM (prefix + step -> eventual success j/k).
#
# Usage:
#   bash examples/ppo_trainer/ppo_gsm8k_actor_value_mc_q.sh

set -e

export ACTOR_VALUE_MC_Q_WM=true
export ACTOR_VALUE_REWARD_WM_LOSS_COEF="${ACTOR_VALUE_REWARD_WM_LOSS_COEF:-0.1}"
export ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH="${ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH:-2048}"
export RUN_NAME="${RUN_NAME:-PPO GSM8K actor-value + MC-Q WM}"

exec bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
