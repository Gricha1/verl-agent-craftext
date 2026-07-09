#!/bin/bash
# PPO AlfWorld: actor-value + online plan-Q WM + response-token entropy.
#
# Usage:
#   bash examples/ppo_trainer/ppo_alfworld_actor_value_plan_q_valid_ent.sh
#   ALFWORLD_PLAN_Q_HORIZON=4 bash examples/ppo_trainer/ppo_alfworld_actor_value_plan_q_valid_ent.sh

set -e

export ACTOR_VALUE_PLAN_Q_WM=true
export ALFWORLD_PLAN_Q_HORIZON="${ALFWORLD_PLAN_Q_HORIZON:-6}"
export ALFWORLD_PLAN_Q_GAMMA="${ALFWORLD_PLAN_Q_GAMMA:-1.0}"
export ALFWORLD_PLAN_Q_MAX_STEPS="${ALFWORLD_PLAN_Q_MAX_STEPS:-50}"
export ALFWORLD_PLAN_Q_MAX_PROMPT_LENGTH="${ALFWORLD_PLAN_Q_MAX_PROMPT_LENGTH:-4096}"
export ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-0.1}"
export RUN_NAME="${RUN_NAME:-PPO AlfWorld actor-value + plan-Q H${ALFWORLD_PLAN_Q_HORIZON} + entropy}"

exec bash examples/ppo_trainer/ppo_alfworld_actor_value_valid_ent.sh
