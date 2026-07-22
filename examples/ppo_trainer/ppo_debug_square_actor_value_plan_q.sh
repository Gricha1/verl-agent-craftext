#!/bin/bash
# PPO debug_square_8x8: single-LLM actor-value + valid entropy + online plan-Q.
# Resources shuffled each reset; GAE along env trajectories (via actor_value script).
#
# After each PPO step:
#   actor PPO -> value CE -> plan-Q SFT -> planner advantage (optional)
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_actor_value_plan_q.sh
#   CRAFTEXT_PLAN_Q_HORIZON=4 bash examples/ppo_trainer/ppo_debug_square_actor_value_plan_q.sh
#   ACTOR_VALUE_ONLINE_PLANNING_WM=false bash ...  # plan-Q only, no planner train

set -e

export ACTOR_VALUE_PLAN_Q_WM=true
export ACTOR_VALUE_ONLINE_PLANNING_WM="${ACTOR_VALUE_ONLINE_PLANNING_WM:-true}"
export ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-0.1}"
export ACTOR_VALUE_PLANNING_WM_LOSS_COEF="${ACTOR_VALUE_PLANNING_WM_LOSS_COEF:-0.1}"
export ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH="${ACTOR_VALUE_REWARD_WM_MAX_PROMPT_LENGTH:-2048}"
export CRAFTEXT_PLAN_Q_HORIZON="${CRAFTEXT_PLAN_Q_HORIZON:-6}"
export CRAFTEXT_PLANNING_WM_HORIZON="${CRAFTEXT_PLANNING_WM_HORIZON:-6}"
export CRAFTEXT_PLAN_Q_GAMMA="${CRAFTEXT_PLAN_Q_GAMMA:-1.0}"
export CRAFTEXT_MC_Q_MAX_STEPS="${CRAFTEXT_MC_Q_MAX_STEPS:-50}"
export RUN_NAME="${RUN_NAME:-PPO Debug Square 8x8 actor-value + valid-ent + plan-Q H${CRAFTEXT_PLAN_Q_HORIZON}}"

exec bash examples/ppo_trainer/ppo_debug_square_actor_value.sh
