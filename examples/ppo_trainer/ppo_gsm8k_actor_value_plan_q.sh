#!/bin/bash
# PPO GSM8K: actor + value + online plan-Q (H=100 only, fixed horizon).
#
# One LLM backbone:
#   actor PPO -> value CE -> plan-Q SFT on first 100 reasoning lines -> return bin target
#
# Usage:
#   bash examples/ppo_trainer/ppo_gsm8k_actor_value_plan_q.sh
#   GSM8K_PLAN_Q_HORIZON=100 bash examples/ppo_trainer/ppo_gsm8k_actor_value_plan_q.sh

set -e

export ACTOR_VALUE_PLAN_Q_WM=true
export USE_ACTOR_LORA=true
export ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-128}"
export ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-128}"
export GSM8K_PLAN_Q_HORIZON="${GSM8K_PLAN_Q_HORIZON:-100}"
export GSM8K_PLAN_Q_GAMMA="${GSM8K_PLAN_Q_GAMMA:-1.0}"
export GSM8K_PLAN_Q_STEP_SPLIT="${GSM8K_PLAN_Q_STEP_SPLIT:-newline}"
export GSM8K_PLAN_Q_MAX_PROMPT_LENGTH="${GSM8K_PLAN_Q_MAX_PROMPT_LENGTH:-32768}"
export ACTOR_VALUE_PLAN_Q_LOSS_COEF="${ACTOR_VALUE_PLAN_Q_LOSS_COEF:-0.1}"
export RUN_NAME="${RUN_NAME:-PPO GSM8K actor-value + plan-Q H${GSM8K_PLAN_Q_HORIZON}}"

exec bash examples/ppo_trainer/ppo_gsm8k_actor_value.sh
