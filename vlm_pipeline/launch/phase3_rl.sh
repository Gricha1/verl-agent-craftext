#!/usr/bin/env bash
# Phase 3: RL fine-tuning of B_ce in veRL (veRL Agents, PPO), starting from a materialised phase-2 checkpoint.
#
#   GPU_ID=3 SETTINGS=achievements_wood MODEL_PATH=$RUNS/<phase2 run>/materialized_epoch_48 \
#     RUN_NAME=slm5m_vision_wood_verl_rl_gpu3_YYYYMMDD_r1 bash vlm_pipeline/launch/phase3_rl.sh
#
# Uses vlm_pipeline/phase3_verl: veRL v0.9.1 + our patches + craftext_integration (it is put first on
# PYTHONPATH, so the image's veRL is not used). The CrafText environment comes from this checkout
# (agent_system + caged_craftext). Any run_ppo_8x8_smoke_docker.sh variable can be overridden.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${SETTINGS:?}" "${MODEL_PATH:?materialised phase-2 checkpoint}" "${RUN_NAME:?}"
[ -f "${MODEL_PATH}/config.json" ] || { echo "missing model ${MODEL_PATH}" >&2; exit 1; }
RUN_ROOT="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_ROOT}"
record_code_version "${RUN_ROOT}"
export RUN_ROOT MODEL_PATH GPU_ID CONTAINER_NAME="${RUN_NAME}" EXPERIMENT_NAME="${RUN_NAME}"
export MODERN_VERL_IMAGE="${VERL_IMAGE}"
export CRAFTEXT_LEGACY_ROOT="${CHECKOUT}"
export CRAF_TEXT_ENV_CONFIG="${SETTINGS}"
# Must match the phase-2 prompt protocol: 0 for vision_image_only_v1, 8 for vision_image_actions8_v1.
export CRAF_TEXT_VISUAL_PROMPT_HISTORY="${CRAF_TEXT_VISUAL_PROMPT_HISTORY:-0}"
# REASONING=1: checkpoint trained on vision_image_reasoning_v1; the model writes <reasoning>...</reasoning>
# then ACTION:, and the whole answer (up to 70 tokens in the CE data) is one PPO response.
if [ "${REASONING:-0}" = 1 ]; then
  export CRAF_TEXT_REASONING_PROMPT=true PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-128}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
fi
if [ "${SETTINGS}" = achievements_all ]; then export MAX_ENV_STEPS="${MAX_ENV_STEPS:-1000}"; else export MAX_ENV_STEPS="${MAX_ENV_STEPS:-400}"; fi
export NUM_ENVS="${NUM_ENVS:-8}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}" VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}" TEST_FREQ="${TEST_FREQ:-5}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}" PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-64}"
export VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-2048}" ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-768}" ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}" USE_REMOVE_PADDING=false USE_TORCH_COMPILE=false
export LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console,file]}"
export CRAF_TEXT_SEED="${CRAF_TEXT_SEED:-20261029}" VALIDATION_SEED="${VALIDATION_SEED:-20261014}"
# Every env step is its own PPO response and GAE stops at its end (next value 0), so the per-step reward
# must already be the discounted return-to-go (as in compat_06c4.env); "immediate" scores an action only
# by its own reward and gives navigation no credit.
export CRAF_TEXT_REWARD_MODE="${CRAF_TEXT_REWARD_MODE:-remaining_return}" CRAF_TEXT_REMAINING_RETURN_GAMMA="${CRAF_TEXT_REMAINING_RETURN_GAMMA:-0.99}"
nohup bash "${CHECKOUT}/vlm_pipeline/phase3_verl/craftext_integration/run_ppo_8x8_smoke_docker.sh" \
  > "${RUN_ROOT}/launcher.log" 2>&1 < /dev/null &
echo "launched ${RUN_NAME} on GPU${GPU_ID}: ${RUN_ROOT}"
