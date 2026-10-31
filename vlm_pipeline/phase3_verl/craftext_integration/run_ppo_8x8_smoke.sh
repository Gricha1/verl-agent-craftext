#!/usr/bin/env bash
# Three-update, one-GPU correctness run.  This is intentionally DP=1 and RSSM-free.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-1.5B-Instruct checkpoint}"
: "${CRAFTEXT_LEGACY_ROOT:?Set CRAFTEXT_LEGACY_ROOT to the existing CrafText checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to a new, empty output directory}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-4}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-1024}"
N_GPUS="${N_GPUS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-320}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${NUM_ENVS}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${NUM_ENVS}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${NUM_ENVS}}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
TEST_FREQ="${TEST_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console]}"
PROJECT_NAME="${PROJECT_NAME:-modern_verl_craftext}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_8x8_reasoning_${N_GPUS}gpu}"
VALIDATION_SEED="${VALIDATION_SEED:-20261014}"
VALIDATION_TRAJECTORY_LIMIT="${VALIDATION_TRAJECTORY_LIMIT:-10}"
PROFILE_STEPS="${PROFILE_STEPS:-}"
PROFILE_SAVE_PATH="${PROFILE_SAVE_PATH:-${RUN_ROOT}/profiles/torch}"
FILE_LOGGER_PATH="${FILE_LOGGER_PATH:-${RUN_ROOT}/logs/metrics.jsonl}"

# A compatibility launcher may mount this non-secret file into RUN_ROOT.  It is
# deliberately sourced after ordinary defaults so it can reproduce an archived
# experiment without changing the defaults used by smoke tests.
if [[ -f "${RUN_ROOT}/compat.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_ROOT}/compat.env"
fi

DATA_RETURN_RAW_CHAT="${DATA_RETURN_RAW_CHAT:-true}"
DATA_FILTER_OVERLONG_PROMPTS="${DATA_FILTER_OVERLONG_PROMPTS:-false}"
DATA_TRUNCATION="${DATA_TRUNCATION:-left}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
# 1e-6 / 1e-5 suit billion-parameter LLMs; small models usually need more.
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1536}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-${NUM_ENVS}}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.12}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE="${REF_LOGPROB_MICRO_BATCH_SIZE:-1}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-auto}"
ROLLOUT_CHAT_FORMAT="${ROLLOUT_CHAT_FORMAT:-chatml}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
CRAF_TEXT_ENV_CONFIG="${CRAF_TEXT_ENV_CONFIG:-debug_square_8x8}"
CRAF_TEXT_RESET_RATIO="${CRAF_TEXT_RESET_RATIO:-1}"
CRAF_TEXT_SEED="${CRAF_TEXT_SEED:-20261013}"
CRAF_TEXT_ACTION_HISTORY_LENGTH="${CRAF_TEXT_ACTION_HISTORY_LENGTH:-50}"
CRAF_TEXT_REASONING_HISTORY_LENGTH="${CRAF_TEXT_REASONING_HISTORY_LENGTH:-3}"
CRAF_TEXT_VISUAL_PROMPT_HISTORY="${CRAF_TEXT_VISUAL_PROMPT_HISTORY:-8}"
CRAF_TEXT_REWARD_MODE="${CRAF_TEXT_REWARD_MODE:-immediate}"
CRAF_TEXT_REMAINING_RETURN_GAMMA="${CRAF_TEXT_REMAINING_RETURN_GAMMA:-1.0}"
CRAF_TEXT_USE_INVALID_ACTION_PENALTY="${CRAF_TEXT_USE_INVALID_ACTION_PENALTY:-false}"
CRAF_TEXT_INVALID_ACTION_PENALTY_COEF="${CRAF_TEXT_INVALID_ACTION_PENALTY_COEF:-0.0}"

mkdir -p "${RUN_ROOT}"/{ray,dataset,checkpoints,logs,profiles}
mkdir -p "${RUN_ROOT}/craftext_runtime_cache"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CRAFTEXT_LEGACY_ROOT
# Do not put the legacy checkout on Python's global import path: it contains
# its own old ``verl`` package.  ``env_adapter`` adds this checkout lazily only
# when constructing the CrafText environment, after modern veRL is imported.
#
# ``vllm_plugin_site`` holds a dist-info whose ``vllm.general_plugins`` entry
# point registers the visual SLM; vLLM loads it in its engine-core and worker
# subprocesses, which never import veRL's ``external_lib``.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/craftext_integration/vllm_plugin_site:${PYTHONPATH:-}"
export RAY_TMPDIR="${RUN_ROOT}/ray"
export RAY_TEMP_DIR="${RUN_ROOT}/ray"
export CRAFTEXT_RUNTIME_CACHE_DIR="${RUN_ROOT}/craftext_runtime_cache"
export COMET_DISABLE_AUTO_LOGGING=1
export WANDB_MODE=disabled
export VERL_FILE_LOGGER_PATH="${FILE_LOGGER_PATH}"
# The environment is JAX-vmapped but intentionally CPU-resident in this first
# PPO correctness run; the sole GPU is reserved for modern veRL actor/critic.
export JAX_PLATFORM_NAME=cpu

if [[ "${LOGGER_BACKENDS}" == *comet* ]]; then
  unset COMET_DISABLE_AUTO_LOGGING
  : "${COMET_API_KEY:?COMET_API_KEY is required when LOGGER_BACKENDS includes comet}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/make_smoke_dataset.py" \
  --output "${RUN_ROOT}/dataset/craftext_smoke.parquet" --rows "${NUM_ENVS}"
# Validation has its own batch size (one placeholder prompt per validation env), so its own rows.
"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/make_smoke_dataset.py" \
  --output "${RUN_ROOT}/dataset/craftext_val.parquet" --rows "${VAL_BATCH_SIZE}"
"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/make_validation_seed_manifest.py" \
  --output "${RUN_ROOT}/dataset/validation_seeds.json" --base-seed "${VALIDATION_SEED}" --episodes "${VAL_BATCH_SIZE}"

PROFILE_ARGS=()
if [[ -n "${PROFILE_STEPS}" ]]; then
  PROFILE_ARGS=(
    global_profiler.tool=torch
    "global_profiler.steps=${PROFILE_STEPS}"
    "global_profiler.save_path=${PROFILE_SAVE_PATH}"
    global_profiler.relocate_results=true
    'actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cpu,cuda,memory]'
    'actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cpu,cuda,memory]'
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=true
  )
fi

# One CrafText action is one PPO response, matching the legacy r8 rollout.
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=false \
  data.train_files="${RUN_ROOT}/dataset/craftext_smoke.parquet" \
  data.val_files="${RUN_ROOT}/dataset/craftext_val.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat="${DATA_RETURN_RAW_CHAT}" \
  data.filter_overlong_prompts="${DATA_FILTER_OVERLONG_PROMPTS}" \
  data.truncation="${DATA_TRUNCATION}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  critic.model.trust_remote_code=true \
  +actor_rollout_ref.model.external_lib=craftext_integration.slm_vision.vllm_model \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +critic.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.calculate_entropy=true \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  critic.optim.lr="${CRITIC_LR}" \
  critic.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  critic.fsdp.param_offload=false \
  critic.fsdp.optimizer_offload=false \
  critic.model.lora_rank="${LORA_RANK}" \
  critic.model.lora_alpha="${LORA_ALPHA}" \
  critic.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  critic.fsdp.use_torch_compile="${USE_TORCH_COMPILE}" \
  critic.forward_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}" \
  actor_rollout_ref.rollout.multi_turn.format="${ROLLOUT_CHAT_FORMAT}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=craftext_reasoning \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${REPO_ROOT}/craftext_integration/configs/agent_loops.yaml" \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=craftext_integration.vector_manager.CraftextVectorAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  trainer.use_v1=true \
  trainer.v1.trainer_mode=sync \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode=disable \
  "trainer.logger=${LOGGER_BACKENDS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
  +ray_kwargs.ray_init.runtime_env.env_vars.CRAFTEXT_LEGACY_ROOT="${CRAFTEXT_LEGACY_ROOT}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.CRAFTEXT_RUNTIME_CACHE_DIR="${CRAFTEXT_RUNTIME_CACHE_DIR}" \
  +ray_kwargs.ray_init._temp_dir="${RUN_ROOT}/ray" \
  +ray_kwargs.timeline_json_file="${RUN_ROOT}/profiles/ray_timeline.json" \
  +craftext.max_env_steps="${MAX_ENV_STEPS}" \
  +craftext.per_turn_max_tokens="${PER_TURN_MAX_TOKENS:-320}" \
  +craftext.num_envs="${NUM_ENVS}" \
  +craftext.seed="${CRAF_TEXT_SEED}" \
  +craftext.env_config_name="${CRAF_TEXT_ENV_CONFIG}" \
  +craftext.reset_ratio="${CRAF_TEXT_RESET_RATIO}" \
  +craftext.action_history_length="${CRAF_TEXT_ACTION_HISTORY_LENGTH}" \
  +craftext.reasoning_history_length="${CRAF_TEXT_REASONING_HISTORY_LENGTH}" \
  +craftext.visual_prompt_history="${CRAF_TEXT_VISUAL_PROMPT_HISTORY}" \
  +craftext.reward_mode="${CRAF_TEXT_REWARD_MODE}" \
  +craftext.remaining_return_gamma="${CRAF_TEXT_REMAINING_RETURN_GAMMA}" \
  +craftext.use_invalid_action_penalty="${CRAF_TEXT_USE_INVALID_ACTION_PENALTY}" \
  +craftext.invalid_action_penalty_coef="${CRAF_TEXT_INVALID_ACTION_PENALTY_COEF}" \
  +craftext.validation_seed="${VALIDATION_SEED}" \
  +craftext.validation_trajectory_dir="${RUN_ROOT}/validation_trajectories" \
  +craftext.validation_trajectory_limit="${VALIDATION_TRAJECTORY_LIMIT}" \
  "${PROFILE_ARGS[@]}"
