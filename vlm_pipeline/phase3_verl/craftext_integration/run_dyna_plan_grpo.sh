#!/usr/bin/env bash
# Stage 2 through veRL (safe_rl_nlp.md rules 18-21): planning in Dreamer 4 + GRPO, one veRL run (plan_manager.py).
# Runs inside the modern veRL image with the one checkout mounted. Two GPUs, roles apart:
#   GPU_ID       veRL: actor (FSDP) + vLLM rollout + reference policy (KL), GRPO (no critic)
#   DREAMER_GPU  dreamer_service.py: Dreamer 4 tokenizer encoding, dreams of the n plans, reward / continue heads, critic V
# veRL keeps the optimiser, the lr schedule and the checkpoints; trainer.resume_mode=auto restores all of it.
set -euo pipefail
: "${MODEL_PATH:?materialised Stage-1 SLM (slm_vision/materialize_d4_checkpoint.py)}"
: "${RUN_ROOT:?new run dir}"
: "${GAME:?crafter or wood}"
: "${D_DYN:?}" "${D_HEADS:?}" "${D_TOK:?}" "${D_CRITIC:?}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # vlm_pipeline/phase3_verl
CHECKOUT="$(cd "${REPO_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"; DREAMER_GPU="${DREAMER_GPU:-1}"
NUM_ENVS="${NUM_ENVS:-32}"; VAL_ENVS="${VAL_ENVS:-16}"
N_PLANS="${N_PLANS:-8}"; PLAN_LEN="${PLAN_LEN:-16}"; TRAIN_EVERY="${TRAIN_EVERY:-4}"; STEPS_PER_ROLLOUT="${STEPS_PER_ROLLOUT:-32}"
EPISODE_CAP="${EPISODE_CAP:-1000}"; VAL_MAX_STEPS="${VAL_MAX_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"; TEST_FREQ="${TEST_FREQ:-20}"; SAVE_FREQ="${SAVE_FREQ:-10}"; VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
ACTOR_LR="${ACTOR_LR:-1e-5}"; CLIP="${CLIP:-0.2}"; KL_COEF="${KL_COEF:-0.01}"; NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-false}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"   # in prompts (lanes), as veRL counts it
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"; MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"; PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:-176}"
TOP_K="${TOP_K:-20}"; TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-170000000}"; VALIDATION_SEED="${VALIDATION_SEED:-150000000}"
D_HORIZON="${D_HORIZON:-8}"; D_K="${D_K:-1}"; D_SAMPLES="${D_SAMPLES:-1}"; D_PESSIMISM="${D_PESSIMISM:-0}"
WORLD_FT_STEPS="${WORLD_FT_STEPS:-0}"
RESUME_MODE=auto
if [[ "${WORLD_FT_STEPS}" != 0 ]]; then
  [[ "${GAME}" == crafter ]] || { echo "Online Dreamer requires Crafter" >&2; exit 1; }
  [[ ! -e "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]] || {
    echo "Online Dreamer has no training-world resume yet; use a new run directory" >&2; exit 1;
  }
  RESUME_MODE=disable
fi
LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console,file]}"
mkdir -p "${RUN_ROOT}"/{ray,dataset,checkpoints,logs,dreamer}
export CRAFTEXT_LEGACY_ROOT="${CHECKOUT}" PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/craftext_integration/vllm_plugin_site:${PYTHONPATH:-}"
export RAY_TMPDIR=/tmp/ray RAY_TEMP_DIR=/tmp/ray   # short: AF_UNIX socket paths are limited to 107 bytes WANDB_MODE=disabled COMET_DISABLE_AUTO_LOGGING=1
export VERL_FILE_LOGGER_PATH="${RUN_ROOT}/logs/metrics.jsonl" JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu

# --- the world model on its own GPU ---------------------------------------------------------------------------------
DREAMER_SOCKET=/tmp/dreamer.sock
CUDA_VISIBLE_DEVICES="${DREAMER_GPU}" PYTHONPATH="${DREAMER_PYTHONPATH:-}:${PYTHONPATH}" "${PYTHON_BIN}" \
  "${CHECKOUT}/vlm_pipeline/phase4_world_model/dreamer_service.py" --dyn "${D_DYN}" --heads "${D_HEADS}" --tok "${D_TOK}" \
  --critic "${D_CRITIC}" ${D_TEXT_TOKENIZER:+--text-tokenizer ${D_TEXT_TOKENIZER}} ${D_INIT:+--init ${D_INIT}} \
  ${D_EXTRA_CRITICS:+--extra-critics ${D_EXTRA_CRITICS}} --critic-pessimism "${D_CRITIC_PESSIMISM:-0}" \
  --horizon "${D_HORIZON}" --K "${D_K}" --dream-samples "${D_SAMPLES}" --pessimism "${D_PESSIMISM}" \
  --critic-ft-steps "${D_CRITIC_FT_STEPS:-0}" --critic-lr "${D_CRITIC_LR:-1e-5}" --critic-anchor "${D_CRITIC_ANCHOR:-0.25}" \
  --world-ft-steps "${WORLD_FT_STEPS}" --world-lr "${WORLD_LR:-1e-5}" \
  --world-batch "${WORLD_BATCH:-16}" --world-buffer "${WORLD_BUFFER:-2048}" \
  --socket "${DREAMER_SOCKET}" --out "${RUN_ROOT}/dreamer" > "${RUN_ROOT}/dreamer/service.log" 2>&1 &
DREAMER_PID=$!
trap 'kill "${DREAMER_PID}" 2>/dev/null || true' EXIT
until [[ -f "${RUN_ROOT}/dreamer/ready" ]]; do kill -0 "${DREAMER_PID}"; sleep 3; done

"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/make_smoke_dataset.py" --output "${RUN_ROOT}/dataset/train.parquet" --rows "${NUM_ENVS}"
"${PYTHON_BIN}" "${REPO_ROOT}/craftext_integration/make_smoke_dataset.py" --output "${RUN_ROOT}/dataset/val.parquet" --rows "${VAL_ENVS}"

# --- veRL: GRPO on the n plans of every train_every-th state ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
  algorithm.use_kl_in_reward=false \
  data.train_files="${RUN_ROOT}/dataset/train.parquet" data.val_files="${RUN_ROOT}/dataset/val.parquet" \
  data.train_batch_size="${NUM_ENVS}" data.val_batch_size="${VAL_ENVS}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=true data.filter_overlong_prompts=false data.truncation=left \
  actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.external_lib=craftext_integration.slm_vision.vllm_model \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.clip_ratio="${CLIP}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP}" actor_rollout_ref.actor.clip_ratio_high="${CLIP}" \
  actor_rollout_ref.actor.policy_loss.loss_mode="${POLICY_LOSS:-vanilla}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE:-token-mean}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN:-131072}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN:-131072}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN:-131072}" \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.calculate_entropy=true actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" actor_rollout_ref.ref.use_torch_compile=false \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" actor_rollout_ref.rollout.top_k="${TOP_K}" actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.calculate_log_probs="${CALC_ROLLOUT_LOGPROBS:-false}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-768}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-512}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_BATCHED_TOKENS:-65536}" \
  actor_rollout_ref.rollout.free_cache_engine=true actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER:-false}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=craftext_reasoning \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${REPO_ROOT}/craftext_integration/configs/agent_loops.yaml" \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=craftext_integration.plan_manager.CraftextPlanVectorAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  trainer.use_v1=true trainer.v1.trainer_mode=sync \
  trainer.total_epochs="${TOTAL_EPOCHS}" trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" trainer.test_freq="${TEST_FREQ}" trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  "trainer.logger=${LOGGER_BACKENDS}" trainer.project_name=stage2_dyna_verl trainer.experiment_name="$(basename "${RUN_ROOT}")" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
  +ray_kwargs.ray_init.runtime_env.env_vars.CRAFTEXT_LEGACY_ROOT="${CHECKOUT}" \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray \
  +craftext.game="${GAME}" +craftext.dreamer_socket="${DREAMER_SOCKET}" +craftext.dreamer_world_ft_steps="${WORLD_FT_STEPS}" \
  +craftext.n_plans="${N_PLANS}" +craftext.plan_len="${PLAN_LEN}" +craftext.train_every="${TRAIN_EVERY}" \
  +craftext.steps_per_rollout="${STEPS_PER_ROLLOUT}" +craftext.episode_cap="${EPISODE_CAP}" +craftext.val_max_steps="${VAL_MAX_STEPS}" \
  +craftext.per_turn_max_tokens="${PER_TURN_MAX_TOKENS}" +craftext.num_envs="${NUM_ENVS}" \
  +craftext.seed="${SEED}" +craftext.validation_seed="${VALIDATION_SEED}" +craftext.visual_tokens=320 \
  +craftext.settings="${SETTINGS:-achievements_wood}" +craftext.teacher_root="${CHECKOUT}/caged_craftext" \
  +craftext.n_chunks="${N_CHUNKS:-4}" +craftext.max_env_steps=1 +craftext.visual_prompt_history=0
