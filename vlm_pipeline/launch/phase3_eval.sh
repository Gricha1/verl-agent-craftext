#!/usr/bin/env bash
# Phase 3 closed-loop eval: export veRL actor checkpoints of an RL run as HF models (same format as the
# materialised B_ce) and evaluate them in the teacher's environment with the phase-2 protocol, so B_ce
# and B_rl are compared on identical episodes.
#
#   GPU_ID=2 SETTINGS=achievements_wood RL_RUN=<phase-3 run name> \
#     START=$RUNS/<phase-2 run>/materialized_epoch_48_v2 bash vlm_pipeline/launch/phase3_eval.sh
#
# STEPS: global steps to evaluate (default: every saved one). WAIT=1 waits for the RL container to exit
# first. Seeds default to the 128-episode wood set (4 tasks x 32 seeds 20264000-031) used for B_ce;
# all-achievements: 22 tasks x 2 seeds, 1000 steps. Results: $RUNS/$RL_RUN/closed_loop/eval_step_<N>.json
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${SETTINGS:?}" "${RL_RUN:?}" "${START:?checkpoint the RL run started from}"
RUN_DIR="${RUNS}/${RL_RUN}"
R="/mnt/runs/${RL_RUN}"
START_IN="/mnt/runs/${START#${RUNS}/}"
if [ "${SETTINGS}" = achievements_all ]; then
  EVAL_SEEDS="${EVAL_SEEDS:-20263000,20263001}"; EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
else
  EVAL_SEEDS="${EVAL_SEEDS:-$(seq -s, 20264000 20264031)}"; EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-400}"
fi
OUT="${RUN_DIR}/closed_loop"
mkdir -p "${OUT}" && chmod 777 "${OUT}"
record_code_version "${OUT}"

chain() {
  if [ "${WAIT:-0}" = 1 ]; then
    # (not "docker ps | grep -q": under pipefail grep's early exit makes the pipeline fail)
    while [ -n "$(docker ps -q --filter "name=^${RL_RUN}\$")" ]; do sleep 60; done
    echo "RL run finished $(date)"
  fi
  local steps="${STEPS:-$(ls "${RUN_DIR}/checkpoints" | sed -n 's/^global_step_//p' | sort -n | tr '\n' ' ')}"
  prepare_bert_cwd "${OUT}/cache"
  write_user_env "${OUT}/eval.env" "JAX_PLATFORMS=cpu
JAX_PLATFORM_NAME=cpu
HF_HOME=${R}/closed_loop/cache/hf
"
  for S in ${steps}; do
    local export_dir="${RUN_DIR}/export_global_step_${S}"
    if [ ! -f "${export_dir}/config.json" ]; then
      mkdir -p "${export_dir}" && chmod 777 "${export_dir}"
      # as our user: written as root, NFS squashes to nobody and model.safetensors stays 0600
      CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RL_RUN}_export_${S}" --user 1007:100 \
        --env-file "${OUT}/eval.env" -v "${CHECKOUT}:/workspace/repo:ro" \
        -v "${RUNS}:/mnt/runs" -w /workspace/repo/vlm_pipeline/phase3_verl/craftext_integration/slm_vision \
        --entrypoint python3 "${VERL_IMAGE}" export_rl_checkpoint.py \
        --actor "${R}/checkpoints/global_step_${S}/actor" --start "${START_IN}" \
        --output "${R}/export_global_step_${S}" > "${OUT}/export_step_${S}.log" 2>&1 || { echo "export ${S} failed"; continue; }
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RL_RUN}_eval_${S}" --network host --user 1007:100 \
      --env-file "${OUT}/eval.env" -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" -w "${R}/closed_loop/cache" \
      --entrypoint python3 "${VERL_IMAGE}" /workspace/repo/vlm_pipeline/phase2_slm/slm_ce/closed_loop_eval.py \
      --mode teacher --no-history --settings "${SETTINGS}" --model "${R}/export_global_step_${S}" \
      --seeds "${EVAL_SEEDS}" --max-steps "${EVAL_MAX_STEPS}" --out "${R}/closed_loop/eval_step_${S}.json" \
      > "${OUT}/eval_step_${S}.log" 2>&1
    echo "step ${S} eval exit $? $(date)"
  done
  echo "chain done $(date)"
}
chain > "${OUT}/chain.log" 2>&1 < /dev/null &
disown
echo "launched closed-loop eval of ${RL_RUN} on GPU${GPU_ID}: ${OUT} (chain.log)"
