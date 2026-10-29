#!/usr/bin/env bash
# Phase 1: train teacher A = PPO + DistilBERT + CNN on one CrafText task set.
#
#   GPU_ID=2 SETTINGS=achievements_wood RUN_NAME=ppo_teacher_wood_gpu2_YYYYMMDD_r1 \
#     bash vlm_pipeline/launch/phase1_teacher_ppo.sh [extra train_teacher_multitask.py args]
#
# Output: $RUNS/$RUN_NAME/{resolved_config.json, train.log, tensorboard/, checkpoints/teacher_{best,latest}.pt}.
# teacher_best.pt is the input of phase1_collect.sh (the collector checks its task set).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${SETTINGS:?achievements_wood | achievements_all | ...}" "${RUN_NAME:?}"
RUN_DIR="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_DIR}"
record_code_version "${RUN_DIR}"
prepare_bert_cwd "${RUN_DIR}/cache"
write_user_env "${RUN_DIR}/container.env"
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup docker run --rm --name "${RUN_NAME}" --network host --user 1007:100 \
  --env-file "${RUN_DIR}/container.env" -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" \
  -w "/mnt/runs/${RUN_NAME}/cache" --entrypoint bash "${JAX_IMAGE}" -c "
    export PYTHONPATH=/workspace/repo/caged_craftext:/workspace/repo/caged_craftext/Craftax:/workspace/repo/caged_craftext/baselines
    python3 -u /workspace/repo/caged_craftext/baselines/train_teacher_multitask.py \
      --run-name ${RUN_NAME} --out-dir /mnt/runs/${RUN_NAME} --craftext-settings ${SETTINGS} $*" \
  > "${RUN_DIR}/launcher.log" 2>&1 &
echo "launched ${RUN_NAME} on GPU${GPU_ID}: ${RUN_DIR}"
