#!/usr/bin/env bash
# Phase 1 -> 2: collect demonstrations with the teacher of the SAME task set.
#
#   GPU_ID=3 SETTINGS=achievements_wood TEACHER=$RUNS/<run>/checkpoints/teacher_best.pt \
#     SEEDS=20261500:20261550 TASKS=0,1,2,3 RUN_NAME=wood_teacher_200_YYYYMMDD_r1 \
#     bash vlm_pipeline/launch/phase1_collect.sh
#
# TEACHER is a Flax teacher_*.pt (phase1_teacher_ppo.sh); the collector refuses a teacher whose
# craftext_settings differ from SETTINGS. For an Orbax ppo_with_instruction.py checkpoint use
# caged_craftext/baselines/collect_ppo_achievements_all_trajectories.py (see launch_r4_achievements_all_collector.py).
# Output: $RUNS/$RUN_NAME/{train.parquet, val.parquet, images/, manifest.json} with teacher logits/probabilities.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${SETTINGS:?}" "${TEACHER:?}" "${SEEDS:?}" "${RUN_NAME:?}"
TASKS="${TASKS:-all}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_DIR="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_DIR}"
record_code_version "${RUN_DIR}"
prepare_bert_cwd "${RUN_DIR}/cache"
write_user_env "${RUN_DIR}/container.env"
TEACHER_IN="/mnt/runs/${TEACHER#${RUNS}/}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup docker run --rm --name "${RUN_NAME}" --network host --user 1007:100 \
  --env-file "${RUN_DIR}/container.env" -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" \
  -w "/mnt/runs/${RUN_NAME}/cache" --entrypoint python3 "${JAX_IMAGE}" -u \
  /workspace/repo/caged_craftext/baselines/collect_flax_teacher_trajectories.py \
  --legacy-root /workspace/repo/caged_craftext --checkpoint "${TEACHER_IN}" --craftext-settings "${SETTINGS}" \
  --output-dir "/mnt/runs/${RUN_NAME}/corpus" --seeds "${SEEDS}" --task-indices "${TASKS}" --max-steps "${MAX_STEPS}" \
  > "${RUN_DIR}/launcher.log" 2>&1 &
echo "launched ${RUN_NAME} on GPU${GPU_ID}: corpus -> ${RUN_DIR}/corpus"
