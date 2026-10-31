#!/usr/bin/env bash
# Phase 2: B = visual 5M SLM + CNN trained on teacher demonstrations from ORIGINAL random weights,
# then materialised for veRL and gated by a closed-loop evaluation in the teacher's environment.
#
#   GPU_ID=2 SETTINGS=achievements_wood CORPUS=$RUNS/<collection>/corpus TARGET=teacher_distribution \
#     RUN_NAME=slm5m_vision4_nohist_wood_distill_gpu2_YYYYMMDD_r1 bash vlm_pipeline/launch/phase2_ce.sh
#
# TARGET=teacher_distribution: expected CE under the teacher's full 17-action distribution (distillation);
# TARGET=argmax: CE on the executed teacher action. Never pass an init checkpoint: phase 2 always starts
# from the original untrained weights (saved to initial_random/).
# Eval defaults: wood 4 tasks x 8 seeds x 400 steps; all 22 tasks x 2 seeds x 1000 steps.
# REASONING=$RUNS/<reasoning run>/reasoning.jsonl (phase2_reasoning.sh): the target becomes
# <reasoning>LLM-teacher sentence</reasoning> + the action (prompt vision_image_reasoning_v1, --mode cot);
# with TARGET=teacher_distribution: CE on the reasoning + distillation of the action given it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${SETTINGS:?}" "${CORPUS:?teacher corpus dir with train/val.parquet and images/}" "${RUN_NAME:?}"
TARGET="${TARGET:-teacher_distribution}"
REASONING="${REASONING:-}"
if [ -n "${REASONING}" ]; then PROMPT=vision_image_reasoning_v1; MODE=cot; else PROMPT="${PROMPT:-vision_image_only_v1}"; MODE=action; fi
EPOCHS="${EPOCHS:-48}"
SEED="${SEED:-7101}"
CHECKPOINTS="${CHECKPOINTS:-best_validation epoch_12 epoch_48}"
if [ "${SETTINGS}" = achievements_all ]; then
  EVAL_SEEDS="${EVAL_SEEDS:-20263000,20263001}"; EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
else
  EVAL_SEEDS="${EVAL_SEEDS:-20263000,20263001,20263002,20263003,20263004,20263005,20263006,20263007}"
  EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-400}"
fi
RUN_DIR="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_DIR}"
record_code_version "${RUN_DIR}"
CORPUS_IN="/mnt/runs/${CORPUS#${RUNS}/}"
REASONING_FLAG=""
[ -n "${REASONING}" ] && REASONING_FLAG="--reasoning /mnt/runs/${REASONING#${RUNS}/}"
R="/mnt/runs/${RUN_NAME}"

cat > "${RUN_DIR}/train.sh" <<EOF
set -euo pipefail
export PYTHONPATH=/workspace/repo/vlm_pipeline/phase2_slm
python3 -m slm_ce.prepare_teacher_dataset --input-dir ${CORPUS_IN} --output-dir ${R}/data \\
  --prompt-protocol ${PROMPT} --split-mode episode --successful-only --split-seed ${SEED} ${REASONING_FLAG}
python3 -m slm_ce.crafter_slm prepare --data ${R}/data
python3 -m slm_ce.crafter_vision_slm train --data ${R}/data --out ${R}/run_5m_vision_action \\
  --size 5m --mode ${MODE} --target ${TARGET} --epochs ${EPOCHS} --seed ${SEED} --visual-tokens 4 --device cuda \\
  --initial-checkpoint-out ${R}/initial_random
EOF
HISTORY_FLAG=""
[ "${PROMPT}" = vision_image_only_v1 ] && HISTORY_FLAG="--no-history"
[ "${PROMPT}" = vision_image_reasoning_v1 ] && HISTORY_FLAG="--no-history --reasoning"
cat > "${RUN_DIR}/eval.sh" <<EOF
set -u
cd ${R}/cache
for C in ${CHECKPOINTS}; do
  [ -f ${R}/materialized_\$C/config.json ] || continue
  python3 /workspace/repo/vlm_pipeline/phase2_slm/slm_ce/closed_loop_eval.py --mode teacher ${HISTORY_FLAG} \\
    --model ${R}/materialized_\$C --settings ${SETTINGS} --seeds ${EVAL_SEEDS} --max-steps ${EVAL_MAX_STEPS} \\
    --out ${R}/eval_\${C}_teacher.json > ${R}/eval_\${C}_teacher.log 2>&1
done
echo EVALDONE
EOF

chain() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RUN_NAME}" --network host --entrypoint bash \
    -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" "${TORCH_IMAGE}" "${R}/train.sh" > "${RUN_DIR}/ce.log" 2>&1
  echo "ce exit $? $(date)"
  for C in ${CHECKPOINTS}; do
    [ -d "${RUN_DIR}/run_5m_vision_action/${C}" ] || continue
    mkdir -p "${RUN_DIR}/materialized_${C}" && chmod 777 "${RUN_DIR}/materialized_${C}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RUN_NAME}_mat_${C}" -v "${CHECKOUT}:/workspace/repo:ro" \
      -v "${RUNS}:/mnt/runs" -w /workspace/repo/vlm_pipeline/phase3_verl/craftext_integration/slm_vision \
      --entrypoint python3 "${VERL_IMAGE}" materialize_ce_checkpoint.py \
      --source "${R}/run_5m_vision_action/${C}" --output "${R}/materialized_${C}" >> "${RUN_DIR}/materialize.log" 2>&1
  done
  prepare_bert_cwd "${RUN_DIR}/cache"
  write_user_env "${RUN_DIR}/eval.env" "JAX_PLATFORMS=cpu
JAX_PLATFORM_NAME=cpu
HF_HOME=${R}/cache/hf
"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RUN_NAME}_eval" --network host --user 1007:100 \
    --entrypoint bash --env-file "${RUN_DIR}/eval.env" -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" \
    "${VERL_IMAGE}" "${R}/eval.sh" > "${RUN_DIR}/eval.log" 2>&1
  echo "chain done $(date)"
}
chain > "${RUN_DIR}/chain.log" 2>&1 < /dev/null &
disown
echo "launched ${RUN_NAME} on GPU${GPU_ID}: ${RUN_DIR} (chain.log)"
