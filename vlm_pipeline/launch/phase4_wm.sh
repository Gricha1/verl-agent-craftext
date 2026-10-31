#!/usr/bin/env bash
# Phase 4: SLM world model (vlm_pipeline/phase4_world_model/train_slm_world_model.py) on a teacher collection.
#
#   GPU_ID=6 CORPUS=$RUNS/<collection> SPLIT_FROM=$RUNS/<phase-2 run>/data INIT=$RUNS/<phase-2 run>/initial_random \
#     H_TOKENS=30 [REASONING=$RUNS/<reasoning run>/reasoning.jsonl] RUN_NAME=<name> bash vlm_pipeline/launch/phase4_wm.sh
#
# SPLIT_FROM: the phase-2 CE data whose validation episodes are reused, so the world model and B_ce share
# one episode split. INIT: the untrained SLM (initial_random) whose tokenizer matches the text the model reads.
# GPU6 and GPU7 are allowed here only: the user assigned the world-model runs to them (2026-11-01).
set -euo pipefail
source "$(dirname "$0")/common.sh"
case "${GPU_ID:-}" in 2|3|6|7) ;; *) echo "GPU_ID must be 2, 3, 6 or 7 for phase 4, got '${GPU_ID:-}'" >&2; exit 1 ;; esac
: "${CORPUS:?}" "${SPLIT_FROM:?}" "${INIT:?}" "${RUN_NAME:?}"
H_SIZE="${H_SIZE:-512}"; H_TOKENS="${H_TOKENS:-30}"; HORIZON="${HORIZON:-16}"; EPOCHS="${EPOCHS:-300}"
RUN_DIR="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_DIR}"
record_code_version "${RUN_DIR}"
in_runs() { echo "/mnt/runs/${1#${RUNS}/}"; }
REASONING_FLAG=""
[ -n "${REASONING:-}" ] && REASONING_FLAG="--reasoning $(in_runs "${REASONING}")"
cat > "${RUN_DIR}/train.sh" <<EOF
set -euo pipefail
python3 /workspace/repo/vlm_pipeline/phase4_world_model/train_slm_world_model.py \\
  --raw-data $(in_runs "${CORPUS}") --split-from $(in_runs "${SPLIT_FROM}") --init-checkpoint $(in_runs "${INIT}") \\
  --out /mnt/runs/${RUN_NAME}/run --h-size ${H_SIZE} --h-tokens ${H_TOKENS} --horizon ${HORIZON} --epochs ${EPOCHS} \\
  ${REASONING_FLAG} ${EXTRA_ARGS:-}
EOF
mkdir -p "${RUN_DIR}/run" && chmod 777 "${RUN_DIR}/run"
# several runs share one card: ~10 GB (30 h tokens) to ~19 GB (100) each at K=16
printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\nPYTHONUNBUFFERED=1\n' > "${RUN_DIR}/container.env"
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup docker run --rm --name "${RUN_NAME}" --entrypoint bash --env-file "${RUN_DIR}/container.env" \
  -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" "${TORCH_IMAGE}" "/mnt/runs/${RUN_NAME}/train.sh" \
  > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
echo "launched ${RUN_NAME} on GPU${GPU_ID}: ${RUN_DIR}"
