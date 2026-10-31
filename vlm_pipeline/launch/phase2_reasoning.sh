#!/usr/bin/env bash
# Phase 2 data: reasoning targets for a teacher collection, ZeroHero recipe (LLM teacher, Gemma-4-26B-A4B-it).
#
#   GPU_ID=2 COLLECTION=$RUNS/wood_teacher_collection_20261028/wood_teacher_200_seeds20261500_r1 \
#     RUN_NAME=wood200_reasoning_gemma4_YYYYMMDD_r1 setsid nohup bash vlm_pipeline/launch/phase2_reasoning.sh &
#
# 1) replay_teacher_states.py replays every episode with the stored teacher actions (jitted; frames must
#    match within 1e-5, float rounding only)
#    and writes the learner's public state as text: $RUNS/$RUN_NAME/states.jsonl
# 2) a vLLM server with Gemma (gorbov_gemma4_vllm, port 8383) labels each transition with a reasoning
#    sentence for the teacher's action: $RUNS/$RUN_NAME/reasoning.jsonl (rerun resumes)
# The vLLM server is started if it is not running and left running (docker rm -f gorbov_gemma4_vllm).
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_our_gpu
: "${COLLECTION:?teacher collection dir}" "${RUN_NAME:?}"
GEMMA="${GEMMA:-${BASE}/hf_models/gemma-4-26B-A4B-it}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.24.0}"
RUN_DIR="${RUNS}/${RUN_NAME}"
new_run_dir "${RUN_DIR}"
record_code_version "${RUN_DIR}"
prepare_bert_cwd "${RUN_DIR}/cache"
write_user_env "${RUN_DIR}/container.env"
REASONING=/workspace/repo/vlm_pipeline/phase2_slm/reasoning

chain() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run --rm --name "${RUN_NAME}_replay" --network host --user 1007:100 \
    --env-file "${RUN_DIR}/container.env" -v "${CHECKOUT}:/workspace/repo:ro" -v "${RUNS}:/mnt/runs" \
    -w "/mnt/runs/${RUN_NAME}/cache" --entrypoint python3 "${JAX_IMAGE}" -u "${REASONING}/replay_teacher_states.py" \
    --legacy-root /workspace/repo/caged_craftext --collection-dir "/mnt/runs/${COLLECTION#${RUNS}/}" \
    --out "/mnt/runs/${RUN_NAME}/states.jsonl" --jit --atol 1e-5 > "${RUN_DIR}/replay.log" 2>&1
  if [ -z "$(docker ps -q --filter name=^gorbov_gemma4_vllm\$)" ]; then
    docker rm gorbov_gemma4_vllm > /dev/null 2>&1 || true
    # text only (no image/video/audio encoder profiling), 0.80 of the card next to an RL run
    CUDA_VISIBLE_DEVICES="${GPU_ID}" docker run -d --name gorbov_gemma4_vllm --network host --ipc host \
      -v "$(dirname "${GEMMA}"):/models:ro" "${VLLM_IMAGE}" --model "/models/$(basename "${GEMMA}")" \
      --served-model-name craftax-gemma --port 8383 --host 127.0.0.1 --gpu-memory-utilization 0.80 \
      --max-model-len 4096 --max-num-seqs 64 --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' > /dev/null
  fi
  until curl -s -m 5 http://127.0.0.1:8383/v1/models | grep -q craftax-gemma; do sleep 20; done
  python3 "${CHECKOUT}/vlm_pipeline/phase2_slm/reasoning/label_reasoning.py" --states "${RUN_DIR}/states.jsonl" \
    --out "${RUN_DIR}/reasoning.jsonl" --concurrency 32
  echo "chain done $(date -u)"
}
echo "running ${RUN_NAME}: ${RUN_DIR}/chain.log"
chain > "${RUN_DIR}/chain.log" 2>&1 < /dev/null
