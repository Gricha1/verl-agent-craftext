#!/usr/bin/env bash
# Stage 2 through veRL on aicenteritl (safe_rl_nlp.md rules 18-21): run_dyna_plan_grpo.sh in the deps-only modern veRL image,
# the one checkout ~/verl-agent-craftext mounted read-only, veRL on GPU_ID and dreamer_service on DREAMER_GPU.
#   GAME=crafter RUN_NAME=<name> GPU_ID=0 DREAMER_GPU=1 [SMOKE=1] [env of run_dyna_plan_grpo.sh] bash stage2_verl_dyna_itl.sh
set -euo pipefail
: "${GAME:?crafter or wood}" "${RUN_NAME:?}"
CHECKOUT=$HOME/verl-agent-craftext
COMMIT=$(git -C "$CHECKOUT" rev-parse --short HEAD)
[ -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=no)" ] || { echo "checkout has local changes" >&2; exit 1; }
IMG=${IMG:-$(docker images modern-verl-craftext --format '{{.Repository}}:{{.Tag}}' | grep ':deps-' | head -1)}
[ -n "$IMG" ] || { echo "no modern-verl-craftext:deps-* image" >&2; exit 1; }
GPU_ID=${GPU_ID:-0}; DREAMER_GPU=${DREAMER_GPU:-1}
R=$HOME/runs/$RUN_NAME
[ ! -e "$R/logs/metrics.jsonl" ] || [ "${RESUME:-0}" = 1 ] || { echo "run exists: $R (RESUME=1 continues it, veRL restores the full state)" >&2; exit 1; }
mkdir -p "$R"; echo "$COMMIT" > "$R/code_version.txt"; cp "$0" "$R/launcher.sh"
A=$HOME/a3mirror
if [ "$GAME" = crafter ]; then
  MODEL=${MODEL:-$HOME/runs/verl_models/slm5m_plan16_d4qwen_crafter64_s0_verl_20261122}
  Q4=$A/run/dreamer4_qwen_crafter64_a3gpu3_20261118_r1
  D4CODE=${D4CODE:-$HOME/runs/slm5m_plan16_crafter_dyna_grpo_tokft_criticft_d4qwen_itlgpu1_20261121_r1/dreamer4}   # patched (text task, continue, init)
  MOUNTS="-v $Q4:/q4:ro -v $Q4:/r:ro -v $A/data/zerohero_crafter_memory_20261106/crafter64/train:/q/train:ro \
    -v $A/run/slm5m_plan16_value_d4qwen_crafter64_r64_scratch_a3gpu1_20261119_r1:/s0:ro -v $A/run:/runs:ro \
    -v $A/run/slm5m_recon_d4wood2200_init_20261118:/c:ro -v $A/run/dreamer_eval_wood2200_a3gpu3_20261112_r1/pylibs:/p/pylibs:ro \
    -v $A/run/crafter_dyna_libs_20261119:/cl:ro -v $D4CODE:/code/dreamer4:ro"
  DENV="-e D_DYN=/r/dyn/checkpoints/latest.pt -e D_TOK=/q4/tok/checkpoints/latest.pt \
    -e D_HEADS=${D_HEADS:-/runs/dreamer4_heads_qwen_crafter64_reward_a3gpu3_20261119_r1/heads/checkpoints/latest.pt} \
    -e D_CRITIC=${D_CRITIC:-/s0/run/checkpoints/step_060000.pt} -e DREAMER_PYTHONPATH=/cl:/p/pylibs:/code/dreamer4 \
    -e EPISODE_CAP=${EPISODE_CAP:-1000} -e VAL_MAX_STEPS=${VAL_MAX_STEPS:-1000} \n    -e PER_TURN_MAX_TOKENS=400 -e MAX_RESPONSE_LENGTH=416 -e ROLLOUT_MAX_MODEL_LEN=1024"   # Stage-1 gen_tokens of this SLM: 400
else
  echo "wood: not wired yet (Stage-1 wood SLM to materialise; prompts had left padding)" >&2; exit 1
fi
SMOKE_ENV=""
if [ "${SMOKE:-0}" = 1 ]; then
  SMOKE_ENV="-e NUM_ENVS=4 -e VAL_ENVS=2 -e N_PLANS=4 -e STEPS_PER_ROLLOUT=4 -e TRAIN_EVERY=2 -e TOTAL_EPOCHS=2 -e TEST_FREQ=2 \
    -e SAVE_FREQ=1 -e VAL_MAX_STEPS=6 -e EPISODE_CAP=20 -e PPO_MINI_BATCH_SIZE=2 -e PPO_MICRO_BATCH_SIZE=8"
fi
PASS=""
for v in ACTOR_LR CLIP KL_COEF NORM_ADV_BY_STD NUM_ENVS VAL_ENVS N_PLANS TRAIN_EVERY STEPS_PER_ROLLOUT TOTAL_EPOCHS TEST_FREQ SAVE_FREQ \
         PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE D_HORIZON D_SAMPLES D_PESSIMISM D_EXTRA_CRITICS D_CRITIC_PESSIMISM VAL_BEFORE_TRAIN; do
  [ -n "${!v:-}" ] && PASS="$PASS -e $v=${!v}"
done
eval docker run -d --rm --name "gorbov_${RUN_NAME:0:60}" --gpus "'\"device=$GPU_ID,$DREAMER_GPU\"'" --shm-size=32g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=gorbov_gv -e LOGNAME=gorbov_gv -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e TRITON_CACHE_DIR=/tmp/triton -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/tmp/torch -e NUMBA_CACHE_DIR=/tmp/numba \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$CHECKOUT:/w:ro" -v "$MODEL:/model:ro" -v "$R:$R" $MOUNTS \
  -e MODEL_PATH=/model -e RUN_ROOT="$R" -e GAME="$GAME" -e GPU_ID=0 -e DREAMER_GPU=1 $DENV $SMOKE_ENV $PASS \
  --entrypoint bash "$IMG" -c "'bash /w/vlm_pipeline/phase3_verl/craftext_integration/run_dyna_plan_grpo.sh >> $R/train.log 2>&1'"
echo "launched $RUN_NAME: veRL on GPU$GPU_ID, Dreamer on GPU$DREAMER_GPU, code $COMMIT, image $IMG, dir $R"
