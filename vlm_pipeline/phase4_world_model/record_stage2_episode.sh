#!/usr/bin/env bash
# Submitted by the local experiment logger. The detached container records the episode on disk.
set -euo pipefail
job=$1
variant=$2
seed=$3
limit=$4
rev=$5
[[ $job =~ ^[a-f0-9]{12}$ && $seed =~ ^[0-9]{1,9}$ && $limit =~ ^[0-9]{1,4}$ && $rev =~ ^[a-f0-9]{40}$ ]]
((limit >= 1 && limit <= 1000))
case "$variant" in
    hl80) mode=hlgauss; step=80 ;;
    hl100) mode=hlgauss; step=100 ;;
    scalar80) mode=scalar; step=80 ;;
    stage1) mode=scalar; step=0 ;;
    *) exit 2 ;;
esac

name=gorbov_crafter_recording_$job
out="$HOME/runs/crafter_episode_recordings/$job"
mkdir -p "$out/code"
if docker inspect "$name" >/dev/null 2>&1; then
    exit 0
fi
repo="$HOME/verl-agent-craftext"
if ! git -C "$repo" cat-file -e "$rev^{commit}" 2>/dev/null; then
    git -C "$repo" fetch origin vlm-ppo-distill-rl
fi
git -C "$repo" archive "$rev" vlm_pipeline/phase4_world_model | tar -x -C "$out/code"
mirror="$HOME/a3mirror/run"
world="$mirror/dreamer4_qwen_crafter64_a3gpu3_20261118_r1"
stage1="$mirror/slm5m_plan16_value_d4qwen_crafter64_r64_scratch_a3gpu1_20261119_r1"
run="$HOME/runs/onegpu_grpo_stage2_joint_${mode}_20261123_r1"
checkpoint="$run/checkpoints/global_step_$step"
critic=/s0/run/checkpoints/step_060000.pt
if [[ $mode == hlgauss ]]; then
    critic=/runs/slm5m_critic_hl_d4qwen_crafter64_from_plan16_a3gpu7_20261121_r1/run/best.pt
fi
mounts=()
weights=()
if ((step)); then
    test -s "$checkpoint/dreamer_online.pt"
    test -s "$checkpoint/actor/model_world_size_1_rank_0.pt"
    mounts=(-v "$checkpoint:/checkpoint:ro")
    weights=(--joint-snapshot /checkpoint/dreamer_online.pt --verl-actor /checkpoint/actor/model_world_size_1_rank_0.pt)
fi
docker run -d --name "$name" --gpus device=0 --memory=8g --user "$(id -u):$(id -g)" \
    -v "$mirror:/runs:ro" -v "$world:/r:ro" -v "$world:/q4:ro" -v "$stage1:/s0:ro" \
    -v "$out:/record" -v "$mirror/slm5m_recon_d4wood2200_init_20261118:/c:ro" \
    -v "$mirror/crafter_dyna_libs_20261119:/cl:ro" \
    -v "$mirror/dreamer_eval_wood2200_a3gpu3_20261112_r1/pylibs:/p/pylibs:ro" \
    -v "$HOME/a3mirror/data/zerohero_crafter_memory_20261106/crafter64/train:/q/train:ro" \
    -v "$HOME/runs/slm5m_plan16_crafter_dyna_grpo_tokft_criticft_d4qwen_itlgpu1_20261121_r1/dreamer4:/code/dreamer4:ro" \
    "${mounts[@]}" -e HOME=/tmp -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" \
    -e PYTHONPATH=/cl:/p/pylibs:/code/dreamer4 -e NUMBA_CACHE_DIR=/tmp/numba \
    -e PYTHONUNBUFFERED=1 -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    --entrypoint python3 modern-verl-craftext:deps-17e7203 \
    /record/code/vlm_pipeline/phase4_world_model/eval_slm_d4_env.py \
    --run /s0 --checkpoint step_060000.pt --critic "$critic" "${weights[@]}" \
    --game crafter --decode topk --temperature 1 --top-k 20 --plans 8 --imagine \
    --score reward --horizon 8 --K 1 --seed 0 --seeds "$seed:$((seed + 1))" --max-steps "$limit" \
    --dyn /r/dyn/checkpoints/latest.pt \
    --heads /runs/dreamer4_heads_qwen_crafter64_reward_a3gpu3_20261119_r1/heads/checkpoints/latest.pt \
    --out /record/summary.json --record-episode /record/episode
