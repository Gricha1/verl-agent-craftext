#!/usr/bin/env bash
# Run on one GPU with the Stage 1 reference files mounted as in the original evaluation.
set -euo pipefail
[[ $# == 6 ]] || {
    echo "Usage: $0 STAGE1_RUN TOKENIZER DYNAMICS HEADS SCALAR_STAGE2_CHECKPOINT OUTPUT_DIR" >&2
    exit 2
}
stage1=$1
tokenizer=$2
dynamics=$3
heads=$4
stage2=$5
out=$6
evaluator="$(dirname "${BASH_SOURCE[0]}")/eval_slm_d4_env.py"
critic="$stage1/run/checkpoints/step_060000.pt"
actor="$stage2/actor/model_world_size_1_rank_0.pt"
world="$stage2/dreamer_online.pt"
for path in "$critic" "$tokenizer" "$dynamics" "$heads" "$actor" "$world"; do
    test -s "$path" || { printf 'Missing input: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$out"
common=(--protocol crafter-all-128 --run "$stage1" --checkpoint step_060000.pt --critic "$critic"
        --tok "$tokenizer" --dyn "$dynamics" --heads "$heads")
for actor_stage in stage1 stage2; do
    for world_stage in stage1 stage2; do
        name="actor_${actor_stage}_dreamer_${world_stage}"
        weights=()
        if [[ $actor_stage == stage2 ]]; then weights+=(--verl-actor "$actor"); fi
        if [[ $world_stage == stage2 ]]; then weights+=(--joint-snapshot "$world"); fi
        "${PYTHON_BIN:-python3}" "$evaluator" "${common[@]}" "${weights[@]}" --out "$out/$name.json" \
            2>&1 | tee "$out/$name.log"
    done
done
