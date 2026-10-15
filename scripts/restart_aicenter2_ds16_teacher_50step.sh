#!/usr/bin/env bash
# Restart only the 16x16 CNN+DistilBERT PPO teacher with the veRL-equivalent
# 50-action external episode cap.  Does not touch the wood-achievements run.
set -euo pipefail

canonical=/home/gorbov_gv/safe_rl_nlp
legacy=/home/gorbov_gv/craftext_cnn_ppo_expert_v1
python_bin="$legacy/venv/bin/python"
run_name=cnn_bert_mlp_ppo_debug_square16_multitask_a100gpu0_20261015_r2_ep50
out="$canonical/runs/$run_name"
export PYTHONPATH="$canonical/caged_craftext:$canonical/caged_craftext/Craftax:$canonical/caged_craftext/baselines${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$out"
cd "$legacy"
CUDA_VISIBLE_DEVICES=0 nohup "$python_bin" -u "$canonical/caged_craftext/baselines/train_teacher_multitask.py" \
  --run-name "$run_name" \
  --out-dir "$out" \
  --craftext-settings debug_square_16x16 \
  --max-episode-length 50 \
  >"$out/train.log" 2>&1 < /dev/null &
echo $! >"$out/pid"
printf 'STARTED run=%s pid=%s output=%s\n' "$run_name" "$(cat "$out/pid")" "$out"
