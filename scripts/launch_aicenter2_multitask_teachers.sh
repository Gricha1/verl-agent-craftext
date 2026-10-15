#!/usr/bin/env bash
# Run on aicenter2.  Uses the canonical git checkout; the old snapshot is only
# the existing virtualenv/HF cache, never the Python source being executed.
set -euo pipefail

canonical=/home/gorbov_gv/safe_rl_nlp
legacy=/home/gorbov_gv/craftext_cnn_ppo_expert_v1
python_bin="$legacy/venv/bin/python"
source_root="$canonical/caged_craftext"
runner="$source_root/baselines/train_teacher_multitask.py"
export PYTHONPATH="$source_root:$source_root/Craftax:$source_root/baselines${PYTHONPATH:+:$PYTHONPATH}"
export COMET_WORKSPACE=gregory-gorbov
export COMET_PROJECT=verl-agent-caged-craftext

CUDA_VISIBLE_DEVICES='' "$python_bin" - <<'PY'
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
print('PRELAUNCH_RUNTIME_IMPORTS_OK', torch.__version__, transformers.__version__)
PY

start() {
  local gpu=$1 name=$2 config=$3
  local out="$canonical/runs/$name"
  mkdir -p "$out"
  # The legacy checkout is cwd exclusively because the frozen DistilBERT cache
  # was previously downloaded there.  $runner and PYTHONPATH remain canonical.
  (
    cd "$legacy"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -u "$runner" \
      --run-name "$name" --out-dir "$out" --craftext-settings "$config" \
      >"$out/train.log" 2>&1 < /dev/null &
    echo $! >"$out/pid"
  )
  printf 'STARTED gpu=%s name=%s pid=%s\n' "$gpu" "$name" "$(cat "$out/pid")"
}

start 0 cnn_bert_mlp_ppo_debug_square16_multitask_a100gpu0_20261015_r1 debug_square_16x16
start 1 cnn_bert_mlp_ppo_wood_all_a100gpu1_20261015_r1 achievements_wood
