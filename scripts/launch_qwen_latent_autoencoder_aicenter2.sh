#!/usr/bin/env bash
# Launch inside `bash -lic`/tmux on aicenter2 so its existing Comet login is inherited.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_NAME SOURCE_COMMIT [trainer args...]" >&2
  exit 2
fi

run_name=$1
source_commit=$2
shift 2

repo_root=/home/gorbov_gv/verl-agent-craftext
output_root=/home/gorbov_gv/latent_wm/runs
mkdir -p "$output_root"

if [[ -z "${COMET_API_KEY:-}" ]]; then
  echo "COMET_API_KEY is not available in this login shell" >&2
  exit 1
fi

exec >"$output_root/${run_name}.log" 2>&1
cd "$repo_root"
exec env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COMET_MODE=ONLINE \
  /home/gorbov_gv/latent_wm_min/bin/python scripts/train_qwen_latent_autoencoder.py \
  --config examples/latent_wm/ds16_qwen_latent_autoencoder_steps15000.yaml \
  --run-name "$run_name" \
  --source-commit "$source_commit" \
  "$@"
