#!/usr/bin/env bash
# Launch inside `bash -lic`/tmux on aicenter2 so its existing Comet login is inherited.
set -euo pipefail

if [[ $# -ge 2 ]]; then
  run_name=$1
  source_commit=$2
  shift 2
else
  run_name=${AE_RUN_NAME:?set AE_RUN_NAME or pass RUN_NAME SOURCE_COMMIT}
  source_commit=${AE_SOURCE_COMMIT:?set AE_SOURCE_COMMIT or pass RUN_NAME SOURCE_COMMIT}
fi

repo_root=/home/gorbov_gv/verl-agent-craftext
output_root=/home/gorbov_gv/latent_wm/runs
mkdir -p "$output_root"
exec >"$output_root/${run_name}.log" 2>&1

if [[ -z "${COMET_API_KEY:-}" ]]; then
  echo "COMET_API_KEY is not available in this login shell" >&2
  exit 1
fi

cd "$repo_root"
extra_args=("$@")
if [[ -n "${AE_SMOKE_STEPS:-}" ]]; then
  extra_args+=(--smoke-steps "$AE_SMOKE_STEPS")
fi

exec env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COMET_MODE=ONLINE \
  /home/gorbov_gv/latent_wm_min/bin/python scripts/train_qwen_latent_autoencoder.py \
  --config examples/latent_wm/ds16_qwen_latent_autoencoder_steps15000.yaml \
  --run-name "$run_name" \
  --source-commit "$source_commit" \
  "${extra_args[@]}"
