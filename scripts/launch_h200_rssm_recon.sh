#!/usr/bin/env bash
# Launch one RSSM+reconstruction training process on a chosen physical H200.
#
# The host has only Docker; the compatible Python/CUDA/JAX stack lives in
# safe_llm_h200_img's ``verl-agent-311`` Conda environment.  Keep the model,
# dataset, runs, and logs outside the container so that they survive restarts.
#
# Usage from the H200 checkout:
#   H200_GPU=6 bash scripts/launch_h200_rssm_recon.sh my_run_name
# Optional overrides:
#   H200_CONFIG=examples/latent_wm/<config>.yaml
#   COMET_API_KEY=...  (otherwise ~/.hosts_comet is read when present)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

host_gpu="${H200_GPU:-6}"
image="${H200_IMAGE:-safe_llm_h200_img}"
config="${H200_CONFIG:-examples/latent_wm/ds16_qwen_rssm_actiononly_h5_z5_reward_hzactor_betaKL01_h200.yaml}"
latent_root="${H200_LATENT_ROOT:-/data/homes/gorbov_gv/latent_wm}"
run_name="${1:-ds16_qwen_rssm_actiononly_h5_z5_reward_hzactor_betaKL01_h200_$(date +%Y%m%d_%H%M%S)}"

[[ -f "$config" ]] || { echo "ERROR: config not found: $config" >&2; exit 2; }
[[ -s "$latent_root/models/qwen2.5-1.5b-instruct/model.safetensors" ]] || {
  echo "ERROR: Qwen model is absent or incomplete under $latent_root/models" >&2
  exit 2
}
[[ -d "$latent_root/dataset_reward_collected_20261007_complete65" ]] || {
  echo "ERROR: RSSM dataset is missing under $latent_root" >&2
  exit 2
}
[[ ! -e "$latent_root/runs/$run_name" ]] || {
  echo "ERROR: run directory already exists: $latent_root/runs/$run_name" >&2
  exit 2
}
docker image inspect "$image" >/dev/null
nvidia-smi -i "$host_gpu" --query-gpu=name,memory.free --format=csv,noheader

# Do not print credentials.  ~/.hosts_comet is a single-line file maintained
# on the compute host; an explicitly supplied environment variable wins.
if [[ -z "${COMET_API_KEY:-}" && -r "$HOME/.hosts_comet" ]]; then
  COMET_API_KEY="$(tr -d '\r\n' < "$HOME/.hosts_comet")"
fi
[[ -n "${COMET_API_KEY:-}" ]] || {
  echo "ERROR: COMET_API_KEY is unset and ~/.hosts_comet is unavailable" >&2
  exit 2
}

mkdir -p "$latent_root/logs" "$latent_root/runs"
log_path="$latent_root/logs/${run_name}.log"
source_commit="$(git rev-parse HEAD)"

echo "Launching $run_name on physical H200 GPU $host_gpu"
echo "Config: $config"
echo "Log:    $log_path"

nohup docker run --rm --network host \
  --memory=200g --shm-size=16g --gpus "device=${host_gpu}" \
  --entrypoint /usr/home/workspace/docker/entrypoint.sh \
  -e COMET_API_KEY -e COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}" \
  -e COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl-agent-caged-craftext}" \
  -e SKIP_ALFWORLD_DATA_SETUP=1 \
  -v "$repo_root:/usr/home/workspace" \
  -v "$latent_root:$latent_root" \
  -w /usr/home/workspace \
  "$image" \
  python scripts/train_qwen_transition_rssm_recon.py \
    --config "$config" --run-name "$run_name" --device cuda:0 \
    --source-commit "$source_commit" \
  >"$log_path" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$latent_root/logs/${run_name}.host_pid"
echo "Started host PID $pid. Follow: tail -f $log_path"
