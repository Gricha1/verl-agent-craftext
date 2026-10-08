#!/usr/bin/env bash
# Launch a two-H200 online RSSM + dual-PPO run.  Two instances of this script
# form the reward/continuation-head ablation on four disjoint physical GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
image="${H200_IMAGE:-safe_llm_h200_img}"
gpus="${H200_GPUS:?set H200_GPUS to two physical GPU IDs, e.g. 0,1}"
config="${H200_CONFIG:?set H200_CONFIG}"
run_name="${1:?run name required}"
latent_root="${H200_LATENT_ROOT:-/data/homes/gorbov_gv/latent_wm}"
[[ "$gpus" =~ ^[0-9]+,[0-9]+$ ]] || { echo "H200_GPUS must be a pair like 0,1" >&2; exit 2; }
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
[[ ! -e "$latent_root/runs/$run_name" ]] || { echo "run already exists: $run_name" >&2; exit 2; }
docker image inspect "$image" >/dev/null
IFS=, read -r gpu_actor gpu_critic <<< "$gpus"
nvidia-smi -i "$gpu_actor,$gpu_critic" --query-gpu=index,name,memory.free --format=csv,noheader
if [[ -z "${COMET_API_KEY:-}" && -r "$HOME/.hosts_comet" ]]; then
  COMET_API_KEY="$(tr -d '\r\n' < "$HOME/.hosts_comet")"
fi
[[ -n "${COMET_API_KEY:-}" ]] || { echo "COMET API credential unavailable" >&2; exit 2; }
mkdir -p "$latent_root/logs" "$latent_root/runs"
log="$latent_root/logs/${run_name}.log"
commit="$(git rev-parse HEAD)"
echo "launching $run_name GPUs=$gpus config=$config"
# Docker's multi-device parser needs the inner double quotes preserved; without
# them it treats `0,1` as both a Count and a DeviceIDs request.
nohup docker run --rm --network host --memory=300g --shm-size=32g --gpus "\"device=$gpus\"" \
  --entrypoint /usr/home/workspace/docker/entrypoint.sh \
  -e COMET_API_KEY -e COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}" \
  -e SKIP_ALFWORLD_DATA_SETUP=1 \
  -e PYTHONPATH="/usr/home/workspace:/usr/home/workspace/caged_craftext:/usr/home/workspace/caged_craftext/Craftax" \
  -e CRAFTAX_RELOAD_TEXTURES=True \
  -v "$repo_root:/usr/home/workspace" -v "$latent_root:$latent_root" -w /usr/home/workspace "$image" \
  python scripts/train_online_rssm_dual_ppo.py --config "$config" --run-name "$run_name" \
    --actor-device cuda:0 --critic-device cuda:1 --source-commit "$commit" \
  >"$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$latent_root/logs/${run_name}.host_pid"
echo "started pid=$pid log=$log"
