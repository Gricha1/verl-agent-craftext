#!/usr/bin/env bash
# Launch the existing verl/Ray/vLLM 16x16 PPO job on exactly three H200s.
# This wrapper does not implement a trainer: it only invokes the repository's
# ppo_debug_square_16x16_shared_av_gt_g099_ce.sh launcher.  The two ablation
# modes differ solely by verl's native trainer.world_model auxiliary update.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
image="${H200_IMAGE:-safe_llm_h200_img}"
gpus="${H200_GPUS:?set H200_GPUS to three physical GPU IDs, e.g. 0,1,2}"
mode="${H200_WM_MODE:?set H200_WM_MODE to reward or none}"
run_name="${1:?run name required}"
runs_root="${H200_VERL_RUNS_ROOT:-/data/homes/gorbov_gv/verl_runs}"

[[ "$gpus" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] || { echo "H200_GPUS must be a triple like 0,1,2" >&2; exit 2; }
[[ "$mode" == "reward" || "$mode" == "none" ]] || { echo "H200_WM_MODE must be reward or none" >&2; exit 2; }
docker image inspect "$image" >/dev/null

if [[ -z "${COMET_API_KEY:-}" && -r "$HOME/.hosts_comet" ]]; then
  COMET_API_KEY="$(tr -d '\r\n' < "$HOME/.hosts_comet")"
fi
[[ -n "${COMET_API_KEY:-}" ]] || { echo "COMET API credential unavailable" >&2; exit 2; }
export COMET_API_KEY

mkdir -p "$runs_root/logs" "$runs_root/checkpoints"
log="$runs_root/logs/${run_name}.log"
[[ ! -e "$log" ]] || { echo "run log already exists: $log" >&2; exit 2; }

wm_args=("trainer.world_model.enable=False")
if [[ "$mode" == "reward" ]]; then
  wm_args=(
    "trainer.world_model.enable=True"
    "trainer.world_model.task=reward"
    "trainer.world_model.loss_coef=0.1"
    "trainer.world_model.filter_unchanged_transitions=False"
  )
fi

checkpoint_dir="$runs_root/checkpoints/$run_name"
ray_temp_dir="/tmp/ray_${gpus//,/}${mode:0:1}"
echo "launching native verl PPO: run=$run_name mode=$mode GPUs=$gpus"
nohup docker run --rm --network host --memory=300g --shm-size=32g --gpus "\"device=$gpus\"" \
  --entrypoint /usr/home/workspace/docker/entrypoint.sh \
  -e COMET_API_KEY -e COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}" \
  -e COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl_agent_caged_craftext}" \
  -e SKIP_ALFWORLD_DATA_SETUP=1 -e CRAFTAX_RELOAD_TEXTURES=True \
  -e CUDA_VISIBLE_DEVICES=0,1,2 -e N_GPUS=3 -e FORCE_NEW_RAY_CLUSTER=1 \
  -e NUM_OPTIMISTIC_ENVS="${H200_NUM_OPTIMISTIC_ENVS:-48}" \
  -e PPO_MINI_BATCH_SIZE="${H200_PPO_MINI_BATCH_SIZE:-48}" \
  -e RAY_TEMP_DIR="$ray_temp_dir" \
  -e PYTHONPATH="/usr/home/workspace:/usr/home/workspace/caged_craftext:/usr/home/workspace/caged_craftext/Craftax" \
  -e RUN_NAME="$run_name" \
  -v "$repo_root:/usr/home/workspace" -v "$runs_root:$runs_root" \
  -w /usr/home/workspace "$image" \
  bash examples/ppo_trainer/ppo_debug_square_16x16_shared_av_gt_g099_ce.sh \
    "trainer.default_local_dir=$checkpoint_dir" "${wm_args[@]}" \
  >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$runs_root/logs/${run_name}.host_pid"
echo "started pid=$pid log=$log checkpoint_dir=$checkpoint_dir"
