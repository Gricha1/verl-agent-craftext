#!/usr/bin/env bash
# Launch the native verl-agent 16x16 text-reasoning baseline on exactly two H200s.
# Unlike the RSSM runs, its policy sees the real text observation: textual
# reasoning would be ill-defined for an H/Z-only policy without decoded text.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image="${H200_IMAGE:-safe_llm_h200_vllm092}"
gpus="${H200_GPUS:?set two physical H200 IDs, e.g. 6,7}"
run_name="${1:?run name required}"
runs_root="${H200_VERL_RUNS_ROOT:-/data/homes/gorbov_gv/verl_rssm_vllm_runs}"
model_dir="${H200_QWEN_MODEL_DIR:-/data/homes/gorbov_gv/latent_wm/models/qwen2.5-1.5b-instruct}"
config="examples/ppo_trainer/config/ppo_debug_square_16x16_actor_value_reasoning_gt_g099_ce_h200.yaml"

[[ "$gpus" =~ ^[0-9]+,[0-9]+$ ]] || { echo "H200_GPUS must contain exactly two IDs" >&2; exit 2; }
[[ -f "$config" && -f "$model_dir/config.json" ]] || { echo "missing config or local Qwen checkpoint" >&2; exit 2; }
docker image inspect "$image" >/dev/null
grep -Fx -- '  - reward_model.use_remaining_return_as_token_reward=True' "$config" >/dev/null
grep -Fx -- '  - reward_model.remaining_return_gamma=0.99' "$config" >/dev/null
grep -Fx -- '  - +env.prompt_template_type=single_token_action_reasoning' "$config" >/dev/null

if [[ -z "${COMET_API_KEY:-}" && -r "$HOME/.hosts_comet" ]]; then
  COMET_API_KEY="$(tr -d '\r\n' < "$HOME/.hosts_comet")"
fi
[[ -n "${COMET_API_KEY:-}" ]] || { echo "Comet credential unavailable" >&2; exit 2; }
export COMET_API_KEY

mkdir -p "$runs_root/logs" "$runs_root/checkpoints"
log="$runs_root/logs/${run_name}.log"
checkpoint_dir="$runs_root/checkpoints/$run_name"
[[ ! -e "$log" ]] || { echo "refusing to overwrite $log" >&2; exit 2; }
run_hash="$(printf '%s' "$run_name" | sha1sum | cut -c1-12)"
ray_temp_dir="/tmp/r${run_hash}"

echo "native verl-agent + vLLM reasoning launch"
echo "  H200s=$gpus; config=$config; run=$run_name"
nvidia-smi -i "$gpus" --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader

nohup docker run --rm --network host --memory=300g --shm-size=32g --gpus "\"device=$gpus\"" \
  --entrypoint /usr/home/workspace/docker/entrypoint.sh \
  -e COMET_API_KEY -e COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}" \
  -e COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl_agent_caged_craftext}" \
  -e SKIP_ALFWORLD_DATA_SETUP=1 -e CRAFTAX_RELOAD_TEXTURES= \
  -e CUDA_VISIBLE_DEVICES=0,1 -e N_GPUS=2 -e FORCE_NEW_RAY_CLUSTER=1 \
  -e RAY_TEMP_DIR="$ray_temp_dir" -e COMET_EXPERIMENT_KEY_FILE="$ray_temp_dir/comet_experiment_key.txt" \
  -e RUN_NAME="$run_name" -e CHECKPOINT_DIR="$checkpoint_dir" \
  -e PYTHONPATH="/usr/home/workspace:/usr/home/workspace/caged_craftext:/usr/home/workspace/caged_craftext/Craftax" \
  -v "$repo_root:/usr/home/workspace" -v "$runs_root:$runs_root" \
  -v "$model_dir:/models/qwen2.5-1.5b-instruct:ro" \
  -w /usr/home/workspace "$image" \
  bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value_reasoning.sh \
    reward_model.use_episode_return_as_token_reward=False \
    reward_model.use_remaining_return_as_token_reward=True \
    reward_model.remaining_return_gamma=0.99 \
    algorithm.gamma=1.0 algorithm.lam=1.0 algorithm.gae_by_trajectory=False \
    actor_rollout_ref.model.path=/models/qwen2.5-1.5b-instruct \
    critic.model.path=/models/qwen2.5-1.5b-instruct \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu=16 critic.ppo_mini_batch_size=64 \
    trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.critic_warmup=0 \
    trainer.resume_mode=disable trainer.save_freq=100 trainer.test_freq=20 \
    trainer.default_local_dir="$checkpoint_dir" \
  >"$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$runs_root/logs/${run_name}.host_pid"
echo "started host PID=$pid log=$log"
