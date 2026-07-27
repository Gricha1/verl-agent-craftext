#!/bin/bash
# Start H200 container with GPUs 6 and 7 (last two on 8-GPU node).
# Base deps (jax/gymnasium/flash-attn/vllm) come from the image: bash docker/build_h200.sh
# First start only does editable installs from the mounted repo (marker: .docker_caged_deps_installed).
#
# IMPORTANT: --network host is required — Docker bridge breaks HTTPS to Comet/HF (SSL handshake timeout).
set -e
cd "$(dirname "$0")/.."

container_postfix=${1:-}
image_name=safe_llm_h200_img
container_name=safe_llm_h200_${container_postfix}
gpus="device=6,7"

echo "image name --- $image_name"
mkdir -p logdir
echo "container name --- $container_name"
echo "gpus in docker --- $gpus"

# Reuse existing container only if it already has host networking (Comet needs it).
if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
  net=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$container_name" 2>/dev/null || echo unknown)
  if [ "$net" = "host" ]; then
    echo "reusing existing container (network=host). Fresh: docker rm -f $container_name && bash docker/start_h200.sh"
    exec docker start -ai "$container_name"
  fi
  echo "Existing container network=$net (Comet HTTPS broken on bridge). Recreating with --network host..."
  docker rm -f "$container_name"
fi

# SKIP_CAGED_CRAFTEXT_SETUP=1 — skip editable installs
# FORCE_CAGED_CRAFTEXT_SETUP=1 — redo editable installs
# SKIP_ALFWORLD_DATA_SETUP=1 — skip AlfWorld dataset download on entry
# Host GPUs 6,7 → inside container they appear as cuda:0 and cuda:1
mkdir -p "$(pwd)/.cache/alfworld"
docker run -it --name "$container_name" --network host \
  --memory="200g" --shm-size=8g --gpus '"device=6,7"' \
  --entrypoint /usr/home/workspace/docker/entrypoint.sh \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env COMET_API_KEY="${COMET_API_KEY:-}" \
  --env COMET_WORKSPACE="${COMET_WORKSPACE:-gregory-gorbov}" \
  --env COMET_PROJECT_NAME="${COMET_PROJECT_NAME:-verl_agent_caged_craftext}" \
  --env SKIP_CAGED_CRAFTEXT_SETUP="${SKIP_CAGED_CRAFTEXT_SETUP:-}" \
  --env FORCE_CAGED_CRAFTEXT_SETUP="${FORCE_CAGED_CRAFTEXT_SETUP:-}" \
  --env SKIP_ALFWORLD_DATA_SETUP="${SKIP_ALFWORLD_DATA_SETUP:-}" \
  --env ALFWORLD_DATA=/root/.cache/alfworld \
  -v "$(pwd)":/usr/home/workspace \
  -v "$(pwd)/logdir":/root/logdir \
  -v "$(pwd)/.cache/alfworld":/root/.cache/alfworld \
  "$image_name"
