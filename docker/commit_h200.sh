#!/bin/bash
# Save current H200 container (with installed deps) into the image so
# `docker rm` + new `start_h200.sh` still has packages without re-download.
set -e
cd "$(dirname "$0")/.."

container_postfix=${1:-}
container_name=safe_llm_h200_${container_postfix}
image_name=safe_llm_h200_img

if ! docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
  echo "ERROR: container $container_name not found"
  exit 1
fi

echo "Committing $container_name -> $image_name ..."
docker commit "$container_name" "$image_name"
echo "Done. New starts from image keep gymnasium/jax/flash-attn/etc."
