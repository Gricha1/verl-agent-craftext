#!/bin/bash
# Build H200 image with base deps baked in (jax/gymnasium/flash-attn/vllm/...).
# On another machine: this script downloads the flash-attn wheel if .wheels/ is empty.
set -euo pipefail
cd "$(dirname "$0")/.."

WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL}"
mkdir -p .wheels
if [ ! -f ".wheels/${WHL}" ]; then
  echo "Downloading flash-attn wheel into .wheels/ (needed for docker build context)..."
  wget -c -O ".wheels/${WHL}" "$URL" || curl -fL --retry 8 --retry-delay 3 -o ".wheels/${WHL}" "$URL"
fi
ls -lh ".wheels/${WHL}"

# Keep an empty placeholder so COPY .wheels/ never fails if only other files exist
touch .wheels/.keep

echo "Building safe_llm_h200_img (this installs jax/vllm/etc into the image)..."
docker build -f docker/Dockerfile.H200 -t safe_llm_h200_img .
echo "Done. Start with: bash docker/start_h200.sh"
