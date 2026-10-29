#!/usr/bin/env bash
set -euo pipefail

docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=4 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --entrypoint /bin/bash \
  modern-verl-craftext:v091-smoke-craftext \
  -lc 'ls -l /dev/nvidia* || true; python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO_CUDA\")"'
