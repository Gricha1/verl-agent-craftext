#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all --entrypoint /bin/bash modern-verl-craftext:v091-smoke-craftext -lc \
  'nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
