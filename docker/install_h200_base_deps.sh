#!/bin/bash
# Base H200 pip stack (no repo mount needed). Used by Dockerfile.H200 and as start fallback.
# Never compiles flash-attn from source.
set -euo pipefail

export CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate verl-agent-311
fi

echo "=== H200 base deps (torch env: $(python -c 'import torch; print(torch.__version__)') ) ==="
pip install -U pip setuptools wheel

pip install \
  psutil packaging datasets tensorboard comet_ml \
  "ray[default]" msgspec cachetools openai \
  gym==0.26.2 gymnasium \
  "jax==0.4.30" "jaxlib==0.4.30" "flax==0.10.4" "chex==0.1.90" \
  "optax==0.2.5" "orbax-checkpoint==0.6.4" "distrax==0.1.5" \
  "tensorstore==0.1.78" "ml_dtypes==0.5.3" \
  imageio matplotlib seaborn pygame gymnax treescope \
  pandas scikit-learn scipy \
  "tensordict<=0.6.2" torchdata "transformers==4.51.1" \
  accelerate codetiming dill hydra-core peft "pyarrow>=19.0.0" \
  pybind11 pylatexenc wandb uvicorn fastapi "qwen-vl-utils[decord]"

pip install --no-deps craftax==1.4.3

# flash-attn: local wheel preferred, else download (no source build)
WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL}"
CANDIDATES=(
  "${FLASH_ATTN_WHEEL:-}"
  "/opt/wheels/${WHL}"
  "/usr/home/workspace/.wheels/${WHL}"
  "$(pwd)/.wheels/${WHL}"
  "/tmp/${WHL}"
)
FOUND=""
for p in "${CANDIDATES[@]}"; do
  [ -n "$p" ] && [ -f "$p" ] && FOUND="$p" && break
done
if [ -z "$FOUND" ]; then
  echo "Downloading ${WHL} ..."
  mkdir -p /opt/wheels
  curl -fL --retry 8 --retry-delay 3 --connect-timeout 30 "$URL" -o "/opt/wheels/${WHL}"
  FOUND="/opt/wheels/${WHL}"
fi
echo "Installing flash-attn from ${FOUND}"
pip install --no-deps "$FOUND"

# vllm matches safe_rl_nlp / setup_conda_venv.sh
pip install "vllm==0.8.5"

echo "=== Verify base deps ==="
python - <<'PY'
import gymnasium, jax, flash_attn, craftax, imageio, vllm, tensordict, transformers
print("gymnasium", gymnasium.__version__)
print("jax", jax.__version__)
print("flash_attn", flash_attn.__version__)
print("vllm", vllm.__version__)
print("tensordict", tensordict.__version__)
print("transformers", transformers.__version__)
print("craftax OK")
PY
echo "=== H200 base deps OK ==="
