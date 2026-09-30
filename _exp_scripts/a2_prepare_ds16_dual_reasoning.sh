#!/usr/bin/env bash
# aicenter2 runtime for 16x16 DUAL PPO with reasoning, single GPU (host GPU 3).
set -euo pipefail

REPO=${REPO:-/storage/gorbov_gv/safe_rl_nlp}
IMAGE=${IMAGE:-verlai/verl:vllm011.latest}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/dev/shm/ray16}
HF_HOME=${HF_HOME:-/storage/gorbov_gv/models/hf}
MODEL_PATH=${MODEL_PATH:-/storage/gorbov_gv/models/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
DATA_DIR=${DATA_DIR:-/storage/gorbov_gv/data/verl-agent}

cd "$REPO"
mkdir -p "$RAY_TEMP_DIR" "$REPO/logdir" "$REPO/training_checkpoints"

echo "[preflight] selected GPUs: $CUDA_VISIBLE_DEVICES"
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  nvidia-smi -i "$gpu" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
done
echo "[preflight] Ray temp: $RAY_TEMP_DIR"
df -h /dev/shm / "$REPO" | head -10

cat <<'EOF'
[READY] 16x16 DUAL with reasoning runtime is wired (single GPU).

To launch after explicitly confirming the GPU allocation, use:
  CONFIRM_LAUNCH=1 bash _exp_scripts/a2_prepare_ds16_dual_reasoning.sh

The container installs only environment-side packages (Gym/JAX/Craftax stack)
at runtime. It does not upgrade Torch, Transformers, vLLM, or xFormers.
EOF

if [[ ${CONFIRM_LAUNCH:-0} != 1 ]]; then
  exit 0
fi

docker run --rm --cap-add=SYS_PTRACE --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" --ipc=host --network host --shm-size=32g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e JAX_PLATFORMS=cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e CRAFTAX_RELOAD_TEXTURES="${CRAFTAX_RELOAD_TEXTURES:-True}" \
  -e RAY_TEMP_DIR="$RAY_TEMP_DIR" \
  -e HF_DATASETS_CACHE=/dev/shm/hf_datasets \
  -e RAY_FORK_CONTEXT=spawn -e VLLM_WORKER_MULTIPROCESSING_START_METHOD=spawn \
  -e HF_DATASETS_NUM_PROC=1 \
  -e HF_HOME="$HF_HOME" -e HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  -e COMET_API_KEY -e COMET_WORKSPACE -e COMET_PROJECT_NAME \
  -e COMET_DISABLED -e VALIDATION_ONLY -e MODEL_PATH="$MODEL_PATH" \
  -v "$REPO:/workspace" -v "$HF_HOME:$HF_HOME:ro" -v "$DATA_DIR:/root/data/verl-agent:ro" \
  -v "$RAY_TEMP_DIR:$RAY_TEMP_DIR" -w /workspace --entrypoint bash "$IMAGE" -lc '
    pip install --no-cache-dir \
      gym==0.26.2 jax==0.4.30 jaxlib==0.4.30 flax==0.10.4 chex==0.1.90 \
      optax==0.2.5 orbax-checkpoint==0.6.4 distrax==0.1.5 tensorstore==0.1.78 \
      ml_dtypes==0.5.3 gymnax==0.0.8 imageio pillow
    extra_args=(actor_rollout_ref.model.path="$MODEL_PATH" critic.model.path="$MODEL_PATH")
    if [ "${VALIDATION_ONLY:-0}" = "1" ]; then
      extra_args=(trainer.val_only=True trainer.val_before_train=True trainer.logger=[console])
    fi
    HYPER_YAML=examples/ppo_trainer/config/ppo_debug_square_16x16_dual_reasoning_grounding.yaml \
      bash examples/ppo_trainer/ppo_debug_square_16x16_reasoning.sh "${extra_args[@]}"
  '
