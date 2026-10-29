# Shared settings for the aicenter3 launchers (sourced, not executed).
#
# aicenter3 docker wrapper: the GPU comes from the caller's CUDA_VISIBLE_DEVICES,
# --gpus/--device are stripped and "-e VAR=val" breaks the command, so variables
# go through --env-file. NFS is root-squashed: run directories are pre-created
# with mode 777, and jobs that write into user-owned trees run as --user 1007:100.

CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${BASE:-/mnt/aicenter1-datasets/gorbov_gv/modern_verl_reasoning_rl_a3_20261016}"
RUNS="${RUNS:-${BASE}/run}"
JAX_IMAGE="${JAX_IMAGE:-modern-verl-craftext:jax0430-ppo-r4-collector}"
TORCH_IMAGE="${TORCH_IMAGE:-modern-verl-craftext:v091-smoke-craftext}"
VERL_IMAGE="${VERL_IMAGE:-modern-verl-craftext:v091-smoke-craftext-trl027}"
# A directory holding models--distilbert-base-uncased: the CrafText encoder uses cache_dir='.'.
DISTILBERT_CACHE="${DISTILBERT_CACHE:-${RUNS}/ppo_mlp_distilbert_cnn_achievements_all_gpu3_20261019_r4}"

# Project rule: our jobs on aicenter3 use only GPU2 and GPU3.
require_our_gpu() {
  case "${GPU_ID:-}" in
    2|3) ;;
    *) echo "GPU_ID must be 2 or 3 (aicenter3 rule), got '${GPU_ID:-}'" >&2; exit 1 ;;
  esac
}

new_run_dir() {
  local dir="$1"
  if [ -e "${dir}" ]; then echo "Refusing to reuse existing run directory ${dir}" >&2; exit 1; fi
  mkdir -p "${dir}" && chmod 777 "${dir}"
}

# Writable cwd with the DistilBERT cache, for anything that builds a CrafText env.
prepare_bert_cwd() {
  local dir="$1"
  mkdir -p "${dir}"
  if [ ! -d "${dir}/models--distilbert-base-uncased" ]; then
    cp -r "${DISTILBERT_CACHE}/models--distilbert-base-uncased" "${dir}/"
  fi
  chmod -R a+rwX "${dir}"
}

write_user_env() {
  # Environment for --user 1007:100 containers (getpass needs USER/LOGNAME).
  printf 'USER=gorbov_gv\nLOGNAME=gorbov_gv\nHOME=/tmp\nPYTHONUNBUFFERED=1\nXLA_PYTHON_CLIENT_PREALLOCATE=false\n%s' "${2:-}" > "$1"
}

record_code_version() {
  # Exact code provenance next to every run.
  {
    echo "checkout=${CHECKOUT}"
    git -C "${CHECKOUT}" rev-parse HEAD
    git -C "${CHECKOUT}" rev-parse --abbrev-ref HEAD
    git -C "${CHECKOUT}" status --short
  } > "$1/code_version.txt" 2>&1
}
