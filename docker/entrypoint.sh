#!/bin/bash
# Runtime entrypoint:
# - base pip stack is in the image (docker/install_h200_base_deps.sh during build)
# - here we only install editable packages from the mounted repo (once per workspace)
set -e

WORKSPACE="${WORKSPACE:-/usr/home/workspace}"
MARKER="${WORKSPACE}/.docker_caged_deps_installed"

cd "$WORKSPACE" || {
  echo "ERROR: mount repo at ${WORKSPACE} (e.g. -v \$(pwd):${WORKSPACE})"
  exit 1
}

# The bind-mounted checkout is owned by the host user, while this entrypoint
# runs as root in the image.  Mark precisely this mount safe so training can
# record its source commit; do not use a broad ``safe.directory=*`` override.
git config --global --add safe.directory "$WORKSPACE" || true

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate verl-agent-311
fi
export CRAFTAX_RELOAD_TEXTURES=True

# Safety net: check base deps without importing JAX/Craftax.
# Importing JAX before AlfWorld's TextWorld workers fork can deadlock.
if ! python - <<'PY' >/dev/null 2>&1
import importlib.util as u
mods = ["gymnasium", "flash_attn", "vllm", "jax", "craftax"]
missing = [m for m in mods if u.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "=== Base deps missing in image — installing (docker/install_h200_base_deps.sh) ==="
  if [ -f docker/install_h200_base_deps.sh ]; then
    bash docker/install_h200_base_deps.sh
  else
    echo "ERROR: docker/install_h200_base_deps.sh not found on mount"
    exit 1
  fi
fi

install_workspace_editables() {
  echo "=== Editable installs from mounted repo ==="
  pip install -e . || echo "WARN: pip install -e . failed"
  pip install --no-deps -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build \
    || pip install --no-deps -e agent_system/environments/env_package/craftext/craftext \
    || echo "WARN: craftext editable install failed"
  if [ -d caged_craftext ]; then
    pip install --no-deps -e caged_craftext || echo "WARN: caged_craftext editable install failed"
  fi
  echo "=== Verify critical imports (fork-safe) ==="
  python - <<'PY'
import importlib.util as u
mods = ["gymnasium", "flash_attn", "jax"]
missing = [m for m in mods if u.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules: {missing}")
print("gymnasium/flash_attn/jax specs OK")
PY
}

if [ "${SKIP_CAGED_CRAFTEXT_SETUP}" = "1" ]; then
  :
elif [ "${FORCE_CAGED_CRAFTEXT_SETUP}" = "1" ]; then
  echo "=== FORCE_CAGED_CRAFTEXT_SETUP ==="
  install_workspace_editables
  touch "$MARKER"
elif [ ! -f "$MARKER" ]; then
  echo "=== First run: editable workspace deps ==="
  install_workspace_editables
  touch "$MARKER"
else
  echo "=== Workspace editables OK (${MARKER}); skip ==="
  echo "    Force again: FORCE_CAGED_CRAFTEXT_SETUP=1 bash docker/start_h200.sh"
fi

# AlfWorld game files (AlfredTWEnv). Persisted via host mount .cache/alfworld.
export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/.cache/alfworld}"
ensure_alfworld_python_deps() {
  # Match working server: pip alfworld==0.4.2 (+ textworld/termcolor). Vendored tree stays on PYTHONPATH.
  if python -c "import importlib.metadata as m, termcolor, textworld; assert m.version('alfworld')=='0.4.2'" >/dev/null 2>&1; then
    echo "=== AlfWorld python deps OK (alfworld==0.4.2, termcolor/textworld); skip ==="
    return 0
  fi
  echo "=== Installing AlfWorld python deps (alfworld==0.4.2, termcolor, textworld[pddl]) ==="
  pip install -q "alfworld==0.4.2" termcolor tqdm "textworld[pddl]>=1.6.1"
  python -c "import importlib.metadata as m, termcolor, textworld; print('alfworld', m.version('alfworld'), 'textworld', textworld.__version__, 'OK')"
}
ensure_alfworld_data() {
  local need=0
  if [ ! -d "${ALFWORLD_DATA}/json_2.1.1/train" ]; then
    need=1
  elif ! find "${ALFWORLD_DATA}/json_2.1.1/train" -name 'game.tw-pddl' -print -quit 2>/dev/null | grep -q .; then
    need=1
  fi
  if [ "$need" = "0" ]; then
    echo "=== AlfWorld data OK (${ALFWORLD_DATA}); skip ==="
    return 0
  fi
  echo "=== AlfWorld data missing — downloading (scripts/download_alfworld_data.sh) ==="
  bash scripts/download_alfworld_data.sh
}

if [ "${SKIP_ALFWORLD_DATA_SETUP}" = "1" ]; then
  echo "=== SKIP_ALFWORLD_DATA_SETUP=1 — not checking AlfWorld data ==="
else
  ensure_alfworld_python_deps
  ensure_alfworld_data
fi

if [ $# -eq 0 ]; then
  exec bash --login
fi
exec "$@"
