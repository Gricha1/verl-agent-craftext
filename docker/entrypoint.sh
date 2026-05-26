#!/bin/bash
# Run once per workspace (marker on host volume). Rebuild image only when base stack changes.
set -e

WORKSPACE="${WORKSPACE:-/usr/home/workspace}"
MARKER="${WORKSPACE}/.docker_caged_deps_installed"

cd "$WORKSPACE" || {
  echo "ERROR: mount repo at ${WORKSPACE} (e.g. -v \$(pwd):${WORKSPACE})"
  exit 1
}

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate verl-agent-311
fi
export CRAFTAX_RELOAD_TEXTURES=True

if [ "${SKIP_CAGED_CRAFTEXT_SETUP}" = "1" ]; then
  :
elif [ "${FORCE_CAGED_CRAFTEXT_SETUP}" = "1" ]; then
  echo "=== FORCE_CAGED_CRAFTEXT_SETUP: full deps install ==="
  bash setup_caged_craftext_deps.sh
  touch "$MARKER"
elif [ ! -f "$MARKER" ]; then
  echo "=== First run: installing caged_craftext deps from mounted repo (~15 min) ==="
  bash setup_caged_craftext_deps.sh
  touch "$MARKER"
else
  echo "=== caged_craftext deps OK (${MARKER}); skip full install ==="
  echo "    After caged_craftext edits: bash docker/setup_caged_craftext_editable.sh"
  echo "    Force full reinstall: FORCE_CAGED_CRAFTEXT_SETUP=1 docker/start.sh ..."
fi

if [ $# -eq 0 ]; then
  exec bash --login
fi
exec "$@"
