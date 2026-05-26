#!/bin/bash
# Fast reinstall after editing caged_craftext / CrafText (no full docker rebuild).
set -e
cd "$(dirname "$0")/.."

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate verl-agent-311 2>/dev/null || true
fi
export CRAFTAX_RELOAD_TEXTURES=True

echo "=== Reinstall editable craftext + caged_craftext ==="
if [ -d agent_system/environments/env_package/craftext/CrafText-super_igor_env_build ]; then
  pip install --no-deps -e agent_system/environments/env_package/craftext/CrafText-super_igor_env_build
elif [ -d agent_system/environments/env_package/craftext/craftext ]; then
  pip install --no-deps -e agent_system/environments/env_package/craftext/craftext
fi
if [ -d caged_craftext ]; then
  pip install --no-deps -e caged_craftext
else
  echo "ERROR: caged_craftext/ not found"
  exit 1
fi
echo "=== Done ==="
