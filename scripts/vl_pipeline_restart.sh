#!/bin/bash
# Restart current VL pipeline phase in tmux safe_vla (infra only, no hyperparam changes).
set -euo pipefail

STATE_FILE="${STATE_FILE:-/home/gorbov_gv/safe_rl_nlp/.cursor/vl_training_pipeline_state.json}"
TMUX_SESSION="${TMUX_SESSION:-safe_vla}"
WORKSPACE="${WORKSPACE:-/usr/home/workspace}"

phase="$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['phase'])")"
script="$(python3 -c "import json; s=json.load(open('$STATE_FILE')); print(s['scripts'][s['phase']])")"

ray stop --force 2>/dev/null || true
tmux send-keys -t "$TMUX_SESSION" C-c 2>/dev/null || true
sleep 2
tmux send-keys -t "$TMUX_SESSION" "cd $WORKSPACE && conda activate verl-agent-311 && export PYTHONPATH=\$PWD && bash $script" C-m

python3 - <<PY
import json, datetime, pathlib
p = pathlib.Path("$STATE_FILE")
s = json.loads(p.read_text())
s["last_restart_at"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
s["last_restart_script"] = "$script"
p.write_text(json.dumps(s, indent=2) + "\n")
print(f"[vl_pipeline_restart] phase=$phase script=$script")
PY
