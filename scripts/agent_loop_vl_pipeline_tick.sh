#!/bin/bash
# Detect VL training state, save crash logs, auto-restart on idle-after-crash, wake agent.
set -euo pipefail

STATE_FILE="${STATE_FILE:-/home/gorbov_gv/safe_rl_nlp/.cursor/vl_training_pipeline_state.json}"
CRASH_FILE="${CRASH_FILE:-/home/gorbov_gv/safe_rl_nlp/.cursor/vl_last_crash.txt}"
TMUX_SESSION="${TMUX_SESSION:-safe_vla}"
SUCCESS_HOURS="${SUCCESS_HOURS:-3}"
SUCCESS_SEC=$((SUCCESS_HOURS * 3600))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$(dirname "$STATE_FILE")" "$(dirname "$CRASH_FILE")"
if [ ! -f "$STATE_FILE" ]; then
  python3 - <<'PY'
import json, datetime, pathlib
p = pathlib.Path("/home/gorbov_gv/safe_rl_nlp/.cursor/vl_training_pipeline_state.json")
p.write_text(json.dumps({
    "phase": "actor_value_vl",
    "phase_started_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "scripts": {
        "actor_value_vl": "examples/ppo_trainer/ppo_debug_square_actor_value_vl.sh",
        "dual_vl": "examples/ppo_trainer/ppo_debug_square_vl.sh",
        "valid_ent_vl": "examples/ppo_trainer/ppo_debug_square_valid_ent_vl.sh",
    },
    "success_hours": 3,
    "restart_count": 0,
}, indent=2) + "\n")
PY
fi

pane="$(tmux capture-pane -t "$TMUX_SESSION" -p -S -300 2>/dev/null || echo 'NO_SESSION')"
tail="$(echo "$pane" | tail -n 100)"

status="unknown"
action_required="false"
auto_restarted="false"

# Crash signatures — tail only so old errors in scrollback don't false-positive while running.
if echo "$tail" | grep -qE 'Traceback \(most recent|OutOfMemoryError|AssertionError|RayTaskError|ray\.exceptions\.|CUDA error: out of memory|ValueError: Attempted to assign|expandable segments are not compatible|HYDRA_FULL_ERROR=1 for a complete'; then
  if echo "$tail" | tail -n 8 | grep -qE 'workspace#|\(verl-agent-311\).*#'; then
    status="idle_after_crash"
    action_required="true"
  else
    status="crashed"
    action_required="true"
  fi
elif echo "$tail" | grep -qE 'Training Progress:[[:space:]]*[1-9]|global_step=[2-9][0-9]*|global_step=[1-9][0-9]{2,}|\[PPO phase.*global_step=[2-9]'; then
  status="running"
elif echo "$tail" | grep -qE 'Training Progress|global_step=1|rollout train|generate_sequences|ROLLOUT:'; then
  status="running"
elif echo "$tail" | grep -qE 'Total training steps|register_center|Loading checkpoint|WorkerDict|Comet INFO'; then
  status="starting"
elif echo "$tail" | grep -qE 'workspace#|\(verl-agent-311\).*#'; then
  status="idle_shell"
  action_required="true"
fi

if [ "$status" = "idle_after_crash" ] || [ "$status" = "crashed" ]; then
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status ==="
    echo "$tail"
  } > "$CRASH_FILE"
  crash_sig="$(echo "$tail" | grep -E 'Error:|Exception:|AssertionError|OutOfMemory|ValueError|Traceback' | tail -n 3 | sha256sum | awk '{print $1}')"
  last_sig="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('last_crash_sig',''))" 2>/dev/null || echo "")"
  restart_count="$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('restart_count',0))" 2>/dev/null || echo 0)"

  # Auto-restart once per unique crash (agent still woken to fix root cause if it repeats).
  if [ "$crash_sig" != "$last_sig" ] || [ "$restart_count" -lt 1 ]; then
    bash "$SCRIPT_DIR/vl_pipeline_restart.sh" || true
    auto_restarted="true"
    python3 - <<PY
import json, pathlib
p = pathlib.Path("$STATE_FILE")
s = json.loads(p.read_text())
s["last_crash_sig"] = "$crash_sig"
s["restart_count"] = int(s.get("restart_count", 0)) + 1
s["last_auto_restart_at"] = __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
p.write_text(json.dumps(s, indent=2) + "\n")
PY
  fi
fi

phase="$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['phase'])")"
script="$(python3 -c "import json; s=json.load(open('$STATE_FILE')); print(s['scripts'][s['phase']])")"
elapsed="$(python3 -c "import json,datetime; s=json.load(open('$STATE_FILE')); t=datetime.datetime.fromisoformat(s['phase_started_at'].replace('Z','+00:00')); print(int((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()))")"
ready_3h="false"
if [ "$elapsed" -ge "$SUCCESS_SEC" ] && [ "$status" = "running" ]; then
  ready_3h="true"
  action_required="true"
fi

export TICK_PHASE="$phase" TICK_SCRIPT="$script" TICK_STATUS="$status" TICK_ELAPSED="$elapsed"
export TICK_READY_3H="$ready_3h" TICK_ACTION="$action_required" TICK_AUTO="$auto_restarted"
export TICK_CRASH_FILE="$CRASH_FILE"

python3 - <<'PY'
import json, os, pathlib
phase = os.environ["TICK_PHASE"]
script = os.environ["TICK_SCRIPT"]
status = os.environ["TICK_STATUS"]
elapsed = int(os.environ["TICK_ELAPSED"])
ready_3h = os.environ["TICK_READY_3H"] == "true"
action_required = os.environ["TICK_ACTION"] == "true"
auto_restarted = os.environ["TICK_AUTO"] == "true"
crash_file = os.environ["TICK_CRASH_FILE"]
crash_tail = ""
if pathlib.Path(crash_file).exists():
    crash_tail = pathlib.Path(crash_file).read_text()[-2500:]

if action_required:
    prompt = (
        f"ACTION REQUIRED. VL pipeline phase={phase}, status={status}, script={script}. "
        "Read .cursor/vl_last_crash.txt if present. Fix CODE (infra only, no train hyperparams: "
        "no env/batch/minibatch/clip/lr changes). Then ray stop and restart script in tmux safe_vla. "
        f"auto_restarted={auto_restarted}. "
        f"If ready_3h={ready_3h} and status=running: stop training, advance phase in "
        ".cursor/vl_training_pipeline_state.json (actor_value_vl->dual_vl->valid_ent_vl), restart next script."
    )
else:
    prompt = (
        f"VL pipeline OK check: phase={phase}, status={status}, elapsed_s={elapsed}. "
        "No action unless status changes. Do not touch hyperparams."
    )

print("AGENT_LOOP_TICK_vl_pipeline", json.dumps({
    "phase": phase,
    "script": script,
    "status": status,
    "elapsed_s": elapsed,
    "ready_3h": ready_3h,
    "action_required": action_required,
    "auto_restarted": auto_restarted,
    "crash_excerpt": crash_tail[-800:] if crash_tail else "",
    "prompt": prompt,
}, ensure_ascii=False))
PY
