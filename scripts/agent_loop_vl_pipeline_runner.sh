#!/bin/bash
# Dynamic-interval loop: fast poll on crash/starting, 15m when training runs.
set -euo pipefail

TICK="${1:-}"
if [ -z "$TICK" ]; then
  TICK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agent_loop_vl_pipeline_tick.sh"
fi

# First tick immediately (no double-run: sleep comes after).
while true; do
  out="$("$TICK" 2>&1)" || out="AGENT_LOOP_TICK_vl_pipeline {\"status\":\"tick_failed\",\"action_required\":true,\"prompt\":\"tick script failed — inspect scripts/agent_loop_vl_pipeline_tick.sh\"}"
  echo "$out"
  status="$(echo "$out" | grep '^AGENT_LOOP_TICK_vl_pipeline' | tail -n1 | python3 -c "import sys,json; line=sys.stdin.read().split(' ',1)[1]; print(json.loads(line).get('status','unknown'))" 2>/dev/null || echo unknown)"
  action="$(echo "$out" | grep '^AGENT_LOOP_TICK_vl_pipeline' | tail -n1 | python3 -c "import sys,json; line=sys.stdin.read().split(' ',1)[1]; print(json.loads(line).get('action_required',False))" 2>/dev/null || echo False)"

  if [ "$action" = "True" ] || [ "$status" = "crashed" ] || [ "$status" = "idle_after_crash" ] || [ "$status" = "idle_shell" ]; then
    sleep 120
  elif [ "$status" = "starting" ]; then
    sleep 300
  else
    sleep 900
  fi
done
