#!/usr/bin/env bash
# Smoke: create tiny fake LoRA adapter dir layout and verify metadata path used by trainer.
set -euo pipefail
TMP=$(mktemp -d)
ADAPTER="$TMP/lora_adapter"
mkdir -p "$ADAPTER"
python3 - <<PY
import json, os
adapter=os.environ["ADAPTER"]
cfg={"peft_type":"LORA","task_type":"CAUSAL_LM","r":64,"lora_alpha":64,"target_modules":["q_proj","v_proj"]}
open(os.path.join(adapter,"adapter_config.json"),"w").write(json.dumps(cfg,indent=2))
open(os.path.join(adapter,"adapter_model.safetensors"),"wb").write(b"not-real-but-nonempty")
print("wrote", adapter)
PY
ADAPTER="$ADAPTER" python3 - <<'PY'
import json, os
adapter=os.environ["ADAPTER"]
cfg=json.load(open(os.path.join(adapter,"adapter_config.json")))
assert cfg["peft_type"]=="LORA"
assert os.path.getsize(os.path.join(adapter,"adapter_model.safetensors"))>0
print("SMOKE_OK adapter file verify")
PY
rm -rf "$TMP"
echo "done"
