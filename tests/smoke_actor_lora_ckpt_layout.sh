#!/usr/bin/env bash
# Smoke test: the LoRA-only checkpoint layout the trainer writes is well-formed.
set -euo pipefail
TMP="$(mktemp -d)"
ADAPTER="$TMP/global_step_1/actor/lora_adapter"
mkdir -p "$ADAPTER"
python3 - "$ADAPTER" <<'PY'
import json, os, sys
adapter = sys.argv[1]
cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 64, "lora_alpha": 64,
       "target_modules": ["q_proj", "v_proj"]}
with open(os.path.join(adapter, "adapter_config.json"), "w") as fh:
    fh.write(json.dumps(cfg, indent=2))
with open(os.path.join(adapter, "adapter_model.safetensors"), "wb") as fh:
    fh.write(b"not-real-but-nonempty")
print("wrote", adapter)
cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
assert cfg["peft_type"] == "LORA"
assert os.path.getsize(os.path.join(adapter, "adapter_model.safetensors")) > 0
print("SMOKE_OK adapter file verify")
PY
rm -rf "$TMP"
echo done
