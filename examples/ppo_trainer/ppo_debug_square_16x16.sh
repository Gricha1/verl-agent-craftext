#!/bin/bash
# PPO on debug_square_16x16: dual actor+critic LLMs.
#
# Same as working 8x8 dual:
# - history_length=50 (executed actions in prompt)
# - Token score = per-step env reward r_t (not G_t / not full episode R)
# - gae_by_trajectory=False (response_len=1 → returns≈r_t)
#
# Hyperparams: examples/ppo_trainer/config/ppo_debug_square_16x16_dual.yaml
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_16x16.sh
#   NUM_OPTIMISTIC_ENVS=32 HISTORY_LENGTH=20 bash examples/ppo_trainer/ppo_debug_square_16x16.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_YAML="${HYPER_YAML:-$SCRIPT_DIR/config/ppo_debug_square_16x16_dual.yaml}"

if [ ! -f "$HYPER_YAML" ]; then
  echo "[ERROR] Hyperparam yaml not found: $HYPER_YAML"
  exit 1
fi

eval "$(python3 - "$HYPER_YAML" <<'PY'
import os, shlex, sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
job = cfg.get("job") or {}

def j(key, default, env=None):
    env = env or key.upper()
    if env in os.environ and os.environ[env] != "":
        return os.environ[env]
    v = OmegaConf.select(job, key)
    return default if v is None else v

def emit(name, value):
    print(f"export {name}={shlex.quote(str(value))}")

emit("NUM_OPTIMISTIC_ENVS", j("num_optimistic_envs", 64, "NUM_OPTIMISTIC_ENVS"))
emit("OPTIMISTIC_RESET_RATIO", j("optimistic_reset_ratio", 8, "OPTIMISTIC_RESET_RATIO"))
emit("USE_ACTOR_LORA", j("use_actor_lora", True, "USE_ACTOR_LORA"))
emit("CRITIC_LORA_RANK", j("critic_lora_rank", 64, "CRITIC_LORA_RANK"))
emit("CRITIC_LORA_ALPHA", j("critic_lora_alpha", 64, "CRITIC_LORA_ALPHA"))
emit("TOTAL_EPOCHS", j("total_epochs", 8000, "TOTAL_EPOCHS"))
emit("HISTORY_LENGTH", j("history_length", 50, "HISTORY_LENGTH"))
emit("CHECKPOINT_DIR", j(
    "checkpoint_dir",
    "training_checkpoints/verl_agent_caged_craftext_debug_square_16x16_dual",
    "CHECKPOINT_DIR",
))
emit("RUN_NAME", j(
    "run_name",
    "PPO Debug Square 16x16 dual LLM + r_t + history",
    "RUN_NAME",
))

overrides = [str(x) for x in (cfg.get("overrides") or [])]
replacements = {
    "+env.optimistic_reset_ratio=": f"+env.optimistic_reset_ratio={j('optimistic_reset_ratio', 8, 'OPTIMISTIC_RESET_RATIO')}",
    "env.history_length=": f"env.history_length={j('history_length', 50, 'HISTORY_LENGTH')}",
    "critic.model.lora_rank=": f"critic.model.lora_rank={j('critic_lora_rank', 64, 'CRITIC_LORA_RANK')}",
    "critic.model.lora_alpha=": f"critic.model.lora_alpha={j('critic_lora_alpha', 64, 'CRITIC_LORA_ALPHA')}",
    "trainer.default_local_dir=": f"trainer.default_local_dir={j('checkpoint_dir', 'training_checkpoints/verl_agent_caged_craftext_debug_square_16x16_dual', 'CHECKPOINT_DIR')}",
}

def apply(item: str) -> str:
    for prefix, replacement in replacements.items():
        if item.startswith(prefix):
            return replacement
    return item

overrides = [apply(o) for o in overrides]
print("HYPER_OVERRIDES=(")
for o in overrides:
    print(f"  {shlex.quote(o)}")
print(")")
PY
)"

_bool() {
  case "${1,,}" in
    1|true|yes|on) echo true ;;
    *) echo false ;;
  esac
}
USE_ACTOR_LORA="$(_bool "$USE_ACTOR_LORA")"

echo "=========================================="
echo "PPO debug_square_16x16 (dual actor+critic LLM)"
echo "=========================================="
echo "[INFO] hyper yaml: $HYPER_YAML"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] HISTORY_LENGTH=$HISTORY_LENGTH (actions already taken → prompt)"
echo "[INFO] token score: per-step r_t (use_episode_return_as_token_reward=False)"
echo "[INFO] gae_by_trajectory=False (returns≈r_t at response_len=1)"
echo "[INFO] actor_value_token=False (separate critic LLM)"
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA CRITIC_LORA_RANK=$CRITIC_LORA_RANK"
echo "[INFO] checkpoints -> $CHECKPOINT_DIR"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  vllm \
  false \
  true \
  "$NUM_OPTIMISTIC_ENVS" \
  1 \
  false \
  false \
  "$TOTAL_EPOCHS" \
  "$USE_ACTOR_LORA" \
  single_token_action \
  0 \
  ascii \
  "${HYPER_OVERRIDES[@]}" \
  "$@"
