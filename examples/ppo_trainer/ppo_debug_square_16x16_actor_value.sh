#!/bin/bash
# PPO on debug_square_16x16: same LLM for actor (action token) and critic (return token).
# Valid-action entropy over 17 action tokens (entropy_over_valid_actions=True).
#
# Hyperparameters live in:
#   examples/ppo_trainer/config/ppo_debug_square_16x16_actor_value.yaml
#
# Map: 16x16, spawn center (8,8), stone/wood/water on inner corners (shuffled each reset).
#
# Usage:
#   bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh
#   NUM_OPTIMISTIC_ENVS=32 bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh
#   HYPER_YAML=path/to/other.yaml bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh
#   bash examples/ppo_trainer/ppo_debug_square_16x16_actor_value.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.45

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPER_YAML="${HYPER_YAML:-$SCRIPT_DIR/config/ppo_debug_square_16x16_actor_value.yaml}"

if [ ! -f "$HYPER_YAML" ]; then
  echo "[ERROR] Hyperparam yaml not found: $HYPER_YAML"
  exit 1
fi

# Load job.* + overrides[] from yaml (env vars override job fields when set).
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
emit("CRITIC_WARMUP", j("critic_warmup", 0, "CRITIC_WARMUP"))
emit("USE_ACTOR_LORA", j("use_actor_lora", True, "USE_ACTOR_LORA"))
emit("TOTAL_EPOCHS", j("total_epochs", 8000, "TOTAL_EPOCHS"))
emit("ENGINE", j("engine", "vllm", "ENGINE"))
emit("NO_REASONING", j("no_reasoning", True, "NO_REASONING"))
emit("AUTO_RESET", j("auto_reset", False, "AUTO_RESET"))
emit("USE_ACTION_HEAD", j("use_action_head", False, "USE_ACTION_HEAD"))
emit("PROMPT_TEMPLATE_TYPE", j("prompt_template_type", "single_token_action", "PROMPT_TEMPLATE_TYPE"))
emit("OBSERVATION_TYPE", j("observation_type", "ascii", "OBSERVATION_TYPE"))
emit("RETURN_BIN_MIN", j("return_bin_min", -26, "RETURN_BIN_MIN"))
emit("RETURN_BIN_MAX", j("return_bin_max", 16, "RETURN_BIN_MAX"))
emit("RETURN_BIN_STEP", j("return_bin_step", 1.5, "RETURN_BIN_STEP"))
emit("ACTOR_VALUE_LOSS_COEF", j("actor_value_loss_coef", 1.0, "ACTOR_VALUE_LOSS_COEF"))
emit("ACTOR_VALUE_SEPARATE_STEPS", j("actor_value_separate_optimizer_steps", True, "ACTOR_VALUE_SEPARATE_STEPS"))
emit("ACTOR_VALUE_TARGET_ENCODING", j("actor_value_target_encoding", "two_hot", "ACTOR_VALUE_TARGET_ENCODING"))
emit("ACTOR_VALUE_ENTROPY_COEF", j("actor_value_entropy_coef", 0.1, "ACTOR_VALUE_ENTROPY_COEF"))
emit("ENTROPY_SINGLE_TOKEN_FASTPATH", j("entropy_single_token_fastpath", True, "ENTROPY_SINGLE_TOKEN_FASTPATH"))
emit("ENTROPY_BAND_ENABLE", j("entropy_band_enable", False, "ENTROPY_BAND_ENABLE"))
emit("ENTROPY_BAND_LOW", j("entropy_band_low", 0.7, "ENTROPY_BAND_LOW"))
emit("ENTROPY_BAND_HIGH", j("entropy_band_high", 1.4, "ENTROPY_BAND_HIGH"))
emit("ENTROPY_BAND_COEF_LR", j("entropy_band_coef_lr", 0.05, "ENTROPY_BAND_COEF_LR"))
emit("ENTROPY_BAND_COEF_LOW", j("entropy_band_coef_low", 0.0, "ENTROPY_BAND_COEF_LOW"))
emit("ENTROPY_BAND_COEF_HIGH", j("entropy_band_coef_high", 1.0, "ENTROPY_BAND_COEF_HIGH"))
emit("CHECKPOINT_DIR", j("checkpoint_dir",
    "training_checkpoints/verl_agent_caged_craftext_debug_square_16x16_actor_value",
    "CHECKPOINT_DIR"))
run_name = j("run_name", "PPO Debug Square 16x16 dual-prompt actor value", "RUN_NAME")
emit("RUN_NAME", run_name)

overrides = list(cfg.get("overrides") or [])
# Keep shell-tunable fields in sync with env/job values.
replacements = {
    "+env.optimistic_reset_ratio=": f"+env.optimistic_reset_ratio={j('optimistic_reset_ratio', 8, 'OPTIMISTIC_RESET_RATIO')}",
    "+env.value_return_min=": f"+env.value_return_min={j('return_bin_min', -26, 'RETURN_BIN_MIN')}",
    "+env.value_return_max=": f"+env.value_return_max={j('return_bin_max', 16, 'RETURN_BIN_MAX')}",
    "+env.value_return_bin_step=": f"+env.value_return_bin_step={j('return_bin_step', 1.5, 'RETURN_BIN_STEP')}",
    "actor_rollout_ref.actor.actor_value_loss_coef=": f"actor_rollout_ref.actor.actor_value_loss_coef={j('actor_value_loss_coef', 1.0, 'ACTOR_VALUE_LOSS_COEF')}",
    "actor_rollout_ref.actor.actor_value_separate_optimizer_steps=": f"actor_rollout_ref.actor.actor_value_separate_optimizer_steps={j('actor_value_separate_optimizer_steps', True, 'ACTOR_VALUE_SEPARATE_STEPS')}",
    "actor_rollout_ref.actor.actor_value_target_encoding=": f"actor_rollout_ref.actor.actor_value_target_encoding={j('actor_value_target_encoding', 'two_hot', 'ACTOR_VALUE_TARGET_ENCODING')}",
    "actor_rollout_ref.actor.actor_value_entropy_coef=": f"actor_rollout_ref.actor.actor_value_entropy_coef={j('actor_value_entropy_coef', 0.1, 'ACTOR_VALUE_ENTROPY_COEF')}",
    "actor_rollout_ref.actor.actor_value_return_min=": f"actor_rollout_ref.actor.actor_value_return_min={j('return_bin_min', -26, 'RETURN_BIN_MIN')}",
    "actor_rollout_ref.actor.actor_value_return_max=": f"actor_rollout_ref.actor.actor_value_return_max={j('return_bin_max', 16, 'RETURN_BIN_MAX')}",
    "actor_rollout_ref.actor.actor_value_return_bin_step=": f"actor_rollout_ref.actor.actor_value_return_bin_step={j('return_bin_step', 1.5, 'RETURN_BIN_STEP')}",
    "actor_rollout_ref.actor.entropy_band.enable=": f"actor_rollout_ref.actor.entropy_band.enable={j('entropy_band_enable', False, 'ENTROPY_BAND_ENABLE')}",
    "actor_rollout_ref.actor.entropy_band.low=": f"actor_rollout_ref.actor.entropy_band.low={j('entropy_band_low', 0.7, 'ENTROPY_BAND_LOW')}",
    "actor_rollout_ref.actor.entropy_band.high=": f"actor_rollout_ref.actor.entropy_band.high={j('entropy_band_high', 1.4, 'ENTROPY_BAND_HIGH')}",
    "actor_rollout_ref.actor.entropy_band.coef_lr=": f"actor_rollout_ref.actor.entropy_band.coef_lr={j('entropy_band_coef_lr', 0.05, 'ENTROPY_BAND_COEF_LR')}",
    "actor_rollout_ref.actor.entropy_band.coef_low=": f"actor_rollout_ref.actor.entropy_band.coef_low={j('entropy_band_coef_low', 0.0, 'ENTROPY_BAND_COEF_LOW')}",
    "actor_rollout_ref.actor.entropy_band.coef_high=": f"actor_rollout_ref.actor.entropy_band.coef_high={j('entropy_band_coef_high', 1.0, 'ENTROPY_BAND_COEF_HIGH')}",
    "actor_rollout_ref.actor.entropy_action_single_token_fastpath=": f"actor_rollout_ref.actor.entropy_action_single_token_fastpath={j('entropy_single_token_fastpath', True, 'ENTROPY_SINGLE_TOKEN_FASTPATH')}",
    "trainer.critic_warmup=": f"trainer.critic_warmup={j('critic_warmup', 0, 'CRITIC_WARMUP')}",
    "trainer.default_local_dir=": f"trainer.default_local_dir={j('checkpoint_dir', 'training_checkpoints/verl_agent_caged_craftext_debug_square_16x16_actor_value', 'CHECKPOINT_DIR')}",
}

def apply_replacements(item: str) -> str:
    for prefix, replacement in replacements.items():
        if item.startswith(prefix):
            return replacement
    return item

overrides = [apply_replacements(str(x)) for x in overrides]
print("HYPER_OVERRIDES=(")
for o in overrides:
    print(f"  {shlex.quote(o)}")
print(")")
PY
)"

# Normalize boolean-ish shell flags to true/false strings expected by the job script.
_bool() {
  case "${1,,}" in
    1|true|yes|on) echo true ;;
    *) echo false ;;
  esac
}
USE_ACTOR_LORA="$(_bool "$USE_ACTOR_LORA")"
NO_REASONING="$(_bool "$NO_REASONING")"
AUTO_RESET="$(_bool "$AUTO_RESET")"
USE_ACTION_HEAD="$(_bool "$USE_ACTION_HEAD")"

echo "=========================================="
echo "PPO debug_square_16x16 (dual-prompt actor-value + valid entropy)"
echo "=========================================="
echo "[INFO] hyper yaml: $HYPER_YAML"
echo "[INFO] NUM_OPTIMISTIC_ENVS=$NUM_OPTIMISTIC_ENVS"
echo "[INFO] OPTIMISTIC_RESET_RATIO=$OPTIMISTIC_RESET_RATIO"
echo "[INFO] Map: 16x16 — stone / wood / water in corners (adjacent = success)"
echo "[INFO] actor prompt: single_token_action, max_response_length=1"
echo "[INFO] critic prompt: return bins [$RETURN_BIN_MIN,$RETURN_BIN_MAX] step=$RETURN_BIN_STEP"
echo "[INFO] critic: token head on same LLM (no separate critic network)"
echo "[INFO] gradient checkpointing: ON (actor + critic) — see yaml"
echo "[INFO] GAE: response-level (gae_by_trajectory=False); token reward already = G_t"
echo "[INFO] value warmup (critic_warmup): $CRITIC_WARMUP PPO steps"
echo "[INFO] entropy: H over 17 action tokens (entropy_over_valid_actions=True)"
echo "[INFO] USE_ACTOR_LORA=$USE_ACTOR_LORA"
echo "[INFO] checkpoints -> $CHECKPOINT_DIR"
echo "[INFO] yaml overrides: ${#HYPER_OVERRIDES[@]} keys"

bash examples/ppo_trainer/run_caged_craftext_lora_job.sh \
  "$ENGINE" \
  false \
  "$NO_REASONING" \
  "$NUM_OPTIMISTIC_ENVS" \
  1 \
  "$AUTO_RESET" \
  "$USE_ACTION_HEAD" \
  "$TOTAL_EPOCHS" \
  "$USE_ACTOR_LORA" \
  "$PROMPT_TEMPLATE_TYPE" \
  "$CRITIC_WARMUP" \
  "$OBSERVATION_TYPE" \
  "${HYPER_OVERRIDES[@]}" \
  "$@"
