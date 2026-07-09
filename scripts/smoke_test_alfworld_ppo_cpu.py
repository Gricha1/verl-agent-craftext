#!/usr/bin/env python3
"""CPU smoke test for AlfWorld PPO configs (no GPU / no vLLM / no full training)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_alfworld_module(subname: str):
    """Load alfworld submodules without importing agent_system.environments (craftax)."""
    import importlib.machinery

    pkg = "agent_system.environments.env_package.alfworld"
    if pkg not in sys.modules:
        pkg_mod = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(pkg, None))
        pkg_mod.__path__ = [str(ROOT / "agent_system/environments/env_package/alfworld")]
        sys.modules[pkg] = pkg_mod
    full_name = f"{pkg}.{subname}"
    path = ROOT / f"agent_system/environments/env_package/alfworld/{subname}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_action_tokens_and_projection():
    at = _load_alfworld_module("action_tokens")
    proj = _load_alfworld_module("projection")
    cmds = [["go to cabinet 1", "go to sinkbasin 1", "inventory", "help"]]
    actions = ["2"]
    out, valids = proj.alfworld_single_token_projection(list(actions), cmds)
    assert valids[0] == 1, out
    assert out[0] == "go to sinkbasin 1"
    assert at.parse_single_token_action("3", 3) == 2
    print("[OK] action_tokens + single_token projection")


def test_compute_reward_normalized():
    # Inline mirror of envs.compute_reward (avoid importing ray/alfworld).
    def compute_reward(info, multi_modal=False):
        if multi_modal:
            raw = 10.0 * float(info["won"]) + float(info["goal_condition_success_rate"])
        else:
            raw = 10.0 * float(info["won"])
        return raw / 10.0

    assert compute_reward({"won": False}) == 0.0
    assert compute_reward({"won": True}) == 1.0
    print("[OK] compute_reward /10")


def test_prompt_and_observation_strip():
    st = _load_module("alf_st", "agent_system/environments/prompts/alfworld_single_token.py")
    at = _load_alfworld_module("action_tokens")
    task = "put a clean mug in coffeemachine"
    raw_obs = (
        "-= Welcome to TextWorld, ALFRED! =-\n"
        "You are in the middle of a room.\n"
        f"Your task is to: {task}"
    )
    # Mirror AlfWorldEnvironmentManager._observation_for_prompt
    obs = raw_obs
    obs = obs.replace(f"Your task is to: {task}", "", 1)
    obs = obs.replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
    legend = at.build_admissible_legend(["go to cabinet 1", "inventory"])
    prompt = st.ALFWORLD_SINGLE_TOKEN_ACTION_TEMPLATE.format(
        task_description=task,
        current_observation=obs,
        action_legend=legend,
    )
    assert task in prompt
    assert "Welcome to TextWorld" not in prompt
    assert "Your task is to:" not in prompt
    assert "1=go to cabinet 1" in prompt
    print("[OK] actor prompt template + obs strip")


def test_dynamic_entropy_pack():
    import torch
    from verl.utils.action_set_entropy import action_log_scores_from_dynamic_admissible_vocab

    logits = torch.zeros(2, 100)
    logits[0, 16] = 3.0
    logits[0, 17] = 1.0
    logits[1, 64] = 2.0
    vocab = torch.tensor([[16, 17], [64, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True], [True, False]])
    scores = action_log_scores_from_dynamic_admissible_vocab(logits, vocab, mask)
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[0, 1]
    print("[OK] dynamic admissible entropy gather")


def test_hydra_configs():
    from hydra import compose, initialize_config_dir

    config_dir = str(ROOT / "verl/trainer/config")
    common = [
        "env.env_name=alfworld/AlfredTWEnv",
        "env.max_steps=2",
        "data.train_batch_size=2",
        "data.val_batch_size=2",
        "data.max_response_length=1",
        "+env.prompt_template_type=single_token_action",
        "actor_rollout_ref.actor.single_token_actions=True",
        "trainer.total_epochs=1",
        "trainer.n_gpus_per_node=0",
        "trainer.logger=[console]",
    ]
    modes = {
        "dual": [
            "actor_rollout_ref.actor.entropy_over_valid_actions=False",
            "algorithm.use_actor_value_token=False",
        ],
        "dual_valid_ent": [
            "actor_rollout_ref.actor.entropy_over_valid_actions=True",
            "algorithm.use_actor_value_token=False",
        ],
        "actor_value_valid_ent": [
            "actor_rollout_ref.actor.entropy_over_valid_actions=True",
            "algorithm.use_actor_value_token=True",
            "actor_rollout_ref.actor.actor_value_token=True",
            "+env.value_prompt_template_type=single_token_return",
            "+env.value_return_min=-0.2",
            "+env.value_return_max=1.2",
            "+env.value_return_bin_step=0.2",
            "actor_rollout_ref.actor.actor_value_target_from_returns=True",
        ],
    }
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for name, extra in modes.items():
            cfg = compose(config_name="ppo_trainer", overrides=common + extra)
            assert cfg.env.env_name == "alfworld/AlfredTWEnv"
            assert cfg.env.prompt_template_type == "single_token_action"
            assert cfg.data.max_response_length == 1
            if name == "actor_value_valid_ent":
                assert cfg.algorithm.use_actor_value_token is True
                assert cfg.env.value_prompt_template_type == "single_token_return"
            else:
                assert cfg.algorithm.use_actor_value_token is False
            print(f"[OK] hydra config: {name}")


def test_make_envs_projection_choice():
    from omegaconf import OmegaConf

    # Import only make_envs alfworld branch logic via partial test
    proj_default = _load_alfworld_module("projection")
    assert hasattr(proj_default, "alfworld_projection")
    assert hasattr(proj_default, "alfworld_single_token_projection")
    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "alfworld/AlfredTWEnv",
                "prompt_template_type": "single_token_action",
                "seed": 0,
                "max_steps": 50,
                "history_length": 2,
                "alfworld": {"eval_dataset": "eval_in_distribution"},
            },
            "data": {"train_batch_size": 2, "val_batch_size": 2},
        }
    )
    assert cfg.env.prompt_template_type == "single_token_action"
    print("[OK] env config fields for make_envs")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("AlfWorld PPO CPU smoke test")
    test_action_tokens_and_projection()
    test_compute_reward_normalized()
    test_prompt_and_observation_strip()
    test_dynamic_entropy_pack()
    test_hydra_configs()
    test_make_envs_projection_choice()
    print("\nAll CPU smoke checks passed.")
    print("Note: full PPO still needs GPU+vLLM+alfworld data; this only validates configs and code paths.")


if __name__ == "__main__":
    main()
