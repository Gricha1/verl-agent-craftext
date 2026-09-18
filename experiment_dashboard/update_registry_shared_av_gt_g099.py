#!/usr/bin/env python3
"""Update registry: stop dual 90deefed, add shared AV G_t 0.99 experiment."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import yaml

REG = Path("experiments/registry.yaml")


def main() -> None:
    data = yaml.safe_load(REG.read_text(encoding="utf-8"))
    exps = data["experiments"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    dual = None
    for e in exps:
        if e.get("id") == "ds8_dual_gt_g099_h50_seed1":
            dual = e
            break
    if dual is None:
        raise SystemExit("dual registry entry missing")

    # Fix live key + mark stopped (process already dead on aicenter3).
    dual["comet_key"] = "90deefed8ddc458da4f27349a14bf4f4"
    dual["comet_url"] = (
        "https://www.comet.com/gregory-gorbov/verl-agent-caged-craftext/"
        "90deefed8ddc458da4f27349a14bf4f4"
    )
    dual["local_log_path"] = (
        "/home/gorbov_gv/safe_rl_nlp/logdir/ppo_debug_square_gt_g099_20260911_070145.log"
    )
    dual["status"] = "stopped_for_next_experiment"
    dual["final_global_step"] = 114
    dual["final_total_env_steps"] = 240544
    dual["success_last"] = 1.0
    dual["success_max"] = 1.0
    dual["gpu_ids"] = [4, 5]
    dual["gpus"] = 2
    dual["host"] = "aicenter3"
    dual["stop_reason"] = (
        "ActorDiedError/worker crash at ~step 114 after success takeoff; "
        "GPUs freed for shared actor-value G_t^0.99 experiment"
    )
    dual["notes"] = (
        (dual.get("notes") or "")
        + "\nStopped for next experiment (shared AV). Live key was 90deefed "
        "(aa4f5c97 was earlier relaunch). Final success_last=max=1.0 at "
        "global_step~114 / total_env_steps~240544."
    )
    tags = list(dual.get("tags") or [])
    dual["tags"] = [t for t in tags if t != "running"] + ["stopped_for_next_experiment"]

    new_id = "ds8_shared_av_gt_g099_h50"
    exps = [e for e in exps if e.get("id") != new_id]
    shared = {
        "id": new_id,
        "name": "shared actor-value G_t gamma=0.99 h50",
        "comet_project": "gregory-gorbov/verl-agent-caged-craftext",
        "comet_key": None,
        "comet_url": None,
        "environment": "debug_square_8x8",
        "architecture": "shared_actor_value",
        "actor_value": True,
        "model": "Qwen2.5-family (LoRA)",
        "finetune": "lora",
        "lora_rank": 64,
        "reward_mode": "discounted_Gt",
        "remaining_return_gamma": 0.99,
        "value_loss": "two_hot_ce",
        "gamma": 0.99,
        "lambda": 1.0,
        "gae_by_trajectory": False,
        "history_length": 50,
        "response_len": 1,
        "batch_size": 64,
        "gpus": 2,
        "host": "aicenter3",
        "gpu_ids": [4, 5],
        "config_path": "examples/ppo_trainer/config/ppo_debug_square_8x8_shared_av_gt_g099.yaml",
        "launch_script": "examples/ppo_trainer/ppo_debug_square_shared_av_gt_g099.sh",
        "local_log_path": None,
        "git_commit": None,
        "start_date": now,
        "status": "planned",
        "success_max": None,
        "success_last": None,
        "checkpoint_mode": "actor_only",
        "checkpoint_schedule": "early/mid/late @ global_step 40/75/110",
        "checkpoints": [],
        "notes": (
            "Main architecture test: dual separate critic vs shared actor-value "
            "under identical discounted G_t^0.99. Value loss=two-hot CE (not MAE). "
            "gae_by_trajectory=False to avoid double-count. Bins [-5,6]/0.4 kept "
            "(same as 4be19); monitor clipping_fraction."
        ),
        "hypothesis": ["H_arch_shared_vs_dual"],
        "parent_experiment": "ds8_dual_gt_g099_h50_seed1",
        "compare_with": [
            "90deefed8ddc458da4f27349a14bf4f4",
            "4be19de5e50949359891ce852700d565",
            "874bc480b5224efb9b92795d271ca81c",
            "2a42e153bdb0448789a56140e70fdf50",
        ],
        "tags": [
            "planned",
            "shared_actor_value",
            "G_t",
            "gamma0.99",
            "8x8",
            "aicenter3",
            "actor_only_ckpt",
        ],
    }
    exps.append(shared)
    data["experiments"] = exps
    data["updated"] = now
    REG.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("updated", REG, "dual_status=", dual["status"], "added", new_id)


if __name__ == "__main__":
    main()
