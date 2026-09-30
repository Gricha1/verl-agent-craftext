#!/usr/bin/env python3
"""Probe why TextWorld AlfWorld episodes terminate early."""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALFWORLD_DATA", "/root/.cache/alfworld")

REPO = Path(__file__).resolve().parents[1]
ALF_PKG = REPO / "agent_system" / "environments" / "env_package" / "alfworld"
sys.path[:0] = [str(REPO), str(ALF_PKG)]

import textworld
import textworld.gym
from agent_system.environments.env_package.alfworld.envs import load_config_file
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
    get_environment,
)

CFG = ALF_PKG / "configs" / "config_tw.yaml"
config = load_config_file(str(CFG))
print("dagger max_nb_steps", config["dagger"]["training"]["max_nb_steps_per_episode"])
base = get_environment(config["env"]["type"])(config, train_eval="eval_in_distribution")
gf = base.game_files[0]
print("sample game", gf)

request_infos = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    moves=True,
    max_score=True,
    score=True,
    extras=["gamefile"],
)

for max_eps in (50, 100, None):
    kwargs = dict(
        request_infos=request_infos,
        batch_size=1,
        asynchronous=False,
    )
    if max_eps is not None:
        kwargs["max_episode_steps"] = max_eps
    env_id = textworld.gym.register_games([gf], **kwargs)
    env = textworld.gym.make(env_id)
    env.seed(0)
    obs, infos = env.reset()
    print(f"\n=== max_episode_steps={max_eps} ===")
    print("reset moves/score/won", infos.get("moves"), infos.get("score"), infos.get("won"))
    for t in range(60):
        cmds = [c for c in infos["admissible_commands"][0] if c != "help"]
        a = cmds[min(1, len(cmds) - 1)]
        obs, scores, dones, infos = env.step([a])
        if t < 5 or dones[0] or t % 10 == 9:
            print(
                f"t={t+1} done={bool(dones[0])} won={infos['won'][0]} "
                f"moves={infos.get('moves')} score={infos.get('score')} action={a!r}"
            )
        if dones[0]:
            print(f"TERMINATED at step {t+1}")
            break
    else:
        print("no done in 60 steps")
    env.close()

# Also: batch_size=8 same game file repeated vs different games
print("\n=== batch=8 different games, max_episode_steps=50, first-action policy ===")
games = base.game_files[:8]
env_id = textworld.gym.register_games(
    games,
    request_infos=request_infos,
    batch_size=8,
    asynchronous=False,
    max_episode_steps=50,
)
env = textworld.gym.make(env_id)
env.seed(0)
obs, infos = env.reset()
print("reset moves", infos.get("moves"))
for t in range(60):
    actions = []
    for i in range(8):
        cmds = [c for c in infos["admissible_commands"][i] if c != "help"]
        actions.append(cmds[0] if cmds else "look")
    obs, scores, dones, infos = env.step(actions)
    nd = sum(bool(d) for d in dones)
    if t < 3 or nd or t % 10 == 9:
        print(f"t={t+1} n_done={nd}/8 moves={infos.get('moves')} won={infos.get('won')}")
    if nd == 8:
        print(f"ALL DONE at {t+1}")
        break
env.close()
