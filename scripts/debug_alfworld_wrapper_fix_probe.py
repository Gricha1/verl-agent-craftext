#!/usr/bin/env python3
"""Verify that per-env wrapper factories fix shared moves counter."""
import os
import random
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
from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredInfos,
    AlfredTWEnv,
)

CFG = str(ALF_PKG / "configs" / "config_tw.yaml")
config = load_config_file(CFG)
base = AlfredTWEnv(config, train_eval="train")
games = base.game_files[:8]
max_nb = config["dagger"]["training"]["max_nb_steps_per_episode"]
request_infos = textworld.EnvInfos(
    won=True, admissible_commands=True, moves=True, score=True, extras=["gamefile"]
)

shuffle = False


def demangler_factory(env=None):
    return AlfredDemangler(env, shuffle=shuffle)


wrappers = [demangler_factory, AlfredInfos]
env_id = textworld.gym.register_games(
    games,
    request_infos,
    batch_size=8,
    asynchronous=False,
    max_episode_steps=max_nb,
    wrappers=wrappers,
)
env = textworld.gym.make(env_id)
env.seed(0)
obs, infos = env.reset()
rng = random.Random(0)
for t in range(12):
    actions = [
        rng.choice([c for c in infos["admissible_commands"][i] if c != "help"] or ["look"])
        for i in range(8)
    ]
    obs, scores, dones, infos = env.step(actions)
    moves = infos["moves"]
    print(f"t={t+1} moves={moves} n_done={sum(map(bool, dones))}")
    if len(set(moves)) != 1:
        print("FAIL: moves diverged / shared counter still broken")
        break
else:
    print("OK: per-slot moves stay aligned (independent counters)")
env.close()
