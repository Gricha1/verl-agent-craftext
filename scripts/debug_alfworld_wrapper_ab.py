#!/usr/bin/env python3
"""Compare AlfredTWEnv.init_env vs raw register_games; log moves at done."""
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
from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredInfos,
    AlfredTWEnv,
)

CFG = str(ALF_PKG / "configs" / "config_tw.yaml")
config = load_config_file(CFG)
base = AlfredTWEnv(config, train_eval="train")
print("n_games", len(base.game_files))

# Patch request_infos to include moves by rebuilding like init_env
domain_randomization = False
wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
request_infos = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    moves=True,
    score=True,
    max_score=True,
    extras=["gamefile"],
)
max_nb = config["dagger"]["training"]["max_nb_steps_per_episode"]
print("register max_episode_steps", max_nb)

# Use only first 8 distinct games (not full list cycling)
games8 = base.game_files[:8]
print("games:")
for g in games8:
    print(" ", g)

env_id = textworld.gym.register_games(
    games8,
    request_infos,
    batch_size=8,
    asynchronous=False,
    max_episode_steps=max_nb,
    wrappers=wrappers,
)
env = textworld.gym.make(env_id)
env.seed(0)
obs, infos = env.reset()
print("\nRESET gamefiles:")
for i in range(8):
    gf = infos["extra.gamefile"][i] if "extra.gamefile" in infos else "?"
    print(f"  slot{i}: moves={infos['moves'][i]} won={infos['won'][i]} game={Path(str(gf)).parent.name}")

import random
rng = random.Random(0)
for t in range(60):
    actions = []
    for i in range(8):
        cmds = [c for c in infos["admissible_commands"][i] if c != "help"]
        actions.append(rng.choice(cmds) if cmds else "look")
    obs, scores, dones, infos = env.step(actions)
    nd = sum(bool(d) for d in dones)
    if t < 3 or nd:
        print(f"\nSTEP {t+1} n_done={nd}/8")
        for i in range(8):
            if t < 3 or dones[i]:
                print(
                    f"  slot{i}: done={bool(dones[i])} moves={infos['moves'][i]} "
                    f"won={infos['won'][i]} score={infos['score'][i]} "
                    f"action={actions[i]!r} obs={obs[i][:70]!r}"
                )
    if nd == 8:
        print("ALL DONE")
        break
env.close()

# Same but WITHOUT wrappers
print("\n\n======== NO WRAPPERS ========")
env_id = textworld.gym.register_games(
    games8,
    request_infos,
    batch_size=8,
    asynchronous=False,
    max_episode_steps=max_nb,
)
env = textworld.gym.make(env_id)
env.seed(0)
obs, infos = env.reset()
rng = random.Random(0)
for t in range(60):
    actions = []
    for i in range(8):
        cmds = [c for c in infos["admissible_commands"][i] if c != "help"]
        actions.append(rng.choice(cmds) if cmds else "look")
    obs, scores, dones, infos = env.step(actions)
    nd = sum(bool(d) for d in dones)
    if t < 2 or nd or t in (6, 7, 8, 49):
        print(f"STEP {t+1} n_done={nd}/8 moves={infos['moves']}")
    if nd == 8:
        print("ALL DONE")
        break
env.close()

# Full AlfredTWEnv.init_env (all games + expert on train)
print("\n\n======== AlfredTWEnv.init_env(train) ========")
env = base.init_env(batch_size=8)
env.seed(0)
obs, infos = env.reset()
# moves may be absent
print("info keys", list(infos.keys()))
rng = random.Random(0)
for t in range(60):
    actions = []
    for i in range(8):
        cmds = [c for c in infos["admissible_commands"][i] if c != "help"]
        actions.append(rng.choice(cmds) if cmds else "look")
    obs, scores, dones, infos = env.step(actions)
    nd = sum(bool(d) for d in dones)
    if t < 2 or nd or t in (6, 7, 8, 49):
        print(f"STEP {t+1} n_done={nd}/8 won={infos.get('won')}")
        if nd:
            for i in range(8):
                if dones[i]:
                    print(f"  done slot{i} action={actions[i]!r} obs={obs[i][:80]!r}")
    if nd == 8:
        print("ALL DONE")
        break
env.close()
