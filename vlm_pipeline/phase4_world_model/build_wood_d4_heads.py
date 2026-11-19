#!/usr/bin/env python3
"""wood2200 training episodes for the Dreamer 4 phase-2 agent heads (reward + policy), task reward.

The frames are those of build_wood_v3.py (the 2165 training episodes, 64 x 64 = the 63 x 63 Craftax frame plus a black row and
column; the adapter cuts the 63 x 63 view), linked, not copied. A raw row is (frame x_t, action a_t, reward / success of
a_t); the frame after the last action is not stored. The task is solved by the last action of a successful episode, so
the +1 is put on the last transition that has an arrival frame:

  rewards[t]   = 1 for t = n - 2 of a successful episode (arrives with x_{n-1}, the frame from which one action finishes the
                 task), else 0.  Dreamer 4 convention: reward of the t -> t+1 transition.
  terminals[t] = the same transition, for every episode that ended with done (success or death; not the 2 cut at 1000).
  tasks        = index of the episode's instruction in the sorted task list (the SLM's order)
  bc           = 1 for successful episodes (the ones behaviour cloning may imitate)
  terminal_rule = "reward"

With a 4th argument "death" (reward v2, 2026-11-19): -1 on the same last transition of an episode that ended with done
and without success (the agent died: health 0 or lava; not the 2 episodes cut at 1000 steps), terminal_rule "abs_reward"
(|r| >= 0.5 ends an imagined episode), so the world model's reward head sees dying.

With a 4th argument "native" (2026-11-19, user: the environment's reward must not be changed): rewards are the recorded
environment rewards (InstructionWrapper on Craftax-Classic: Craftax reward / 50, i.e. 0.02 per achievement and 0.002 per
health point gained or lost; no reward of its own for the task or for dying): rewards[t] = r_t for t <= n - 2, and the
reward of the last action, which has no arrival frame, is added to t = n - 2 (the same one-step shift as the +1 above).
terminals as above; terminal_rule "native" (no terminal from the reward: train_heads --cont_head 1 learns the end).

  python build_wood_d4_heads.py <raw collection> <wood2200_v3 dir> <out dir> [death|native]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_slm_world_model import load_episodes  # noqa: E402

raw, v3, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
death = len(sys.argv) > 4 and sys.argv[4] == "death"
native = len(sys.argv) > 4 and sys.argv[4] == "native"
deaths = 0
out.mkdir(parents=True, exist_ok=True)
episodes = load_episodes(raw, None, None)
man = json.loads((v3 / "manifest.json").read_text())
tasks = sorted({ep[0]["instruction"] for ep in episodes.values()})
N = sum(man["lengths"])
actions = np.load(v3 / "actions.npy")
rewards, terminals = np.zeros(N, np.float32), np.zeros(N, bool)
task_of, bc = [], []
for key, first, n in zip(man["episode_ids"], man["starts"], man["lengths"]):
    ep = episodes[int(key)]
    assert len(ep) == n, (key, len(ep), n)
    assert all(int(it["action"]) == int(actions[first + t]) for t, it in enumerate(ep))
    ok = bool(ep[-1]["success"])
    if n >= 2:
        died = (not ok) and bool(ep[-1]["done"]) and n < 1000
        if native:
            r = np.array([float(it["reward"]) for it in ep], np.float32)
            rewards[first:first + n - 1] = r[:n - 1]
            rewards[first + n - 2] += r[n - 1]
        else:
            rewards[first + n - 2] = 1.0 if ok else (-1.0 if death and died else 0.0)
        deaths += died
        terminals[first + n - 2] = bool(ep[-1]["done"]) and n < 1000
    task_of.append(tasks.index(ep[0]["instruction"]))
    bc.append(int(ok))
np.save(out / "actions.npy", actions)
np.save(out / "rewards.npy", rewards)
np.save(out / "terminals.npy", terminals)
if not (out / "frames.npy").exists():
    os.symlink((v3 / "frames.npy").resolve(), out / "frames.npy")
(out / "wood_manifest.json").write_text(json.dumps(dict(
    starts=man["starts"], lengths=man["lengths"], episode_ids=man["episode_ids"], n_actions=17, tasks=task_of, bc=bc,
    task_names=tasks, terminal_rule="native" if native else "abs_reward" if death else "reward",
    reward_rule=("native environment reward (Craftax / 50), last action's reward added to the transition into x_{n-1}"
                 if native else "+1 on the transition into x_{n-1} of a successful episode (one action before the task is done)"
                 + ("; -1 there for an episode that ended in death" if death else "")))))
print("reward sum", float(rewards.sum()), "nonzero", int((rewards != 0).sum()), flush=True)
print("episodes", len(task_of), "frames", N, "successes", sum(bc), "deaths", deaths, "terminals", int(terminals.sum()),
      "tasks", {t: task_of.count(i) for i, t in enumerate(tasks)}, flush=True)
