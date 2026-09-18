#!/usr/bin/env python3
"""Diagnose early-done in AlfWorld TextWorld train envs with a random policy.

No LLM / PPO — only env load + admissible random (or fixed) actions.
Mirrors AlfworldWorker construction (AlfredTWEnv + expert_plan on train).

Usage (inside training container):
  export ALFWORLD_DATA=/root/.cache/alfworld
  export PYTHONPATH=/usr/home/workspace:/usr/home/workspace/agent_system/environments/env_package/alfworld:$PYTHONPATH
  python scripts/debug_alfworld_random_rollout.py
  python scripts/debug_alfworld_random_rollout.py --mode look --max-steps 10
  python scripts/debug_alfworld_random_rollout.py --eval  # no expert wrapper
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Avoid JAX/os.fork issues if anything imports jax transitively.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALFWORLD_DATA", os.environ.get("ALFWORLD_DATA", "/root/.cache/alfworld"))

REPO = Path(__file__).resolve().parents[1]
ALF_PKG = REPO / "agent_system" / "environments" / "env_package" / "alfworld"
for p in (str(REPO), str(ALF_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

CFG = ALF_PKG / "configs" / "config_tw.yaml"


def _load_tw_env(batch_size: int, is_train: bool, seed: int, eval_dataset: str):
    from agent_system.environments.env_package.alfworld.envs import load_config_file
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
        get_environment,
    )

    config = load_config_file(str(CFG))
    env_type = config["env"]["type"]
    train_eval = "train" if is_train else eval_dataset
    base = get_environment(env_type)(config, train_eval=train_eval)
    print(
        f"[env] type={env_type} train_eval={train_eval} "
        f"n_games={len(base.game_files)} batch_size={batch_size} "
        f"ALFWORLD_DATA={os.environ.get('ALFWORLD_DATA')}"
    )
    if not base.game_files:
        raise FileNotFoundError("0 AlfWorld games — check ALFWORLD_DATA / json_2.1.1")
    env = base.init_env(batch_size=batch_size)
    env.seed(seed)
    return env


def _info_at(infos: dict, idx: int) -> dict:
    out = {}
    for k, v in infos.items():
        if isinstance(v, (list, tuple)) and len(v) > idx:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _pick_action(pool, mode: str, rng: random.Random) -> str:
    cmds = [c for c in (pool or []) if c and c != "help"]
    if not cmds:
        return "look"
    if mode == "look":
        return "look" if "look" in cmds else cmds[0]
    if mode == "first":
        return cmds[0]
    return rng.choice(cmds)


def _snip(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " | ")
    return s if len(s) <= n else s[: n - 3] + "..."


def run_episode(
    env,
    *,
    batch_size: int,
    max_steps: int,
    mode: str,
    seed: int,
    log_slots: int,
    sticky_probe_steps: int,
):
    rng = random.Random(seed)
    obs, infos = env.reset()
    episode_len = [0] * batch_size
    episode_won = [False] * batch_size
    first_done_step = [None] * batch_size
    still_alive = [True] * batch_size
    sticky_hits = [0] * batch_size  # post-done steps that still report done=True

    print("\n=== RESET ===")
    for i in range(min(log_slots, batch_size)):
        info = _info_at(infos, i)
        adm = info.get("admissible_commands") or []
        game = (info.get("extra.gamefile") or info.get("extra", {}) or {})
        if isinstance(game, dict):
            game = game.get("gamefile", "?")
        print(
            f"  slot{i}: won={info.get('won')} n_adm={len(adm)} "
            f"game={Path(str(game)).name if game else '?'} "
            f"obs={_snip(obs[i])}"
        )

    for t in range(max_steps):
        pools = infos.get("admissible_commands")
        actions = []
        for i in range(batch_size):
            pool = pools[i] if isinstance(pools, (list, tuple)) else None
            actions.append(_pick_action(pool, mode, rng))

        obs, scores, dones, infos = env.step(actions)

        # Convert dones to plain bools
        dones_list = [bool(d) for d in dones]
        n_done = sum(dones_list)
        newly = []

        for i in range(batch_size):
            info = _info_at(infos, i)
            if still_alive[i]:
                episode_len[i] += 1
                if info.get("won"):
                    episode_won[i] = True
                if dones_list[i]:
                    still_alive[i] = False
                    first_done_step[i] = t + 1
                    newly.append(i)
            elif dones_list[i]:
                sticky_hits[i] += 1

        # Log first few slots every step while any early-done risk, else only on new dones / first 3 steps
        should_log = (t < 3) or newly or (n_done == batch_size)
        if should_log:
            print(f"\n=== STEP {t + 1}  dones={n_done}/{batch_size} newly={newly} ===")
            for i in range(min(log_slots, batch_size)):
                info = _info_at(infos, i)
                print(
                    f"  slot{i}: action={actions[i]!r} done={dones_list[i]} "
                    f"won={info.get('won')} score={scores[i] if hasattr(scores, '__getitem__') else scores} "
                    f"alive_was={still_alive[i] or (first_done_step[i] == t + 1)} "
                    f"obs={_snip(obs[i])}"
                )

        if all(not a for a in still_alive):
            print(f"\n[batch] all slots done after {t + 1} env steps")
            # Sticky-done probe: keep stepping finished batch
            if sticky_probe_steps > 0:
                print(f"\n=== STICKY-DONE PROBE ({sticky_probe_steps} extra steps) ===")
                for k in range(sticky_probe_steps):
                    pools = infos.get("admissible_commands")
                    actions = [
                        _pick_action(
                            pools[i] if isinstance(pools, (list, tuple)) else None,
                            mode,
                            rng,
                        )
                        for i in range(batch_size)
                    ]
                    obs, scores, dones, infos = env.step(actions)
                    dones_list = [bool(d) for d in dones]
                    for i in range(batch_size):
                        if dones_list[i]:
                            sticky_hits[i] += 1
                    print(
                        f"  probe+{k + 1}: dones={sum(dones_list)}/{batch_size} "
                        f"sample_obs={_snip(obs[0])}"
                    )
            break

    print("\n=== SUMMARY ===")
    lens = episode_len
    print(f"lengths: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.3f}")
    print(f"won: {sum(episode_won)}/{batch_size}")
    print(f"first_done_step hist: {Counter(first_done_step)}")
    print(f"length hist: {Counter(lens)}")
    print(f"sticky_done extra hits (sum per slot): {sticky_hits}")
    early = sum(1 for L in lens if L <= 2)
    print(f"episodes with length<=2: {early}/{batch_size}")
    return {
        "lengths": lens,
        "won": episode_won,
        "first_done_step": first_done_step,
        "sticky_hits": sticky_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["random", "look", "first"], default="random")
    ap.add_argument("--eval", action="store_true", help="Use valid_seen (no expert wrapper)")
    ap.add_argument("--eval-dataset", default="eval_in_distribution")
    ap.add_argument("--log-slots", type=int, default=8)
    ap.add_argument("--sticky-probe", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1, help="Independent resets to average")
    args = ap.parse_args()

    is_train = not args.eval
    print(
        f"config={CFG} mode={args.mode} is_train={is_train} "
        f"batch={args.batch_size} max_steps={args.max_steps}"
    )

    env = _load_tw_env(
        args.batch_size,
        is_train=is_train,
        seed=args.seed,
        eval_dataset=args.eval_dataset,
    )

    all_lens = []
    try:
        for r in range(args.repeats):
            if args.repeats > 1:
                print(f"\n########## REPEAT {r + 1}/{args.repeats} ##########")
            # re-seed so each repeat is a fresh shuffle of TW batch
            env.seed(args.seed + r)
            stats = run_episode(
                env,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                mode=args.mode,
                seed=args.seed + 1000 * r,
                log_slots=args.log_slots,
                sticky_probe_steps=args.sticky_probe,
            )
            all_lens.extend(stats["lengths"])
    finally:
        try:
            env.close()
        except Exception:
            pass

    if args.repeats > 1:
        print(
            f"\n=== ALL REPEATS lengths mean={sum(all_lens)/len(all_lens):.3f} "
            f"len<=2={sum(1 for L in all_lens if L <= 2)}/{len(all_lens)} ==="
        )


if __name__ == "__main__":
    main()
