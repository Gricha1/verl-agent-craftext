"""Replay a teacher collection and write the public state of every transition as text.

The PPO teacher A gives actions only; reasoning targets are written later by an LLM
teacher (ZeroHero recipe, label_reasoning.py) from what the learner can see.  The
collection stores frames and actions but not the Craftax state, so each episode is
replayed exactly as ``collect_flax_teacher_trajectories.py`` ran it (PRNGKey(seed),
reset with the task index, one key split per step) with the stored teacher actions.
Every replayed frame must equal the stored one, otherwise the script stops.

  python replay_teacher_states.py --legacy-root caged_craftext --collection-dir <dir> --out states.jsonl

Output: one JSON row per transition (episode_id, step, ...), with ``observation_text``:
the 7 x 9 visible map, mobs, facing block, neighbours, nearest visible objects,
inventory and vitals; the same information as the learner's 63 x 63 frame.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BLOCKS = ["invalid", "out of bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron",
          "diamond", "crafting table", "furnace", "sand", "lava", "sapling", "ripe plant"]
SYMBOLS = "?X.~#TwPcid+Fs%pr"
DIRECTION_NAMES = {1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}
# (row, column) step of each direction, as Craftax-Classic DIRECTIONS
DIRECTION_STEPS = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
ACHIEVEMENTS = ["collect wood", "place table", "eat cow", "collect sapling", "collect drink", "make wood pickaxe",
                "make wood sword", "place plant", "defeat zombie", "collect stone", "place stone", "eat plant",
                "defeat skeleton", "make stone pickaxe", "make stone sword", "wake up", "place furnace",
                "collect coal", "collect iron", "collect diamond", "make iron pickaxe", "make iron sword"]
INVENTORY = ["wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe", "stone_pickaxe",
             "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]
MOBS = [("zombies", "zombie", "Z"), ("cows", "cow", "C"), ("skeletons", "skeleton", "S"), ("arrows", "arrow", "a")]
VIEW = (7, 9)  # Craftax-Classic OBS_DIM: rows, columns
HINDSIGHT = 8


def offset_words(d_row: int, d_col: int) -> str:
    parts = []
    if d_col:
        parts.append(f"{abs(d_col)} {'left' if d_col < 0 else 'right'}")
    if d_row:
        parts.append(f"{abs(d_row)} {'up' if d_row < 0 else 'down'}")
    return ", ".join(parts) or "here"


def describe(env_state) -> dict:
    """Public view of one Craftax-Classic state (what the 63 x 63 frame shows)."""
    world = np.asarray(env_state.map)
    row, col = (int(v) for v in np.asarray(env_state.player_position))
    top, left = row - VIEW[0] // 2, col - VIEW[1] // 2
    grid = [[1] * VIEW[1] for _ in range(VIEW[0])]
    for r in range(VIEW[0]):
        for c in range(VIEW[1]):
            wr, wc = top + r, left + c
            if 0 <= wr < world.shape[0] and 0 <= wc < world.shape[1]:
                grid[r][c] = int(world[wr, wc])
    symbols = [[SYMBOLS[b] for b in line] for line in grid]
    mobs = []
    for field, name, symbol in MOBS:
        group = getattr(env_state, field)
        for (mr, mc), alive in zip(np.asarray(group.position), np.asarray(group.mask)):
            r, c = int(mr) - top, int(mc) - left
            if alive and 0 <= r < VIEW[0] and 0 <= c < VIEW[1]:
                symbols[r][c] = symbol
                mobs.append((name, int(mr) - row, int(mc) - col))
    centre = (VIEW[0] // 2, VIEW[1] // 2)
    symbols[centre[0]][centre[1]] = "@"
    facing = DIRECTION_NAMES.get(int(env_state.player_direction), "UP")

    def block_at(d_row, d_col):
        r, c = centre[0] + d_row, centre[1] + d_col
        return BLOCKS[grid[r][c]]

    neighbours = {name: block_at(*step) for name, step in DIRECTION_STEPS.items()}
    for mob, d_row, d_col in mobs:
        for name, step in DIRECTION_STEPS.items():
            if (d_row, d_col) == step:
                neighbours[name] = mob
    nearest = {}
    for r in range(VIEW[0]):
        for c in range(VIEW[1]):
            name = BLOCKS[grid[r][c]]
            if name in ("grass", "out of bounds", "invalid", "path", "sand"):
                continue
            distance = abs(r - centre[0]) + abs(c - centre[1])
            if distance and (name not in nearest or distance < nearest[name][0]):
                nearest[name] = (distance, offset_words(r - centre[0], c - centre[1]))
    for mob, d_row, d_col in mobs:
        distance = abs(d_row) + abs(d_col)
        if mob not in nearest or distance < nearest[mob][0]:
            nearest[mob] = (distance, offset_words(d_row, d_col))
    inventory = {name: int(np.asarray(getattr(env_state.inventory, name))) for name in INVENTORY}
    vitals = {name: int(np.asarray(getattr(env_state, f"player_{name}"))) for name in ("health", "food", "drink", "energy")}
    achieved = [ACHIEVEMENTS[i] for i, flag in enumerate(np.asarray(env_state.achievements)) if flag]
    return dict(grid="\n".join(" ".join(line) for line in symbols), facing=facing,
                facing_block=neighbours[facing], neighbours=neighbours,
                nearest={k: v[1] for k, v in sorted(nearest.items(), key=lambda kv: kv[1][0])},
                inventory={k: v for k, v in inventory.items() if v}, vitals=vitals, achieved=achieved,
                sleeping=bool(np.asarray(env_state.is_sleeping)))


def observation_text(view: dict) -> str:
    legend = ("Legend: @ you, . grass, T tree, ~ water, # stone, s sand, P path, c coal, i iron, d diamond, "
              "w wood block, + crafting table, F furnace, % lava, p sapling, r ripe plant, X edge of the world, "
              "Z zombie, C cow, S skeleton, a arrow.")
    nearest = "; ".join(f"{name}: {where}" for name, where in view["nearest"].items()) or "nothing notable"
    inventory = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in view["inventory"].items()) or "empty"
    vitals = ", ".join(f"{k} {v}/9" for k, v in view["vitals"].items())
    return (f"Visible area (7 rows x 9 columns, up is the top row):\n{view['grid']}\n{legend}\n"
            f"You face {view['facing']}; the cell you face: {view['facing_block']}.\n"
            "Next to you: " + ", ".join(f"{k} {v}" for k, v in view["neighbours"].items()) + ".\n"
            f"Nearest visible: {nearest}.\nInventory: {inventory}.\nVitals: {vitals}."
            + (" You are asleep." if view["sleeping"] else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--collection-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all (smoke runs use a few)")
    parser.add_argument("--jit", action="store_true", help="jitted step; frames are then compared with --atol")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    sys.path[:0] = [str(legacy), str(legacy / "Craftax"), str(legacy / "baselines")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    from craftax.craftax_env import make_craftax_env_from_name
    from craftext.environment.craftext_wrapper import InstructionWrapper

    manifest = json.loads((args.collection_dir / "manifest.json").read_text())
    # both collectors: PRNGKey(seed), reset with the task index, one key split per step, same env
    if manifest["collector"] not in ("collect_flax_teacher_trajectories.py", "collect_ppo_achievements_all_trajectories.py"):
        raise RuntimeError(f"replay mirrors the wood / all teacher collectors, not {manifest['collector']}")
    rows = defaultdict(dict)
    for split in ("train", "val"):
        for row in pq.read_table(args.collection_dir / f"{split}.parquet").to_pylist():
            info = row["extra_info"]
            rows[info["episode_id"]][info["step"]] = info
    base_env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
    env_params = base_env.default_params
    env = InstructionWrapper(base_env, manifest["craftext_settings"])
    action_ids = {name: i for i, name in enumerate(
        ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE",
         "PLACE_PLANT", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
         "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"])}
    # eager, as the collector, reproduces the stored frames bit for bit (~1.5 s per step); a jitted
    # step is ~100x faster and its frames differ by float rounding only, checked against --atol
    step_fn = jax.jit(env.step) if args.jit else env.step
    max_diff = 0.0

    trajectories = manifest["trajectories"][: args.max_episodes or None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w") as out:
        for trajectory in trajectories:
            episode = rows[trajectory["episode_id"]]
            steps = sorted(episode)
            if steps != list(range(trajectory["steps"])):
                raise RuntimeError(f"episode {trajectory['episode_id']}: rows do not cover its {trajectory['steps']} steps")
            actions = [episode[s]["teacher_action"] for s in steps]
            key = jax.random.PRNGKey(trajectory["seed"])
            obs, state = env.reset(key, env_params, instruction_idx=trajectory["task_index"])
            for step in steps:
                info = episode[step]
                stored = np.load(args.collection_dir / "images" / Path(info["observation_path"]).name)["observation"]
                diff = np.abs(stored.astype(np.float64) - np.asarray(obs, dtype=np.float64))
                max_diff = max(max_diff, float(diff.max()))
                if diff.max() > args.atol:
                    raise RuntimeError(f"episode {trajectory['episode_id']} step {step}: replayed frame differs "
                                       f"(max {diff.max():.4g}, {int((diff > 0).sum())} values)")
                view = describe(state.env_state)
                out.write(json.dumps(dict(
                    episode_id=trajectory["episode_id"], step=step, seed=trajectory["seed"],
                    task_index=trajectory["task_index"], instruction=trajectory["instruction"],
                    episode_steps=trajectory["steps"], episode_success=trajectory["success"],
                    teacher_action=info["teacher_action"],
                    teacher_probs=info["teacher_action_probabilities"],
                    next_actions=actions[step + 1: step + 1 + HINDSIGHT],
                    view=view, observation_text=observation_text(view))) + "\n")
                written += 1
                key, step_key = jax.random.split(key)
                obs, state, _, _, _ = step_fn(step_key, state, jnp.int32(action_ids[info["teacher_action"]]), env_params)
            print(json.dumps(dict(episode_id=trajectory["episode_id"], steps=len(steps), frames_match=True,
                                  max_frame_diff=max_diff)), flush=True)
    print(json.dumps(dict(episodes=len(trajectories), transitions=written, max_frame_diff=max_diff,
                          out=str(args.out))), flush=True)


if __name__ == "__main__":
    main()
