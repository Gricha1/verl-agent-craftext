#!/usr/bin/env python3
"""Smoke-test debug_square_8x8: termination conditions and shortest possible episodes."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAGED = os.path.join(REPO, "caged_craftext")
if CAGED not in sys.path:
    sys.path.insert(0, CAGED)
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

from craftext.environment.craftext_constants import Achievement, AchievementState
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.environment.scenarious.checkers.achivments import checker_acvievments
from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP
from craftext.environment.states.state_classic import GameDataClassic

from agent_system.environments.env_package.caged_craftext.envs import (
    _craftax_env_params,
    _make_craftax_classic_pixels_env,
)

ACTION_NAMES = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP",
    "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)


def _mask_summary(mask):
    mask = jnp.array(mask, dtype=jnp.int32)
    need = [i for i in range(len(mask)) if int(mask[i]) == AchievementState.NEED_TO_ACHIEVE]
    avoid = [i for i in range(len(mask)) if int(mask[i]) == AchievementState.AVOID_TO_ACHIEVE]
    return need, avoid


def main():
    env_kwargs = {"config_name": "debug_square_8x8", "use_debug_square_map": True}
    env = _make_craftax_classic_pixels_env(env_kwargs)
    wrapper = CMDPInstructionWrapper(
        env=env,
        config_name="debug_square_8x8",
        scenario_handler_class=ScenariosNoLambdaCMDP,
        encode_model_class=DistilBertEncode,
        encode_form=EncodeForm.EMBEDDING,
    )
    params = _craftax_env_params(wrapper.env)
    n_instr = len(wrapper.scenario_handler.scenario_data.instructions_list)
    print(f"instructions: {n_instr}")

    key = jax.random.PRNGKey(0)
    step_fn = jax.jit(wrapper.step, static_argnames=["env_params"])

    # 1) instruction_done at reset (before any action)?
    print("\n=== done_at_reset (should all be False) ===")
    for idx in range(n_instr):
        key, sub = jax.random.split(key)
        _, state = wrapper.reset(sub, params, instruction_idx=idx)
        ts = wrapper.batched_ts.select(state.idx)
        need, avoid = _mask_summary(ts.achievements.achievement_mask)
        gd = GameDataClassic.from_state(state.env_state, state.env_state, jnp.int32(0))
        done_at_reset = bool(checker_acvievments(gd, ts.achievements))
        pos = tuple(map(int, state.env_state.player_position))
        print(f"  idx={idx} pos={pos} need={need} avoid={avoid} done_at_reset={done_at_reset}")
        if done_at_reset:
            print("  ERROR: checker True at reset — would allow length-0/1 metrics")

    # 2) done on first env step for every (instruction, action)
    print("\n=== done on first step (instruction_idx, action) ===")
    early = []
    for idx in range(n_instr):
        for action in range(17):
            key, sub = jax.random.split(key)
            _, state = wrapper.reset(sub, params, instruction_idx=idx)
            key, sub = jax.random.split(key)
            _, state, reward, done, _ = step_fn(sub, state, jnp.int32(action), env_params=params)
            if bool(done):
                ach = jnp.array(state.env_state.achievements)
                early.append((idx, action, float(reward), [int(i) for i in range(len(ach)) if bool(ach[i])]))
    if early:
        for idx, action, reward, ach_on in early:
            print(
                f"  DONE idx={idx} action={ACTION_NAMES[action]} ({action}) "
                f"reward={reward:.3f} achievements_on={ach_on}"
            )
        print(f"\nFound {len(early)} spurious first-step terminations.")
    else:
        print("  None — shortest real done is >= 2 steps (run multi-step scan next)")

    # 3) shortest number of steps until done (greedy: DO, then moves toward corner)
    print("\n=== shortest path scan (BFS actions, all instructions) ===")
    for idx in range(n_instr):
        best = None
        # BFS up to depth 12
        from collections import deque

        key, sub = jax.random.split(key)
        _, start = wrapper.reset(sub, params, instruction_idx=idx)
        q = deque([(start, [])])
        visited = set()
        while q:
            state, seq = q.popleft()
            if len(seq) > 12:
                continue
            sig = (tuple(map(int, state.env_state.player_position)), tuple(state.env_state.achievements.tolist()))
            if sig in visited:
                continue
            visited.add(sig)
            for action in range(17):
                key, sub = jax.random.split(key)
                _, nstate, reward, done, _ = step_fn(sub, state, jnp.int32(action), env_params=params)
                if bool(done):
                    if best is None or len(seq) + 1 < best[0]:
                        best = (len(seq) + 1, list(seq) + [action], float(reward))
                    continue
                if len(seq) + 1 <= 12:
                    q.append((nstate, seq + [action]))
        if best:
            n_steps, actions, reward = best
            names = [ACTION_NAMES[a] for a in actions]
            print(f"  idx={idx} shortest_done_steps={n_steps} reward={reward:.3f} actions={names}")
        else:
            print(f"  idx={idx} no done within 12 steps")

    print("\nDone.")


if __name__ == "__main__":
    main()
