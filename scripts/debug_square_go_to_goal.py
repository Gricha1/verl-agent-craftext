#!/usr/bin/env python3
"""
Fast manual test: load debug_square_8x8 and walk to the goal (no LLM).

Same Craftext/CMDP stack as PPO; fixed action sequence UP/LEFT/RIGHT/DOWN.

Usage:
  python3 scripts/debug_square_go_to_goal.py
  python3 scripts/debug_square_go_to_goal.py --task wood
  python3 scripts/debug_square_go_to_goal.py --task wood --gif gif/wood.gif
  bash examples/ppo_trainer/debug_square_go_to_wood.sh
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from typing import List, Sequence, Tuple

# Craftax Action enum ids
ACTION_NAMES = (
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP",
    "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD",
)
UP, DOWN, LEFT, RIGHT = 3, 4, 1, 2

TASKS: dict[str, dict] = {
    "stone": {
        "instruction_idx": 0,
        "goal_cell": (1, 1),
        "actions": (UP, UP, UP, LEFT, LEFT),
    },
    "wood": {
        "instruction_idx": 1,
        "goal_cell": (1, 6),
        "actions": (UP, UP, UP, RIGHT),
    },
    "water": {
        "instruction_idx": 2,
        "goal_cell": (6, 1),
        "actions": (DOWN, DOWN, LEFT, LEFT),
    },
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _setup_paths() -> None:
    root = _repo_root()
    caged = os.environ.get("CAGED_CRAFTEXT_PATH", os.path.join(root, "caged_craftext"))
    craftax = os.path.join(caged, "Craftax")
    for p in (craftax, caged, root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _pos(craftax_state) -> Tuple[int, int]:
    p = craftax_state.player_position
    return int(p[0]), int(p[1])


def _goal_debug(text_state, instruction_idx: int) -> dict:
    import jax.numpy as jnp

    from craftext.environment.debug_square_rewards import (
        chebyshev_distance,
        debug_square_adjacent_to_goal,
        debug_square_resolve_target_cell,
    )

    ach_mask = text_state.target_state.achievements.achievement_mask
    craftax = text_state.env_state
    target = debug_square_resolve_target_cell(ach_mask, instruction_idx=instruction_idx)
    return {
        "goal_cell": tuple(int(x) for x in jnp.asarray(target).tolist()),
        "chebyshev": int(chebyshev_distance(craftax.player_position, target)),
        "goal_ok": bool(
            debug_square_adjacent_to_goal(
                craftax.player_position,
                ach_mask,
                env_state=craftax,
                instruction_idx=instruction_idx,
            )
        ),
    }


def _make_debug_square_base_env():
    """Minimal copy of caged_craftext envs._make_craftax_classic_pixels_env (no agent_system import)."""
    import importlib.util

    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnvNoAutoReset
    from craftax.craftax_classic.envs.craftax_state import StaticEnvParams

    caged_root = os.environ.get(
        "CAGED_CRAFTEXT_PATH", os.path.join(_repo_root(), "caged_craftext")
    )
    gen_path = os.path.join(
        caged_root, "Craftax/craftax/craftax_classic/debug_square_world_gen.py"
    )
    spec = importlib.util.spec_from_file_location("debug_square_world_gen", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)
    generate_debug_square_world = gen_mod.generate_debug_square_world

    static_params = StaticEnvParams(map_size=(8, 8))
    env = CraftaxClassicPixelsEnvNoAutoReset(static_env_params=static_params)
    env._caged_debug_env_params = env.default_params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )

    def reset_env(rng, params):
        state = generate_debug_square_world(rng, params, env.static_env_params)
        return env.get_obs(state), state

    env.reset_env = reset_env
    return env


def _make_env():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
    from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
    from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
    from craftext.environment.debug_square_rewards import is_debug_square_config
    from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
    from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
    from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

    def render_frame(state):
        return np.asarray(render_classic(state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN))

    def render_text(state):
        pos = tuple(int(x) for x in np.asarray(jax.device_get(state.player_position)).tolist())
        return f"(ascii map omitted — pos={pos}; use --gif for pixels)"

    base_env = _make_debug_square_base_env()
    wrapper = CMDPInstructionWrapper(
        env=base_env,
        config_name="debug_square_8x8",
        scenario_handler_class=ScenariosNoLambdaCMDP,
        encode_model_class=DistilBertEncode,
        encode_form=EncodeForm.EMBEDDING,
    )
    assert is_debug_square_config(wrapper.config_name)
    params = wrapper.env.default_params
    step_fn = jax.jit(wrapper.step, static_argnames=["env_params"])
    reset_fn = jax.jit(wrapper.reset, static_argnames=["env_params"])
    return wrapper, params, step_fn, reset_fn, render_frame, render_text


def run_episode(
    *,
    task: str,
    seed: int,
    gif_path: str | None,
    extra_steps: int,
) -> bool:
    _setup_paths()
    # Unset CRAFTAX_RELOAD_TEXTURES unless explicitly enabled (string "False" is truthy in Python).
    if os.environ.get("CRAFTAX_RELOAD_TEXTURES", "").lower() not in ("1", "true", "yes"):
        os.environ.pop("CRAFTAX_RELOAD_TEXTURES", None)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    spec = TASKS[task]
    instruction_idx = int(spec["instruction_idx"])
    planned_actions: Sequence[int] = spec["actions"]
    goal_cell = spec["goal_cell"]

    wrapper, params, step_fn, reset_fn, render_frame, render_text = _make_env()
    instructions = wrapper.scenario_handler.scenario_data.instructions_list
    instruction = instructions[instruction_idx]

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    _, state = reset_fn(sub, params, instruction_idx=instruction_idx)

    print("=" * 72)
    print(f"task={task!r}  instruction_idx={instruction_idx}")
    print(f"instruction: {instruction}")
    print(f"expected goal cell: {goal_cell}")
    print(f"planned: {[ACTION_NAMES[a] for a in planned_actions]}")
    print("=" * 72)

    frames: List = []
    if gif_path:
        frames.append(render_frame(state.env_state))

    total_reward = 0.0
    steps = 0
    finished = False

    def _print_step(action_id: int, reward, done, text_state) -> None:
        dbg = _goal_debug(text_state, instruction_idx)
        pos = _pos(text_state.env_state)
        print(
            f"step {steps:2d}  {ACTION_NAMES[action_id]:5s}  pos={pos}  "
            f"reward={float(reward):+6.3f}  cum={total_reward:+6.3f}  "
            f"chebyshev={dbg['chebyshev']}  goal_ok={dbg['goal_ok']}  "
            f"instruction_done={bool(text_state.instruction_done)}  done={bool(done)}"
        )

    for action_id in planned_actions:
        steps += 1
        key, sub = jax.random.split(key)
        _, state, reward, done, _ = step_fn(
            sub, state, jnp.int32(action_id), env_params=params
        )
        total_reward += float(reward)
        _print_step(action_id, reward, done, state)
        if gif_path:
            frames.append(render_frame(state.env_state))
        if bool(done):
            finished = True
            print("  -> episode ended")
            break

    if not finished and extra_steps > 0:
        print(f"-- extra NOOP (max {extra_steps}) --")
        for _ in range(extra_steps):
            if finished:
                break
            steps += 1
            key, sub = jax.random.split(key)
            _, state, reward, done, _ = step_fn(
                sub, state, jnp.int32(0), env_params=params
            )
            total_reward += float(reward)
            _print_step(0, reward, done, state)
            if gif_path:
                frames.append(render_frame(state.env_state))
            if bool(done):
                finished = True
                print("  -> episode ended")
                break

    dbg = _goal_debug(state, instruction_idx)
    print("-" * 72)
    print(
        f"final pos={_pos(state.env_state)}  goal={dbg['goal_cell']}  "
        f"chebyshev={dbg['chebyshev']}  goal_ok={dbg['goal_ok']}  "
        f"finished={finished}  total_reward={total_reward:+.3f}"
    )
    ascii_obs = render_text(jax.device_get(state.env_state))
    print("-" * 72)
    print(ascii_obs)

    if gif_path and frames:
        import imageio

        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        imageio.mimsave(gif_path, frames, duration=0.4, loop=0)
        print(f"[INFO] saved GIF: {gif_path} ({len(frames)} frames)")

    if finished:
        print("OK: task completed")
    else:
        print("FAIL: episode did not terminate at goal")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk agent to debug_square goal (no LLM)")
    parser.add_argument("--task", choices=tuple(TASKS.keys()), default="wood")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", type=str, default="", help="optional GIF path")
    parser.add_argument("--extra-noop-steps", type=int, default=3)
    args = parser.parse_args()

    ok = run_episode(
        task=args.task,
        seed=args.seed,
        gif_path=args.gif or None,
        extra_steps=max(0, args.extra_noop_steps),
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
