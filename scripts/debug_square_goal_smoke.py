#!/usr/bin/env python3
"""Smoke test: wood task completes when agent stands left of wooden block."""
from __future__ import annotations

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAGED = os.path.join(REPO, "caged_craftext")
CRAFTAX = os.path.join(CAGED, "Craftax")
for p in (CRAFTAX, CAGED, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
from craftext.environment.debug_square_rewards import (
    debug_square_adjacent_to_goal,
    debug_square_at_goal_from_map,
    is_debug_square_config,
)
from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

from agent_system.environments.env_package.caged_craftext.envs import (
    _craftax_env_params,
    _make_craftax_classic_pixels_env,
)

WOOD_IDX = 1
# Known adjacency cells for wood at (1, 6) on the fixed debug map.
WOOD_ADJACENT = [(1, 5), (2, 5), (2, 6)]


def main() -> None:
    env_kwargs = {
        "config_name": "debug_square_8x8",
        "use_debug_square_map": True,
        "observation_type": "ascii",
    }
    env = _make_craftax_classic_pixels_env(env_kwargs)
    wrapper = CMDPInstructionWrapper(
        env=env,
        config_name="debug_square_8x8",
        scenario_handler_class=ScenariosNoLambdaCMDP,
        encode_model_class=DistilBertEncode,
        encode_form=EncodeForm.EMBEDDING,
    )
    assert is_debug_square_config(wrapper.config_name), wrapper.config_name
    params = _craftax_env_params(wrapper.env)
    step = jax.jit(wrapper.step, static_argnames=["env_params"])
    reset = jax.jit(wrapper.reset, static_argnames=["env_params"])

    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    _, state = reset(sub, params, instruction_idx=WOOD_IDX)
    craftax = state.env_state
    print(f"spawn={tuple(map(int, craftax.player_position))} wood@ (1,6)")

    for target in WOOD_ADJACENT:
        craftax = craftax.replace(player_position=jnp.array(target, dtype=jnp.int32))
        state = state.replace(env_state=craftax)
        ach_mask = state.target_state.achievements.achievement_mask
        done_map = bool(debug_square_at_goal_from_map(craftax, ach_mask, instruction_idx=WOOD_IDX))
        done_all = bool(
            debug_square_adjacent_to_goal(
                craftax.player_position,
                ach_mask,
                env_state=craftax,
                instruction_idx=WOOD_IDX,
            )
        )
        print(f"  pos={target} map_ok={done_map} goal_ok={done_all}")
        assert done_all, f"expected success at {target}"

    # One env step while standing at (1, 5) should terminate the episode.
    craftax = craftax.replace(player_position=jnp.array([1, 5], dtype=jnp.int32))
    state = state.replace(env_state=craftax)
    key, sub = jax.random.split(key)
    _, state, reward, done, _ = step(sub, state, jnp.int32(0), env_params=params)
    print(f"step@ (1,5): done={bool(done)} reward={float(reward):.3f}")
    assert bool(done), "episode must end when adjacent to wooden block"
    print("OK: debug_square wood goal detection works")

if __name__ == "__main__":
    main()
