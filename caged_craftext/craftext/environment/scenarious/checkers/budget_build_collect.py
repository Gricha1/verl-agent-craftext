import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_budget_energy_level import EnergyLevelState
from craftext.environment.craftext_constants import BlockType

def checker_budget_build_collect(game_data: Union[GameDataClassic, GameData],  target_state: EnergyLevelState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    block_type = target_state.level
    return new_item(game_data, block_type)

def new_item(game_data: Union[GameDataClassic, GameData], block_type: int):
    block_type  = block_type

    def get_item(index, inventory):
        leaves, _ = tree_util.tree_flatten(inventory)
        leaves = jnp.stack(leaves)
        return leaves[index]
    
    
    
    
    curr = get_item(block_type, game_data.states[0].inventory)
    prev = get_item(block_type, game_data.states[1].inventory)
    return (curr - prev) > 0


