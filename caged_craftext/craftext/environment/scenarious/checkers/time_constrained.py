import jax.numpy as jnp
import jax.lax as lax
import jax
from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import TimeCosntrainedPlacmentState
from craftext.environment.scenarious.checkers.utils import safe_dynamic_slice



def checker_time_placement(game_data: Union[GameData, GameDataClassic],  target_state: TimeCosntrainedPlacmentState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    time_state = target_state.time_state
    # return jax.lax.select(target_state.need_to_achieve, 
    #                time_place_checker(game_data, block_index, radius, time_state),
    #                jnp.array(False))
    return at_time_block_placed(game_data, block_index, radius, time_state)

def at_time_block_placed(game_data: Union[GameData, GameDataClassic], block_index, radius, time_state) -> jax.Array:
    
    x, y = game_data.states[0].variables.player_position
    region = safe_dynamic_slice(game_data.states[0].map.game_map, x, y, radius, 5)
    in_range = jnp.abs(game_data.states[0].variables.light_level - jax.lax.clamp(0, time_state, 1)) <= 0.2
    return in_range & (region == block_index).any()