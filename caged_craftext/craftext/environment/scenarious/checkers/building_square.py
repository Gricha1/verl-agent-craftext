import jax

from jax import (
    numpy as jnp,
    lax
)
from flax.struct import dataclass
from typing import Tuple

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import BuildSquareState
from craftext.environment.scenarious.checkers.squeres import (
    check_square_2x2, 
    check_square_3x3, 
    check_square_4x4
)

@dataclass
class Carry:
    region: jax.Array
    region_size: int
    size: int

def checker_square(game_data: Union[GameDataClassic, GameData],  target_state: BuildSquareState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    
    # return jax.lax.select(target_state.need_to_achieve, 
    #                is_square_formed(game_data, block_index, radius, size),
    #                jnp.array(False))
    return is_square_formed(game_data, block_index, radius, size)

def is_square_formed(game_data: Union[GameDataClassic, GameData], block_index: int, radius: int, size: int) -> jax.Array:

    game_map = game_data.states[0].map.game_map
    
    binary_map = (game_map == block_index).astype(jnp.int32)

    player_position = game_data.states[0].variables.player_position
    
    x, y = player_position

    region_size = 2 * 10 + 1

    region = lax.dynamic_slice(
        binary_map,
        start_indices=(x - radius, y - radius),
        slice_sizes=(region_size, region_size)
    )

    indices = jnp.arange(region_size * region_size)

    carry = Carry(region, region_size, size)
    
    _, squares = lax.scan(scan_square_function, carry, indices)
    
    return jnp.any(squares)

def check_square_by_size(center: Tuple[int, int], region: jax.Array, size: int):
    i, j = center

    return jax.lax.switch(
        size - 2, 
        [
            lambda: check_square_2x2((i, j), region),
            lambda: check_square_3x3((i, j), region),
            lambda: check_square_4x4((i, j), region)
        ]
    )

def scan_square_function(carry: Carry, x):
    region = carry.region
    region_size = carry.region_size
    size = carry.size

    i, j = x // region_size, x % region_size
    is_square = check_square_by_size(center=(i, j), region=region, size=size)
    return carry, is_square
