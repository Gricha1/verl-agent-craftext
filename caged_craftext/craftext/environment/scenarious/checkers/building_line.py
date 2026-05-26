import jax

from jax import (
    numpy as jnp,
    lax
)

from typing import Tuple, Callable 

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import BuildLineState
from craftext.environment.scenarious.checkers.lines import check_line_2, check_line_3, check_line_4
from craftext.environment.scenarious.checkers.utils import extract_square_region

from flax.struct import dataclass

def checker_line(game_data: Union[GameDataClassic, GameData],  target_state: BuildLineState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    check_diagonal = target_state.is_diagonal

    # return jax.lax.select(target_state.need_to_achieve, 
    #                checker_line(game_data, block_index, radius, size, check_diagonal),
    #                jnp.array(False))
    return is_line_formed(game_data, block_index, radius, size, check_diagonal)

@dataclass
class Carry:
    region: jax.Array
    region_size: int
    size: int
    check_diagonal: bool

def is_line_formed(game_data: Union[GameDataClassic, GameData], block_index: int, radius: int, length: int, check_diagonal: bool) -> jax.Array:
    
    game_map =  game_data.states[0].map.game_map
    
    binary_map = (game_map == block_index).astype(jnp.int32)
    
    player_position = game_data.states[0].variables.player_position
    
    x, y = player_position

    # Static region_size: required under jit (generic_check lax.switch traces every branch).
    region_size = 21
    region = extract_square_region(binary_map, x, y, radius, region_size, pad_value=0)

    indices = jnp.arange(region_size * region_size)

    carry = Carry(region, region_size, length, check_diagonal)
    
    _, lines = lax.scan(scan_line_function, carry, indices)
    return jnp.any(lines)

def check_line_by_size(center: Tuple[int, int], region: jax.Array, size: int, check_diagonal: bool) -> Callable:
    i, j = center

    func = jax.lax.switch(
        size - 2, 
        [
            lambda: check_line_2((i, j), region, check_diagonal),
            lambda: check_line_3((i, j), region, check_diagonal),
            lambda: check_line_4((i, j), region, check_diagonal)
        ]
    )
        
    return func

def scan_line_function(carry: Carry, x):
    region = carry.region
    region_size = carry.region_size
    size = carry.size
    check_diagonal = carry.check_diagonal
    
    i, j = x // region_size, x % region_size
    is_line = check_line_by_size((i, j), region, size, check_diagonal)
    return carry, is_line