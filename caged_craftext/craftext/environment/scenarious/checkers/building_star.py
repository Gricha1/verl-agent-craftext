import jax.numpy as jnp
import jax.lax as lax
import jax

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import BuildStarState

from functools import partial


import jax
import jax.numpy as jnp

def checker_star(game_data: Union[GameDataClassic, GameData],  target_state: BuildStarState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    cross_type = target_state.cross_type

    # return jax.lax.select(target_state.need_to_achieve, 
                #    is_cross_formed(10, 7, game_data, block_index, radius, size, cross_type),
                #    jnp.array(False))
    return is_cross_formed(10, 7, game_data, block_index, radius, size, cross_type)

@partial(jax.jit, static_argnums=(0,1))
def is_cross_formed(
    max_radius: int,
    max_size:   int,
    game_data,
    block_index: int,    
    cross_type:  int,    
    radius:      int,
    size:        int     
):

    x, y = game_data.states[0].variables.player_position
    R   = max_radius
    FULL = 2*R + 1
    padded = jnp.pad(
        game_data.states[0].map.game_map,
        ((R, R), (R, R)),
        constant_values=-1
    )
    region_full = lax.dynamic_slice(padded, (x, y), (FULL, FULL))  

    coords = jnp.arange(-R, R+1)                         
    mask1d = jnp.abs(coords) <= radius                   
    mask2d = mask1d[:, None] & mask1d[None, :]           
    region = jnp.where(mask2d, region_full, -1)

    B = (region == block_index).astype(jnp.float32)[None, None, ...]

    S = max_size
    C = S // 2  

    idxs = jnp.arange(S)  

    half = size // 2      
    start = C - half      
    end   = start + size  

    mask_range = (idxs >= start) & (idxs < end) 

    row_idx = idxs[:, None]      
    col_idx = idxs[None, :]      
    filt_h = (row_idx == C) & mask_range[None, :]  
    filt_v = (col_idx == C) & mask_range[:, None]  
    filt_d1 = (row_idx == col_idx) & mask_range[:, None] & mask_range[None, :]
    filt_d2 = (row_idx + col_idx == 2*C) & mask_range[:, None] & mask_range[None, :]

    kh = filt_h.astype(jnp.float32)
    kv = filt_v.astype(jnp.float32)
    kd1 = filt_d1.astype(jnp.float32)
    kd2 = filt_d2.astype(jnp.float32)

    conv = partial(lax.conv_general_dilated,
                   window_strides=(1,1),
                   padding="VALID",
                   dimension_numbers=("NCHW","OIHW","NCHW"))

    h_out  = conv(B, kh [None, None])[0,0]  
    v_out  = conv(B, kv [None, None])[0,0]
    d1_out = conv(B, kd1[None, None])[0,0]
    d2_out = conv(B, kd2[None, None])[0,0]

    straight = (h_out  == size) & (v_out  == size)
    diagonal = (d1_out == size) & (d2_out == size)

    mask = lax.cond(
        cross_type == 0,
        lambda _: straight,
        lambda _: lax.cond(
            cross_type == 1,
            lambda _: diagonal,
            lambda _: straight | diagonal,
            operand=None
        ),
        operand=None
    )

    return jnp.any(mask)

