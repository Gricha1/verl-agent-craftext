import jax

from jax import (
    numpy as jnp,
)
from typing import Tuple

from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import TargetOfInterest


def checker_last_visible_target(game_data: GameDataClassic,  target_state: TargetOfInterest) -> Tuple[TargetOfInterest, jax.Array]:
    type_of_interest = target_state.object_of_interest
    distance = target_state.far_from_agent
    last_visible_target_position = target_state.last_visible_target_position
    last_visible_target_position, is_cost = is_mob_near_than_n(game_data, type_of_interest=type_of_interest, distance=distance, last_visible_target_position=last_visible_target_position)
    
    return TargetOfInterest(object_of_interest=type_of_interest, far_from_agent=target_state.far_from_agent, last_visible_target_position=last_visible_target_position), is_cost

def is_mob_near_than_n(
    game_data: GameDataClassic, 
    type_of_interest: jax.Array,
    distance: int, 
    last_visible_target_position: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    
    player_position = game_data.states[0].variables.player_position
    is_too_far = (jnp.sum(jnp.abs(player_position - last_visible_target_position)) > distance)

    # type_of_interest = type_of_interest.squeeze()

    def call_water_checker(_):
        return water_checker(game_data, distance, last_visible_target_position)

    def call_food_checker(_):
        return food_checker(game_data, distance, last_visible_target_position)

    near_target, is_cost = jax.lax.cond(
        type_of_interest == 0,
        call_water_checker,
        call_food_checker,
        operand=None
    )
    
    return near_target, is_too_far & is_cost
    
from craftax.craftax_classic.constants import BlockType
from .utils import safe_dynamic_slice



def water_checker(game_data: GameDataClassic, distance: int, last_visible_target_position: jax.Array):
    player_position = game_data.states[0].variables.player_position
    game_map = game_data.states[0].map.game_map
    x, y = player_position

    MAX_RADIUS = 4  # должно совпадать с max_radius в вызове safe_dynamic_slice

    local_view = safe_dynamic_slice(
        game_map=game_map,
        x=x,
        y=y,
        radius=distance,
        max_radius=MAX_RADIUS
    )

    is_water_in_sight = (local_view == BlockType.WATER.value) & (local_view != -1)
    
    max_candidates = (2 * MAX_RADIUS + 1) ** 2
    local_indices = jnp.argwhere(is_water_in_sight, size=max_candidates, fill_value=-1)

    is_valid = local_indices[:, 0] != -1
    is_water_found = is_valid.any()

    def find_closest_water(_):
        # Преобразуем локальные индексы в глобальные
        offsets = local_indices - MAX_RADIUS  # потому что центр — [MAX_RADIUS, MAX_RADIUS]
        global_coords = player_position + offsets

        # Манхэттенское расстояние
        distances = jnp.sum(jnp.abs(offsets), axis=1)
        distances = jnp.where(is_valid, distances, jnp.inf)
        closest_idx = jnp.argmin(distances)
        return global_coords[closest_idx]

    new_target_position = jax.lax.cond(
        is_water_found,
        find_closest_water,
        lambda _: last_visible_target_position,
        operand=None
    )

    return new_target_position, jnp.logical_not(is_water_found)

def stone_checker(game_data: GameDataClassic, distance: int, last_visible_target_position: jax.Array):
    player_position = game_data.states[0].variables.player_position
    game_map = game_data.states[0].map.game_map

    local_view = safe_dynamic_slice(
        game_map=game_map,
        x=player_position[0],
        y=player_position[1],
        radius=distance,
        max_radius=10
    )

    is_water_in_sight = (local_view == BlockType.STONE.value) & (local_view != -1)
    local_indices = jnp.argwhere(is_water_in_sight) 

    is_water_found = local_indices.shape[0] > 0

    def find_closest_water(_):
        offset = player_position - 10 
        global_coords = local_indices + offset  

        deltas = jnp.abs(global_coords - player_position)
        distances_sq = jnp.sum(deltas, axis=1)

        closest_idx = jnp.argmin(distances_sq)
        return global_coords[closest_idx]

    new_target_position = jax.lax.cond(
        is_water_found,
        find_closest_water,
        lambda _: last_visible_target_position,
        operand=None
    )

    return new_target_position, jnp.logical_not(is_water_found)

def tree_checker(game_data: GameDataClassic, distance: int, last_visible_target_position: jax.Array):
    player_position = game_data.states[0].variables.player_position
    game_map = game_data.states[0].map.game_map

    local_view = safe_dynamic_slice(
        game_map=game_map,
        x=player_position[0],
        y=player_position[1],
        radius=distance,
        max_radius=10
    )

    is_water_in_sight = (local_view == BlockType.TREE.value) & (local_view != -1)
    local_indices = jnp.argwhere(is_water_in_sight) 

    is_water_found = jax.lax.cond(local_indices.shape[0] > 0, lambda: True, lambda: False)

    def find_closest_water(_):
        offset = player_position 
        global_coords = local_indices + offset  

        deltas = jnp.abs(global_coords - player_position)
        distances_sq = jnp.sum(deltas, axis=1)

        closest_idx = jnp.argmin(distances_sq)
        return global_coords[closest_idx]

    new_target_position = jax.lax.cond(
        is_water_found,
        find_closest_water,
        lambda _: last_visible_target_position,
        operand=None
    )

    return new_target_position, jnp.array([jnp.logical_not(is_water_found)], dtype=jnp.bool_)

def food_checker(game_data: GameDataClassic, distance: int, last_visible_target_position: jax.Array):
    player_position = game_data.states[0].variables.player_position
    cows = game_data.states[0].cows.position 


    deltas = cows - player_position 
    distances = jnp.sum(jnp.abs(deltas), axis=1)


    within_radius = distances <= distance


    is_cow_found = within_radius.any() 

    def find_closest_cow(_):

        masked_distances = jnp.where(within_radius, distances, jnp.inf)
        closest_idx = jnp.argmin(masked_distances)
        return cows[closest_idx] 

    new_target_position = jax.lax.cond(
        is_cow_found,
        find_closest_cow,
        lambda _: last_visible_target_position,
        operand=None
    )

    return new_target_position, jnp.logical_not(is_cow_found)