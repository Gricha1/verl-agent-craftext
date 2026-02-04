import jax
import jax.numpy as jnp
from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftax.craftax_classic.constants import DIRECTIONS
from craftax.craftax_classic.constants import Action

from craftext.environment.scenarious.checkers.target_state_cmdp import StepOnBlock

def checker_dont_sleep_near_monsters(game_data: Union[GameDataClassic, GameData]) -> jax.Array:
    is_sleeping = game_data.states[0].variables.is_sleeping
    distance = 4
    monsters_are_nearby = is_mob_near_than_n(game_data, distance=distance)
    
    return jnp.logical_and(is_sleeping, monsters_are_nearby)

def is_mob_near_than_n(game_data: GameDataClassic, distance: int):
    player_position = game_data.states[0].variables.player_position
    
    zombies_positions = game_data.states[0].zombies.position
    skeletones_positions = game_data.states[0].skeletons.position
    
    deltas_zombies = (jnp.sum(jnp.abs(zombies_positions - player_position), axis=1) < (distance)).any()
    deltas_skeletons = (jnp.sum(jnp.abs(skeletones_positions - player_position), axis=1) < (distance)).any()

    return jnp.logical_or(deltas_zombies, deltas_skeletons) 


