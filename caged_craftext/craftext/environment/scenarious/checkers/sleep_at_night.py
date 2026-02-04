import jax

from jax import (
    numpy as jnp,
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_night_day_light import LightLevelState


from typing import Union



def checker_sleep_at_night(game_data: Union[GameDataClassic, GameData],  night_constraint_level: LightLevelState, day_constraint_level: LightLevelState) -> jax.Array:
    night = night_constraint_level.level
    day = day_constraint_level.level
    return is_sleep_at_night(game_data, night, day)

def is_sleep_at_night(game_data: Union[GameDataClassic, GameData], night, day):
    is_sleep = game_data.states[0].variables.is_sleeping               # jax.Array с целочисленным кодом действия
    light_level = game_data.states[0].variables.light_level  # jax.Array с уровнем освещённости


    #night
    threshold = night
    is_dark_enough = light_level < threshold
    

    # day
    threshold = day
    is_light_ebove = light_level > threshold

    return jnp.logical_and(is_dark_enough, jnp.logical_not(is_sleep)) | (jnp.logical_and(is_light_ebove, is_sleep))
    

