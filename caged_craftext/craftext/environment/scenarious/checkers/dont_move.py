import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_night_light import LightLevelState

from craftax.craftax.constants import Action as ActionExtend
from craftax.craftax_classic.constants import Action as ActionClassic
from typing import Union

# class Action(ActionExtend, ActionClassic):
#     pass

def checker_moveing_at_night_level(game_data: Union[GameDataClassic, GameData],  target_state: LightLevelState) -> jax.Array:
    # raise NotImplementedError("checker_budget_build_collect is not implemented yet")
    light_level = target_state.level
    return is_moveing_at_night(game_data, light_level)

def is_moveing_at_night(game_data: Union[GameDataClassic, GameData], night):
    action = game_data.states[0].action               # jax.Array с целочисленным кодом действия
    light_level = game_data.states[0].variables.light_level  # jax.Array с уровнем освещённости

    # Порог из целевого состояния:
    threshold = night                     # число или jax.Array

    # Допустимые коды действий:
    allowed = jnp.array([
        ActionClassic.UP.value,
        ActionClassic.DOWN.value,
        ActionClassic.LEFT.value,
        ActionClassic.RIGHT.value,
    ], dtype=int)

    # Проверяем условия:
    is_dark_enough = light_level <= threshold
    is_allowed_action = jnp.isin(action, allowed)

    return jnp.logical_and(is_dark_enough, is_allowed_action)
    

