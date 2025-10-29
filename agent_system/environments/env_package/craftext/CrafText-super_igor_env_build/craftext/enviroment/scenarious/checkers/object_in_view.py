import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# Предполагаем, что GameData и GameDataClassic имеют одинаковую структуру 
# для карты и позиции игрока.
from craftext.enviroment.states.state import GameData
from craftext.enviroment.states.state_classic import GameDataClassic
from typing import Union

# Эта функция теперь принимает GameData и корректно работает с его структурой
@partial(jax.jit, static_argnames=['view_radius'])
def is_object_in_view(
    game_data: Union[GameData, GameDataClassic], 
    object_id: int, 
    view_radius: int
) -> jax.Array:
    """
    Проверяет, находится ли объект с заданным ID в квадрате обзора агента.
    Работает со структурой GameData.
    """
    region_size = 2 * view_radius + 1

    # ИЗМЕНЕНИЕ: Доступ к данным через структуру GameData
    x, y = game_data.states[0].variables.player_position
    game_map = game_data.states[0].map.game_map

    padded_map = jnp.pad(
        game_map,
        ((view_radius, view_radius), (view_radius, view_radius)),
        constant_values=-1
    )

    region = lax.dynamic_slice(
        padded_map,
        (x, y),
        (region_size, region_size)
    )

    return jnp.any(region == object_id)