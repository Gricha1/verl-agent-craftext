import jax
from jax import (
    numpy as jnp,
    lax
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import LocalizaPlacingState
from functools import partial


@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked


def cheker_localization(game_data: Union[GameDataClassic, GameData], target_state: LocalizaPlacingState) -> jax.Array:
    
    object_name = target_state.object_name
    target_object_name = target_state.target_object_name 
    side = target_state.side 
    distance = target_state.distance

    # return jax.lax.select(target_state.need_to_achieve, 
    #                localization_checker(game_data=game_data, object_name=object_name, target_object_name=target_object_name, side=side, distance=distance),
    #                jnp.array(False))
    return place_object_relevant_to(game_data=game_data, object_name=object_name, target_object_name=target_object_name, side=side, distance=distance)


from functools import partial
# Фиксированный радиус, которым мы захватываем все возможные distance ≤ MAX_RADIUS
MAX_RADIUS = 5
REGION_SIZE = 2 * MAX_RADIUS + 1  # статичный

@jax.jit
def place_object_relevant_to(
    game_data: Union[GameDataClassic, GameData], 
    object_name: str, 
    target_object_name: str, 
    side: int,      # tracer-скаляр: 0=Right,1=Left,2=Top,3=Bottom
    distance: int   # tracer-скаляр, ≤ MAX_RADIUS
) -> jax.Array:
    # 1) Собираем единый REGION вокруг игрока размером REGION_SIZE×REGION_SIZE
    x, y = game_data.states[0].variables.player_position
    padded_map = jnp.pad(
        game_data.states[0].map.game_map,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=-1  # любая метка за границей
    )
    region = lax.dynamic_slice(
        padded_map,
        (x, y),
        (REGION_SIZE, REGION_SIZE)
    )  # → shape [REGION_SIZE, REGION_SIZE]

    # 2) Делаем булевы карты для цели и объекта
    #    (можно сразу сравнить с .value, или если у вас int-коды — без .value)
    tgt_mask = (region == target_object_name)
    obj_mask = (region == object_name)

    # 3) Чтобы сдвинуть obj_mask на (side, distance), ещё раз паддим region/obj_mask
    #    на MAX_RADIUS ║ MAX_RADIUS, и будем «секурно» брать кусок REGION_SIZE×REGION_SIZE
    padded_obj = jnp.pad(
        obj_mask,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=False
    )

    # 4) Вычисляем динамический сдвиг в координатах padded_obj
    #    Право  (0):  di= 0, dj=+distance
    #    Лево  (1):  di= 0, dj=-distance
    #    Верх  (2):  di=-distance, dj=0
    #    Низ   (3):  di=+distance, dj=0
    di = jnp.where(side==2, -distance,
         jnp.where(side==3,  distance, 0))
    dj = jnp.where(side==0,  distance,
         jnp.where(side==1, -distance, 0))

    # Стартовый индекс в padded_obj: центр + (di,dj)
    start_i = MAX_RADIUS + di
    start_j = MAX_RADIUS + dj

    # 5) Единичный dynamic_slice, **фиксированный** REGION_SIZE×REGION_SIZE
    shifted_obj = lax.dynamic_slice(
        padded_obj,
        (start_i, start_j),
        (REGION_SIZE, REGION_SIZE)
    )

    # 6) Проверяем, есть ли позиция (i,j), где одновременно tgt_mask[i,j] и shifted_obj[i,j]
    hit = tgt_mask & shifted_obj

    # 7) Нужен ли `need_to_achieve`? Если да, поднимайте его снаружи через select()  
    return jnp.any(hit)
