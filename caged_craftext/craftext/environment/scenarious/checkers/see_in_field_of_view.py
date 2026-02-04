import jax

from jax import (
    numpy as jnp,
    lax
)
from jax import tree_util

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import BlockType

from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_see_in_field_of_view import SeeInView
from jax import numpy as jnp
from jax.lax import dynamic_slice
def checker_what_in_field_of_view(game_data: GameDataClassic,  target_state: SeeInView) -> jax.Array:
    block_type = target_state.block_type
    return is_block_in_fov(game_data, block_type=block_type)

def is_block_in_fov(game_data: GameDataClassic, block_type: BlockType) -> jax.Array:
    player_position = game_data.states[0].variables.player_position  # [x, y]
    game_map = game_data.states[0].map.game_map  # shape (H, W)

    px, py = player_position[0], player_position[1]  # assuming [x, y]

    H, W = game_map.shape
    fov_width, fov_height = 9, 7

    # Центр окна — на игроке → стартовые координаты
    x_start = px - 4  # 9 // 2 = 4
    y_start = py - 3  # 7 // 2 = 3

    # Гарантируем, что не выйдем за границы
    x_start = jnp.clip(x_start, 0, W - fov_width)
    y_start = jnp.clip(y_start, 0, H - fov_height)

    # Вырезаем поле зрения
    fov_map = dynamic_slice(
        operand=game_map,
        start_indices=(y_start, x_start),  # (y, x) — как форма массива
        slice_sizes=(fov_height, fov_width)
    )

    # Проверяем наличие блока
    return (fov_map != block_type).all()