import jax

from jax import numpy as jnp, lax
from functools import partial


def extract_square_region(game_map, x, y, radius, region_size: int, pad_value=-1):
    """
    JIT-safe region extract for lax.switch-traced checkers.
    Pads maps smaller than region_size (e.g. 8x8 debug) so slice_sizes stay static.
    """
    map_h, map_w = game_map.shape
    pad_h = jnp.maximum(0, region_size - map_h)
    pad_w = jnp.maximum(0, region_size - map_w)
    padded = jnp.pad(game_map, ((0, pad_h), (0, pad_w)), constant_values=pad_value)
    start_x = jnp.clip(x - radius, 0, region_size - 1)
    start_y = jnp.clip(y - radius, 0, region_size - 1)
    return lax.dynamic_slice(
        padded,
        start_indices=(start_x, start_y),
        slice_sizes=(region_size, region_size),
    )


@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1
    region = extract_square_region(game_map, x, y, radius, full_region_size, pad_value=-1)

    # Затем обрезаем (маскируем) лишнее, так как radius может быть меньше max_radius
    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked