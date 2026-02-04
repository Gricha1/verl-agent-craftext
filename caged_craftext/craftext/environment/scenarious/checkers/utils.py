import jax

from jax import numpy as jnp, lax
from functools import partial

@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1
    # padded_map = jnp.pad(game_map, max_radius, mode='constant')

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    # Затем обрезаем (маскируем) лишнее, так как radius может быть меньше max_radius
    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked