import jax.numpy as jnp
import jax
from typing import Tuple

def check_square_2x2(center: Tuple[int, int], region: jax.Array) -> jax.Array:

    i, j = center

    return jnp.all(jnp.array([
        region[i    ,     j],
        region[i + 1,     j],
        region[i    , j + 1],
        region[i + 1, j + 1],

        region[i - 1, j - 1] != 1,
        region[i - 1, j    ] != 1,
        region[i - 1, j + 1] != 1,
        region[i - 1, j + 2] != 1,
        region[i    , j - 1] != 1,
        region[i + 1, j - 1] != 1,
        region[i + 2, j - 1] != 1,
        region[i + 2, j    ] != 1,
        region[i + 2, j + 1] != 1,
        region[i + 2, j + 2] != 1,
        region[i    , j + 2] != 1,
        region[i + 1, j + 2] != 1
    ]))


def check_square_3x3(center: Tuple[int, int], region: jax.Array) -> jax.Array:
    i, j = center

    return jnp.all(jnp.array([
        region[i, j]   ,
        region[i-1, j] ,
        region[i+1, j] ,
        region[i, j-1] ,
        region[i, j+1] ,
        region[i-1, j-1],
        region[i-1, j+1],
        region[i+1, j-1],
        region[i+1, j+1],

        region[i-2, j-2] != 1,
        region[i-2, j-1] != 1,
        region[i-2, j]   != 1,
        region[i-2, j+1] != 1,
        region[i-2, j+2] != 1,
        region[i-1, j-2] != 1,
        region[i, j-2]  != 1,
        region[i+1, j-2] != 1,
        region[i+2, j-2] != 1,
        region[i+2, j-1] != 1,
        region[i+2, j]  != 1,
        region[i+2, j+1] != 1,
        region[i+2, j+2] != 1,
        region[i-1, j+2] != 1,
        region[i, j+2] != 1,
        region[i+1, j+2] != 1
    ]))


def check_square_4x4(center: Tuple[int, int], region: jax.Array) -> jax.Array:
    i, j = center

    return jnp.all(jnp.array([
        region[i, j] ,
        region[i+1, j] ,
        region[i+2, j] ,
        region[i+3, j] ,
        region[i, j+1] ,
        region[i+1, j+1] ,
        region[i+2, j+1] ,
        region[i+3, j+1] ,
        region[i, j+2] ,
        region[i+1, j+2] ,
        region[i+2, j+2] ,
        region[i+3, j+2] ,
        region[i, j+3] ,
        region[i+1, j+3] ,
        region[i+2, j+3] ,
        region[i+3, j+3] ,

        region[i-1, j-1] != 1,
        region[i-1, j] != 1,
        region[i-1, j+1] != 1,
        region[i-1, j+2] != 1,
        region[i-1, j+3] != 1,
        region[i-1, j+4] != 1,
        region[i, j-1] != 1,
        region[i+1, j-1] != 1,
        region[i+2, j-1] != 1,
        region[i+3, j-1] != 1,
        region[i+4, j-1] != 1,
        region[i+4, j] != 1,
        region[i+4, j+1] != 1,
        region[i+4, j+2] != 1,
        region[i+4, j+3] != 1,
        region[i+4, j+4] != 1,
        region[i, j+4] != 1,
        region[i+1, j+4] != 1,
        region[i+2, j+4] != 1,
        region[i+3, j+4] != 1
    ]))