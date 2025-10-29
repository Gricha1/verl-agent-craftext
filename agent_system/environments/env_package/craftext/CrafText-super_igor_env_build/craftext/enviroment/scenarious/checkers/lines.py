import jax
import jax.numpy as jnp
from typing import Tuple

def check_line_2(center: Tuple[int, int], region: jax.Array, check_diagonal: bool = False, stone_index: int = 0):

    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1]
        ])

        return (jnp.sum(region) == 3) & jnp.all(diagonal_1 == stone_index) | \
               (jnp.sum(region) == 3) & jnp.all(diagonal_2 == stone_index)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1]
        ])

        return (jnp.sum(region) == stone_index) & jnp.all(vertical == stone_index) | \
               (jnp.sum(region) == stone_index) & jnp.all(horizontal == stone_index)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)


def check_line_3(center: Tuple[int, int], region: jax.Array, check_diagonal: bool = False, stone_index: int = 0):

    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1],
            region[i + 2, j + 2]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1],
            region[i + 2, j - 2]
        ])

        return (jnp.sum(region) == 3) & jnp.all(diagonal_1 == 1) | \
               (jnp.sum(region) == 3) & jnp.all(diagonal_2 == 1)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j],
            region[i + 2, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1],
            region[i, j + 2]
        ])

        return (jnp.sum(region) == 3) & jnp.all(vertical == 1) | \
               (jnp.sum(region) == 3) & jnp.all(horizontal == 1)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)


def check_line_4(center: Tuple[int, int], region: jax.Array, check_diagonal=False, stone_index: int = 0):

    i, j = center

    def check_diagonal_lines(_):
        diagonal_1 = jnp.array([
            region[i, j],
            region[i + 1, j + 1],
            region[i + 2, j + 2],
            region[i + 3, j + 3]
        ])
        diagonal_2 = jnp.array([
            region[i, j],
            region[i + 1, j - 1],
            region[i + 2, j - 2],
            region[i + 3, j - 3]
        ])

        return (jnp.sum(region) == 4) & jnp.all(diagonal_1 == 1) | \
               (jnp.sum(region) == 4) & jnp.all(diagonal_2 == 1)

    def check_straight_lines(_):
        vertical = jnp.array([
            region[i, j],
            region[i + 1, j],
            region[i + 2, j],
            region[i + 3, j]
        ])
        horizontal = jnp.array([
            region[i, j],
            region[i, j + 1],
            region[i, j + 2],
            region[i, j + 3]
        ])

        return (jnp.sum(region) == 4) & jnp.all(vertical == 1) | \
               (jnp.sum(region) == 4) & jnp.all(horizontal == 1)

    return jax.lax.cond(check_diagonal, check_diagonal_lines, check_straight_lines, None)



    