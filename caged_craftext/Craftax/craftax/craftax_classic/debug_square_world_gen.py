"""Debug arena: tree border, grass interior, resources in inner corners (shuffled each reset).

Supported map sizes: 8x8 and 16x16.
"""

import jax
import jax.numpy as jnp

from craftax.craftax_classic.constants import Achievement, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax_classic.game_logic import calculate_light_level

_SUPPORTED_MAP_SIZES = ((8, 8), (16, 16))


def debug_square_inner_corner_cells(h: int, w: int) -> jnp.ndarray:
    """Four inner corners just inside the tree border (away from center spawn)."""
    return jnp.array(
        [
            [1, 1],  # top-left
            [1, w - 2],  # top-right
            [h - 2, 1],  # bottom-left
            [h - 2, w - 2],  # bottom-right
        ],
        dtype=jnp.int32,
    )


def generate_debug_square_world(rng, params, static_params, fixed_layout: bool = False):
    """
    Layout (NxN with N in {8, 16}, 0-indexed):
      - Border (row/col 0 and N-1): TREE
      - Inner floor: GRASS
      - Player spawn: map center (N//2, N//2)
      - STONE / WOOD / WATER on three distinct inner corners (permutation each reset,
        or fixed canonical corners when ``fixed_layout=True``)
    No mobs, ores, lava, procedural noise, or starting saplings in inventory.
    """
    h, w = static_params.map_size
    assert (h, w) in _SUPPORTED_MAP_SIZES, (
        f"debug square world requires map_size in {_SUPPORTED_MAP_SIZES}, got {(h, w)}"
    )

    map = jnp.full((h, w), BlockType.GRASS.value, dtype=jnp.int32)

    # Tree border
    map = map.at[0, :].set(BlockType.TREE.value)
    map = map.at[h - 1, :].set(BlockType.TREE.value)
    map = map.at[:, 0].set(BlockType.TREE.value)
    map = map.at[:, w - 1].set(BlockType.TREE.value)

    corners = debug_square_inner_corner_cells(h, w)
    if fixed_layout:
        # Canonical, reproducible layout: top-left=STONE, top-right=WOOD,
        # bottom-left=WATER; bottom-right remains GRASS.
        stone_rc, wood_rc, water_rc = corners[0], corners[1], corners[2]
    else:
        perm = jax.random.permutation(rng, corners.shape[0])
        stone_rc = corners[perm[0]]
        wood_rc = corners[perm[1]]
        water_rc = corners[perm[2]]
    map = map.at[stone_rc[0], stone_rc[1]].set(BlockType.STONE.value)
    map = map.at[wood_rc[0], wood_rc[1]].set(BlockType.WOOD.value)
    map = map.at[water_rc[0], water_rc[1]].set(BlockType.WATER.value)

    player_position = jnp.array([h // 2, w // 2], dtype=jnp.int32)
    map = map.at[player_position[0], player_position[1]].set(BlockType.GRASS.value)

    off_map = jnp.full((2,), -1, dtype=jnp.int32)
    zombies = Mobs(
        position=jnp.tile(off_map, (static_params.max_zombies, 1)),
        health=jnp.ones(static_params.max_zombies, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_zombies, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
    )
    skeletons = Mobs(
        position=jnp.tile(off_map, (static_params.max_skeletons, 1)),
        health=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_skeletons, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
    )
    arrows = Mobs(
        position=jnp.tile(off_map, (static_params.max_arrows, 1)),
        health=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_arrows, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
    )
    cows = Mobs(
        position=jnp.tile(off_map, (static_params.max_cows, 1)),
        health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
        mask=jnp.zeros(static_params.max_cows, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
    )

    return EnvState(
        map=map,
        mob_map=jnp.zeros(static_params.map_size, dtype=bool),
        player_position=player_position,
        player_direction=Action.UP.value,
        player_health=9,
        player_food=9,
        player_drink=9,
        player_energy=9,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        is_sleeping=False,
        inventory=Inventory(sapling=0),
        zombies=zombies,
        skeletons=skeletons,
        arrows=arrows,
        arrow_directions=jnp.ones((static_params.max_arrows, 2), dtype=jnp.int32),
        cows=cows,
        growing_plants_positions=jnp.zeros(
            (static_params.max_growing_plants, 2), dtype=jnp.int32
        ),
        growing_plants_age=jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32),
        growing_plants_mask=jnp.zeros(static_params.max_growing_plants, dtype=bool),
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=calculate_light_level(0, params),
        state_rng=rng,
        timestep=0,
    )
