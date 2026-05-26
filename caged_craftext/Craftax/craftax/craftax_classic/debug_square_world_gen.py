"""Fixed 8x8 debug arena: tree border, grass interior, corner resource blocks, player at center."""

import jax.numpy as jnp

from craftax.craftax_classic.constants import Achievement, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax_classic.game_logic import calculate_light_level


def generate_debug_square_world(rng, params, static_params):
    """
    Layout (8x8, 0-indexed):
      - Border (row/col 0 and 7): TREE
      - Inner floor: GRASS
      - Player spawn: map center (4, 4)
      - Corner resources (one cell inside the tree ring):
          (1, 1) STONE
          (1, 6) WOOD
          (6, 1) WATER
          (6, 6) empty GRASS (unused corner tile)
    No mobs, ores, lava, procedural noise, or starting saplings in inventory.
    """
    h, w = static_params.map_size
    assert (h, w) == (8, 8), f"debug square world requires map_size (8, 8), got {(h, w)}"

    map = jnp.full((h, w), BlockType.GRASS.value, dtype=jnp.int32)

    # Tree border
    map = map.at[0, :].set(BlockType.TREE.value)
    map = map.at[h - 1, :].set(BlockType.TREE.value)
    map = map.at[:, 0].set(BlockType.TREE.value)
    map = map.at[:, w - 1].set(BlockType.TREE.value)

    # Corner resources (adjacent to tree wall, not on the outermost ring)
    map = map.at[1, 1].set(BlockType.STONE.value)
    map = map.at[1, w - 2].set(BlockType.WOOD.value)
    map = map.at[h - 2, 1].set(BlockType.WATER.value)
    map = map.at[h - 2, w - 2].set(BlockType.GRASS.value)

    player_position = jnp.array([h // 2, w // 2], dtype=jnp.int32)
    map = map.at[player_position[0], player_position[1]].set(BlockType.GRASS.value)

    zombies = Mobs(
        position=jnp.zeros((static_params.max_zombies, 2), dtype=jnp.int32),
        health=jnp.ones(static_params.max_zombies, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_zombies, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
    )
    skeletons = Mobs(
        position=jnp.zeros((static_params.max_skeletons, 2), dtype=jnp.int32),
        health=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_skeletons, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
    )
    arrows = Mobs(
        position=jnp.zeros((static_params.max_arrows, 2), dtype=jnp.int32),
        health=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_arrows, dtype=bool),
        attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
    )
    cows = Mobs(
        position=jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32),
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
