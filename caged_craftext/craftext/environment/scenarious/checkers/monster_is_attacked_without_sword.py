import jax.numpy as jnp
from craftax.craftax_classic.constants import Action, DIRECTIONS


def unarmed_monster_attack(action, player_position, player_direction, is_sleeping, inventory, zombies, skeletons):
    """Match the Craftax DO interaction using the state before mobs move or die."""
    target = player_position + DIRECTIONS[player_direction]
    has_sword = (inventory.wood_sword > 0) | (inventory.iron_sword > 0)

    def mob_at_target(mobs):
        return jnp.any(mobs.mask & jnp.all(mobs.position == target, axis=-1))

    return (
        (action == Action.DO.value)
        & ~is_sleeping
        & ~has_sword
        & (mob_at_target(zombies) | mob_at_target(skeletons))
    )


def checker_monster_is_attacked_without_sword(game_data):
    state = game_data.states[1]
    return unarmed_monster_attack(
        game_data.states[0].action,
        state.variables.player_position,
        state.variables.player_direction,
        state.variables.is_sleeping,
        state.inventory,
        state.zombies,
        state.skeletons,
    )
