import jax
from jax import numpy as jnp

from craftext.environment.craftext_constants import MobType
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance
from craftext.environment.states.state_classic import GameDataClassic


def active_mob_near(state, mob_type, distance):
    position = state.variables.player_position

    def near(mobs):
        within_radius = jnp.abs(mobs.position - position).sum(axis=-1) <= distance
        return jnp.any(mobs.mask & within_radius)

    include_zombies = (mob_type == MobType.ZOMBIE) | (mob_type == MobType.ALL)
    include_skeletons = (mob_type == MobType.SKELETON) | (mob_type == MobType.ALL)
    return (include_zombies & near(state.zombies)) | (include_skeletons & near(state.skeletons))


def checker_relactional_avoid_mob_distance(
    game_data: GameDataClassic, target_state: AvoidMobDistance
) -> jax.Array:
    return is_mob_near_than_n(game_data, target_state.mob, target_state.distance)


def is_mob_near_than_n(game_data: GameDataClassic, mob_type: int, distance: int):
    return active_mob_near(game_data.states[0], mob_type, distance)
