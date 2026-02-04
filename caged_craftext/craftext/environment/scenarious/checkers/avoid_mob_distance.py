import jax

from jax import (
    numpy as jnp,
)

from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance

def checker_relactional_avoid_mob_distance(game_data: GameDataClassic,  target_state: AvoidMobDistance) -> jax.Array:
    mob_type = target_state.mob
    distance = target_state.distance
    return is_mob_near_than_n(game_data, mob_type=mob_type, distance=distance)

def is_mob_near_than_n(game_data: GameDataClassic, mob_type: int, distance: int):
    player_position = game_data.states[0].variables.player_position
    
    zombies_positions = game_data.states[0].zombies.position
    skeletones_positions = game_data.states[0].skeletons.position
    
    deltas_zombies = jnp.abs(zombies_positions - player_position)
    # deltas_zombies_xs = (deltas_zombies[:, 0] <= distance).any()
    # deltas_zombies_ys = (deltas_zombies[:, 1] <= (distance)).any()
    is_in_radius_zombies = (jnp.sum(deltas_zombies, axis=1) <= distance).any() #deltas_zombies_xs & deltas_zombies_ys 
    
    deltas_skeletons = jnp.abs(skeletones_positions - player_position) 
    # deltas_skeletons_xs = (deltas_skeletons[:, 0] <= distance).any()
    # deltas_skeletons_ys = (deltas_skeletons[:, 1] <= distance).any()
    is_in_radius_skeletones = (jnp.sum(deltas_skeletons, axis=1) <= distance).any() #deltas_skeletons_xs & deltas_skeletons_ys
    return jax.lax.cond(mob_type == 2, 
                        lambda: jnp.logical_or(is_in_radius_skeletones, is_in_radius_zombies), 
                        lambda: jax.lax.cond(mob_type == 0, 
                                             lambda: is_in_radius_zombies, 
                                             lambda: is_in_radius_skeletones
                                )
            )    

