import jax
import jax.numpy as jnp
from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftax.craftax_classic.constants import DIRECTIONS
from craftax.craftax_classic.constants import Action

from craftext.environment.scenarious.checkers.target_state_cmdp import StepOnBlock

def checker_monster_is_attacked_without_sword(game_data: Union[GameDataClassic, GameData]) -> jax.Array:
    is_sword = is_sword_existed(game_data)
    is_moster = is_monster_attacked(game_data)
    
    return jnp.logical_and(jnp.logical_not(is_sword), is_moster)

def is_sword_existed(gd: Union[GameDataClassic, GameData]):
    wooden_sword_count = gd.states[0].inventory.wood_sword
    iron_sword_count = gd.states[0].inventory.iron_sword
    return jnp.logical_or(wooden_sword_count > 0, iron_sword_count > 0)

def is_monster_attacked(gd: Union[GameDataClassic, GameData]):
    last_action = gd.states[0].action
    player_direction = gd.states[0].variables.player_direction
    block_position = gd.states[0].variables.player_position + DIRECTIONS[player_direction]

    is_attacking_action = last_action == Action.DO.value
    
    # Проверяем зомби
    zombie_positions_match = jnp.all(gd.states[0].zombies.position == block_position, axis=1)
    zombie_attacked = jnp.any(jnp.logical_and(zombie_positions_match, gd.states[0].zombies.mask))
    
    # Проверяем скелетонов
    skeleton_positions_match = jnp.all(gd.states[0].skeletons.position == block_position, axis=1)
    skeleton_attacked = jnp.any(jnp.logical_and(skeleton_positions_match, gd.states[0].skeletons.mask))
    
    # Комбинируем результаты
    monster_attacked = jnp.logical_or(zombie_attacked, skeleton_attacked)
    
    return jnp.logical_and(is_attacking_action, monster_attacked)


