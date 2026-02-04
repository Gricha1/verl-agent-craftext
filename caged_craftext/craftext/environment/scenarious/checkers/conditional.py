import jax.numpy as jnp
import jax 
from jax import lax

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.states.state import PlayerInventory
from craftext.environment.states.state_classic import PlayerInventory as PlayerInventoryClassic

from craftext.environment.scenarious.checkers.target_state import ConditionalPlacingState

def checker_conditional_placement(game_data: Union[GameDataClassic, GameData],  target_state: ConditionalPlacingState) -> jax.Array:
    
    object_inventory_enum = target_state.object_inventory_enum
    object_to_place = target_state.object_to_place
    count_to_collect = target_state.count_to_collect
    count_to_stand = target_state.count_to_stand

    return conditional_placing(game_data, object_inventory_enum, object_to_place, count_to_collect, count_to_stand)

def conditional_placing(gd: Union[GameDataClassic, GameData], object_inventory_enum: int, object_to_place: int, count_to_collect: int, count_to_stand: int) -> jax.Array:

    previous_state = gd.states[0]
    current_state = gd.states[1]
    
    prev_inventory_check = check_inventory(previous_state.inventory, object_inventory_enum, count_to_collect)
    
    curr_inventory_check = check_inventory(current_state.inventory, object_inventory_enum, count_to_collect)
    
    placed_check = check_map(current_state.map.game_map, object_to_place, count_to_stand)
    
    return jnp.logical_and(jnp.logical_and(jnp.logical_not(prev_inventory_check), curr_inventory_check), placed_check)
    
    
def check_inventory(inventory: Union[PlayerInventoryClassic, PlayerInventory], object_inventory_id: int, count_to_collect: int):
    
    def get_item(index, inventory):
        return lax.switch(index,
            [
                lambda: inventory.wood,
                lambda: inventory.stone,
                lambda: inventory.coal,
                lambda: inventory.iron,
                lambda: inventory.diamond,
                lambda: inventory.sapling,
                
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                lambda: inventory.diamond,
                
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.pickaxe, lambda: inventory.wood_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.sword, lambda: inventory.stone_pickaxe]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.bow,   lambda: inventory.iron_pickaxe]), 
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.arrows, lambda: inventory.wood_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.armour, lambda: inventory.stone_sword]),
                # lambda: lax.switch(inventory.inventory, [lambda: inventory.torches, lambda: inventory.iron_sword]),
            
                lambda: inventory.ruby,
                lambda: inventory.sapphire,
                
                lambda: inventory.diamond, #lambda: inventory.potions
                lambda: inventory.diamond, #lambda: inventory.books
            ]
        )
        
    collected_count = get_item(object_inventory_id, inventory)
    return collected_count >= count_to_collect


def check_map(game_map: jax.Array, object_to_place: int, count_to_stand: int):

    placed_count = jnp.sum(game_map == object_to_place)
    return placed_count >= count_to_stand