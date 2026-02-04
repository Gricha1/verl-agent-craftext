import jax
import jax.numpy as jnp

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.states.state import PlayerInventory
from craftext.environment.states.state_classic import PlayerInventory as PlayerInventoryClassic

def was_item_collected_after_another(game_data: Union[GameDataClassic, GameData], first_item: str, second_item: str) -> jax.Array:
    """
    JAX-compatible version to check if `second_item` was collected after `first_item`.

    Args:
    - game_data (GameData): The game data object containing states and actions.
    - first_item (str): The first item to check in the inventory.
    - second_item (str): The second item to check in the inventory.

    Returns:
    - bool: True if `second_item` was collected after `first_item`, otherwise False.
    """

    def extract_item_collected(inventory: Union[PlayerInventoryClassic, PlayerInventory], item_name: str) -> jax.Array:
        print("Inventory: ", inventory)
        item_quantity = getattr(inventory, item_name, None)
        print("Item item_quantity: ", item_quantity)
        return jnp.any(item_quantity > 0) if item_quantity is not None else jnp.array(False)

    # Vectorized check across all game states
    first_item_collected  = jnp.array([extract_item_collected(state.inventory, "wood") for state in game_data.states])
    second_item_collected = jnp.array([extract_item_collected(state.inventory, "stone") for state in game_data.states])

    # # Find the first occurrence (index) where each item was collected
    first_item_idx  = jnp.argmax(first_item_collected)
    second_item_idx = jnp.argmax(second_item_collected)

    # # Ensure both items are found
    first_item_found  = jnp.any(first_item_collected)
    second_item_found = jnp.any(second_item_collected)

    # Return True if both items were found and second item was collected after first item
    return jnp.logical_and(first_item_found, jnp.logical_and(second_item_found, second_item_idx >= first_item_idx))