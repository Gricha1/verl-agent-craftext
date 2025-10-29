from typing import List, Tuple, Optional, Union
import jax.numpy as jnp
from flax import struct
import jax

@struct.dataclass
class PlayerVariables:
    player_position: jax.Array
    player_direction: int
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float
    light_level: float
    state_rng: jax.Array
    timestep: int
    
@struct.dataclass
class PlayerAchievements:
    achievements: List[str]

@struct.dataclass
class PlayerInventory:
    inventory = 1
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int
     # Just for jax for correct invemtory check
    pickaxe: jax.Array
    sword: jax.Array
    bow: jax.Array
    arrows: jax.Array
    armour: jax.Array
    torches: jax.Array
    ruby: jax.Array
    sapphire: jax.Array
    potions: jax.Array
    books: jax.Array



@struct.dataclass
class GameMap:
    game_map: jax.Array


@struct.dataclass
class PlayerState:
    variables: PlayerVariables
    achievements: PlayerAchievements
    inventory: PlayerInventory
    map: GameMap
    action: int

    @classmethod
    def from_state(cls, state, action):
        variables = PlayerVariables(
            player_position=jnp.array(state.player_position) if hasattr(state, 'player_position') else None,
            player_direction=state.player_direction,
            player_health=state.player_health,
            player_food=state.player_food,
            player_drink=state.player_drink,
            player_energy=state.player_energy,
            is_sleeping=state.is_sleeping,
            player_recover=state.player_recover,
            player_hunger=state.player_hunger,
            player_thirst=state.player_thirst,
            player_fatigue=state.player_fatigue,
            light_level=state.light_level,
            state_rng=state.state_rng,
            timestep=state.timestep,
        )

        achievements = PlayerAchievements(
            achievements=jnp.array(state.achievements) if hasattr(state, 'achievements') else None
        )

        inventory = PlayerInventory(
            wood=state.inventory.wood,
            stone=state.inventory.stone,
            coal=state.inventory.coal,
            iron=state.inventory.iron,
            diamond=state.inventory.diamond,
            sapling=state.inventory.sapling,
            wood_pickaxe=state.inventory.wood_pickaxe,
            stone_pickaxe=state.inventory.stone_pickaxe,
            iron_pickaxe=state.inventory.iron_pickaxe,
            wood_sword=state.inventory.wood_sword,
            stone_sword=state.inventory.stone_sword,
            iron_sword=state.inventory.iron_sword,
            # Just for jax for correct invemtory check
            pickaxe=state.inventory.iron,
            sword=state.inventory.iron,
            bow=state.inventory.iron,
            arrows=state.inventory.iron,
            armour=state.inventory.iron,
            torches=state.inventory.iron,
            ruby=state.inventory.iron,
            sapphire=state.inventory.iron,
            potions=state.inventory.iron,
            books=state.inventory.iron,
        )

        game_map = GameMap(
            game_map=jnp.array(state.map) if hasattr(state, 'map') else None
        )

        return cls(
            variables=variables,
            achievements=achievements,
            inventory=inventory,
            map=game_map,
            action=action
        )

@struct.dataclass
class GameDataClassic:
    states: list

    @classmethod
    def from_state(cls, previos_state, current_state, action):
        player_state_current = PlayerState.from_state(current_state, action)
        player_state_previos = PlayerState.from_state(previos_state, action)
        return cls(states=[player_state_current, player_state_previos])
    