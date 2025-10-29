from typing import List
import jax.numpy as jnp
from flax import struct
import jax

@struct.dataclass
class PlayerVariables:
    player_position: jax.Array 
    # player_level: int
    # player_direction: int
    # player_health: float
    # player_food: int 
    # player_drink: int 
    # player_energy: int 
    # player_mana: int
    # is_sleeping: bool 
    # is_resting: bool 
    # player_recover: float 
    # player_hunger: float 
    # player_thirst: float
    # player_fatigue: float 
    # player_recover_mana: float 
    # player_xp: int
    # player_dexterity: int 
    # player_strength: int 
    # player_intelligence: int 
    # learned_spells: jax.Array
    # sword_enchantment: int
    # bow_enchantment: int
    # boss_progress: int 
    # boss_timesteps_to_spawn_this_round: int 
    light_level: jax.Array 
    # light_level_dinamic: jax.Array

    # state_rng: jax.Array 
    # timestep: int 

@struct.dataclass
class PlayerAchievements:
    achievements: List[str] 

@struct.dataclass
class PlayerInventory:
    inventory = 0
    wood: jax.Array 
    stone: jax.Array 
    coal: jax.Array 
    iron: jax.Array 
    diamond: jax.Array 
    sapling: jax.Array 
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
    
    # Just for jax for correct invemtory check (conditional tasks)
    wood_pickaxe: jax.Array  
    stone_pickaxe: jax.Array  
    iron_pickaxe: jax.Array  
    wood_sword: jax.Array  
    stone_sword: jax.Array  
    iron_sword: jax.Array  

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
        action = action
        variables = PlayerVariables(
            player_position=jnp.array(state.player_position) if hasattr(state, 'player_position') else None,
            # player_level=state.player_level,
            # player_direction=state.player_direction,
            # player_health=state.player_health,
            # player_food= state.player_food,
            # player_drink=state.player_drink,
            # player_energy=state.player_energy,
            # player_mana=state.player_mana,
            # is_sleeping=state.is_sleeping,
            # is_resting=state.is_resting,
            # player_recover=state.player_recover,
            # player_hunger=state.player_hunger,
            # player_thirst=state.player_thirst,
            # player_fatigue=state.player_fatigue,
            # player_recover_mana=state.player_recover_mana,
            # player_xp=state.player_xp,
            # player_dexterity=state.player_dexterity,
            # player_strength=state.player_strength,
            # player_intelligence=state.player_intelligence,
            # learned_spells=state.learned_spells,
            # sword_enchantment=state.sword_enchantment,
            # bow_enchantment=state.bow_enchantment,
            # boss_progress=state.boss_progress,
            light_level=state.light_level,
            # light_level_dinamic=jnp.array(0),
            # state_rng=state.state_rng,
            # timestep=state.timestep,
        )

        achievements = PlayerAchievements(
            achievements=state.achievements
        )

        inventory = PlayerInventory(
                wood=jnp.array(state.inventory.wood),
                stone=jnp.array(state.inventory.stone),
                coal=jnp.array(state.inventory.coal),
                iron=jnp.array(state.inventory.iron),
                diamond=jnp.array(state.inventory.diamond),
                sapling=jnp.array(state.inventory.sapling),
                
                pickaxe=jnp.array(state.inventory.pickaxe),
                sword=jnp.array(state.inventory.sword),
                bow=jnp.array(state.inventory.bow),
                arrows=jnp.array(state.inventory.arrows),
                armour=jnp.array(state.inventory.armour),
                torches=jnp.array(state.inventory.torches),
                ruby=jnp.array(state.inventory.ruby),
                sapphire=jnp.array(state.inventory.sapphire),
                potions=jnp.array(state.inventory.potions),
                books=jnp.array(state.inventory.books),
                
                # Just for jax for correct invemtory check (conditional tasks)
                wood_pickaxe=jnp.array(state.inventory.pickaxe),
                stone_pickaxe=jnp.array(state.inventory.sword),
                iron_pickaxe=jnp.array(state.inventory.bow),
                wood_sword=jnp.array(state.inventory.arrows),
                stone_sword=jnp.array(state.inventory.armour),
                iron_sword=jnp.array(state.inventory.torches),
            )

        game_map = GameMap(
            game_map=jnp.array(state.map[0]) if hasattr(state, 'map') else None
            
        )

        return cls(
            variables=variables,
            achievements=achievements,
            inventory=inventory,
            map=game_map,
            action=action
        )

@struct.dataclass
class GameData:
    states: list[PlayerState]

    @classmethod
    def from_state(cls, previos_state: PlayerState, current_state: PlayerState, action):
        player_state_current = PlayerState.from_state(current_state, action)
        player_state_previos = PlayerState.from_state(previos_state, action)
        return cls(states=[player_state_current, player_state_previos])
    