from flax import struct
import jax.numpy as jnp
from craftext.environment.craftext_constants import Achievement, AchievementState, BlockType, TimeState
import jax 

@struct.dataclass
class BuildLineState:

    block_type: int = BlockType.INVALID
    size:int = 3
    radius: int = 3
    is_diagonal:bool = False

@struct.dataclass
class BuildSquareState:
    
    block_type: int = BlockType.INVALID
    size: int = 3
    radius: int = 5
    
@struct.dataclass
class BuildStarState:
    
    block_type: int = BlockType.INVALID
    size: int = 3
    radius: int = 3
    cross_type: int = -1

@struct.dataclass
class ConditionalPlacingState:
    
    object_inventory_enum: int = -1
    object_to_place: int = 0
    count_to_collect: int = 0
    count_to_stand: int = 1
    
@struct.dataclass
class LocalizaPlacingState:
    
    object_name: int = -1
    target_object_name: int = -1
    side: int = -1
    distance: int = 5

@struct.dataclass
class Achievements:
    
    achievement_mask: tuple = struct.field(default_factory=lambda: tuple([AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]))

@struct.dataclass
class TimeCosntrainedPlacmentState:
    
    block_type: int = BlockType.INVALID
    time_state: int = TimeState.DAY
    radius: int = 5

@struct.dataclass
class UnifiedPatternState:
    
    block_type: int = BlockType.INVALID
    pattern_type: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1))
    size: int = 3
    radius: int = 3

@struct.dataclass
class TargetState:
    achievements: Achievements = struct.field(default_factory=Achievements)
    building_line: BuildLineState =  struct.field(default_factory=BuildLineState)
    building_square: BuildSquareState = struct.field(default_factory=BuildSquareState)
    building_star: BuildStarState =  struct.field(default_factory=BuildStarState)
    conditional_placing: ConditionalPlacingState = struct.field(default_factory=ConditionalPlacingState)
    Localization_placing: LocalizaPlacingState = struct.field(default_factory=LocalizaPlacingState)
    time_placement: TimeCosntrainedPlacmentState = struct.field(default_factory=TimeCosntrainedPlacmentState)
    unified_pattern_state: UnifiedPatternState = struct.field(default_factory=UnifiedPatternState)

    @classmethod
    def stack(cls, lst: 'TargetState') -> 'TargetState':
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lst)

    def select(self, idx: jnp.ndarray) -> 'TargetState':
        return jax.tree_util.tree_map(lambda arr: arr[idx], self)
    
    #building_line: Tuple[AchievementState, BuildLineAchievement]
    # collect_wood: int = AchievementState.NOT_MATTER.value
    # place_table: int = AchievementState.NOT_MATTER.value
    # eat_cow: int = AchievementState.NOT_MATTER.value
    # collect_sapling: int = AchievementState.NOT_MATTER.value
    # collect_drink: int = AchievementState.NOT_MATTER.value
    # make_wood_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_wood_sword: int = AchievementState.NOT_MATTER.value
    # place_plant: int = AchievementState.NOT_MATTER.value
    # defeat_zombie: int = AchievementState.NOT_MATTER.value
    # collect_stone: int = AchievementState.NOT_MATTER.value
    # place_stone: int = AchievementState.NOT_MATTER.value
    # eat_plant: int = AchievementState.NOT_MATTER.value
    # defeat_skeleton: int = AchievementState.NOT_MATTER.value
    # make_stone_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_stone_sword: int = AchievementState.NOT_MATTER.value
    # wake_up: int = AchievementState.NOT_MATTER.value
    # place_furnace: int = AchievementState.NOT_MATTER.value
    # collect_coal: int = AchievementState.NOT_MATTER.value
    # collect_iron: int = AchievementState.NOT_MATTER.value
    # collect_diamond: int = AchievementState.NOT_MATTER.value
    # make_iron_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_iron_sword: int = AchievementState.NOT_MATTER.value
    # make_arrow: int = AchievementState.NOT_MATTER.value
    # make_torch: int = AchievementState.NOT_MATTER.value
    # place_torch: int = AchievementState.NOT_MATTER.value
    # collect_sapphire: int = AchievementState.NOT_MATTER.value
    # collect_ruby: int = AchievementState.NOT_MATTER.value
    # make_diamond_pickaxe: int = AchievementState.NOT_MATTER.value
    # make_diamond_sword: int = AchievementState.NOT_MATTER.value
    # make_iron_armour: int = AchievementState.NOT_MATTER.value
    # make_diamond_armour: int = AchievementState.NOT_MATTER.value
    # enter_gnomish_mines: int = AchievementState.NOT_MATTER.value
    # enter_dungeon: int = AchievementState.NOT_MATTER.value
    # enter_sewers: int = AchievementState.NOT_MATTER.value
    # enter_vault: int = AchievementState.NOT_MATTER.value
    # enter_troll_mines: int = AchievementState.NOT_MATTER.value
    # enter_fire_realm: int = AchievementState.NOT_MATTER.value
    # enter_ice_realm: int = AchievementState.NOT_MATTER.value
    # enter_graveyard: int = AchievementState.NOT_MATTER.value
    # defeat_gnome_warrior: int = AchievementState.NOT_MATTER.value
    # defeat_gnome_archer: int = AchievementState.NOT_MATTER.value
    # defeat_orc_solider: int = AchievementState.NOT_MATTER.value
    # defeat_orc_mage: int = AchievementState.NOT_MATTER.value
    # defeat_lizard: int = AchievementState.NOT_MATTER.value
    # defeat_kobold: int = AchievementState.NOT_MATTER.value
    # defeat_knight: int = AchievementState.NOT_MATTER.value
    # defeat_archer: int = AchievementState.NOT_MATTER.value
    # defeat_troll: int = AchievementState.NOT_MATTER.value
    # defeat_deep_thing: int = AchievementState.NOT_MATTER.value
    # defeat_pigman: int = AchievementState.NOT_MATTER.value
    # defeat_fire_elemental: int = AchievementState.NOT_MATTER.value
    # defeat_frost_troll: int = AchievementState.NOT_MATTER.value
    # defeat_ice_elemental: int = AchievementState.NOT_MATTER.value
    # damage_necromancer: int = AchievementState.NOT_MATTER.value
    # defeat_necromancer: int = AchievementState.NOT_MATTER.value
    # eat_bat: int = AchievementState.NOT_MATTER.value
    # eat_snail: int = AchievementState.NOT_MATTER.value
    # find_bow: int = AchievementState.NOT_MATTER.value
    # fire_bow: int = AchievementState.NOT_MATTER.value
    # learn_fireball: int = AchievementState.NOT_MATTER.value
    # cast_fireball: int = AchievementState.NOT_MATTER.value
    # learn_iceball: int = AchievementState.NOT_MATTER.value
    # cast_iceball: int = AchievementState.NOT_MATTER.value
    # open_chest: int = AchievementState.NOT_MATTER.value
    # drink_potion: int = AchievementState.NOT_MATTER.value
    # enchant_sword: int = AchievementState.NOT_MATTER.value
    # enchant_armour: int = AchievementState.NOT_MATTER.value
    # smth: int = AchievementState.NOT_MATTER.value
    # end: int = AchievementState.NOT_MATTER.value