from flax.struct import dataclass, field
from craftext.environment.craftext_constants import MobType
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import TargetOfInterest
from craftext.environment.scenarious.checkers.target_state_cmdp_budget_hp_level import HPLevelState
from craftext.environment.scenarious.checkers.target_state_cmdp_math_budget_by_action import BudgetByAction
from jax import numpy as jnp
import jax


@dataclass
class CMDPTargetState(TargetState):
    budget_food_by_action: BudgetByAction = field(default_factory=BudgetByAction)
    budget_wood_by_action: BudgetByAction = field(default_factory=BudgetByAction)
    avoid_mob_distance: AvoidMobDistance = field(default_factory=AvoidMobDistance)
    target_of_interest_water: TargetOfInterest = field(default_factory=TargetOfInterest)
    target_of_interest_food: TargetOfInterest = field(default_factory=TargetOfInterest)
    level_hp_budget: HPLevelState = field(default_factory=HPLevelState)
    level_away_monsters: HPLevelState = field(default_factory=HPLevelState)
    
    
    
