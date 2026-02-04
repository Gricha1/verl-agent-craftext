from flax.struct import dataclass, field
from craftext.environment.craftext_constants import MobType
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_avoid_mob import AvoidMobDistance
from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import TargetOfInterest

from jax import numpy as jnp
import jax


@dataclass
class CMDPTargetState(TargetState):
    avoid_mob_distance: AvoidMobDistance = field(default_factory=AvoidMobDistance)
    target_of_interest_water: TargetOfInterest = field(default_factory=TargetOfInterest)
    target_of_interest_food: TargetOfInterest = field(default_factory=TargetOfInterest)
    
    
    
