from flax import struct
from craftext.environment.craftext_constants import MobType
from craftext.environment.scenarious.checkers.target_state import TargetState

@struct.dataclass
class AvoidMobDistance:
    mob: MobType = struct.field(default_factory=MobType.ZOMBIE)
    distance: int = struct.field(default_factory=int)
    
@struct.dataclass
class CMDPTargetState(TargetState):
    avoid_mob_distance: AvoidMobDistance =  struct.field(default_factory=AvoidMobDistance)
    
    
    
