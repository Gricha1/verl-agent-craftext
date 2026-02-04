from flax import struct
from craftext.environment.craftext_constants import BlockType
from craftext.environment.scenarious.checkers.target_state import TargetState

@struct.dataclass
class SeeInView:
    block_type: BlockType = struct.field(default_factory=BlockType.GRASS)
        
@struct.dataclass
class CMDPTargetState(TargetState):
    see_in_fov: SeeInView =  struct.field(default_factory=SeeInView)
    
    
    
