from flax import struct
from craftext.environment.scenarious.checkers.target_state import TargetState

@struct.dataclass
class LightLevelState:
    level: float = 0

@struct.dataclass
class CMDPTargetState(TargetState):
    # step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)
    night_constraint_level: LightLevelState = struct.field(default_factory=LightLevelState)
