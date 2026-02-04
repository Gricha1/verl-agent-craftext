from flax import struct
import jax.numpy as jnp
from craftext.environment.craftext_constants import BlockType, TimeState
from craftext.environment.scenarious.checkers.target_state import TargetState
import jax 

@struct.dataclass
class HungryLevelState:
    level: int = 100

@struct.dataclass
class CMDPTargetState(TargetState):
    # step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)
    hungry_level_state: HungryLevelState = struct.field(default_factory=HungryLevelState)