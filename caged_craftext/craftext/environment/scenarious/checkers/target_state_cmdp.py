from flax import struct
import jax.numpy as jnp
from craftext.environment.craftext_constants import BlockType, TimeState
from craftext.environment.scenarious.checkers.target_state import TargetState
import jax 

@struct.dataclass
class StepOnBlock:
    block_type: int = BlockType.INVALID

@struct.dataclass
class CMDPTargetState(TargetState):
    step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)