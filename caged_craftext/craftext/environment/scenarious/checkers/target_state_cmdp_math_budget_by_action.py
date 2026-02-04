from flax import struct
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftax.craftax_classic.constants import Action
from jax import numpy as jnp

@struct.dataclass
class TargetAction:
    FOOD = 0
    WOOD = 1

@struct.dataclass
class BudgetByAction:
    target_action: int = 0
    budget: int = 100
    delimiter_actions: jnp.ndarray = struct.field(default_factory=jnp.zeros_like(len(Action.__dict__.keys()), dtype=jnp.int16))

@struct.dataclass
class CMDPTargetState(TargetState):
    # step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)
    budget_by_action: BudgetByAction = struct.field(default_factory=BudgetByAction)