from flax import struct
from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_math_budget_by_action import BudgetByAction
from craftax.craftax_classic.constants import Action
from jax import numpy as jnp

@struct.dataclass
class CMDPTargetState(TargetState):
    # step_on_block: StepOnBlock =  struct.field(default_factory=StepOnBlock)
    budget_food_by_action: BudgetByAction = struct.field(default_factory=BudgetByAction)
    budget_wood_by_action: BudgetByAction = struct.field(default_factory=BudgetByAction)