from flax.struct import dataclass, field
from craftext.environment.scenarious.checkers.target_state import TargetState 
from typing import Literal
from enum import Enum
from jax import numpy as jnp
import jax

@dataclass
class TypeOfInterest:
    FOOD = 0,
    WATER = 1,
    TREE = 2,
    STONE = 3
    
@dataclass
class TargetOfInterest:
    object_of_interest: int = field(default_factory=TypeOfInterest.FOOD)
    far_from_agent: int = field(default_factory=5)
    last_visible_target_position: jax.Array = field(default_factory=jnp.zeros(shape=(1, 2), dtype=jnp.int16))

@dataclass
class CMDPTargetState(TargetState):
    target_of_interest: TargetOfInterest = field(default_factory=TargetOfInterest)
