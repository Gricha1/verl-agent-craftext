# instruction_wrapper.py

from dataclasses import dataclass
from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from gym import Wrapper
from jax import tree_map

import jax.numpy as jnp
from typing import List, TypeVar, Type

T = TypeVar("T")

def list_to_array(lst: List[T]) -> T:
    """Convert a list of dataclass instances to a batched version with jnp.arrays."""
    if not lst:
        raise ValueError("Input list is empty.")

    cls: Type[T] = type(lst[0])  
    converted_data = {}

    for k, field in cls.__dataclass_fields__.items():
        values = [getattr(v, k) for v in lst]

        if isinstance(values[0], jnp.ndarray):
            converted_data[k] = jnp.stack(values, axis=0)  
        elif isinstance(values[0], (int, float, bool)):  
            converted_data[k] = jnp.array(values) 
        else:
            converted_data[k] = list_to_array(values) 
    return cls(**converted_data)

@struct.dataclass
class TextEnvState:
    env_state: Any
    craftext_state: Any
    timestep: int
    full_instruction: Optional[jax.Array]
    step_embedding: Optional[jax.Array]
    step_idx: int
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int
    checker_id: int

from gymnax.environments import spaces, environment
from craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
)

def check_subvector(x, y):
    return jnp.all(x == y)

class SIPlanning(Wrapper):
    def __init__(self, env):
        super().__init__(env)  # CrafText wrapper should be last
        self.end_embedding = (
            self.env.scenario_handler.scenario_data_jax.embeddings_list[0][-1]
        )
        self.steps = 0
        self.encoded_instruction = self.scenario_handler.scenario_data_jax.embeddings_list[0][:1]
        print(self.encoded_instruction.shape)

    @property
    def num_actions(self) -> int:
        return 18

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        return spaces.Discrete(18)

    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Add to the TextEnvState embedding of the step.
        """
        obs, craftext_state = self.env.reset(
            _rng, env_params, instruction_idx
        )

        plan_embeddings_list = craftext_state.instruction
        step_idx = 0
        instruction_idx = craftext_state.idx

        si_state = TextEnvState(
            env_state=craftext_state.env_state,  # Should be Craftax state
            craftext_state = craftext_state,
            timestep=craftext_state.timestep,
            full_instruction=plan_embeddings_list,
            step_embedding=plan_embeddings_list[step_idx],
            step_idx=step_idx,
            idx=instruction_idx,
            environment_key=craftext_state.environment_key,
            success_rate=0.0,
            total_success_rate=0.0,
            rng=_rng,
            checker_id=(
                self.env.scenario_handler.scenario_data_jax
                .scenario_checker[instruction_idx]
            )
        )
        return obs, si_state

    def step(self, _rng, si_env_state, action, env_params):
        craftext_env_state = si_env_state.env_state
        step_idx = si_env_state.step_idx

        mask = jnp.where(action == 17, True, False)

        # Action 17 is interpreted as noop + move to next step.
        new_step_idx = jax.lax.cond(
            mask, lambda _: step_idx + 1, lambda _: step_idx, operand=None
        )
        action = jax.lax.cond(
            mask, lambda _: 0, lambda _: action, operand=None
        )

        obs, craftext_state, reward, done, info = self.env.step(
            _rng, si_env_state.craftext_state, action, env_params
        )

        # If reward >= 0.99, instruction was completed — remove bonus.
        instruction_done_on_step = reward >= 0.99
        agent_die = jnp.logical_and(done, reward < 0.99)

        reward = jnp.where(instruction_done_on_step, reward - 1.0, reward)

        plans_ends = jnp.allclose(
            si_env_state.full_instruction[step_idx],
            self.end_embedding,
            atol=1e-2,
            rtol=0.0
        )

        need_give_reward = jnp.logical_and(
            instruction_done_on_step, plans_ends
        )
        reward = jax.lax.cond(
            need_give_reward,
            lambda _: reward + 1.0,
            lambda _: reward,
            operand=None
        )

        # Episode ends either on final embedding or if agent dies.
        done = jnp.logical_or(plans_ends, agent_die)

        new_episode_sr = si_env_state.success_rate + jnp.float32(need_give_reward)
        new_step_idx = jax.lax.cond(
            plans_ends, lambda _: 0, lambda _: new_step_idx, operand=None
        )

        new_si_state = TextEnvState(
            env_state=craftext_state.env_state,
            craftext_state = craftext_state,
            timestep=si_env_state.timestep,
            full_instruction=craftext_state.instruction,
            step_embedding=craftext_state.instruction[step_idx],
            step_idx=new_step_idx,
            idx=craftext_state.idx,
            environment_key=craftext_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=(
                si_env_state.total_success_rate * (1 - done)
                + new_episode_sr * done
            ),
            rng=craftext_state.rng,
            checker_id=si_env_state.checker_id
        )

        info.update({
            "SR": new_si_state.total_success_rate,
            "steps": self.steps
        })
        self.steps += 1

        return obs, new_si_state, reward, done, info
    
    
class CustomInstructionWrapper(Wrapper):
    def __init__(self, env, instruction):
        self.env = env
        self.castom_initial_instruction = jnp.array(self.env.scenario_handler.castom_initial_instruction(instruction)[0])
        print("CASTOM INSTRUCTION SHAPE: ", self.castom_initial_instruction.shape)
    
    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        state = TextEnvState(
            env_state=state.env_state,
            timestep=state.timestep,
            full_instruction=self.castom_initial_instruction,
            instruction=self.castom_initial_instruction[0],
            step_idx=0,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            total_success_rate=0.0,
            rng=_rng
        )
         
        return obs, state
    
    def step(self, _rng, env_state, action, env_params):
         obs, state, reward, done, info = self.env.step(_rng, env_state, action, env_params)
         return obs, state, reward, done, info
        
