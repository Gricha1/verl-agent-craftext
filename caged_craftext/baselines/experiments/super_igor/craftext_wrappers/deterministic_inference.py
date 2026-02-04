import argparse
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
from functools import partial
from flax import linen as nn, struct

from craftax.craftax_env import make_craftax_env_from_name
from typing import Any, Optional

#from craftext.instruction.wrappers.craftext_wrapper import InstructionWrapper

import numpy as np

import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp


def generate_all_instruction_rng_pairs(rng, n_rngs, n_instructions):
    """
    Generate all possible (instruction, rng) pairs based on the settings.

    Args:
        rng (jax.random.PRNGKey): PRNG key.
        n_rngs (int): Total number of RNG seeds available.
        n_instructions (int): Total number of instructions.

    Returns:
        tuple: (instruction_rng_pairs, rng_keys), where:
            - instruction_rng_pairs: jnp.ndarray of all instruction indices.
            - rng_keys: jnp.ndarray of PRNG keys corresponding to each instruction.
    """
    total_pairs = n_rngs * n_instructions
    indices = jnp.arange(total_pairs)

    # Generate instruction indices (1-indexed)
    instruction_indices = indices % n_instructions + 1

    # Generate RNG indices (numerical seeds)
    rng_indices = indices // n_instructions

    # Convert rng_indices to PRNG keys
    rng_keys = jax.vmap(lambda seed: jax.random.fold_in(rng, seed))(rng_indices)

    return instruction_indices, rng_keys

@struct.dataclass
class DetermEnvState:
    env_state: Any
    indecec_vector: int
    rngs: int
    instructions: int
    v1: int
    v2: int
    v3: int

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class DetermOptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int, n_instructions:int=2, n_seeds:int=4):
        super().__init__(env)

        self.num_envs = num_envs
        self.n_instructions = n_instructions
        self.n_seeds = n_seeds
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None, 0))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):

        instructions, rngs = generate_all_instruction_rng_pairs(rng, self.n_seeds, self.n_instructions,)#self.n_instructions, self.n_seeds)
        obs, env_state = self.reset_fn(rngs[:self.num_envs], params, instructions[:self.num_envs])
        indecec_vector = jnp.ones_like(instructions)
        indecec_vector = indecec_vector.at[:self.num_envs].set(0)
        state = DetermEnvState(env_state=env_state,
                               indecec_vector=indecec_vector, 
                               rngs = rngs[:self.num_envs],
                               instructions=instructions[:self.num_envs],
                               v1=indecec_vector, 
                               v2=indecec_vector ,
                               v3=indecec_vector)

        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        # Generate vector of all possible pairs (seed, nstruction_id)
        instructions, rngs = generate_all_instruction_rng_pairs(rng,  self.n_seeds, self.n_instructions)#self.n_instructions, self.n_seeds)
        
        # Run STEP with RNGS we sampled in last reset 
        rngs_step = state.rngs
        obs_st, state_st, reward, done, info = self.step_fn(rngs_step, state.env_state, action, params)

         # Mask for indices of pairs (seed, instruction_id) we already used (0 - used, 1 - may use in iteration)
        last_vector = state.indecec_vector
        
        # Indices of all possible indices of pairs
        possible_indices = jnp.arange(len(instructions))  
        
        # Sample indices, use last_vector as propability, so indices we already used will have probability = 0
        # NOT CORRECT p CALCULATION
        
        indices_ = jax.random.choice(rng, possible_indices, (self.num_envs,), p=last_vector/jnp.sum(last_vector))
        
        # Run reset with sampled RNG + Instruction_ID
        obs_re, state_re = self.reset_fn(rngs[indices_], params, instructions[indices_])
        
        # Replace indices to -1 if corresponded environment dont DONE. Indices with -1 will not be used (becouse its env will not reset)
        indices_correct = jax.numpy.place(indices_, ~done, -1, inplace=False) 
    
        # Calculate vector of indices we use on this iteration (if RESET was)
        used_instructions = jnp.isin( jnp.arange(len(instructions)), indices_correct)
        
        # Calculate new vector of vector with indices we already used
        last_vector = last_vector - used_instructions.astype(last_vector.dtype)
        
        # rng__, _rng = jax.random.split(rng)
        # reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        # being_reset = jax.random.choice(
        #     _rng,
        #     jnp.arange(self.num_envs),
        #     shape=(self.num_resets,),
        #     p=done,
        #     replace=False,
        # )
        # reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        # obs_re = obs_re[reset_indexes]
        # state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)
        state = DetermEnvState(env_state=state,
                               indecec_vector=last_vector,
                               rngs=rngs[indices_],
                               instructions=instructions[indices_],
                               v1=instructions,
                               v2=rngs,
                               v3=last_vector/jnp.sum(last_vector))
        return obs, state, reward, done, info


# class MinimalExperiment:
#     def __init__(self, args):
#         self.args = args
#         self.env = self._initialize_environment()

#     def _initialize_environment(self):
#         env_name = "Craftax-Classic-Pixels-v1"
#         env = make_craftax_env_from_name(env_name, False)
#         env = InstructionWrapper(env, self.args.craftext_settings)
#         env = DetermOptimisticResetVecEnvWrapper(env, self.args.num_envs, self.args.ratio, n_instructions=env.n_instructions, n_seeds=5) 
#         return env

#     def run(self):
#         rng = jax.random.PRNGKey(42)
#         n_actions = 17 # action space for Craftax-Classic

#         obs, env_state = self.env.reset(rng, self.env.default_params)
#         step_fn = jax.jit(self.env.step)
#         steps = 0
#         instr_rngs = dict()
#         while steps < 500:
#             # Random action selection
#             actions = jax.random.randint(rng, (obs.shape[0],), 0, n_actions)

#             obs, env_state, reward, done, info = step_fn(rng, env_state, actions, self.env.default_params)
            
            
                
#             print("INSTRUCTIONS: ", env_state.env_state.idx)
#             print("RNGS: ", env_state.env_state.rng)
#             print("EVALUATED: ", env_state.indecec_vector)
#             print("used_instructions", env_state.v1)
#             print("possible_indices", env_state.v2)
            
#             indicec_used = np.array(env_state.env_state.idx)
#             rngs_used = np.array(env_state.env_state.rng)
#             for i, instruction in enumerate(indicec_used):
#                 if instruction not in instr_rngs:
#                     instr_rngs[instruction] = []
                    
#                 instr_rngs[instruction].append(tuple(rngs_used[i].tolist()))
        
#             steps += 1
        
#         for instruction in instr_rngs:
#             print("- - -"*20)
#             print("Instruction: ", instruction)
#             print("RNGS: ", set(instr_rngs[instruction]))
#             print("- - -"*20)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--craftext_settings", type=str, default="simple")
#     parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
#     parser.add_argument("--ratio", type=int, default=1)

#     args = parser.parse_args()

#     experiment = MinimalExperiment(args)
#     experiment.run()
