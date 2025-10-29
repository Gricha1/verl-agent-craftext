from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from flax import struct
from gym import Wrapper


from craftext.enviroment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.enviroment.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.enviroment.scenarious.manager import ScenariosNoLambda

from craftext.enviroment.states.state import GameData
from craftext.enviroment.states.state_classic import GameDataClassic

from craftext.enviroment.scenarious.checkers.achivments       import checker_acvievments
from craftext.enviroment.scenarious.checkers.time_constrained import checker_time_placement
from craftext.enviroment.scenarious.checkers.building_star    import checker_star
from craftext.enviroment.scenarious.checkers.building_line    import checker_line
from craftext.enviroment.scenarious.checkers.building_square  import checker_square
from craftext.enviroment.scenarious.checkers.conditional      import checker_conditional_placement
from craftext.enviroment.scenarious.checkers.relevant         import cheker_localization
from craftext.enviroment.scenarious.checkers.object_in_view   import is_object_in_view
from craftext.enviroment.scenarious.checkers.target_state     import TargetState
from typing import Union

@struct.dataclass
class TextEnvState:
    env_state: Any
    timestep: int
    instruction: Optional[jax.Array]
    def_instruction: Optional[jax.Array] #Everitime there will be regular CrafText embedding of instruction
    idx: int
    success_rate: float
    total_success_rate: float
    environment_key: int
    rng: int
    instruction_done: bool
    craftax_reward: int
    checker_id: int
    

def generic_check(game_data: Union[GameData, GameDataClassic], target_state: TargetState, idx: int) -> jnp.ndarray:
    

    def ca(ts: TargetState):   return checker_acvievments(game_data, ts.achievements)
    def cp(ts: TargetState):   return checker_conditional_placement(game_data, ts.conditional_placing)
    def port(ts: TargetState): return cheker_localization(game_data, ts.Localization_placing)
    def ilf(ts: TargetState):  return checker_line(game_data, ts.building_line)
    def isf(ts: TargetState):  return checker_square(game_data, ts.building_square)
    def icf(ts: TargetState):  return checker_star(game_data, ts.building_star)
    def atp(ts: TargetState):  return checker_time_placement(game_data, ts.time_placement)

    fns = (ca, cp, port, ilf, isf, icf, atp, ca)

    return lax.switch(idx, fns, target_state)


import jax
import jax.numpy as jnp


#sample_index_with_hyperbolic_tail_jax(_rng, length=len(self.scenario_handler.scenario_data_jax.embeddings_list), temperature=0.3, decay_k=0.8), #

def sample_index_with_hyperbolic_tail_jax(rng, length: int, temperature: float, decay_k: float = 1.0):
    """
    Безопасная JAX-реализация: семплирует индекс по температурно-гиперболическому распределению.
    """

    def full_weight_sampler(length, temperature, decay_k):
        core_len = jnp.maximum(1, jnp.int32(temperature * length))
        tail_len = length - core_len

        # равномерная часть
        uniform_weights = jnp.full((core_len,), 1.0)

        # хвостовая часть
        tail_indices = jnp.arange(1, tail_len + 1)
        tail_weights = 1.0 / (1.0 + decay_k * tail_indices)

        # объединяем
        weights = jnp.concatenate([uniform_weights, tail_weights], axis=0)
        probs = weights / jnp.sum(weights)

        return probs

    # чтобы не писать cond, можно использовать чистую версию
    probs = full_weight_sampler(length, temperature, decay_k)
    return jax.random.choice(rng, a=length, p=probs)



class InstructionWrapper(Wrapper):
    def __init__(self, env, sample_range: tuple, config_name=None, scenario_handler_class=ScenariosNoLambda,
                  encode_model_class=DistilBertEncode, encode_form=EncodeForm.EMBEDDING):
        """
        Initializes the InstructionWrapper with the environment, creating EncodeModel and CrafTextScenarios.
        
        Parameters:
        - env: The environment to wrap.
        - config_name: Optional configuration name for scenarios.
        - encode_model_class: A class for the encoding model. Defaults to DistilBertEncode.
        - encode_form: The form of encoding (EMBEDDING or TOKEN). Defaults to EMBEDDING.
        """
        super().__init__(env)

        # self.default_encoder = DistilBertEncode(form_to_use=encode_form)
        # self.default_sc_handler = ScenariosNoLambda(self.default_encoder, config_name)
        
        self.encode_model = encode_model_class(form_to_use=encode_form)

        # Initialize the scenario handler with the encoding model
        self.scenario_handler = scenario_handler_class(self.encode_model, config_name)
        #initial_instruction 
        self.encoded_instruction = self.scenario_handler.scenario_data_jax.embeddings_list[0]
        self.scenario_arguments = self.scenario_handler.scenario_data_jax.arguments
        self.batched_ts = TargetState.stack(self.scenario_arguments)

        # mine
        self.sample_range = sample_range

        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        print("Initialized Instruction Wrapper with environment key:", self.environment_key)
        # print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)
        def_instruction_emb = self.scenario_handler.scenario_data_jax.original_inst_emb[0]
        
        print("Def instr shape:", def_instruction_emb.shape)
        
  
    def reset(self, _rng, env_params, instruction_idx=-1):
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        idx = jax.lax.cond(
                instruction_idx == -1, 
                # lambda:  jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.scenario_handler.scenario_data_jax.embeddings_list)),
                lambda:  jax.random.randint(_rng, shape=(), minval=self.sample_range[0], maxval=min(self.sample_range[1], len(self.scenario_handler.scenario_data_jax.embeddings_list))),
                lambda: instruction_idx
            )
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]
        def_instruction_emb = self.scenario_handler.scenario_data_jax.original_inst_emb[idx]
        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=instructions_emb,
            def_instruction = def_instruction_emb,
            idx=idx,
            environment_key=self.environment_key,
            success_rate=0.0,
            total_success_rate=0.0,
            craftax_reward = 0.0,
            rng=_rng,
            instruction_done=False,
            checker_id=self.scenario_handler.scenario_data_jax.scenario_checker[idx]
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        craftax_reward = reward
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
                    
        ts = self.batched_ts.select(env_state.idx)
       # print(ts)
        instruction_done = generic_check(game_data_vector, ts, env_state.checker_id)
        
        # If EXPLORE mode - give craftAx reward
        reward = lax.cond(
                    env_state.checker_id < 7,
                    lambda r: r / 50,
                    lambda r: r / 10,
                    reward
                )
       
        reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        
        done = instruction_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            instruction=env_state.instruction,
            def_instruction = env_state.def_instruction,
            idx=env_state.idx,
            environment_key=env_state.environment_key,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=env_state.rng,
            craftax_reward = craftax_reward,
            instruction_done=instruction_done,
            checker_id=env_state.checker_id
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        info.update({"Cheker_id": env_state.checker_id})
        self.steps += 1
        return obs, state, reward, done, info
 
 
class ResetUntilObjectWrapper(Wrapper):
    def __init__(self, env, object_id: int, view_radius: int = 5, max_resets: int = 1000):
        super().__init__(env)
        self.object_id = object_id
        self.view_radius = view_radius
        self.max_resets = max_resets
        self._check_fn = partial(is_object_in_view, object_id=self.object_id, view_radius=self.view_radius)

    # --- НОВЫЙ МЕТОД, КОТОРЫЙ РЕШАЕТ ПРОБЛЕМУ ---
    def step(self, _rng, env_state, action, env_params):
        """
        Просто передает вызов step к обернутой среде.
        Эта обертка изменяет только поведение reset.
        """
        return self.env.step(_rng, env_state, action, env_params)

    # Метод reset остается без изменений
    def reset(self, _rng, env_params, instruction_idx=-1):
        init_rng, loop_rng = jax.random.split(_rng)

        def reset_body_fun(loop_state):
            _, _, rng, count, _ = loop_state
            rng, reset_rng = jax.random.split(rng)
            new_obs, new_text_env_state = self.env.reset(reset_rng, env_params, instruction_idx)
            game_data = self.env.StateStructure.from_state(new_text_env_state.env_state, new_text_env_state.env_state, 0)
            object_found = self._check_fn(game_data)
            return new_obs, new_text_env_state, rng, count + 1, object_found

        def reset_cond_fun(loop_state):
            _, _, _, count, object_found = loop_state
            return jnp.logical_and(jnp.logical_not(object_found), count < self.max_resets)

        initial_obs, initial_text_env_state = self.env.reset(init_rng, env_params, instruction_idx)
        initial_game_data = self.env.StateStructure.from_state(initial_text_env_state.env_state, initial_text_env_state.env_state, 0)
        initial_found = self._check_fn(initial_game_data)
        initial_loop_state = (initial_obs, initial_text_env_state, loop_rng, 0, initial_found)

        final_obs, final_state, _, final_count, _ = lax.while_loop(
            reset_cond_fun,
            reset_body_fun,
            initial_loop_state
        )

        return final_obs, final_state