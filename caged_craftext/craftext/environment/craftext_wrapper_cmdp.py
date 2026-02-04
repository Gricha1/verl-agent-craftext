from typing import Any, Optional
import jax
import jax.numpy as jnp
from jax import lax

from flax import struct
from gym import Wrapper


from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode

from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import Scenarios 
from craftext.environment.scenarious.checkers.achivments       import checker_acvievments
from craftext.environment.scenarious.checkers.time_constrained import checker_time_placement
from craftext.environment.scenarious.checkers.building_star    import checker_star
from craftext.environment.scenarious.checkers.building_line    import checker_line
from craftext.environment.scenarious.checkers.building_square  import checker_square
from craftext.environment.scenarious.checkers.conditional      import checker_conditional_placement
from craftext.environment.scenarious.checkers.relevant         import cheker_localization
from craftext.environment.scenarious.checkers.step_on_block    import checker_step_on_block
from typing import Union

from craftext.environment.craftext_wrapper import InstructionWrapper, TextEnvState
from craftext.environment.scenarious.checkers.budget_build_collect import checker_budget_build_collect
from craftext.environment.scenarious.checkers.dont_move import checker_moveing_at_night_level
from craftext.environment.scenarious.checkers.sleep_at_night import checker_sleep_at_night
from craftext.environment.scenarious.checkers.monster_is_attacked_without_sword import checker_monster_is_attacked_without_sword
from craftext.environment.scenarious.checkers.away_from_monsters_when_hp_low import checker_away_from_monsters_when_hp_low
from craftext.environment.scenarious.checkers.dont_sleep_near_monsters import checker_dont_sleep_near_monsters
from craftext.environment.scenarious.checkers.drink_level import checker_budget_drink_level
from craftext.environment.scenarious.checkers.hp_level import checker_budget_hp_level
from craftext.environment.scenarious.checkers.hungry_level import checker_budget_hungry_level
from craftext.environment.scenarious.checkers.energy_level import checker_budget_energy_level

from craftext.environment.scenarious.checkers.avoid_mob_distance import checker_relactional_avoid_mob_distance
# from craftext.environment.scenarious.checkers.see_in_field_of_view import checker_what_in_field_of_view
from craftext.environment.scenarious.checkers.budget_by_action import checker_budget_by_action
from craftext.environment.scenarious.checkers.last_visible_target import checker_last_visible_target

from craftext.environment.scenarious.checkers.target_state_cmdp_relactional_point_of_intereset import CMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_relational_all import CMDPTargetState as RelationalAllCMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_math_all import CMDPTargetState as MathAllCMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_math_budget_by_action import CMDPTargetState as MATHCMDPTargetState
from craftext.environment.scenarious.checkers.target_state_cmdp_caged import CMDPTargetState as CagedCMDPTargetState

@struct.dataclass
class TextEnvStateCMDP(TextEnvState):
    textual_constraint: Optional[jax.Array]
    cost_type: int
    idx: int
    success_rate: float
    episode_cost: float
    cost: float
    target_state: CMDPTargetState
    

class CMDPInstructionWrapper(InstructionWrapper):
    def __init__(self, env, config_name=None, scenario_handler_class=ScenariosNoLambdaCMDP,
                  encode_model_class=DistilBertEncode, encode_form=EncodeForm.EMBEDDING):
        super().__init__(env, config_name=config_name, scenario_handler_class=scenario_handler_class,
                  encode_model_class=encode_model_class, encode_form=encode_form)

        self.config_name = config_name
        self.encoded_textual_constraint = self.scenario_handler.scenario_data_jax.constraints_embeddings_list[0]

    def reset(self, _rng, env_params, instruction_idx=-1):
        obs, state = super().reset(_rng, env_params, instruction_idx=instruction_idx)

        textual_constraint_emb = self.scenario_handler.scenario_data_jax.constraints_embeddings_list[state.idx]
        cost_type = self.scenario_handler.scenario_data_jax.cost_types[state.idx]
        state = TextEnvStateCMDP(
            env_state=state.env_state,
            timestep=state.timestep,
            instruction=state.instruction,
            textual_constraint=textual_constraint_emb,
            cost_type=cost_type,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            episode_cost=0.,
            cost=0.,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id,
            target_state=state.target_state
        )
        return obs, state

    def step(self, _rng, env_state, action, env_params):
        obs, state, reward, done, info = super().step(_rng, env_state, action, env_params)

        # set cost
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state.env_state, action)
        ts = env_state.target_state

        if self.config_name == "simple_achivments_safe":
            cost = checker_step_on_block(game_data_vector, ts.step_on_block).astype(float)
        # budgetary
        elif self.config_name == "achievements_safe_budget_dont_move_night":
            cost = checker_moveing_at_night_level(game_data_vector, ts.night_constraint_level).astype(float)
        elif self.config_name == "build_squere_simple_safe_budget":
            cost = checker_budget_build_collect(game_data_vector, ts.build_budget_state).astype(float)
        elif self.config_name == "achievements_safe_budget_drink":
            cost = checker_budget_drink_level(game_data_vector, ts.drink_level_state.level).astype(float)
        elif self.config_name == "achievements_easy_safe_budget_drink":
            cost = checker_budget_drink_level(game_data_vector, ts.drink_level_state.level).astype(float)
        elif self.config_name in ("achievements_safe_budget_energy", "achievements_safe_budget_energy_collect_wood"):
            cost = checker_budget_energy_level(game_data_vector, ts.energy_level_state.level).astype(float)
        elif self.config_name == "achievements_safe_budget_hp":
            cost = checker_budget_hp_level(game_data_vector, ts.hp_level_state.level).astype(float)
        elif self.config_name == "achievements_safe_budget_hungry" or \
             self.config_name == "achievements_safe_budget_hungry_multi_limit" or \
             self.config_name == "cmdp_hard_achievements_budget_hungry":
            cost = checker_budget_hungry_level(game_data_vector, ts.hungry_level_state.level).astype(float)
        elif self.config_name == "achievements_safe_budget_sleep_at_night":
            cost = checker_sleep_at_night(game_data_vector, ts.night_constraint_level, ts.day_constraint_level).astype(float)
        # sequential
        elif self.config_name == "achievements_safe_sequential_defeat_monster":
            cost = checker_monster_is_attacked_without_sword(game_data_vector).astype(float)
        elif self.config_name == "achievements_safe_sequential_dont_sleep_near_monsters":
            cost = checker_dont_sleep_near_monsters(game_data_vector).astype(float)
        elif self.config_name == "achievements_safe_sequential_away_monsters_when_hp":
            cost = checker_away_from_monsters_when_hp_low(game_data_vector, ts.hp_level_state).astype(float)
        elif self.config_name == "achievements_safe_sequential_all":
            #    env_state.cost_type
            #    "sequential_dont_sleep_near_monsters": 0,
            #    "sequential_defeat_monster": 1, 
            #    "sequential_away_monsters_when_hp": 2
            cost = jax.lax.switch(
                env_state.cost_type,
                [
                    # case 0: sequential_dont_sleep_near_monsters
                    lambda: checker_dont_sleep_near_monsters(game_data_vector).astype(float),
                    # case 1: sequential_defeat_monster  
                    lambda: checker_monster_is_attacked_without_sword(game_data_vector).astype(float),
                    # case 2: sequential_away_monsters_when_hp
                    lambda: checker_away_from_monsters_when_hp_low(game_data_vector, ts.hp_level_state).astype(float)
                ]
            )
        elif self.config_name in ("achievements_safe_math_wood_budget", "achievements_safe_math_food_budget"):
            new_target_state, cost = checker_budget_by_action(game_data_vector, ts.budget_by_action)
            ts = MATHCMDPTargetState(achievements=ts.achievements, budget_by_action=new_target_state)
            cost = cost.astype(float)
        elif self.config_name == "achievements_safe_math_all":
             #    env_state.cost_type
            #    "math_food_budget": 0,
            #    "math_wood_budget": 1, 
            new_target_state_food, cost_food = checker_budget_by_action(game_data_vector, ts.budget_food_by_action)
            new_target_state_wood, cost_wood = checker_budget_by_action(game_data_vector, ts.budget_wood_by_action)
            ts = MathAllCMDPTargetState(achievements=ts.achievements, budget_food_by_action=new_target_state_food, 
                                                                   budget_wood_by_action=new_target_state_wood)
            cost = jax.lax.switch(
                env_state.cost_type,
                [
                    # case 0: math_food_budget
                    lambda: cost_food.astype(float),   
                    # case 1: math_wood_budget  
                    lambda: cost_wood.astype(float),
                ]
            )
        elif self.config_name == "achievements_easy_relational_avoid_enemy_by_radius":
            cost = checker_relactional_avoid_mob_distance(game_data_vector, ts.avoid_mob_distance).astype(float)    
        elif self.config_name in ("achievements_easy_relational_last_food_location", "achievements_easy_relational_last_water_location"):
            new_target_state, cost = checker_last_visible_target(game_data_vector, ts.target_of_interest)
            ts = CMDPTargetState(achievements=ts.achievements, target_of_interest=new_target_state)
            cost = cost.astype(float)
        elif self.config_name == "achievements_easy_relational_all":
            #    env_state.cost_type
            #    "relational_last_food_location": 0,
            #    "relational_last_water_location": 1, 
            #    "relational_avoid_enemy_by_radius": 2
            new_target_state_food, cost_food = checker_last_visible_target(game_data_vector, ts.target_of_interest_food)
            new_target_state_water, cost_water = checker_last_visible_target(game_data_vector, ts.target_of_interest_water)
            ts = RelationalAllCMDPTargetState(achievements=ts.achievements, avoid_mob_distance=ts.avoid_mob_distance, 
                                    target_of_interest_water=new_target_state_water, target_of_interest_food=new_target_state_food)
            cost = jax.lax.switch(
                env_state.cost_type,
                [
                    # case 0: relational_last_food_location
                    lambda: cost_food.astype(float),   
                    # case 1: relational_last_water_location  
                    lambda: cost_water.astype(float),
                    # case 2: relational_avoid_enemy_by_radius
                    lambda: checker_relactional_avoid_mob_distance(game_data_vector, ts.avoid_mob_distance).astype(float),
                ]
            )
        elif self.config_name == "achievements_safe_budget_all":
            #    env_state.cost_type
            #    "budget_hp": 0,
            #    "budget_drink": 1, 
            #    "budget_energy": 2
            #    "budget_hungry": 3
            cost = jax.lax.switch(
                env_state.cost_type,
                [
                    # case 0: budget_hp
                    lambda: checker_budget_hp_level(game_data_vector, ts.hp_level_state.level).astype(float),
                    # case 1: budget_drink  
                    lambda: checker_budget_drink_level(game_data_vector, ts.hp_level_state.level).astype(float),
                    # case 2: budget_energy
                    lambda: checker_budget_energy_level(game_data_vector, ts.hp_level_state.level).astype(float),
                    # case 3: budget_hungry
                    lambda: checker_budget_hungry_level(game_data_vector, ts.hp_level_state.level).astype(float)
                ]
            )
        elif self.config_name == "achievements_safe_caged":
             #    env_state.cost_type
            #    "budget_hp": 0, 
            #    "budget_drink": 1, 
            #    "budget_energy": 2, 
            #    "budget_hungry": 3, 
            #    "sequential_away_monsters_when_hp": 4, 
            #    "sequential_dont_sleep_near_monsters": 5, 
            #    "sequential_defeat_monster": 6
            #    "relational_last_food_location": 7, 
            #    "relational_last_water_location": 8,
            #    "relational_avoid_enemy_by_radius": 9,
            #    "math_food_budget": 10, 
            #    "math_wood_budget": 11
            new_target_state_math_food, cost_math_food = checker_budget_by_action(game_data_vector, ts.budget_food_by_action)
            new_target_state_math_wood, cost_math_wood = checker_budget_by_action(game_data_vector, ts.budget_wood_by_action)
            new_target_state_relational_food, cost_relational_food = checker_last_visible_target(game_data_vector, ts.target_of_interest_food)
            new_target_state_relational_water, cost_relational_water = checker_last_visible_target(game_data_vector, ts.target_of_interest_water)
            
            ts = CagedCMDPTargetState(achievements=ts.achievements,
                                      budget_food_by_action=new_target_state_math_food, 
                                      budget_wood_by_action=new_target_state_math_wood,
                                      avoid_mob_distance=ts.avoid_mob_distance,
                                      target_of_interest_water=new_target_state_relational_water,
                                      target_of_interest_food=new_target_state_relational_food,
                                      level_hp_budget=ts.level_hp_budget, 
                                      level_away_monsters=ts.level_away_monsters)
            cost = jax.lax.switch(
                env_state.cost_type,
                [   # case 0: budget_hp
                    lambda: checker_budget_hp_level(game_data_vector, ts.level_hp_budget.level).astype(float),
                    # case 1: budget_drink  
                    lambda: checker_budget_drink_level(game_data_vector, ts.level_hp_budget.level).astype(float),
                    # case 2: budget_energy
                    lambda: checker_budget_energy_level(game_data_vector, ts.level_hp_budget.level).astype(float),
                    # case 3: budget_hungry
                    lambda: checker_budget_hungry_level(game_data_vector, ts.level_hp_budget.level).astype(float),
                    # case 4: sequential_dont_sleep_near_monsters
                    lambda: checker_dont_sleep_near_monsters(game_data_vector).astype(float),
                    # case 5: sequential_defeat_monster  
                    lambda: checker_monster_is_attacked_without_sword(game_data_vector).astype(float),
                    # case 6: sequential_away_monsters_when_hp
                    lambda: checker_away_from_monsters_when_hp_low(game_data_vector, ts.level_away_monsters).astype(float),
                     # case 7: relational_last_food_location
                    lambda: cost_relational_food.astype(float),   
                    # case 8: relational_last_water_location  
                    lambda: cost_relational_water.astype(float),
                    # case 9: relational_avoid_enemy_by_radius
                    lambda: checker_relactional_avoid_mob_distance(game_data_vector, ts.avoid_mob_distance).astype(float),
                    # case 10: math_food_budget
                    lambda: cost_math_food.astype(float),   
                    # case 11: math_wood_budget  
                    lambda: cost_math_wood.astype(float),
                    
                ]
            )
        else:
            assert 1 == 0, f"unknow config name: {self.config_name}, need assign cost function for this config"
            

        state = TextEnvStateCMDP(
            env_state=state.env_state,
            timestep=state.timestep,
            instruction=state.instruction,
            textual_constraint=env_state.textual_constraint,
            cost_type=env_state.cost_type,
            idx=state.idx,
            environment_key=state.environment_key,
            success_rate=state.success_rate,
            episode_cost=env_state.episode_cost + cost,
            cost=cost,
            total_success_rate=state.total_success_rate,
            rng=state.rng,
            instruction_done=state.checker_id,
            checker_id=state.checker_id,
            target_state=ts
        )

        return obs, state, reward, done, info