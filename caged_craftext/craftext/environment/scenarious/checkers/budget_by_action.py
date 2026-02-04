import jax

from jax import (
    numpy as jnp,
    lax
)

from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state_cmdp_math_budget_by_action import BudgetByAction, TargetAction
from typing import Tuple
from craftext.environment.craftext_constants import Achievement

def checker_budget_by_action(game_data: GameDataClassic,  target_state: BudgetByAction) -> Tuple[BudgetByAction, jax.Array]:
    current_budget = target_state.budget
    actions = target_state.delimiter_actions
    target_action = target_state.target_action
    new_budget, is_cost = apply_action_cost_if_wood_collected(game_data, current_budget=current_budget, actions=actions, target_action=target_action)
    target_state = BudgetByAction(budget=new_budget, delimiter_actions=actions)
    return target_state, is_cost

def apply_action_cost_if_wood_collected(game_data: GameDataClassic, current_budget: int, actions: jnp.ndarray, target_action: int):
    
    current_minus = actions[game_data.states[0].action].astype(int)
    
    
    def cow(_):
        is_eat_cow = game_data.states[0].achievements.achievements[Achievement.EAT_COW]
        new_budget = jax.lax.cond(is_eat_cow > 0, lambda: current_budget - 10, lambda: current_budget) 
        return new_budget, (is_eat_cow > 0) & (new_budget < 0)
    
    def wood(_):
        is_collect_wood = game_data.states[0].achievements.achievements[Achievement.COLLECT_WOOD] #game_data.states[0].inventory.wood - game_data.states[1].inventory.wood
        new_budget = jax.lax.cond(is_collect_wood > 0, lambda: current_budget - 10, lambda: current_budget) 
        return new_budget, is_collect_wood
        # return new_budget, (is_collect_wood > 0) & (new_budget < 0)
    
   
    
    return jax.lax.cond(target_action == TargetAction.WOOD, wood, cow, operand=None)