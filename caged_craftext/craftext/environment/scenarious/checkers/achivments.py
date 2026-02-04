import jax
import jax.numpy as jnp
from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import Achievements, AchievementState

def checker_acvievments(game_data: Union[GameDataClassic, GameData],  target_state: Achievements) -> jax.Array:
    
    achievement_mask  = target_state.achievement_mask
    
    # return jax.lax.select(target_state.need_to_achieve, 
    #                conditional_achivments(game_data, achievement_mask),
    #                jnp.array(False))
    return conditional_achivments(game_data, achievement_mask)

def conditional_achivments(gd: Union[GameDataClassic, GameData], achievement_mask):
    # 0) Убедимся, что маска — это JAX‑массив
    mask = jnp.array(achievement_mask, dtype=jnp.int32)

    current_state      = gd.states[0]
    state_achievements = current_state.achievements.achievements  # jnp.array of 0/1

    # 1) там, где mask == NEED, но ещё не достигнуто (state==0) → ошибка
    must_achieve     = (mask == AchievementState.NEED_TO_ACHIEVE) & (state_achievements == 0)
    # 2) там, где mask == AVOID, но уже достигнуто (state==1) → ошибка
    must_not_achieve = (mask == AchievementState.AVOID_TO_ACHIEVE) & (state_achievements == 1)

    # если хоть раз ошибились — fail=True
    fail_condition = jnp.any(must_achieve) | jnp.any(must_not_achieve)

    # True только когда нет ни одного fail
    return jnp.logical_not(fail_condition)


