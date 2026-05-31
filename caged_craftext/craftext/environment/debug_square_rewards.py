"""Reward shaping and goal checks for the fixed 8x8 debug_square map."""
from __future__ import annotations

import jax.numpy as jnp

from craftext.environment.craftext_constants import Achievement, AchievementState

# Corner resource cells (row, col) — see debug_square_world_gen.py
_DEBUG_ACHIEVEMENT_IDS = jnp.array(
    [
        Achievement.COLLECT_STONE,
        Achievement.COLLECT_WOOD,
        Achievement.COLLECT_DRINK,
    ],
    dtype=jnp.int32,
)
_DEBUG_TARGET_CELLS = jnp.array(
    [
        [1, 1],  # stone
        [1, 6],  # wooden block (WOOD tile, not border trees)
        [6, 1],  # water
    ],
    dtype=jnp.int32,
)

TASK_COMPLETION_REWARD = jnp.float32(1.0)


def debug_square_target_cell(achievement_mask) -> jnp.ndarray:
    """(row, col) of the resource block for the current task."""
    mask = jnp.asarray(achievement_mask, dtype=jnp.int32)
    is_need = mask[_DEBUG_ACHIEVEMENT_IDS] == AchievementState.NEED_TO_ACHIEVE
    weights = is_need.astype(jnp.float32)
    weights = weights / (jnp.sum(weights) + 1e-8)
    return jnp.sum(weights[:, None] * _DEBUG_TARGET_CELLS, axis=0).astype(jnp.int32)


def manhattan_distance(player_pos, target_cell) -> jnp.ndarray:
    player_pos = jnp.asarray(player_pos, dtype=jnp.int32).reshape(2)
    target_cell = jnp.asarray(target_cell, dtype=jnp.int32).reshape(2)
    return jnp.abs(player_pos[0] - target_cell[0]) + jnp.abs(player_pos[1] - target_cell[1])


def debug_square_adjacent_to_goal(player_pos, achievement_mask) -> jnp.ndarray:
    """
    True when the player is orthogonally adjacent to the task block (L1 distance 1)
    or standing on it (distance 0). Facing direction does not matter.
    """
    target = debug_square_target_cell(achievement_mask)
    return manhattan_distance(player_pos, target) <= 1


def debug_square_navigation_reward(prev_player_pos, new_player_pos, achievement_mask) -> jnp.ndarray:
    """Potential-based shaping: d(s_t, goal) - d(s_{t+1}, goal) in grid cells (L1)."""
    target = debug_square_target_cell(achievement_mask)
    prev_d = manhattan_distance(prev_player_pos, target).astype(jnp.float32)
    new_d = manhattan_distance(new_player_pos, target).astype(jnp.float32)
    return prev_d - new_d


def debug_square_step_reward(
    craftax_reward,
    prev_player_pos,
    new_player_pos,
    achievement_mask,
    instruction_done,
) -> jnp.ndarray:
    """
    debug_square_8x8:
      - Craftax achievement reward disabled
      - navigation reward (distance decrease)
      - bonus when adjacent to the goal cell
    """
    del craftax_reward
    nav = debug_square_navigation_reward(prev_player_pos, new_player_pos, achievement_mask)
    task_bonus = jnp.where(instruction_done, TASK_COMPLETION_REWARD, jnp.float32(0.0))
    return nav + task_bonus
