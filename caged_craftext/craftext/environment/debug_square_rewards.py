"""Reward shaping and goal checks for debug_square maps (resources may move each reset)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from craftext.environment.craftext_constants import Achievement, AchievementState

# Goal tiles: inner corners only; assignment shuffled each reset (see debug_square_world_gen.py).
_DEBUG_ACHIEVEMENT_IDS = jnp.array(
    [
        Achievement.COLLECT_STONE,
        Achievement.COLLECT_WOOD,
        Achievement.COLLECT_DRINK,
    ],
    dtype=jnp.int32,
)
# Fallback goal cells for 8x8 when live map is unavailable (stone/wood/water order).
_DEBUG_TARGET_CELLS = jnp.array(
    [
        [1, 1],  # stone
        [1, 6],  # wooden block (WOOD tile, not border trees)
        [6, 1],  # water
    ],
    dtype=jnp.int32,
)
# craftax_classic BlockType values for the three goal tiles
_DEBUG_GOAL_BLOCK_TYPES = jnp.array(
    [
        4,  # STONE
        6,  # WOOD (placed block, not TREE)
        3,  # WATER
    ],
    dtype=jnp.int32,
)
# Fixed task order when use_paraphrases=False (see jax_debug_square_*/instructions.py).
_DEBUG_GOAL_BLOCK_BY_IDX = jnp.array([4, 6, 3], dtype=jnp.int32)
_DEBUG_TARGET_CELLS_BY_IDX = jnp.array(
    [
        [1, 1],
        [1, 6],
        [6, 1],
    ],
    dtype=jnp.int32,
)
# Player cell + 8-neighborhood (orthogonal + diagonal) for live-map goal checks.
_NEIGHBOR_OFFSETS = jnp.array(
    [
        [0, 0],
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

TASK_COMPLETION_REWARD = jnp.float32(1.0)


def is_debug_square_config(config_name) -> bool:
    """True for debug_square_* scenario configs (Hydra/OmegaConf-safe)."""
    return config_name is not None and "debug_square" in str(config_name)


def _normalize_achievement_mask(achievement_mask) -> jnp.ndarray:
    """Flax TargetState.select may leave a tuple of scalar arrays; flatten to (N,)."""
    mask = jnp.asarray(achievement_mask, dtype=jnp.int32)
    return mask.reshape(-1)


def debug_square_find_resource_cell(game_map, block_type_id) -> jnp.ndarray:
    """(row, col) of the unique inner STONE/WOOD/WATER tile on the current map."""
    game_map = jnp.asarray(game_map, dtype=jnp.int32)
    block_type = jnp.asarray(block_type_id, dtype=jnp.int32).reshape(())
    h, w = game_map.shape[0], game_map.shape[1]
    rows = jnp.arange(1, h - 1, dtype=jnp.int32)
    cols = jnp.arange(1, w - 1, dtype=jnp.int32)
    rr, cc = jnp.meshgrid(rows, cols, indexing="ij")
    flat_r = rr.reshape(-1)
    flat_c = cc.reshape(-1)
    matches = (game_map[flat_r, flat_c] == block_type).astype(jnp.float32)
    weights = matches / (jnp.sum(matches) + 1e-8)
    target_r = jnp.sum(weights * flat_r.astype(jnp.float32))
    target_c = jnp.sum(weights * flat_c.astype(jnp.float32))
    return jnp.array([target_r, target_c], dtype=jnp.int32)


def debug_square_target_cell(achievement_mask) -> jnp.ndarray:
    """(row, col) of the resource block for the current task."""
    mask = _normalize_achievement_mask(achievement_mask)
    is_need = mask[_DEBUG_ACHIEVEMENT_IDS] == AchievementState.NEED_TO_ACHIEVE
    weights = is_need.astype(jnp.float32)
    weights = weights / (jnp.sum(weights) + 1e-8)
    return jnp.sum(weights[:, None] * _DEBUG_TARGET_CELLS, axis=0).astype(jnp.int32)


def debug_square_goal_block_type(achievement_mask, instruction_idx=None) -> jnp.ndarray:
    """craftax_classic BlockType id for the active goal tile."""
    if instruction_idx is not None:
        idx = jnp.asarray(instruction_idx, dtype=jnp.int32).reshape(())
        in_range = (idx >= 0) & (idx < _DEBUG_GOAL_BLOCK_BY_IDX.shape[0])
        from_idx = _DEBUG_GOAL_BLOCK_BY_IDX[jnp.clip(idx, 0, _DEBUG_GOAL_BLOCK_BY_IDX.shape[0] - 1)]
        return jnp.where(in_range, from_idx, jnp.int32(_DEBUG_GOAL_BLOCK_TYPES[0]))

    mask = _normalize_achievement_mask(achievement_mask)
    is_need = mask[_DEBUG_ACHIEVEMENT_IDS] == AchievementState.NEED_TO_ACHIEVE
    weights = is_need.astype(jnp.float32)
    weights = weights / (jnp.sum(weights) + 1e-8)
    from_mask = jnp.sum(weights * _DEBUG_GOAL_BLOCK_TYPES).astype(jnp.int32)
    has_need = jnp.sum(is_need) > 0
    return jnp.where(has_need, from_mask, jnp.int32(_DEBUG_GOAL_BLOCK_TYPES[0]))


def manhattan_distance(player_pos, target_cell) -> jnp.ndarray:
    player_pos = jnp.asarray(player_pos, dtype=jnp.int32).reshape(2)
    target_cell = jnp.asarray(target_cell, dtype=jnp.int32).reshape(2)
    return jnp.abs(player_pos[0] - target_cell[0]) + jnp.abs(player_pos[1] - target_cell[1])


def chebyshev_distance(player_pos, target_cell) -> jnp.ndarray:
    player_pos = jnp.asarray(player_pos, dtype=jnp.int32).reshape(2)
    target_cell = jnp.asarray(target_cell, dtype=jnp.int32).reshape(2)
    return jnp.maximum(
        jnp.abs(player_pos[0] - target_cell[0]),
        jnp.abs(player_pos[1] - target_cell[1]),
    )


def debug_square_target_cell_for_idx(
    instruction_idx, game_map=None
) -> jnp.ndarray:
    """Goal cell from scenario index (stone=0, wood=1, water=2). Uses live map when provided."""
    idx = jnp.asarray(instruction_idx, dtype=jnp.int32).reshape(())
    in_range = (idx >= 0) & (idx < _DEBUG_GOAL_BLOCK_BY_IDX.shape[0])
    clipped = jnp.clip(idx, 0, _DEBUG_GOAL_BLOCK_BY_IDX.shape[0] - 1)
    block_type = _DEBUG_GOAL_BLOCK_BY_IDX[clipped]

    if game_map is not None:
        from_map = debug_square_find_resource_cell(game_map, block_type)
        fixed = _DEBUG_TARGET_CELLS_BY_IDX[clipped]
        target = jnp.where(in_range, from_map, _DEBUG_TARGET_CELLS[0])
        return target

    target = _DEBUG_TARGET_CELLS_BY_IDX[clipped]
    return jnp.where(in_range, target, _DEBUG_TARGET_CELLS[0])


def debug_square_resolve_target_cell(
    achievement_mask, instruction_idx=None, game_map=None
) -> jnp.ndarray:
    """Prefer instruction_idx on debug_square; fall back to achievement mask."""
    if instruction_idx is not None:
        return debug_square_target_cell_for_idx(instruction_idx, game_map=game_map)
    return debug_square_target_cell(achievement_mask)


def debug_square_at_goal_from_map(
    env_state, achievement_mask, instruction_idx=None
) -> jnp.ndarray:
    """
    True when the goal block type appears on the player cell or any 8-neighbor cell
    (orthogonal + diagonal). Border TREE tiles are not goals; only STONE/WOOD/WATER blocks count.
    """
    goal_block = debug_square_goal_block_type(achievement_mask, instruction_idx=instruction_idx)
    player_pos = jnp.asarray(env_state.player_position, dtype=jnp.int32).reshape(2)
    game_map = env_state.map
    map_h, map_w = game_map.shape[0], game_map.shape[1]

    def _check_offset(_unused, offset):
        pos = player_pos + offset
        in_bounds = (
            (pos[0] >= 0)
            & (pos[0] < map_h)
            & (pos[1] >= 0)
            & (pos[1] < map_w)
        )
        block = game_map[pos[0], pos[1]]
        return None, jnp.logical_and(in_bounds, block == goal_block)

    _, hits = jax.lax.scan(_check_offset, None, _NEIGHBOR_OFFSETS)
    return hits.any()


def debug_square_adjacent_to_goal(
    player_pos,
    achievement_mask,
    env_state=None,
    instruction_idx=None,
) -> jnp.ndarray:
    """
    Success when adjacent to the task resource tile (Chebyshev distance <= 1).

    Primary: goal block type (from instruction_idx) appears in the 8-neighborhood on
    the live map. Fallback: distance to the fixed target cell.
    """
    if env_state is not None:
        map_ok = debug_square_at_goal_from_map(
            env_state, achievement_mask, instruction_idx=instruction_idx
        )
        game_map = env_state.map
    else:
        map_ok = jnp.array(False)
        game_map = None

    target = debug_square_resolve_target_cell(
        achievement_mask, instruction_idx=instruction_idx, game_map=game_map
    )
    coord_ok = chebyshev_distance(player_pos, target) <= 1
    return jnp.logical_or(map_ok, coord_ok)


def debug_square_navigation_reward(
    prev_player_pos,
    new_player_pos,
    achievement_mask,
    instruction_idx=None,
    game_map=None,
) -> jnp.ndarray:
    """Potential-based shaping: d(s_t, goal) - d(s_{t+1}, goal) in grid cells (L1)."""
    target = debug_square_resolve_target_cell(
        achievement_mask, instruction_idx=instruction_idx, game_map=game_map
    )
    prev_d = manhattan_distance(prev_player_pos, target).astype(jnp.float32)
    new_d = manhattan_distance(new_player_pos, target).astype(jnp.float32)
    return prev_d - new_d


def debug_square_step_reward(
    craftax_reward,
    prev_player_pos,
    new_player_pos,
    achievement_mask,
    instruction_done,
    instruction_idx=None,
    game_map=None,
) -> jnp.ndarray:
    """
    debug_square (8x8 / 16x16):
      - Craftax achievement reward disabled
      - navigation reward (distance decrease)
      - bonus when adjacent to the goal cell
    """
    del craftax_reward
    nav = debug_square_navigation_reward(
        prev_player_pos,
        new_player_pos,
        achievement_mask,
        instruction_idx=instruction_idx,
        game_map=game_map,
    )
    task_bonus = jnp.where(instruction_done, TASK_COMPLETION_REWARD, jnp.float32(0.0))
    return nav + task_bonus
