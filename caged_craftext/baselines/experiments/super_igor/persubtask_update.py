import jax
import jax.numpy as jnp

# # -----------------------------
# # 1. Инициализация матриц
# # -----------------------------
# def initialize_reward_tracking(num_goals: int, max_subgoals: int):
#     shape = (num_goals, max_subgoals)
#     reward_sum_matrix = jnp.zeros(shape, dtype=jnp.float32)
#     episode_count_matrix = jnp.zeros(shape, dtype=jnp.int32)
#     return reward_sum_matrix, episode_count_matrix

# # -----------------------------
# # 2. Вычисление маски смены подзадачи
# # -----------------------------
# def get_update_mask(prev_subgoals: jnp.ndarray,
#                     curr_subgoals: jnp.ndarray,
#                     curr_goals: jnp.ndarray,
#                     num_goals: int,
#                     max_subgoals: int):
#     changed_mask = prev_subgoals != curr_subgoals

#     # Нужно отметить именно ПРЕДЫДУЩУЮ подзадачу!
#     valid_indices = jnp.where(changed_mask)[0]
#     changed_goals = curr_goals[valid_indices]
#     changed_prev_subgoals = prev_subgoals[valid_indices]

#     flat_indices = changed_goals * max_subgoals + changed_prev_subgoals
#     num_total = num_goals * max_subgoals

#     flat_mask = jnp.zeros((num_total,), dtype=jnp.bool_)
#     flat_mask = flat_mask.at[flat_indices].set(True)

#     return flat_mask.reshape((num_goals, max_subgoals))


# # -----------------------------
# # 3. Обновление матрицы сумм
# # -----------------------------
# def update_reward_sum_matrix_simple(reward_sum_matrix: jnp.ndarray,
#                                     rewards: jnp.ndarray,
#                                     curr_goals: jnp.ndarray,
#                                     curr_subgoals: jnp.ndarray):
#     """
#     Просто суммирует награды по текущим (goal, subgoal) без проверок смены.
#     """
#     num_envs = rewards.shape[0]

#     def body_fn(i, acc_matrix):
#         g = curr_goals[i]
#         sg = curr_subgoals[i]
#         r = rewards[i]
#         return acc_matrix.at[g, sg].add(r)

#     return jax.lax.fori_loop(0, num_envs, body_fn, reward_sum_matrix)

# def update_reward_sum_matrix_vectorized(reward_sum_matrix: jnp.ndarray,
#                                         rewards: jnp.ndarray,
#                                         curr_goals: jnp.ndarray,
#                                         curr_subgoals: jnp.ndarray):
#     """
#     Обновляет reward_sum_matrix, где каждое значение — это вектор вознаграждений
#     по (goal, subgoal). rewards имеет размер (num_envs, reward_dim).
#     """
#     num_envs = rewards.shape[0]

#     def body_fn(i, acc_matrix):
#         g = curr_goals[i]
#         sg = curr_subgoals[i]
#         r = rewards[i]  # shape: (reward_dim,)
#         return acc_matrix.at[g, sg].add(r)  # добавляет вектор

#     return jax.lax.fori_loop(0, num_envs, body_fn, reward_sum_matrix)


# def update_episode_count_matrix(episode_count_matrix: jnp.ndarray,
#                                 dones: jnp.ndarray,
#                                 curr_goals: jnp.ndarray,
#                                 curr_subgoals: jnp.ndarray,
#                                 update_mask: jnp.ndarray):
#     """
#     Обновляет счётчики эпизодов по (goal, subgoal), только если:
#     - эпизод завершён
#     - подцель не сменилась (т.е. update_mask == 0)
#     """
#     num_envs = dones.shape[0]

#     def body_fn(i, acc_matrix):
#         g = curr_goals[i]
#         sg = curr_subgoals[i]
#         d = dones[i]
#         should_update = (d == 1) & (~update_mask[g, sg])
#         add_one = jnp.where(should_update, 1, 0)
#         return acc_matrix.at[g, sg].add(add_one)

#     updated_matrix = jax.lax.fori_loop(0, num_envs, body_fn, episode_count_matrix)
#     return updated_matrix



# -----------------------------
# 1. Initialization
# -----------------------------
def initialize_reward_tracking(num_goals: int, max_subgoals: int, reward_dim: int = 1):
    shape = (num_goals, max_subgoals) if reward_dim == 1 else (num_goals, max_subgoals, reward_dim)
    reward_sum_matrix = jnp.zeros(shape, dtype=jnp.float32)
    episode_count_matrix = jnp.zeros((num_goals, max_subgoals), dtype=jnp.int32)
    return reward_sum_matrix, episode_count_matrix

# -----------------------------
# 2. Compute Subgoal Change Mask
# -----------------------------
def get_update_mask(prev_subgoals: jnp.ndarray,
                    curr_subgoals: jnp.ndarray,
                    curr_goals: jnp.ndarray,
                    num_goals: int,
                    max_subgoals: int):
    changed_mask = prev_subgoals != curr_subgoals
    valid_indices = jnp.where(changed_mask)[0]
    changed_goals = curr_goals[valid_indices]
    changed_prev_subgoals = prev_subgoals[valid_indices]

    flat_indices = changed_goals * max_subgoals + changed_prev_subgoals
    num_total = num_goals * max_subgoals

    flat_mask = jnp.zeros((num_total,), dtype=jnp.bool_)
    flat_mask = flat_mask.at[flat_indices].set(True)

    return flat_mask.reshape((num_goals, max_subgoals))

# -----------------------------
# 3. Reward Sum Update (Scalar)
# -----------------------------
def update_reward_sum_matrix_simple(reward_sum_matrix: jnp.ndarray,
                                         rewards: jnp.ndarray,
                                         curr_goals: jnp.ndarray,
                                         curr_subgoals: jnp.ndarray):
    """
    Updates scalar reward matrix using vectorized scatter_add.
    """
    max_subgoals = reward_sum_matrix.shape[1]
    indices = curr_goals * max_subgoals + curr_subgoals

    flat_matrix = reward_sum_matrix.reshape(-1)
    updated_flat = flat_matrix.at[indices].add(rewards)
    return updated_flat.reshape(reward_sum_matrix.shape)

# -----------------------------
# 4. Reward Sum Update (Vector)
# -----------------------------
def update_reward_sum_matrix_vectorized(reward_sum_matrix: jnp.ndarray,
                                             rewards: jnp.ndarray,
                                             curr_goals: jnp.ndarray,
                                             curr_subgoals: jnp.ndarray):
    """
    Vectorized reward update for reward_dim > 1
    """
    num_goals, max_subgoals, reward_dim = reward_sum_matrix.shape
    indices = curr_goals * max_subgoals + curr_subgoals

    flat_matrix = reward_sum_matrix.reshape(-1, reward_dim)
    updated_flat = flat_matrix.at[indices].add(rewards)
    return updated_flat.reshape(reward_sum_matrix.shape)

# -----------------------------
# 5. Episode Count Update
# -----------------------------
def update_episode_count_matrix(episode_count_matrix: jnp.ndarray,
                                     dones: jnp.ndarray,
                                     curr_goals: jnp.ndarray,
                                     curr_subgoals: jnp.ndarray,
                                     update_mask: jnp.ndarray):
    """
    Updates count matrix if (done == 1) and subgoal did NOT change.
    """
    num_goals, max_subgoals = update_mask.shape
    flat_mask = update_mask[curr_goals, curr_subgoals]  # shape: (num_envs,)
    should_update = (dones == 1) & (~flat_mask)         # shape: (num_envs,)

    indices = curr_goals * max_subgoals + curr_subgoals
    updates = should_update.astype(jnp.int32)

    flat_matrix = episode_count_matrix.reshape(-1)
    updated_flat = flat_matrix.at[indices].add(updates)
    return updated_flat.reshape((num_goals, max_subgoals))