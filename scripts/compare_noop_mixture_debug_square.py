#!/usr/bin/env python3
"""
Compare two simple stochastic policies on debug_square_8x8 using the batched optimistic env.

Policy A: with prob p_random choose a uniform random valid action in {0..16},
          with prob (1-p_random) choose INVALID_ACTION_ID (-1) which effectively behaves like NOOP.
Policy B: always choose a uniform random valid action in {0..16}.

This is meant to sanity-check the hypothesis:
removing invalid actions (standing still) can improve reward even if actions are otherwise uniform random.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np


INVALID_ACTION_ID = -1
NUM_ACTIONS = 17  # 0..16 inclusive


@dataclass(frozen=True)
class RolloutStats:
    steps: int
    envs: int
    mean_reward_per_step: float
    done_frac_per_step: float


def _ensure_import_paths() -> None:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    caged = os.path.join(repo, "caged_craftext")
    if caged not in sys.path:
        sys.path.insert(0, caged)
    # Prefer repo Craftax over any pip craftax.
    craftax_repo = os.path.join(caged, "Craftax")
    if craftax_repo not in sys.path:
        sys.path.insert(0, craftax_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _make_env(num_envs: int, seed: int, optimistic_reset_ratio: int) -> object:
    import jax
    import jax.numpy as jnp

    from craftax.craftax_classic.debug_square_world_gen import generate_debug_square_world
    from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnvNoAutoReset
    from craftax.craftax_classic.envs.craftax_state import StaticEnvParams

    from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
    from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
    from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
    from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

    # Optimistic wrapper from caged_craftext baselines (jit + vmap).
    from wrappers_cmdp import OptimisticResetVecEnvWrapper

    static_params = StaticEnvParams(map_size=(8, 8))
    base_env = CraftaxClassicPixelsEnvNoAutoReset(static_env_params=static_params)

    # Disable mob spawns on the tiny arena.
    params = base_env.default_params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )

    # Patch reset to use fixed debug world (procedural gen breaks on 8x8).
    def reset_env(rng, _params):
        state = generate_debug_square_world(rng, _params, base_env.static_env_params)
        return base_env.get_obs(state), state

    base_env.reset_env = reset_env

    wrapper = CMDPInstructionWrapper(
        env=base_env,
        config_name="debug_square_8x8",
        scenario_handler_class=ScenariosNoLambdaCMDP,
        encode_model_class=DistilBertEncode,
        encode_form=EncodeForm.EMBEDDING,
    )

    if optimistic_reset_ratio <= 0:
        optimistic_reset_ratio = 1
    if num_envs % optimistic_reset_ratio != 0:
        optimistic_reset_ratio = 1

    vec = OptimisticResetVecEnvWrapper(wrapper, num_envs=int(num_envs), reset_ratio=int(optimistic_reset_ratio))
    key = jax.random.PRNGKey(int(seed))
    state = None

    class _Env:
        env_num = int(num_envs)

        def reset(self):
            nonlocal key, state
            key, sub = jax.random.split(key)
            _obs, state = vec.reset(sub, params)
            infos = [{} for _ in range(self.env_num)]
            return [None] * self.env_num, infos

        def step(self, actions: list[int]):
            nonlocal key, state
            key, sub = jax.random.split(key)
            action_arr = jnp.asarray(actions, dtype=jnp.int32)
            _obs, state, reward, done, _info, _state_pre_reset = vec.step(sub, state, action_arr, params)
            rewards = [float(x) for x in np.asarray(jax.device_get(reward)).tolist()]
            dones = [bool(x) for x in np.asarray(jax.device_get(done)).tolist()]
            infos = [{} for _ in range(self.env_num)]
            return [None] * self.env_num, rewards, dones, infos

    return _Env()


def _rollout(env, *, steps: int, rng: np.random.RandomState, p_random_valid: float) -> RolloutStats:
    # reset once; optimistic wrapper auto-swaps done envs internally
    _obs, _infos = env.reset()

    total_reward = 0.0
    total_done = 0
    for _t in range(steps):
        # Sample actions for each env slot
        u = rng.rand(env.env_num)
        valid_actions = rng.randint(0, NUM_ACTIONS, size=env.env_num, dtype=np.int32)
        actions = np.where(u < p_random_valid, valid_actions, INVALID_ACTION_ID).astype(np.int32)

        _obs, rewards, dones, _infos = env.step(actions.tolist())
        total_reward += float(np.sum(rewards))
        total_done += int(np.sum(dones))

    mean_reward_per_step = total_reward / (steps * env.env_num)
    done_frac_per_step = total_done / (steps * env.env_num)
    return RolloutStats(
        steps=steps,
        envs=env.env_num,
        mean_reward_per_step=mean_reward_per_step,
        done_frac_per_step=done_frac_per_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimistic-reset-ratio", type=int, default=8)
    parser.add_argument("--p-random-valid", type=float, default=0.85)
    args = parser.parse_args()

    _ensure_import_paths()

    env = _make_env(args.num_envs, args.seed, args.optimistic_reset_ratio)
    rng = np.random.RandomState(args.seed)

    stats_mixture = _rollout(env, steps=args.steps, rng=rng, p_random_valid=float(args.p_random_valid))
    # Recreate env for a clean second run (same seed)
    env2 = _make_env(args.num_envs, args.seed, args.optimistic_reset_ratio)
    rng2 = np.random.RandomState(args.seed)
    stats_random = _rollout(env2, steps=args.steps, rng=rng2, p_random_valid=1.0)

    print("=== debug_square_8x8 random-vs-invalid mixture ===")
    print(f"envs={args.num_envs} steps={args.steps} seed={args.seed} reset_ratio={args.optimistic_reset_ratio}")
    print("")
    print(f"Policy A: p_random_valid={args.p_random_valid:.3f}, else INVALID(-1)≈NOOP")
    print(f"  mean_reward_per_step={stats_mixture.mean_reward_per_step:.6f}")
    print(f"  done_frac_per_step={stats_mixture.done_frac_per_step:.6f}")
    print("")
    print("Policy B: p_random_valid=1.000 (uniform random valid action 0..16)")
    print(f"  mean_reward_per_step={stats_random.mean_reward_per_step:.6f}")
    print(f"  done_frac_per_step={stats_random.done_frac_per_step:.6f}")
    print("")
    delta = stats_random.mean_reward_per_step - stats_mixture.mean_reward_per_step
    print(f"Delta (B - A) mean_reward_per_step = {delta:.6f}")


if __name__ == "__main__":
    main()

