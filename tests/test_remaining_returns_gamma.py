#!/usr/bin/env python3
"""Unit tests for discounted remaining returns G_t^gamma."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path("/home/gorbov_gv/safe_rl_nlp")
sys.path.insert(0, str(ROOT))

from agent_system.environments.env_package.caged_craftext.return_tokens import (  # noqa: E402
    compute_remaining_returns,
)
from verl.utils.actor_value_token import compute_remaining_return_scalars  # noqa: E402


def almost_eq(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) < tol, (a, b)


def test_gamma1_example():
    rewards = [-1.0, 1.0, 1.0]
    G = compute_remaining_returns(rewards, gamma=1.0)
    assert G == [1.0, 2.0, 1.0], G
    print("PASS gamma=1 example", G)


def test_gamma099_example():
    rewards = [-1.0, 1.0, 1.0]
    g = 0.99
    G = compute_remaining_returns(rewards, gamma=g)
    g0 = -1 + g * 1 + g * g * 1
    g1 = 1 + g * 1
    g2 = 1.0
    almost_eq(G[0], g0)
    almost_eq(G[1], g1)
    almost_eq(G[2], g2)
    # bad first action not exactly cancelled: -1 + g*1 != 0
    assert abs((-1.0 + g * 1.0) - 0.0) > 1e-12
    print("PASS gamma=0.99 example", G)


def test_terminal_boundary_and_traj_uid():
    # Two trajectories in one batch, shuffled order; must not cross boundary.
    # traj A steps 0,1,2 rewards [-1,1,1]
    # traj B steps 0,1 rewards [2,3]
    traj_uids = ["B", "A", "A", "B", "A"]
    step_rewards = [2.0, -1.0, 1.0, 3.0, 1.0]
    episode_step_idx = [0, 0, 1, 1, 2]
    g = 0.99
    out = compute_remaining_return_scalars(
        traj_uids, step_rewards, episode_step_idx=episode_step_idx, gamma=g
    )
    # Expected for A: same as single traj
    Ga = compute_remaining_returns([-1.0, 1.0, 1.0], gamma=g)
    Gb = compute_remaining_returns([2.0, 3.0], gamma=g)
    # Map back by row order
    expect = {
        0: Gb[0],  # B step0
        1: Ga[0],  # A step0
        2: Ga[1],  # A step1
        3: Gb[1],  # B step1
        4: Ga[2],  # A step2
    }
    for i, e in expect.items():
        almost_eq(out[i], e)
    print("PASS multi-traj / step_idx", out.tolist())


def test_duplicate_step_idx_safe():
    # Duplicate rows (adjust_batch) share same step_idx; same G.
    traj_uids = ["T", "T", "T"]
    step_rewards = [1.0, 1.0, 2.0]
    episode_step_idx = [0, 0, 1]  # first two duplicates of step 0
    out = compute_remaining_return_scalars(
        traj_uids, step_rewards, episode_step_idx=episode_step_idx, gamma=1.0
    )
    # Unique steps: step0 reward from first occurrence (=1), step1 (=2) => G=[3,3,2]
    almost_eq(out[0], 3.0)
    almost_eq(out[1], 3.0)
    almost_eq(out[2], 2.0)
    print("PASS duplicate step_idx", out.tolist())


def test_default_gamma_backward_compat():
    rewards = [1.0, 2.0, 3.0]
    assert compute_remaining_returns(rewards) == compute_remaining_returns(rewards, gamma=1.0)
    print("PASS default gamma=1 backward compatible")


if __name__ == "__main__":
    test_gamma1_example()
    test_gamma099_example()
    test_terminal_boundary_and_traj_uid()
    test_duplicate_step_idx_safe()
    test_default_gamma_backward_compat()
    print("ALL TESTS PASSED")
