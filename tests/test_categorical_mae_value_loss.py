#!/usr/bin/env python3
"""Unit tests for categorical MAE over return-bin distributions (one-hot / two-hot)."""
from __future__ import annotations

import torch

from verl.utils.actor_value_token import (
    categorical_mae_loss,
    return_token_value_loss,
    two_hot_return_distribution,
    ReturnBinSpec,
)


def test_one_hot_mae_equals_one_minus_py():
    # y=[0,1,0,0], p=[0.1,0.4,0.2,0.3] => 0.5*L1 = 0.6 = 1-0.4
    p = torch.tensor([[0.1, 0.4, 0.2, 0.3]], dtype=torch.float32)
    y = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    mae = categorical_mae_loss(p, y)
    assert torch.isclose(mae, torch.tensor(0.6), atol=1e-6), float(mae)
    assert torch.isclose(mae, torch.tensor(1.0 - 0.4), atol=1e-6)


def test_two_hot_mae_exact():
    # y=[0,0.3,0.7,0], p=[0.1,0.2,0.5,0.2]
    # |p-y| = [0.1, 0.1, 0.2, 0.2] sum=0.6 => 0.5*L1=0.3
    p = torch.tensor([[0.1, 0.2, 0.5, 0.2]], dtype=torch.float32)
    y = torch.tensor([[0.0, 0.3, 0.7, 0.0]], dtype=torch.float32)
    mae = categorical_mae_loss(p, y)
    assert torch.isclose(mae, torch.tensor(0.3), atol=1e-6), float(mae)


def test_two_hot_helper_matches_manual():
    # bins [0,3], value=1.3 => low=1 weight=0.7, high=2 weight=0.3
    v = torch.tensor([1.3], dtype=torch.float32)
    y = two_hot_return_distribution(v, max_bin=3)
    assert torch.allclose(y, torch.tensor([[0.0, 0.7, 0.3, 0.0]]), atol=1e-5), y


def test_mae_backward_finite_and_decreases():
    # Optimize logits toward two-hot target with MAE
    torch.manual_seed(0)
    target = torch.tensor([[0.0, 0.3, 0.7, 0.0]], dtype=torch.float32)
    logits = torch.zeros(1, 4, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=1.0)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        probs = torch.softmax(logits, dim=-1)
        loss = categorical_mae_loss(probs, target)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert losses[-1] < 0.15, losses[-1]


def test_return_token_value_loss_mae_on_bin_logits_not_vocab():
    # 5 return bins only (not full vocab)
    spec = ReturnBinSpec(vmin=0.0, vmax=0.8, step=0.2)  # bins 0..4
    # target return 0.3 => two-hot between bin 1 and 2 with alpha on left=0.5? 
    # (0.3-0)/0.2 = 1.5 => low=1 w_low=0.5, high=2 w_high=0.5
    logits = torch.zeros(2, 5, dtype=torch.float32)
    logits[0, 1] = 5.0  # peak near target
    logits[1, 4] = 5.0  # peak far from target
    targets = torch.tensor([0.3, 0.3], dtype=torch.float32)
    loss_mae, hard = return_token_value_loss(
        logits,
        targets,
        target_encoding="two_hot",
        value_loss_type="mae",
        spec=spec,
    )
    loss_ce, _ = return_token_value_loss(
        logits,
        targets,
        target_encoding="two_hot",
        value_loss_type="ce",
        spec=spec,
    )
    assert logits.shape[-1] == 5
    assert hard.numel() == 2
    # first example closer => MAE of batch should be finite and CE != MAE numerically
    assert torch.isfinite(loss_mae)
    assert torch.isfinite(loss_ce)
    assert not torch.isclose(loss_mae, loss_ce, atol=1e-4)


def test_ce_default_unchanged_one_hot():
    logits = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0.2], dtype=torch.float32)  # with vmin=0 step=0.2 -> bin 1
    spec = ReturnBinSpec(vmin=0.0, vmax=0.4, step=0.2)
    loss, hard = return_token_value_loss(
        logits, targets, target_encoding="one_hot", value_loss_type="ce", spec=spec
    )
    logp = torch.log_softmax(logits, dim=-1)
    expected = -logp[0, 1]
    assert hard.item() == 1
    assert torch.isclose(loss, expected, atol=1e-5)


if __name__ == "__main__":
    test_one_hot_mae_equals_one_minus_py()
    test_two_hot_mae_exact()
    test_two_hot_helper_matches_manual()
    test_mae_backward_finite_and_decreases()
    test_return_token_value_loss_mae_on_bin_logits_not_vocab()
    test_ce_default_unchanged_one_hot()
    print("ALL MAE TESTS PASSED")
