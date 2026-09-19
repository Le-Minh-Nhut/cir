from __future__ import annotations

import pytest
import torch

from losses.dac import (
    dac_groups,
    dac_responsibility_weights,
    dac_stage,
    validate_dac_candidate_count,
)


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (0, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        (1, ((0, 1, 2, 3), (4, 5, 6, 7))),
        (2, ((0, 1), (2, 3), (4, 5), (6, 7))),
        (3, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))),
    ),
)
def test_k8_hierarchy_matches_dac_binary_partition(
    stage: int, expected: tuple[tuple[int, ...], ...]
) -> None:
    assert dac_groups(8, stage) == expected


@pytest.mark.parametrize(
    ("global_step", "expected_stage"),
    (
        (0, 0),
        (1999, 0),
        (2000, 1),
        (3999, 1),
        (4000, 2),
        (5999, 2),
        (6000, 3),
        (12345, 3),
    ),
)
def test_dac_schedule_uses_exact_optimizer_step_boundaries(
    global_step: int, expected_stage: int
) -> None:
    assert dac_stage(global_step, split_interval_steps=2000, num_candidates=8) == expected_stage


def test_dac_rejects_non_power_of_two_candidate_count() -> None:
    with pytest.raises(ValueError, match="power of two"):
        validate_dac_candidate_count(6)


def test_dac_rejects_invalid_schedule_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        dac_stage(-1, split_interval_steps=2000, num_candidates=8)
    with pytest.raises(ValueError, match="positive"):
        dac_stage(0, split_interval_steps=0, num_candidates=8)


@pytest.mark.parametrize(
    ("stage", "losses", "expected_gradient", "expected_group"),
    (
        (
            0,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.125] * 8,
            0,
        ),
        (
            1,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            0,
        ),
        (
            2,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            1,
        ),
        (
            3,
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            3,
        ),
    ),
)
def test_dac_routes_exact_gradient_to_every_winning_group_member(
    stage: int,
    losses: list[float],
    expected_gradient: list[float],
    expected_group: int,
) -> None:
    candidate_losses = torch.tensor([losses], requires_grad=True)

    weights, winning_groups = dac_responsibility_weights(candidate_losses, stage)
    row_loss = (weights * candidate_losses).sum()
    row_loss.backward()

    assert winning_groups.tolist() == [expected_group]
    assert torch.equal(candidate_losses.grad, torch.tensor([expected_gradient]))


def test_dac_winning_group_loss_is_mean_not_argmin_loss() -> None:
    losses = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )
    weights, _ = dac_responsibility_weights(losses, stage=1)

    value = (weights * losses).sum()

    assert value == torch.tensor(2.5)
    assert value != losses[0, 3]


def test_dac_selects_right_hand_group_for_candidate_six_winner() -> None:
    losses = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 3.0]])

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)

    assert winning_groups.tolist() == [1]
    assert torch.equal(weights, torch.tensor([[0.0] * 4 + [0.25] * 4]))


def test_dac_routes_different_batch_rows_to_different_groups() -> None:
    losses = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 3.0],
        ]
    )

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)

    assert winning_groups.tolist() == [0, 1]
    assert torch.equal(
        weights,
        torch.tensor([[0.25] * 4 + [0.0] * 4, [0.0] * 4 + [0.25] * 4]),
    )


def test_dac_assignment_is_stop_gradient() -> None:
    losses = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )

    weights, winning_groups = dac_responsibility_weights(losses, stage=1)
    (weights * losses).sum().backward()

    assert not weights.requires_grad
    assert not winning_groups.requires_grad
    assert torch.equal(
        losses.grad, torch.tensor([[0.25] * 4 + [0.0] * 4])
    )


def test_dac_leaf_matches_hard_wta_numerically_and_in_gradient() -> None:
    dac_losses = torch.tensor(
        [[3.0, 1.0, 4.0, 2.0, 8.0, 7.0, 6.0, 5.0]], requires_grad=True
    )
    hard_losses = dac_losses.detach().clone().requires_grad_(True)

    dac_weights, _ = dac_responsibility_weights(dac_losses, stage=3)
    hard_weights = torch.nn.functional.one_hot(
        hard_losses.detach().argmin(dim=-1), num_classes=8
    ).float()
    dac_value = (dac_weights * dac_losses).sum()
    hard_value = (hard_weights * hard_losses).sum()
    dac_value.backward()
    hard_value.backward()

    assert torch.equal(dac_value, hard_value)
    assert torch.equal(dac_weights, hard_weights)
    assert torch.equal(dac_losses.grad, hard_losses.grad)
