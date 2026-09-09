from __future__ import annotations

from dataclasses import replace

import torch

import losses.objective as objective_module
from losses.objective import (
    IAGSRMEObjective,
    ObjectiveConfig,
    awta_temperature,
    compute_candidate_credit_weights,
)


def test_awta_weights_match_source_formula() -> None:
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    temperature = 0.7

    weights = compute_candidate_credit_weights(losses, "awta", temperature)

    assert torch.allclose(weights, torch.softmax(-losses / temperature, dim=-1))


def test_awta_assignment_is_stop_gradient() -> None:
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    weights = compute_candidate_credit_weights(losses, "awta", temperature=1.0)
    expected_gradient = weights.clone()

    (weights * losses).sum().backward()

    assert not weights.requires_grad
    # If assignment were not detached, the softmax derivative would add another term.
    assert torch.allclose(losses.grad, expected_gradient)


def test_awta_high_temperature_is_nearly_uniform() -> None:
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weights = compute_candidate_credit_weights(losses, "awta", temperature=1e6)

    assert torch.allclose(weights, torch.full_like(weights, 0.25), atol=1e-6)


def test_awta_low_temperature_selects_loss_argmin_more_strongly() -> None:
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    high = compute_candidate_credit_weights(losses, "awta", temperature=10.0)
    low = compute_candidate_credit_weights(losses, "awta", temperature=0.01)

    assert low.argmax(dim=-1).item() == losses.argmin(dim=-1).item()
    assert low.max() > high.max()


def test_hard_wta_is_exact_one_hot_loss_argmin() -> None:
    losses = torch.tensor([[3.0, 1.0, 4.0, 2.0], [0.0, 5.0, 2.0, 3.0]])
    weights = compute_candidate_credit_weights(losses, "hard_wta", temperature=1.0)

    assert torch.equal(
        weights,
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )


def test_hard_and_awta_distribute_candidate_gradients_differently() -> None:
    hard_losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    hard_weights = compute_candidate_credit_weights(
        hard_losses, "hard_wta", temperature=1.0
    )
    (hard_weights * hard_losses).sum().backward()

    awta_losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    awta_weights = compute_candidate_credit_weights(
        awta_losses, "awta", temperature=10.0
    )
    (awta_weights * awta_losses).sum().backward()

    assert torch.equal(hard_losses.grad, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.all(awta_losses.grad != 0)
    assert torch.allclose(awta_losses.grad, awta_weights)


def test_awta_temperature_schedule_is_exponential_and_floored() -> None:
    assert awta_temperature(0, 1.0, 0.85, 0.05) == 1.0
    assert awta_temperature(3, 1.0, 0.85, 0.05) == 1.0 * 0.85**3
    temperatures = [awta_temperature(epoch, 1.0, 0.85, 0.05) for epoch in range(100)]
    assert min(temperatures) == 0.05
    assert all(value >= 0.05 for value in temperatures)


def test_invalid_teacher_rows_are_skipped_without_nonfinite_diagnostics() -> None:
    batch, candidates, dimension = 3, 4, 5
    query = torch.randn(batch, dimension, requires_grad=True)
    candidate_queries = torch.randn(
        batch, candidates, dimension, requires_grad=True
    )
    output = {
        "query": query,
        "steps": [
            {
                "live_indices": torch.arange(batch),
                "current_query": torch.randn(batch, dimension),
                "candidate_queries": candidate_queries,
                "scores": torch.randn(batch, candidates, requires_grad=True),
                "delta_q": torch.randn(batch, candidates, dimension),
                "stopped_now": torch.zeros(batch, dtype=torch.bool),
                "selected_idx": torch.zeros(batch, dtype=torch.long),
            }
        ],
    }
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="awta",
            lambda_candidate_credit=1.0,
        )
    )

    # Every target is a positive for every row, so no valid negative exists.
    components = objective(output, torch.randn(batch, dimension), ["same"] * batch)

    for name in (
        "candidate_credit_loss",
        "candidate_credit_weighted",
        "awta_weight_entropy",
        "awta_max_weight",
        "awta_effective_k",
        "total",
    ):
        assert torch.isfinite(components[name])
        assert components[name] == 0
    assert components["teacher_invalid_rows"] == batch


def test_candidate_credit_is_normalized_per_valid_row_across_timesteps(
    monkeypatch,
) -> None:
    candidate_loss_rows = iter(
        (
            torch.tensor([[1.0, 3.0], [2.0, 5.0]]),
            torch.tensor([[4.0, 6.0]]),
        )
    )

    def valid_teacher(current, candidates, targets, positive, negative, temperature):
        del current, targets, positive, negative, temperature
        return (
            candidates.new_zeros(candidates.shape[:-1]),
            torch.ones(candidates.shape[0], dtype=torch.bool),
        )

    def prescribed_candidate_losses(
        candidates, targets, positive, negative, temperature
    ):
        del targets, positive, negative, temperature
        values = next(candidate_loss_rows).to(candidates)
        return candidates[..., 0] * 0.0 + values

    monkeypatch.setattr(objective_module, "marginal_teacher_utilities", valid_teacher)
    monkeypatch.setattr(
        objective_module, "teacher_retrieval_loss", prescribed_candidate_losses
    )
    query = torch.randn(2, 3)
    steps = []
    for live_indices in (torch.tensor([0, 1]), torch.tensor([0])):
        live = live_indices.numel()
        steps.append(
            {
                "live_indices": live_indices,
                "current_query": torch.randn(live, 3),
                "candidate_queries": torch.randn(live, 2, 3, requires_grad=True),
                "scores": torch.randn(live, 2),
                "delta_q": torch.randn(live, 2, 3),
                "stopped_now": torch.zeros(live, dtype=torch.bool),
                "selected_idx": torch.zeros(live, dtype=torch.long),
            }
        )
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="hard_wta",
            lambda_candidate_credit=1.0,
        )
    )

    components = objective(
        {"query": query, "steps": steps}, torch.randn(2, 3), ["a", "b"]
    )

    # Hard-WTA row losses are 1, 2, and 4: average rows, not timesteps or sums.
    assert torch.allclose(components["candidate_credit_loss"], torch.tensor(7.0 / 3.0))


def test_none_mode_preserves_original_total_and_skips_candidate_credit(
    model, features, monkeypatch
) -> None:
    model.config = replace(model.config, max_steps=1)
    output = model.forward_from_features(*features)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            candidate_credit_mode="none",
            lambda_candidate_credit=0.0,
        )
    )

    def unexpected_candidate_credit_call(*args, **kwargs):
        raise AssertionError("none mode must not compute differentiable candidate credit")

    monkeypatch.setattr(
        objective_module, "teacher_retrieval_loss", unexpected_candidate_credit_call
    )
    components = objective(output, torch.randn(3, 12), ["a", "b", "c"])
    expected = (
        objective.config.terminal_weight * components["terminal"]
        + objective.config.lambda_pair * components["pair"]
        + objective.config.lambda_gain * components["gain"]
        + objective.config.lambda_c * components["concept_loss"]
        + objective.config.lambda_bind * components["bind_loss"]
        + objective.config.lambda_rel * components["rel_ortho_loss"]
        + components["dpp_weighted"]
    )

    assert torch.equal(components["total"], expected)
    assert components["candidate_credit_loss"] == 0
    assert components["candidate_credit_weighted"] == 0


def test_candidate_credit_detaches_targets_but_not_candidate_queries(
    model, features
) -> None:
    model.config = replace(model.config, max_steps=1)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    output = model.forward_from_features(*features)
    candidate_queries = output["steps"][0]["candidate_queries"]
    candidate_queries.retain_grad()
    targets = torch.randn(3, 12, requires_grad=True)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            candidate_credit_mode="awta",
            lambda_candidate_credit=1.0,
            awta_temperature_init=10.0,
        )
    )

    components = objective(output, targets, ["a", "b", "c"])
    components["total"].backward()

    assert candidate_queries.grad is not None
    assert torch.all(candidate_queries.grad.abs().sum(dim=-1) > 0)
    assert targets.grad is None or targets.grad.count_nonzero() == 0
    assert components["awta_effective_k"] > 1
