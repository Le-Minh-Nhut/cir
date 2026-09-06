from __future__ import annotations

import inspect
from dataclasses import replace

import torch

import losses.objective as objective_module
from losses.objective import (
    IAGSRMEObjective,
    ObjectiveConfig,
    functional_dpp_loss,
    update_executed_history,
)


def _grad_sum(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.abs().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return sum(values, torch.tensor(0.0))


def _dpp(effects: torch.Tensor, history: torch.Tensor | None, utility: torch.Tensor):
    return functional_dpp_loss(
        effects,
        history,
        utility,
        kappa=2.0,
        sigma=0.5,
        tau=0.2,
        useful_threshold=0.6,
        jitter=1e-4,
    )


def test_dpp_uses_live_delta_q_gradient_but_detaches_quality_and_history() -> None:
    torch.manual_seed(4)
    delta_q = torch.randn(4, 6, requires_grad=True)
    history = torch.randn(2, 6, requires_grad=True)
    teacher_utility = torch.full((4,), 2.0, requires_grad=True)
    result = _dpp(delta_q, history, teacher_utility)
    result["loss"].backward()

    assert bool(result["valid"])
    assert delta_q.grad is not None and delta_q.grad.abs().sum() > 0
    assert history.grad is None
    assert teacher_utility.grad is None
    assert not result["quality"].requires_grad


def test_dpp_is_permutation_invariant_and_penalizes_redundant_useful_effects() -> None:
    utility = torch.full((4,), 2.0)
    identical = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    distinct = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    identical_loss = _dpp(identical, None, utility)["loss"]
    distinct_loss = _dpp(distinct, None, utility)["loss"]
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_loss = _dpp(distinct[permutation], None, utility[permutation])["loss"]

    assert identical_loss > distinct_loss
    assert torch.allclose(distinct_loss, permuted_loss, atol=1e-6)


def test_dpp_skips_when_fewer_than_two_candidates_are_useful() -> None:
    effects = torch.randn(4, 6, requires_grad=True)
    # sigmoid(2/.2) is useful; all negative entries are below the threshold.
    utility = torch.tensor([2.0, -2.0, -2.0, -2.0])
    result = _dpp(effects, None, utility)

    assert not bool(result["valid"])
    assert result["useful_count"] == 1
    assert result["loss"] == 0


def test_history_contains_only_detached_executed_effects_and_stop_adds_nothing() -> None:
    delta_q = torch.randn(2, 4, 5, requires_grad=True)
    histories: list[list[torch.Tensor]] = [[], []]
    live = torch.tensor([0, 1])
    selected = torch.tensor([2, 4])  # K is the STOP sentinel.
    update_executed_history(histories, live, selected, delta_q, num_candidates=4)

    expected = delta_q[0, 2] / (delta_q[0, 2].norm() + 1e-8)
    assert len(histories[0]) == 1
    assert len(histories[1]) == 0
    assert torch.allclose(histories[0][0], expected)
    assert not histories[0][0].requires_grad
    assert not torch.equal(histories[0][0], delta_q[0, 0].detach())


def test_kappa_and_external_lambda_are_distinct_configuration_fields() -> None:
    config = ObjectiveConfig(kappa_dpp=3.0, lambda_dpp=0.07)
    assert config.kappa_dpp == 3.0
    assert config.lambda_dpp == 0.07
    assert "lambda" not in inspect.signature(functional_dpp_loss).parameters

    effects = torch.eye(4)
    utility = torch.ones(4)
    low = functional_dpp_loss(
        effects,
        None,
        utility,
        kappa=1.0,
        sigma=0.5,
        tau=0.2,
        useful_threshold=0.6,
        jitter=1e-4,
    )["loss"]
    high = functional_dpp_loss(
        effects,
        None,
        utility,
        kappa=3.0,
        sigma=0.5,
        tau=0.2,
        useful_threshold=0.6,
        jitter=1e-4,
    )["loss"]
    assert not torch.allclose(low, high)


def test_dpp_routes_through_functional_consequence_not_scorenet_or_target(
    model, features, monkeypatch
) -> None:
    model.config = replace(model.config, max_steps=1)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    output = model.forward_from_features(*features)

    def useful_teacher(current, candidates, targets, positive, negative, temperature):
        del current, targets, positive, negative, temperature
        utilities = candidates.new_full(candidates.shape[:-1], 1.0)
        valid = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
        return utilities, valid

    monkeypatch.setattr(objective_module, "marginal_teacher_utilities", useful_teacher)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            dpp_enabled=True,
            lambda_dpp=1.0,
            useful_threshold=0.55,
        ),
        state_dim=16,
    )
    targets = torch.randn(3, 12, requires_grad=True)
    components = objective(output, targets, ["a", "b", "c"])
    assert components["dpp_valid_timestep_count"] == 3
    components["total"].backward()

    for module in (model.proposal, model.grounder, model.action_fusion, model.executor):
        assert _grad_sum(module) > 0
    assert _grad_sum(model.score_net) == 0
    assert targets.grad is None or targets.grad.count_nonzero() == 0
