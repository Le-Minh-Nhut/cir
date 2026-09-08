from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from data.images import ImageBatch
from diagnose_slot_specialization import _gradient_attribution
from diagnostics.specialization import (
    concept_mil_responsibility,
    exact_shapley_values,
    functional_specialization_summary,
    gradient_interference_summary,
    semantic_specialization_summary,
)
from losses.objective import IAGSRMEObjective, ObjectiveConfig


def test_exact_shapley_efficiency() -> None:
    utility = torch.tensor(
        [[0.5, 0.1, -0.2, 0.3], [-0.1, 0.4, 0.2, 0.0]], dtype=torch.float64
    )
    shapley, coalition = exact_shapley_values(utility)
    assert torch.allclose(shapley.sum(dim=-1), coalition[:, -1], atol=1e-12)


def test_redundant_candidates_split_shapley_credit() -> None:
    utility = torch.tensor([[0.5, 0.5, 0.4, -0.1]], dtype=torch.float64)
    shapley, _ = exact_shapley_values(utility)
    assert shapley[0, 0] == pytest.approx(shapley[0, 1])
    assert shapley[0, 0] > 0
    assert shapley[0, 2] > 0
    assert shapley[0, 3] == pytest.approx(0.0)


def test_single_useful_candidate_has_effective_k_one() -> None:
    utility = torch.tensor(
        [[-0.3, -0.2, -0.1, 0.5], [-0.2, -0.4, -0.1, 0.2]]
    )
    summary = functional_specialization_summary(utility)
    coalition = summary["coalition_oracle"]
    assert coalition["full_set_value"] == pytest.approx(coalition["c3_only_value"])
    assert summary["shapley"]["contribution_per_slot"][:3] == pytest.approx([0, 0, 0])
    assert summary["shapley"]["contribution_per_slot"][3] > 0
    assert summary["effective_functional_k"] == pytest.approx(1.0)


def test_complementary_specialists_raise_full_oracle_and_effective_k() -> None:
    utility = torch.tensor(
        [
            [0.7, -0.1, -0.2, 0.1],
            [-0.1, 0.8, -0.2, 0.1],
            [-0.1, -0.2, 0.9, 0.1],
            [-0.1, -0.2, 0.0, 1.0],
        ]
    )
    summary = functional_specialization_summary(utility)
    full = summary["coalition_oracle"]["full_set_value"]
    assert full > max(summary["coalition_oracle"]["single_slot_values"])
    assert all(value > 0 for value in summary["shapley"]["contribution_per_slot"])
    assert summary["effective_functional_k"] > 1.0


def test_concept_conditioned_shapley_recovers_synthetic_niches() -> None:
    utility = torch.tensor(
        [
            [0.8, 0.0, 0.0, 0.1],
            [0.7, 0.0, 0.0, 0.1],
            [0.0, 0.9, 0.0, 0.1],
            [0.0, 0.8, 0.0, 0.1],
        ]
    )
    shapley, _ = exact_shapley_values(utility)
    concepts = [("red",), ("red",), ("sleeve",), ("sleeve",)]
    summary = semantic_specialization_summary(
        utility,
        shapley,
        concepts,
        min_support=2,
        bootstrap_samples=50,
        seed=13,
    )
    by_concept = {row["concept"]: row for row in summary["reported_concepts"]}
    assert by_concept["red"]["conditional_shapley_mean"][0] > 0
    assert by_concept["red"]["conditional_shapley_mean"][1] == pytest.approx(0)
    assert by_concept["sleeve"]["conditional_shapley_mean"][1] > 0
    assert by_concept["sleeve"]["conditional_shapley_mean"][0] == pytest.approx(0)


def _build_gradient_case(model, features):
    model.config = replace(model.config, max_steps=1, stop_enabled=False)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    output = model.forward_from_features(*features)
    proposals = output["steps"][0]["proposals"]
    targets = output["steps"][0]["current_query"].detach().clone()
    objective = IAGSRMEObjective(
        ObjectiveConfig(terminal_weight=1.0, lambda_pair=1.0, lambda_gain=1.0)
    )
    components = objective(output, targets, ["a", "b", "c"])
    return proposals, components


def test_gradient_measurement_is_read_only_and_does_not_populate_grad(model, features) -> None:
    proposals, components = _build_gradient_case(model, features)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    terminal_gradient = torch.autograd.grad(
        components["terminal"], proposals, allow_unused=True
    )[0]
    assert terminal_gradient is not None
    assert terminal_gradient.abs().sum() > 0
    assert all(
        torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_pair_gain_are_upstream_detached_from_candidate_proposals(model, features) -> None:
    proposals, components = _build_gradient_case(model, features)
    pair_gradient = torch.autograd.grad(
        components["pair"], proposals, retain_graph=True, allow_unused=True
    )[0]
    gain_gradient = torch.autograd.grad(
        components["gain"], proposals, allow_unused=True
    )[0]
    assert pair_gradient is None or pair_gradient.abs().max() == 0
    assert gain_gradient is None or gain_gradient.abs().max() == 0


def test_concept_mil_responsibility_is_exact_candidate_softmax() -> None:
    logits = torch.tensor(
        [[[1.0, -1.0], [2.0, 0.0], [0.5, 3.0], [-0.5, 2.0]]]
    )
    expected = torch.softmax(logits / 0.7, dim=1)
    actual = concept_mil_responsibility(logits, tau_mil=0.7)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual.sum(dim=1), torch.ones_like(actual.sum(dim=1)))


def test_zero_gradient_interference_has_no_nan_or_inf() -> None:
    gradients = {
        "terminal": torch.zeros(3, 4, 5),
        "concept": torch.randn(3, 4, 5),
        "bind": torch.zeros(3, 4, 5),
        "dpp": torch.randn(3, 4, 5),
    }
    summary = gradient_interference_summary(gradients)
    for comparison in summary.values():
        for slot in comparison:
            assert slot["valid_comparison_count"] == 0
            assert slot["mean_cosine"] is None
            assert slot["median_cosine"] is None
            assert slot["negative_cosine_fraction"] is None
            assert slot["mean_aux_to_terminal_norm_ratio"] is None


def test_functional_summary_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(71)
    utility = torch.randn(32, 4, generator=generator)
    first = functional_specialization_summary(utility)
    second = functional_specialization_summary(utility)
    first.pop("per_sample_shapley")
    second.pop("per_sample_shapley")
    assert first == second


def test_gradient_attribution_smoke_uses_autograd_without_parameter_mutation(model) -> None:
    model.config = replace(model.config, max_steps=1, stop_enabled=False)
    torch.nn.init.normal_(model.executor.action_mlp[-1].weight, std=0.03)
    torch.nn.init.normal_(model.executor.state_up.weight, std=0.03)
    model.eval()
    objective = IAGSRMEObjective(
        ObjectiveConfig(terminal_weight=1.0, lambda_pair=1.0, lambda_gain=1.0)
    ).eval()
    batch = ImageBatch(
        sample_ids=["a", "b", "c"],
        reference_ids=["ra", "rb", "rc"],
        target_ids=["ta", "tb", "tc"],
        modification_texts=["red", "blue", "green"],
        categories=["dress", "dress", "dress"],
        reference_pixels=torch.randn(3, 3, 3, 3),
        target_pixels=torch.randn(3, 3, 3, 3),
        input_ids=torch.randint(0, 32, (3, 6)),
        attention_mask=torch.ones(3, 6, dtype=torch.bool),
        content_mask=torch.ones(3, 6, dtype=torch.bool),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    precision = SimpleNamespace(autocast_enabled=False, autocast_dtype=torch.float32)

    report, sample_ids = _gradient_attribution(
        model,
        objective,
        [batch],
        device=torch.device("cpu"),
        precision=precision,
        max_batches=1,
    )

    assert report["available"]
    assert sample_ids == ["a", "b", "c"]
    assert report["score_loss_upstream_detach_control"]["pair"]["passed"]
    assert report["score_loss_upstream_detach_control"]["gain"]["passed"]
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
