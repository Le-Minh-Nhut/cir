from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from data.images import ImageBatch
from diagnose_slot_specialization import (
    _automatic_flags,
    _concept_forward_with_precision,
    _gradient_attribution,
)
from diagnostics.specialization import (
    cluster_bootstrap_mean_interval,
    cluster_bootstrap_row_indices,
    concept_mil_responsibility,
    exact_shapley_values,
    functional_cluster_bootstrap_summary,
    functional_specialization_summary,
    gradient_interference_summary,
    semantic_specialization_summary,
    valid_teacher_cluster_ids,
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


def test_tie_aware_oracle_occupancy_splits_exact_winners() -> None:
    utility = torch.tensor(
        [[0.5, 0.5, -0.1, -0.2], [-0.2, -0.1, 0.7, 0.7]]
    )
    summary = functional_specialization_summary(utility)
    tie_aware = [
        row["tie_aware_oracle_occupancy_given_oracle_execute"]
        for row in summary["per_slot"]
    ]
    tiebroken = [
        row["argmax_tiebroken_oracle_occupancy_given_oracle_execute"]
        for row in summary["per_slot"]
    ]
    assert tie_aware == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert tiebroken == pytest.approx([0.5, 0.0, 0.5, 0.0])
    assert summary["oracle_exact_tie_fraction"] == pytest.approx(1.0)
    assert summary["mean_oracle_tie_size_given_execute"] == pytest.approx(2.0)


def test_tie_aware_oracle_reporting_does_not_change_shapley() -> None:
    utility = torch.tensor([[0.5, 0.5, 0.2, -0.1], [0.3, 0.3, 0.3, 0.3]])
    expected, _ = exact_shapley_values(utility)
    summary = functional_specialization_summary(utility)
    assert summary["shapley"]["contribution_per_slot"] == pytest.approx(
        expected.mean(dim=0).tolist()
    )


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
        torch.tensor([0, 0, 1, 1]),
        min_support=2,
        bootstrap_samples=50,
        seed=13,
    )
    by_concept = {row["concept"]: row for row in summary["reported_concepts"]}
    assert by_concept["red"]["conditional_shapley_mean"][0] > 0
    assert by_concept["red"]["conditional_shapley_mean"][1] == pytest.approx(0)
    assert by_concept["sleeve"]["conditional_shapley_mean"][1] > 0
    assert by_concept["sleeve"]["conditional_shapley_mean"][0] == pytest.approx(0)
    assert by_concept["red"]["bootstrap_method"] == "teacher_batch_cluster"
    assert by_concept["red"]["support_teacher_cluster_count"] == 1


def test_semantic_shapley_ci_uses_teacher_cluster_bootstrap_plan() -> None:
    utility = torch.ones(6, 2)
    shapley = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [5.0, 10.0], [5.0, 10.0], [5.0, 10.0], [9.0, 18.0]]
    )
    clusters = torch.tensor([0, 0, 1, 1, 1, 2])
    samples = 80
    seed = 23
    summary = semantic_specialization_summary(
        utility,
        shapley,
        [("shared",)] * 6,
        clusters,
        min_support=1,
        bootstrap_samples=samples,
        seed=seed,
    )
    concept = summary["reported_concepts"][0]
    plan = cluster_bootstrap_row_indices(clusters, samples=samples, seed=seed)
    estimates = torch.stack([shapley[indices].mean(dim=0) for indices in plan])
    assert concept["conditional_shapley_ci_low"] == pytest.approx(
        torch.quantile(estimates, 0.025, dim=0).tolist()
    )
    assert concept["conditional_shapley_ci_high"] == pytest.approx(
        torch.quantile(estimates, 0.975, dim=0).tolist()
    )
    assert concept["valid_bootstrap_replicates"] == samples


def test_cluster_bootstrap_resamples_complete_clusters() -> None:
    cluster_ids = torch.tensor([0, 0, 1, 1, 1, 2])
    plans = cluster_bootstrap_row_indices(cluster_ids, samples=20, seed=17)
    original_sizes = {0: 2, 1: 3, 2: 1}
    for indices in plans:
        sampled_clusters = cluster_ids[indices]
        for cluster, size in original_sizes.items():
            assert int(sampled_clusters.eq(cluster).sum()) % size == 0


def test_valid_filtering_retains_original_teacher_batch_id() -> None:
    valid = torch.tensor([True, False, True, True, False, True, True, True])
    cluster_ids = valid_teacher_cluster_ids(valid, teacher_batch_id=7)
    assert cluster_ids.tolist() == [7, 7, 7, 7, 7, 7]


def test_cluster_bootstrap_is_deterministic_and_value_independent() -> None:
    cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    first = cluster_bootstrap_row_indices(cluster_ids, samples=12, seed=31)
    second = cluster_bootstrap_row_indices(cluster_ids, samples=12, seed=31)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))

    values_a = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    values_b = -values_a
    low_a, high_a, count_a = cluster_bootstrap_mean_interval(
        values_a, cluster_ids, samples=12, seed=31
    )
    low_b, high_b, count_b = cluster_bootstrap_mean_interval(
        values_b, cluster_ids, samples=12, seed=31
    )
    assert count_a == count_b == 12
    assert torch.allclose(low_a, -high_b)
    assert torch.allclose(high_a, -low_b)


def test_clear_non_c3_complementarity_has_positive_cluster_ci() -> None:
    utility = torch.tensor([[0.8, -0.2, -0.1, 0.1]] * 24)
    clusters = torch.arange(24) // 4
    summary = functional_cluster_bootstrap_summary(
        utility, clusters, bootstrap_samples=100, seed=5
    )
    complementarity = summary["complementarity"]["all_minus_c3"]
    assert complementarity["estimate"] == pytest.approx(0.7)
    assert complementarity["ci_low"] > 0


def test_c3_dominance_has_exact_zero_complementarity_ci() -> None:
    utility = torch.tensor([[0.1, -0.2, 0.2, 0.8]] * 24)
    clusters = torch.arange(24) // 4
    summary = functional_cluster_bootstrap_summary(
        utility, clusters, bootstrap_samples=100, seed=5
    )
    complementarity = summary["complementarity"]["all_minus_c3"]
    assert complementarity["estimate"] == pytest.approx(0.0)
    assert complementarity["ci_low"] == pytest.approx(0.0)
    assert complementarity["ci_high"] == pytest.approx(0.0)


def test_complementarity_flags_use_cluster_ci_not_tiny_point_threshold() -> None:
    utility = torch.tensor([[0.8, -0.2, -0.1, 0.1]] * 24)
    clusters = torch.arange(24) // 4
    functional = functional_specialization_summary(utility)
    functional.pop("per_sample_shapley")
    functional["cluster_bootstrap"] = functional_cluster_bootstrap_summary(
        utility, clusters, bootstrap_samples=100, seed=5
    )
    flags = _automatic_flags(
        functional,
        {"available": False},
        {"available": False},
        {"reported_concepts": [{}]},
        relative_complementarity_threshold=0.0,
        gradient_skew_threshold=0.75,
        mil_skew_threshold=0.60,
    )
    assert "NON_C3_COMPLEMENTARITY_PRESENT" in {flag["code"] for flag in flags}


def test_zero_value_bootstraps_do_not_fabricate_effective_k() -> None:
    utility = torch.zeros(16, 4)
    clusters = torch.arange(16) // 4
    summary = functional_cluster_bootstrap_summary(
        utility, clusters, bootstrap_samples=50, seed=9
    )
    effective_k = summary["effective_functional_k"]
    assert effective_k == {
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "valid_bootstrap_replicates": 0,
    }


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


def test_concept_mil_forward_uses_configured_autocast_for_mixed_dtype() -> None:
    class MockConcept(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 3)

        def forward(
            self, proposals: torch.Tensor, modification_texts: list[str]
        ) -> dict[str, torch.Tensor]:
            del modification_texts
            return {"candidate_logits": self.projection(proposals)}

    concept = MockConcept()
    proposals = torch.randn(2, 4, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        concept(proposals, ["red", "blue"])

    output = _concept_forward_with_precision(
        concept,
        proposals,
        ["red", "blue"],
        device=torch.device("cpu"),
        precision=SimpleNamespace(
            autocast_enabled=True,
            autocast_dtype=torch.bfloat16,
        ),
    )
    assert output["candidate_logits"].shape == (2, 4, 3)
    assert output["candidate_logits"].dtype == torch.bfloat16


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
    assert "full_objective_gradient_wrt_t0_proposals" in report
    assert (
        report["parameter_level_aggregate_query_row_gradient"]["terminal"][
            "observation_count"
        ]
        == 1
    )
    assert "not a local t0-only" in report["scope_notes"][
        "full_objective_gradient_wrt_t0_proposals"
    ]
    assert "not one sample" in report["scope_notes"][
        "parameter_level_aggregate_query_row_gradient"
    ]
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
