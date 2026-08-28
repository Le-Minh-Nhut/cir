from __future__ import annotations

import pytest
import torch

from training.taper_mag_health import (
    QueryGradientTracker,
    query_delta_clone_geometry,
    dynamic_frozen_metrics,
    repeat_staleness_metrics,
    response_effective_rank,
    utility_health_metrics,
)
from training.taper_mag_reports import EpochHealthAccumulator
from training.taper_mag_engine import CurriculumStage, EngineConfig
from test_taper_mag_training_contract import _end_to_end_fixture


def test_utility_health_regret_stop_pairwise_and_calibration() -> None:
    teacher = torch.tensor(
        [[[2.0, 1.0], [0.5, -0.5]], [[1.0, 2.0], [-0.2, -0.1]]]
    )
    predicted = torch.tensor(
        [[[2.0, 1.0], [0.4, -0.2]], [[2.0, 1.0], [0.2, 0.1]]]
    )
    metrics = utility_health_metrics(
        predicted,
        teacher,
        active=torch.ones(2, 2, dtype=torch.bool),
        near_tie_band=0.05,
        calibration_bins=3,
    )
    assert metrics["top1_agreement"] == pytest.approx(0.5)
    assert metrics["mean_regret"] > 0
    assert metrics["median_regret"] >= 0
    assert metrics["p95_regret"] >= metrics["p90_regret"]
    assert metrics["false_continue_rate"] > 0
    assert metrics["false_stop_rate"] == 0
    assert 0 <= metrics["pairwise_accuracy"] <= 1
    assert 0 <= metrics["confident_pair_accuracy"] <= 1
    assert metrics["near_tie_band"] == 0.05
    assert len(metrics["calibration_by_predicted_gain"]) == 3


def test_response_rank_candidate_variance_and_clone_controls() -> None:
    current = torch.tensor([[1.0, 0.0, 0.0]])
    candidates = torch.tensor(
        [[[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]]
    )
    rank = response_effective_rank(current, candidates)
    assert rank["mean_effective_rank"] > 1
    deltas = candidates - current[:, None]
    controls = query_delta_clone_geometry(deltas, torch.tensor([0]))
    assert controls["clone_all_best_effective_rank"] == pytest.approx(1.0)
    assert controls["clone_all_mean_effective_rank"] == pytest.approx(1.0)
    assert controls["operator_zero_delta_norm"] == 0
    assert controls["repeat_best_delta_norm"] > 0
    assert controls["mean_repeat_delta_norm"] > 0


def test_dynamic_frozen_ordering_and_repeat_staleness() -> None:
    dynamic_actions = torch.tensor([[0, 1, 1], [1, 0, 2]])
    frozen_actions = torch.tensor([[0, 0, 0], [1, 1, 2]])
    dynamic_values = torch.tensor(
        [[[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 3.0, 2.0]]] * 2
    )
    frozen_values = torch.tensor(
        [[[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]] * 2
    )
    metrics = dynamic_frozen_metrics(
        dynamic_actions,
        frozen_actions,
        dynamic_values,
        frozen_values,
        dynamic_retrieval=torch.tensor([2.0, 2.0]),
        frozen_retrieval=torch.tensor([1.0, 1.0]),
        dynamic_realized_gain=torch.tensor([1.0, 1.0]),
        frozen_realized_gain=torch.tensor([0.0, 0.0]),
        stop_index=2,
    )
    assert metrics["ordering_change_rate"] > 0
    assert metrics["top1_action_change_rate"] > 0
    assert metrics["retrieval_difference"] == pytest.approx(1.0)
    gains = torch.tensor(
        [[[1.0, 0.0], [0.1, 0.5], [0.0, -0.2]], [[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]]]
    )
    repeat = repeat_staleness_metrics(
        dynamic_actions,
        gains,
        torch.ones(2, 3, dtype=torch.bool),
        stop_index=2,
    )
    assert repeat["repeat_frequency"] > 0
    assert repeat["teacher_gain_before_repeat"] > repeat["teacher_gain_when_reconsidered"]
    assert repeat["repeated_gain_non_positive_fraction"] > 0


def test_rolling_per_query_gradient_coverage() -> None:
    queries = torch.nn.Parameter(torch.zeros(4, 3))
    tracker = QueryGradientTracker(4)
    queries.grad = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    tracker.update(queries)
    queries.grad = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    tracker.update(queries)
    report = tracker.report()
    assert report["rolling_query_gradient_coverage"] == 1.0
    assert report["per_query_zero_gradient_fraction"] == [0.5] * 4
    assert all(value > 0 for value in report["per_query_gradient_norm_mean"])


def test_epoch_health_accumulator_honors_calibration_bin_config() -> None:
    _, _, policy, supervision, engine = _end_to_end_fixture()
    result = engine.step(
        policy,
        supervision,
        EngineConfig(stage=CurriculumStage.ACTOR_WARMUP, horizon=1),
    )
    accumulator = EpochHealthAccumulator(
        near_tie_band=0.01,
        step_cost=0.0,
        calibration_bins=2,
    )
    accumulator.update(result.model_output, result.teacher_gain)
    actor, utility = accumulator.report()
    assert actor["candidate_outcome_variance"] >= 0
    assert len(utility["calibration_by_predicted_gain"]) in {1, 2}
