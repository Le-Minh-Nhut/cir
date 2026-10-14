"""Adversarial tests for the paper-metric gate.

The point of these tests is the boundary, not the happy path: exactly 5 percent, just inside and just
outside it, scale conventions, zero-valued metrics, a mean that passes while R@10 fails, and the
three ways a comparison must be refused before arithmetic (mock, unverified reference, mismatched
protocol or checkpoint).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import paper_gate as gate

PAPER = {"r10": 57.07, "r50": 77.27, "mean": 67.17}
META = {
    "protocol_id": "fashioniq_original_split",
    "checkpoint_id": "fashioniq",
    "checkpoint_training_noise_pct": 0,
    "evaluation_noise_pct": 0,
    "split": "val",
    "dataset": "FashionIQ",
}


def run(metrics, meta=None, **kwargs):
    return gate.evaluate_gate(PAPER, metrics, paper_meta=META, run_meta=meta or dict(META),
                              real_execution=True, **kwargs)


def test_exact_match_passes():
    verdict = run(dict(PAPER))
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert verdict.passed and not verdict.variant
    assert all(c.within_tolerance for c in verdict.comparisons)


def test_boundary_just_inside_tolerance_passes():
    # 4.99% below the paper value on every metric.
    metrics = {k: v * (1 - 0.0499) for k, v in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    for c in verdict.comparisons:
        assert c.relative_error_pct < 5.0
        assert c.within_tolerance


def test_boundary_exactly_five_percent_passes():
    metrics = {k: v * 0.95 for k, v in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    for c in verdict.comparisons:
        assert c.relative_error_pct == pytest.approx(5.0, abs=1e-9)
        assert c.within_tolerance


def test_boundary_just_outside_tolerance_alerts():
    metrics = {k: v * (1 - 0.0501) for k, v in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed
    assert all(c.relative_error_pct > 5.0 for c in verdict.comparisons)


def test_negative_and_positive_deviations_are_symmetric():
    low = run({k: v * 0.94 for k, v in PAPER.items()})
    high = run({k: v * 1.06 for k, v in PAPER.items()})
    assert low.status == high.status == gate.ALERT_METRIC_DEVIATION
    for a, b in zip(low.comparisons, high.comparisons):
        assert a.relative_error_pct == pytest.approx(b.relative_error_pct, abs=1e-9)


def test_mean_passing_cannot_carry_a_failing_r10():
    # The classic false pass: average looks fine, one headline metric is broken.
    metrics = {"r10": 45.0, "r50": 77.27, "mean": 67.17}
    verdict = run(metrics)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert "r10" in " ".join(verdict.blockers)


def test_scale_convention_is_normalised_before_comparison():
    # 0.5707 and 57.07 are the same Recall under different conventions.
    verdict = run({"r10": 0.5707, "r50": 0.7727, "mean": 0.6717})
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT


def test_exactly_one_is_read_as_one_percent_not_one_hundred():
    verdict = gate.evaluate_gate({"r10": 50.0}, {"r10": 1.0}, paper_meta=META, run_meta=dict(META),
                                 real_execution=True)
    comparison = verdict.comparisons[0]
    assert comparison.reproduced == 1.0
    assert comparison.within_tolerance is False


def test_zero_paper_metric_requires_exact_equality():
    identical = gate.evaluate_gate({"r10": 0.0}, {"r10": 0.0}, paper_meta=META, run_meta=dict(META),
                                   real_execution=True)
    assert identical.status == gate.PASS_WITHIN_5_PERCENT
    assert identical.comparisons[0].relative_error_pct is None
    different = gate.evaluate_gate({"r10": 0.0}, {"r10": 0.5}, paper_meta=META, run_meta=dict(META),
                                   real_execution=True)
    assert different.status == gate.ALERT_METRIC_DEVIATION


def test_missing_reproduced_metric_fails_rather_than_passing():
    verdict = run({"r10": 57.07, "r50": None, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_reproduced_metric_never_passes(bad):
    verdict = run({"r10": bad, "r50": 77.27, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_nan_paper_metric_is_not_a_match():
    verdict = gate.evaluate_gate({"r10": float("nan"), "r50": 77.27, "mean": 67.17},
                                 {"r10": 57.07, "r50": 77.27, "mean": 67.17},
                                 paper_meta=META, run_meta=dict(META), real_execution=True)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION


def test_mock_result_is_not_run():
    verdict = run(dict(PAPER), data_kind="mock")
    assert verdict.status == gate.NOT_RUN
    assert verdict.comparisons == []


def test_absent_execution_is_not_run():
    verdict = gate.evaluate_gate(PAPER, dict(PAPER), paper_meta=META, run_meta=dict(META))
    assert verdict.status == gate.NOT_RUN


def test_unverified_paper_reference_blocks_comparison():
    verdict = gate.evaluate_gate({"r10": None, "r50": None, "mean": None}, dict(PAPER),
                                 paper_meta=META, run_meta=dict(META), real_execution=True)
    assert verdict.status == gate.METRIC_REFERENCE_UNVERIFIED


def test_protocol_mismatch_blocks_before_comparison():
    meta = dict(META, protocol_id="fashioniq_val_split")
    verdict = run(dict(PAPER), meta)
    assert verdict.status == gate.BLOCKED_PROTOCOL_MISMATCH
    assert verdict.comparisons == []


def test_checkpoint_noise_mismatch_blocks():
    meta = dict(META, checkpoint_training_noise_pct=50)
    verdict = run(dict(PAPER), meta)
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH


def test_unknown_provenance_field_blocks_rather_than_matching():
    meta = dict(META)
    meta.pop("evaluation_noise_pct")
    verdict = run(dict(PAPER), meta)
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH
    assert not verdict.passed


def test_staged_alias_labels_the_pass_as_a_variant():
    meta = dict(META, caption_alias_staged=True)
    verdict = run(dict(PAPER), meta)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT_VARIANT
    assert verdict.passed and verdict.variant


def test_variant_deviation_is_still_an_alert_not_a_variant_pass():
    meta = dict(META, caption_alias_staged=True)
    verdict = run({k: v * 0.5 for k, v in PAPER.items()}, meta)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_acceptance_intervals_from_the_registry_are_reproducible():
    """The reviewer's stated intervals must equal what the rule itself produces."""
    half = 0.05
    assert PAPER["r10"] * (1 - half) == pytest.approx(54.2165, abs=5e-5)
    assert PAPER["r10"] * (1 + half) == pytest.approx(59.9235, abs=5e-5)
    assert PAPER["r50"] * (1 - half) == pytest.approx(73.4065, abs=5e-5)
    assert PAPER["r50"] * (1 + half) == pytest.approx(81.1335, abs=5e-5)
    assert PAPER["mean"] * (1 - half) == pytest.approx(63.8115, abs=5e-5)
    assert PAPER["mean"] * (1 + half) == pytest.approx(70.5285, abs=5e-5)
    assert math.isclose(PAPER["mean"], (PAPER["r10"] + PAPER["r50"]) / 2, abs_tol=0.01)
