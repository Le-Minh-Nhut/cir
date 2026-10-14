"""Tests for the single numerical acceptance gate.

Expected outcomes are computed by hand from the published values, never by calling the gate to
produce them. The set covers both sides of the 5 percent boundary, the ambiguous scale, zero
baselines, non-finite inputs, and every refusal the gate must make before it compares anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import benchmark_references as benchmarks
from workbench.backend import paper_gate as gate

#: A complete published row so a case tests the rule rather than the reference's completeness.
PAPER = {"r10": 57.07, "r50": 77.27, "mean": 67.17}
META = {
    "protocol_id": "fashioniq_original_split",
    "checkpoint_id": "fashioniq",
    "checkpoint_training_noise_pct": 0,
    "evaluation_noise_pct": 0,
    "split": "val",
    "dataset": "fashioniq",
}


def reference(paper=None, status=benchmarks.REFERENCE_VERIFIED, **overrides):
    payload = {
        "reference_id": overrides.pop("reference_id", "test_row"),
        "model_id": "testmodel",
        "gallery_protocol": META["protocol_id"],
        "checkpoint_id": META["checkpoint_id"],
        "checkpoint_training_noise_pct": META["checkpoint_training_noise_pct"],
        "evaluation_noise_pct": META["evaluation_noise_pct"],
        "split": META["split"],
        "dataset": META["dataset"],
        "reported_metrics": dict(paper if paper is not None else PAPER),
        "derived_metrics": [],
        "reference_status": status,
        **overrides,
    }
    return benchmarks.BenchmarkReference("test_row", "testmodel", "p", "T", payload, "digest", "v1")


def run(metrics, ref=None, meta=None, **kwargs):
    return gate.evaluate_gate(ref or reference(), metrics,
                              run_meta=meta or dict(META), real_execution=True, **kwargs)


# ---------------------------------------------------------------- the 5 percent rule

def test_exact_agreement_passes():
    verdict = run(dict(PAPER))
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert verdict.passed and not verdict.variant


def test_one_percent_error_passes():
    metrics = {key: value * 1.01 for key, value in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    for comparison in verdict.comparisons:
        assert comparison.relative_error_pct == pytest.approx(1.0, abs=1e-9)


def test_just_inside_the_boundary_passes():
    metrics = {key: value * (1 - 0.0499) for key, value in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert all(comparison.relative_error_pct < 5.0 for comparison in verdict.comparisons)


def test_exactly_at_the_boundary_passes():
    metrics = {key: value * 0.95 for key, value in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    for comparison in verdict.comparisons:
        assert comparison.relative_error_pct == pytest.approx(5.0, abs=1e-9)
        assert comparison.within_tolerance


def test_just_outside_the_boundary_alerts():
    metrics = {key: value * (1 - 0.0501) for key, value in PAPER.items()}
    verdict = run(metrics)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_ten_percent_error_alerts():
    metrics = {key: value * 0.9 for key, value in PAPER.items()}
    assert run(metrics).status == gate.ALERT_METRIC_DEVIATION


def test_both_deviation_directions_are_symmetric():
    low = run({key: value * 0.94 for key, value in PAPER.items()})
    high = run({key: value * 1.06 for key, value in PAPER.items()})
    assert low.status == high.status == gate.ALERT_METRIC_DEVIATION
    for a, b in zip(low.comparisons, high.comparisons):
        assert a.relative_error_pct == pytest.approx(b.relative_error_pct, abs=1e-9)


def test_a_passing_mean_cannot_carry_a_failing_r10():
    verdict = run({"r10": 45.0, "r50": 77.27, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert "r10" in " ".join(verdict.blockers)


def test_a_failing_category_mean_cannot_carry_a_passing_overall_mean():
    # The overall mean is computed from R@10 and R@50, so a category failure shows up in whichever of
    # those it moved. This asserts the aggregate itself is not consulted as a shortcut.
    verdict = run({"r10": 57.07, "r50": 77.27, "mean": 40.0})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert "mean" in " ".join(verdict.blockers)


# ---------------------------------------------------------------- scale

def test_a_declared_fraction_scale_is_normalised():
    # The declared scale applies to BOTH sides: the registry stores the published value in the same
    # convention the parser used, so both 0.5707 (run) and 0.5707 (paper) mean 57.07 percent.
    published = {"r10": 0.5707, "r50": 0.7727, "mean": 0.6717}
    verdict = run({"r10": 0.5707, "r50": 0.7727, "mean": 0.6717},
                  reference(published), metric_scale=gate.SCALE_FRACTION)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert verdict.comparisons[0].reproduced == pytest.approx(57.07, abs=1e-9)
    assert verdict.comparisons[0].paper == pytest.approx(57.07, abs=1e-9)


def test_a_declared_percentage_scale_is_not_rescaled():
    # 0.5707 under the percentage convention means 0.57 percent, which is nowhere near 57.07.
    verdict = run({"r10": 0.5707, "r50": 0.7727, "mean": 0.6717},
                  metric_scale=gate.SCALE_PERCENTAGE)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION


def test_one_point_zero_is_never_silently_rescaled():
    # Under the fraction convention `1.0` is one percent; under the percentage convention it is one
    # percent too. Neither may become 100.
    fraction = run({"r10": 1.0, "r50": 77.27, "mean": 67.17}, metric_scale=gate.SCALE_FRACTION)
    assert fraction.comparisons[0].reproduced == 100.0
    percent = run({"r10": 1.0, "r50": 77.27, "mean": 67.17}, metric_scale=gate.SCALE_PERCENTAGE)
    assert percent.comparisons[0].reproduced == 1.0


def test_an_unknown_scale_refuses_rather_than_guessing():
    verdict = run(dict(PAPER), metric_scale="furlongs")
    assert verdict.status == gate.FAILED_OUTPUT_VALIDATION
    assert "scale" in " ".join(verdict.blockers)


def test_normalize_scale_rejects_an_undeclared_convention():
    with pytest.raises(ValueError, match="scale"):
        gate.normalize_scale(57.0, "guess")


# ---------------------------------------------------------------- degenerate values

def test_zero_baseline_requires_exact_equality():
    paper = {"r10": 0.0, "r50": 75.0, "mean": 65.0}
    identical = run({"r10": 0.0, "r50": 75.0, "mean": 65.0}, reference(paper))
    assert identical.status == gate.PASS_WITHIN_5_PERCENT
    assert identical.comparisons[0].relative_error_pct is None
    different = run({"r10": 0.5, "r50": 75.0, "mean": 65.0}, reference(paper))
    assert different.status == gate.ALERT_METRIC_DEVIATION


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_reproduced_value_never_passes(bad):
    verdict = run({"r10": bad, "r50": 77.27, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_a_negative_recall_never_passes():
    verdict = run({"r10": -5.0, "r50": 77.27, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION


def test_a_missing_reproduced_metric_fails():
    verdict = run({"r10": 57.07, "r50": None, "mean": 67.17})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_missing_metrics_on_both_sides_are_reported_not_compared():
    paper = {"r10": 57.07, "r50": None, "mean": 67.17}
    verdict = run({"r10": 57.07, "r50": None, "mean": 67.17}, reference(paper))
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert "not compared" in verdict.detail


# ---------------------------------------------------------------- refusals before arithmetic

def test_a_mock_result_is_not_run():
    verdict = run(dict(PAPER), data_kind="mock")
    assert verdict.status == gate.NOT_RUN
    assert verdict.comparisons == []


def test_absent_execution_is_not_run():
    verdict = gate.evaluate_gate(reference(), dict(PAPER), run_meta=dict(META))
    assert verdict.status == gate.NOT_RUN


def test_a_failed_evaluation_is_not_a_metric_verdict():
    verdict = gate.evaluate_gate(reference(), dict(PAPER), run_meta=dict(META),
                                 real_execution=True, evaluation_completed=False)
    assert verdict.status == gate.FAILED_EVALUATION


def test_an_empty_extraction_is_an_output_failure():
    verdict = gate.evaluate_gate(reference(), {}, run_meta=dict(META), real_execution=True)
    assert verdict.status == gate.FAILED_OUTPUT_VALIDATION


def test_no_reference_means_unverified_not_a_failure():
    verdict = gate.evaluate_gate(None, dict(PAPER), run_meta=dict(META), real_execution=True)
    assert verdict.status == gate.METRIC_REFERENCE_UNVERIFIED
    assert not verdict.passed


def test_an_unverified_reference_cannot_certify_a_run():
    verdict = run(dict(PAPER), reference(status=benchmarks.REFERENCE_UNVERIFIED))
    assert verdict.status == gate.METRIC_REFERENCE_UNVERIFIED
    assert not verdict.passed


def test_a_reference_with_no_published_metric_is_unverified():
    verdict = run({"r10": 1.0}, reference({"r10": None, "r50": None, "mean": None}))
    assert verdict.status == gate.METRIC_REFERENCE_UNVERIFIED


def test_protocol_mismatch_blocks_before_any_comparison():
    verdict = run(dict(PAPER), meta={**META, "protocol_id": "fashioniq_val_split"})
    assert verdict.status == gate.BLOCKED_PROTOCOL_MISMATCH
    assert verdict.comparisons == []


def test_split_mismatch_blocks():
    verdict = run(dict(PAPER), meta={**META, "split": "test"})
    assert verdict.status == gate.BLOCKED_PROTOCOL_MISMATCH


def test_dataset_mismatch_blocks():
    verdict = run(dict(PAPER), meta={**META, "dataset": "cirr"})
    assert verdict.status == gate.BLOCKED_PROTOCOL_MISMATCH


def test_checkpoint_noise_mismatch_blocks():
    verdict = run(dict(PAPER), meta={**META, "checkpoint_training_noise_pct": 50})
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH


def test_wrong_checkpoint_identity_blocks():
    verdict = run(dict(PAPER), meta={**META, "checkpoint_id": "some_other_variant"})
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH


def test_evaluation_noise_mismatch_blocks():
    verdict = run(dict(PAPER), meta={**META, "evaluation_noise_pct": 20})
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH


def test_an_unknown_run_field_blocks_rather_than_matching():
    meta = dict(META)
    meta.pop("checkpoint_training_noise_pct")
    verdict = run(dict(PAPER), meta=meta)
    assert verdict.status == gate.BLOCKED_CHECKPOINT_MISMATCH


def test_a_condition_the_reference_does_not_state_is_reported_not_assumed():
    # A paper that never says which protocol it used cannot silently match whichever one is claimed:
    # the unstated field is reported so a reader sees what was not checked, and the comparison still
    # proceeds on the conditions both sides do state.
    ref = reference(gallery_protocol=None)
    verdict = run(dict(PAPER), ref)
    assert verdict.status in (gate.PASS_WITHIN_5_PERCENT, gate.FAILED_OUTPUT_VALIDATION)
    assert any("not stated by the reference" in blocker for blocker in verdict.blockers)
    assert verdict.comparisons, "the stated conditions still permit a comparison"


# ---------------------------------------------------------------- variants and evidence

def test_a_declared_variant_passes_but_is_labelled():
    verdict = run(dict(PAPER), meta={**META, "caption_alias_staged": True})
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT_VARIANT
    assert verdict.passed and verdict.variant


def test_a_variant_that_deviates_is_an_alert_not_a_variant_pass():
    verdict = run({key: value * 0.5 for key, value in PAPER.items()},
                  meta={**META, "caption_alias_staged": True})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed


def test_a_pass_records_the_reference_evidence():
    verdict = run(dict(PAPER))
    evidence = verdict.reference_evidence
    assert evidence["reference_id"] == "test_row"
    assert evidence["registry_digest"] == "digest"
    assert evidence["registry_version"] == "v1"
    assert evidence["metric_provenance"]["r10"] == benchmarks.PAPER_REPORTED


def test_a_derived_value_is_labelled_as_derived():
    ref = reference({"r10": 55.91, "r50": 77.82, "mean": None})
    ref.payload["derived_metrics"] = [{"key": "mean", "value": 66.87,
                                      "derivation_formula": "(r10 + r50) / 2",
                                      "provenance": "WORKBENCH_DERIVED"}]
    verdict = run({"r10": 55.91, "r50": 77.82, "mean": 66.87}, ref)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert verdict.reference_evidence["metric_provenance"]["mean"] == benchmarks.WORKBENCH_DERIVED


def test_verdict_serialises_for_a_report():
    payload = run(dict(PAPER)).to_dict()
    assert set(payload) >= {"status", "passed", "variant", "blockers", "comparisons", "reference_evidence"}
    assert payload["comparisons"][0]["metric"] == "r10"
    assert payload["comparisons"][0]["paper_provenance"] == benchmarks.PAPER_REPORTED


# ---------------------------------------------------------------- the registry itself

def test_the_committed_registry_loads_and_is_versioned():
    data = benchmarks.load_registry()
    assert data["references"], "the registry must carry the committed references"
    assert benchmarks.registry_version() != "unversioned"
    for record in data["references"]:
        assert record["reference_id"]
        assert record["model_id"]
        assert record["reference_status"] in {benchmarks.REFERENCE_VERIFIED,
                                              benchmarks.REFERENCE_CONFLICT,
                                              benchmarks.REFERENCE_UNVERIFIED}


def test_every_registry_row_covers_at_least_one_metric():
    for record in benchmarks.load_registry()["references"]:
        if record["reference_status"] != benchmarks.REFERENCE_VERIFIED:
            continue
        reported = record.get("reported_metrics") or {}
        derived = record.get("derived_metrics") or []
        values = [reported.get(key) for key in gate.METRIC_KEYS]
        values += [d.get("value") for d in derived if d.get("key") in gate.METRIC_KEYS]
        assert any(value is not None for value in values), record["reference_id"]


def test_a_verified_row_never_states_a_derived_metric_as_paper_reported():
    for record in benchmarks.load_registry()["references"]:
        for item in record.get("derived_metrics") or []:
            assert item.get("provenance") == benchmarks.WORKBENCH_DERIVED, record["reference_id"]
            assert item.get("derivation_formula"), record["reference_id"]
            assert (record.get("reported_metrics") or {}).get(item["key"]) is None, \
                f"{record['reference_id']}: {item['key']} is both reported and derived"


def test_selecting_a_reference_requires_the_condition_to_match():
    exact, why = benchmarks.select_reference("habit", checkpoint_training_noise_pct=50,
                                              protocol_id="fashioniq_full_gallery_ref_excluded")
    assert exact is not None and exact.value_for("r10") == 52.52, why
    absent, why = benchmarks.select_reference("habit", checkpoint_training_noise_pct=99)
    assert absent is None and "no verified reference" in why
    unknown, why = benchmarks.select_reference("no_such_model")
    assert unknown is None and "no committed paper reference" in why


# ---------------------------------------------------------------- report integration

def test_the_report_helper_selects_by_condition_not_by_headline_value():
    """`_benchmark_gate_verdict` must judge a noisy checkpoint against its own noise row.

    Comparing HABIT's 0%-noise numbers against the 50%-noise reference is the mistake this guards: the
    two rows differ by about 5.4 percent, which is close enough to look like a good reproduction if the
    wrong reference is picked.
    """
    from argparse import Namespace

    from workbench.scripts.evaluate_models import _benchmark_gate_verdict

    class Extraction:
        observed_metrics = {"r10": 55.38, "r50": 75.20, "mean": 65.29}

    clean = Namespace(model_id="habit", protocol="fashioniq_full_gallery_ref_excluded",
                      checkpoint_id="fiq_n00", checkpoint_training_noise_pct=0)
    verdict = _benchmark_gate_verdict(clean, Extraction(), 0)
    assert verdict["status"] == gate.PASS_WITHIN_5_PERCENT
    assert verdict["reference_evidence"]["reference_id"] == "habit_aaai26_table1_fiq_n00"

    noisy = Namespace(model_id="habit", protocol="fashioniq_full_gallery_ref_excluded",
                      checkpoint_id="fiq_n05", checkpoint_training_noise_pct=50)
    verdict = _benchmark_gate_verdict(noisy, Extraction(), 0)
    assert verdict["status"] == gate.ALERT_METRIC_DEVIATION, verdict
    assert verdict["reference_evidence"]["reference_id"] == "habit_aaai26_table1_fiq_n50"


def test_the_report_helper_reports_a_failed_run_as_not_a_metric_verdict():
    from argparse import Namespace

    from workbench.scripts.evaluate_models import _benchmark_gate_verdict

    class Extraction:
        observed_metrics = None

    plan = Namespace(model_id="csmcir", protocol="fashioniq_original_split",
                     checkpoint_id="fashioniq", checkpoint_training_noise_pct=0)
    verdict = _benchmark_gate_verdict(plan, Extraction(), 1)
    assert verdict["status"] == gate.FAILED_EVALUATION
    assert not verdict["passed"]
