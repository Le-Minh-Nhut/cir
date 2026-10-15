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
    # The test fixture describes a checkpoint whose mapping IS verified; a run whose mapping is
    # unverified is covered by its own test below.
    "checkpoint_mapping_verified": True,
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
    # The missing metric is reported as missing evidence, never as a paper deviation: only a
    # comparable measured number can carry a relative error.
    missing = next(c for c in verdict.comparisons if c.metric == "r50")
    assert missing.reproduced is None
    assert missing.relative_error_pct is None
    assert missing.within_tolerance is False
    assert "no usable reproduced value" in (missing.note or "")

def test_a_missing_metric_never_becomes_a_paper_deviation_label():
    # The frontend reflag (and the backend deviations helper) may only flag numbers. A recorded
    # None relative error must stay out of every deviation flag at any threshold.
    verdict = run({"r10": 57.07, "r50": None, "mean": 67.17})
    for threshold in (0.0, 5.0, 100.0):
        flag = gate.deviations(verdict.comparisons, threshold)
        # r50 (missing evidence) is never the flagged worst metric: no relative error was recorded.
        if flag["deviates"]:
            assert flag["worst_metric"] != "r50"
            assert flag["max_relative_error_pct"] is not None


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
    # A paper that never says which protocol it used cannot silently match whichever one is claimed.
    # The comparison proceeds on the conditions both sides state, and the unchecked ones are surfaced
    # in the detail and in the reference evidence -- not in `blockers`, because a passing verdict must
    # not carry refusal text that a consumer would mishandle.
    ref = reference(gallery_protocol=None)
    verdict = run(dict(PAPER), ref)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert verdict.blockers == []
    assert "not checked" in verdict.detail
    assert verdict.comparisons, "the stated conditions still permit a comparison"


def test_a_reference_that_states_no_condition_cannot_certify_a_run():
    """A row that describes no experiment is refused, however close the numbers look.

    Treating every unstated field as "not checked" made a blank row pass any run, including one on a
    different protocol, dataset and split. A reference has to constrain something to be evidence.
    """
    blank = reference({"r10": 55.0, "r50": 75.0, "mean": 65.0}, reference_id="blank",
                      gallery_protocol=None, checkpoint_id=None, checkpoint_training_noise_pct=None,
                      evaluation_noise_pct=None, split=None, dataset=None)
    run_meta = {"protocol_id": "fashioniq_val_split", "checkpoint_id": "other",
                "checkpoint_training_noise_pct": 99, "evaluation_noise_pct": 42,
                "split": "test", "dataset": "cirr"}
    verdict = gate.evaluate_gate(blank, {"r10": 55.0, "r50": 75.0, "mean": 65.0},
                                 run_meta=run_meta, real_execution=True)
    assert verdict.status == gate.BLOCKED_REFERENCE_INCOMPLETE
    assert not verdict.passed


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

def test_mapping_unverified_and_category_gap_together_report_the_mapping_status():
    """When both caveats hold, the status names the stronger one (identity), detail names both."""
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = gate.evaluate_gate(
        ref, {"r10": 57.07, "r50": 77.27, "mean": 67.17},
        run_meta={**META, "checkpoint_mapping_verified": False}, real_execution=True,
        reproduced_category_metrics=None)
    assert verdict.status == gate.PASS_METRICS_ONLY_MAPPING_UNVERIFIED
    assert verdict.passed is False
    assert "UNVERIFIED" in verdict.detail
    assert "were not compared" in verdict.detail

def test_an_unverified_checkpoint_mapping_is_metrics_only_not_certified():
    """Numerical comparability and official identity are separate claims.

    With an UNVERIFIED mapping the gate does not know which upstream asset produced the numbers, so
    a metric match cannot certify an official reproduction: the status names the gap and `passed`
    stays False even though every comparable metric agrees.
    """
    verdict = run(dict(PAPER), meta={**META, "checkpoint_mapping_verified": False})
    assert verdict.status == gate.PASS_METRICS_ONLY_MAPPING_UNVERIFIED
    assert verdict.passed is False
    assert verdict.variant is False
    assert "UNVERIFIED" in verdict.detail
    # The comparisons stay on the verdict so the numbers remain inspectable.
    assert any(c.metric == "r10" and c.within_tolerance for c in verdict.comparisons)

def test_variant_and_deviation_together_stay_an_alert():
    """A declared variant whose numbers deviate is an alert, not a labelled pass."""
    verdict = run({key: value * 1.2 for key, value in PAPER.items()},
                  meta={**META, "caption_alias_staged": True, "declared_variant": True})
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    assert not verdict.passed
    assert verdict.variant


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
                      checkpoint_id="fiq_n00", checkpoint_training_noise_pct=0,
                      checkpoint_mapping_verified=True)
    verdict = _benchmark_gate_verdict(clean, Extraction(), 0)
    # The habit row publishes per-category metrics and the extraction carries none, so the pass is
    # explicitly incomplete rather than a plain aggregate-only pass.
    assert verdict["status"] == gate.PASS_WITH_UNCHECKED_CATEGORIES
    assert verdict["passed"] is False
    assert verdict["reference_evidence"]["reference_id"] == "habit_aaai26_table1_fiq_n00"

    noisy = Namespace(model_id="habit", protocol="fashioniq_full_gallery_ref_excluded",
                      checkpoint_id="fiq_n05", checkpoint_training_noise_pct=50,
                      checkpoint_mapping_verified=True)
    verdict = _benchmark_gate_verdict(noisy, Extraction(), 0)
    assert verdict["status"] == gate.ALERT_METRIC_DEVIATION, verdict
    assert verdict["reference_evidence"]["reference_id"] == "habit_aaai26_table1_fiq_n50"


def test_the_report_helper_reports_a_failed_run_as_not_a_metric_verdict():
    from argparse import Namespace

    from workbench.scripts.evaluate_models import _benchmark_gate_verdict

    class Extraction:
        observed_metrics = None

    plan = Namespace(model_id="csmcir", protocol="fashioniq_original_split",
                     checkpoint_id="fashioniq", checkpoint_training_noise_pct=0,
                     checkpoint_mapping_verified=True)
    verdict = _benchmark_gate_verdict(plan, Extraction(), 1)
    assert verdict["status"] == gate.FAILED_EVALUATION
    assert not verdict["passed"]


def test_a_derived_entry_is_never_a_second_copy_of_a_printed_value():
    """Extractors name derivations after the arithmetic; the registry must name them after the metric.

    A derived copy of a printed value is unreachable (the reported value wins in `value_for`), and
    where the two disagree it asserts arithmetic the paper contradicts. Air-Know's 20% row had exactly
    that: a derived 75.72 against the paper's printed 75.71.
    """
    for record in benchmarks.load_registry()["references"]:
        reported = record.get("reported_metrics") or {}
        for item in record.get("derived_metrics") or []:
            assert item["key"] in gate.METRIC_KEYS, (record["reference_id"], item["key"])
            assert reported.get(item["key"]) is None, (
                f"{record['reference_id']}: {item['key']} is derived while the paper prints it")


def test_no_committed_row_states_no_condition_at_all():
    """A row that constrains nothing could pass any run, so the gate refuses it. Keep the registry clean."""
    from workbench.backend import paper_gate as gate_module

    for record in benchmarks.load_registry()["references"]:
        stated = sum(1 for field in gate_module.COMPARABILITY_FIELDS
                     if {
                         "protocol_id": record.get("gallery_protocol"),
                         "checkpoint_id": record.get("checkpoint_id"),
                         "checkpoint_training_noise_pct": record.get("checkpoint_training_noise_pct"),
                         "evaluation_noise_pct": record.get("evaluation_noise_pct"),
                         "split": record.get("split"),
                         "dataset": record.get("dataset"),
                     }.get(field) is not None)
        assert stated > 0, record["reference_id"]

# ---------------------------------------------------------------- per-category metrics

def run_with_categories(metrics, categories, ref=None, **kwargs):
    return gate.evaluate_gate(ref or reference(), metrics,
                              run_meta=dict(META), real_execution=True,
                              reproduced_category_metrics=categories, **kwargs)

def test_a_deviating_category_fails_even_when_the_aggregate_passes():
    # Scenario E: the aggregate hides a broken category. Paper dress R@10 is 50.00; the run's dress
    # is 42.00 (16 percent off), while every aggregate metric is exact.
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = run_with_categories(
        {"r10": 57.07, "r50": 77.27, "mean": 67.17},
        {"dress": {"r10": 42.00, "r50": 70.00}},
        ref)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    dress = next(c for c in verdict.comparisons if c.metric == "dress_r10")
    assert not dress.within_tolerance
    assert dress.direction == "lower"
    assert dress.relative_error_pct == pytest.approx(16.0)

def test_categories_compare_only_where_both_sides_publish():
    # The paper prints a dress row; the run observed no categories. Nothing is compared against a
    # number that was never observed, and the pass is explicitly incomplete: the status names the
    # published-but-unchecked categories instead of claiming a full reproduction check.
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = run_with_categories({"r10": 57.07, "r50": 77.27, "mean": 67.17}, None, ref)
    assert verdict.status == gate.PASS_WITH_UNCHECKED_CATEGORIES
    # Incomplete category coverage is not full reproduction verification.
    assert verdict.passed is False
    # The metric comparisons stay on the verdict so the incomplete result remains inspectable.
    assert any(c.metric == "r10" and c.within_tolerance for c in verdict.comparisons)
    assert "dress_r10" in verdict.detail and "not compared" in verdict.detail

def test_a_run_category_with_no_paper_counterpart_is_not_compared():
    # An observed category cannot be judged against a row that never printed one.
    ref = reference(**{"category_metrics": None})
    verdict = run_with_categories(
        {"r10": 57.07, "r50": 77.27, "mean": 67.17},
        {"dress": {"r10": 10.00, "r50": 10.00}},
        ref)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    assert not any(c.metric == "dress_r10" for c in verdict.comparisons)

def test_matching_categories_keep_the_pass():
    ref = reference(**{"category_metrics": {
        "dress": {"r10": 52.45, "r50": 74.81},
        "shirt": {"r10": 57.70, "r50": 75.76},
        "toptee": {"r10": 61.14, "r50": 80.98},
    }})
    verdict = run_with_categories(
        {"r10": 57.10, "r50": 77.25, "mean": 67.15},
        {"dress": {"r10": 52.44, "r50": 74.80},
         "shirt": {"r10": 57.69, "r50": 75.77},
         "toptee": {"r10": 61.15, "r50": 80.97}},
        ref)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT

# ---------------------------------------------------------------- the threshold helper

def test_deviations_helper_reflags_recorded_comparisons():
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = run_with_categories(
        {"r10": 57.07, "r50": 77.27, "mean": 67.17},
        {"dress": {"r10": 48.50, "r50": 70.00}},  # dress_r10 is 3 percent off: inside the gate
        ref)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT
    flag = gate.deviations(verdict.comparisons, 2.0)
    assert flag["deviates"] and flag["worst_metric"] == "dress_r10"
    loose = gate.deviations(verdict.comparisons, 10.0)
    assert not loose["deviates"] and loose["worst_metric"] is None

# ---------------------------------------------------------------- parity now follows the gate rule

def test_parity_uses_relative_error_not_absolute_tolerance():
    # 57.07 vs 60.00 is 2.93pp apart: the legacy check (`abs(diff) > 0.05` with values already in
    # percent units) called that a pass, but it is 5.1 percent relative -- a FAIL under the canonical
    # rule. And 57.07 vs 57.20 (0.13pp) stays a pass: 0.23 percent relative.
    from workbench.backend.metrics_extraction import _check_parity
    assert _check_parity({"r10": 60.00, "r50": 77.27, "mean": 67.17},
                         {"r10": 57.07, "r50": 77.27, "mean": 67.17}) == "PARITY_MISMATCH"
    assert _check_parity({"r10": 57.20, "r50": 77.27, "mean": 67.17},
                         {"r10": 57.07, "r50": 77.27, "mean": 67.17}) == "PARITY_VERIFIED"
    assert _check_parity({"r10": 57.07, "r50": 77.27},  # paper missing mean
                         {"r10": 57.07, "r50": 77.27, "mean": 67.17}) == "PARITY_NOT_EVALUATED"


def test_a_deviating_category_still_fails_when_the_aggregate_is_absent():
    # Scenario: the parser could not validate the aggregate line (observed_metrics=None), but the
    # per-category tables were parsed and forwarded. The category evidence must reach the gate: a
    # 16% dress deviation cannot vanish just because the aggregate line is missing.
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = run_with_categories({}, {"dress": {"r10": 42.00, "r50": 70.00}}, ref)
    assert verdict.status == gate.ALERT_METRIC_DEVIATION
    dress = next(c for c in verdict.comparisons if c.metric == "dress_r10")
    assert not dress.within_tolerance
    assert dress.relative_error_pct == pytest.approx(16.0)


def test_no_metrics_at_all_still_fails_output_validation():
    # With neither an aggregate nor category evidence there is nothing to compare: the refusal
    # state is unchanged by the category-evidence fix.
    ref = reference(**{"category_metrics": {"dress": {"r10": 50.00, "r50": 70.00}}})
    verdict = run_with_categories({}, None, ref)
    assert verdict.status == gate.FAILED_OUTPUT_VALIDATION
