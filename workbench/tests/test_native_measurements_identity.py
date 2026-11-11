"""R01 / R07a / R07b / R08: identity joins, per-metric units, and dedup by identity.

Each test here is a reproduction of a defect the independent review executed against this module, and
each asserts the fixed behaviour. The failure mode is quoted in every docstring so the regression is
recognisable if it returns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend import native_measurements as nm

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _collect(tmp_path: Path, files: dict[str, dict]) -> dict:
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return nm.list_native_measurements(tmp_path)


def _row(payload: dict, model: str, path_contains: str = "") -> dict:
    rows = [row for row in payload["measurements"]
            if row.get("metrics") and row["model_id"] == model
            and path_contains in row["source"]["path"]]
    assert len(rows) == 1, (model, path_contains, rows)
    return rows[0]


ROW = {"model_id": "dcnet", "checkpoint_id": "checkpoint_A", "protocol_id": "protocol_A",
       "aggregate": {"macro_r10": 12.0, "macro_r50": 30.0}, "unit": "percent"}

UNRELATED_REPORT = {"model_id": "dcnet", "checkpoint_id": "checkpoint_B",
                    "protocol_id": "protocol_B", "run_id": "unrelated_run_B",
                    "observed_metrics": {"macro_r10": 88.0, "macro_r50": 99.0}}


# ------------------------------------------------------------------------------- R01

def test_a_report_of_another_identity_cannot_back_a_row(tmp_path: Path) -> None:
    """BEFORE: the A row kept its A label but served B's `r10=88.0`, `artifact_backed=true`.

    The join fell back to "any report for this model". Policy implemented: REFUSE the replacement —
    an artifact must not silently relabel itself — so the row keeps its own number and its own
    backing state, and states why nothing was substituted.
    """
    payload = _collect(tmp_path, {
        "checkpoint_selection/measurement.json": ROW,
        "reports/unrelated_aggregate.json": UNRELATED_REPORT,
    })
    row = _row(payload, "dcnet", "measurement.json")
    assert row["metrics"]["r10"] == pytest.approx(12.0), row["metrics"]
    assert row["raw_metrics"] is None, row
    # `artifact_backed` stays exactly as it was — the row is its own artifact, and B did not back it.
    assert row["artifact_backed"] is True, row
    assert row["backed_by"] == row["source"]["path"], row
    assert row.get("backed_by_run_id") is None, row
    assert row["checkpoint_id"] == "checkpoint_A", row
    assert row["protocol_id"] == "protocol_A", row
    # Nothing was replaced, so no `raw_*` identity is invented either.
    assert row["raw_checkpoint_id"] is None and row["raw_source_path"] is None, row


def test_the_positive_control_keeps_its_own_number(tmp_path: Path) -> None:
    """BEFORE: this case already passed. It must keep passing, or the fix over-refused."""
    payload = _collect(tmp_path, {"checkpoint_selection/measurement.json": ROW})
    row = _row(payload, "dcnet")
    assert row["metrics"]["r10"] == pytest.approx(12.0), row
    assert row["raw_metrics"] is None, row
    assert row["backed_by"] == row["source"]["path"], row


def test_a_report_of_the_same_identity_backs_the_row_and_keeps_the_old_label(tmp_path: Path) -> None:
    """A replacement is allowed only on a full identity match, and then the original label survives."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/measurement.json": ROW,
        "reports/matching_aggregate.json": {**UNRELATED_REPORT,
                                            "checkpoint_id": "checkpoint_A",
                                            "protocol_id": "protocol_A",
                                            "observed_metrics": {"macro_r10": 12.5,
                                                                 "macro_r50": 30.5}},
    })
    row = _row(payload, "dcnet", "measurement.json")
    assert row["metrics"]["r10"] == pytest.approx(12.5), row["metrics"]
    assert row["raw_metrics"]["r10"] == pytest.approx(12.0), row
    assert row["artifact_backed"] is True, row
    assert row["backed_by"].endswith("matching_aggregate.json"), row
    assert row["identity_join"] == "exact", row
    # The label never changes with the number it serves.
    assert row["checkpoint_id"] == "checkpoint_A", row
    assert row["raw_checkpoint_id"] == "checkpoint_A", row
    assert row["raw_protocol_id"] == "protocol_A", row
    assert row["raw_source_path"].endswith("measurement.json"), row


def test_two_candidate_reports_never_substitute_and_say_why(tmp_path: Path) -> None:
    """R01 corollary: 2+ candidates is not "pick one", it is "refuse and show the ambiguity"."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/measurement.json": ROW,
        "reports/first_aggregate.json": {**UNRELATED_REPORT, "run_id": "first"},
        "reports/second_aggregate.json": {**UNRELATED_REPORT, "run_id": "second"},
    })
    row = _row(payload, "dcnet", "measurement.json")
    assert row["metrics"]["r10"] == pytest.approx(12.0), row["metrics"]
    assert row["raw_metrics"] is None, row
    assert row["backed_by"] == row["source"]["path"], row
    assert len(row["identity_ambiguous"]) == 2, row["identity_ambiguous"]


def test_a_single_candidate_joins_a_row_that_states_no_protocol_or_checkpoint(
        tmp_path: Path) -> None:
    """The one permitted relaxation: no protocol stated, one candidate, checkpoint agrees.

    The row states neither a protocol nor a checkpoint, so the single report is the only measurement
    it could mean; its protocol is taken from that report and the label says so.
    """
    # The row omits the protocol; the single report states one, so the join IS a relaxation.
    row = {"model_id": "dcnet", "aggregate": {"macro_r10": 12.0, "macro_r50": 30.0}}
    payload = _collect(tmp_path, {
        "checkpoint_selection/measurement.json": row,
        "reports/only_aggregate.json": {**UNRELATED_REPORT, "checkpoint_id": None},
    })
    served = _row(payload, "dcnet", "measurement.json")
    assert served["metrics"]["r10"] == pytest.approx(88.0), served["metrics"]
    assert served["identity_join"] == "single_candidate_protocol_unknown", served
    assert served["raw_protocol_id"] is None, served


def test_a_single_candidate_cannot_back_a_row_that_names_another_checkpoint(
        tmp_path: Path) -> None:
    """R01 through the side door: no protocol, but `checkpoint_A` vs a `checkpoint_B` report.

    BEFORE: the single-candidate branch skipped `_identity_matches` entirely, so this row served 88.0
    under its own `checkpoint_A` label. A row that names a checkpoint is a specific claim, not a
    wildcard — the replacement is refused and the candidate is named as why.
    """
    row = {key: value for key, value in ROW.items() if key != "protocol_id"}
    payload = _collect(tmp_path, {
        "checkpoint_selection/measurement.json": row,
        "reports/only_aggregate.json": UNRELATED_REPORT,
    })
    served = _row(payload, "dcnet", "measurement.json")
    assert served["metrics"]["r10"] == pytest.approx(12.0), served["metrics"]
    assert served["raw_metrics"] is None, served
    assert served["backed_by"] == served["source"]["path"], served
    assert served["identity_ambiguous"], served
    assert served["checkpoint_id"] == "checkpoint_A", served


# ------------------------------------------------------------------------------- R07a / R07b

def test_a_declared_percent_sibling_wins_over_the_fraction_beside_it(tmp_path: Path) -> None:
    """R07a BEFORE: served 0.2793 while labelling it `percent` (`unit_source="sibling_key"`).

    The row path scanned the block for ANY `*_percent` suffix and paired it with a value parsed from
    a different key. Now one parser takes the suffix-ordered value, so value and unit agree.
    """
    payload = _collect(tmp_path, {"checkpoint_selection/dcnet.json": {
        "model_id": "dcnet", "protocol_id": "protocol_A",
        "aggregate": {"macro_r10": 0.2793, "macro_r10_percent": 27.93,
                      "macro_r50": 0.5373, "macro_r50_percent": 53.73},
    }})
    row = _row(payload, "dcnet")
    assert row["metrics"]["r10"] == pytest.approx(27.93), row["metrics"]
    assert row["unit"] == "percent", row
    assert row["unit_source"] == "sibling_key", row
    assert row["per_metric_units"] == {"r10": "_percent", "r50": "_percent"}, row
    # The exact defect: a fraction travelling under a percent label.
    assert not (row["metrics"]["r10"] < 1.0 and row["unit"] == "percent"), row


def test_a_suffix_of_another_metric_does_not_declare_the_recall_unit(tmp_path: Path) -> None:
    """R07b BEFORE: `dead_parameters_pct` declared the RECALL scale, so 0.2793 was called percent."""
    payload = _collect(tmp_path, {"checkpoint_selection/dcnet.json": {
        "model_id": "dcnet", "protocol_id": "protocol_A",
        "aggregate": {"macro_r10": 0.2793, "macro_r50": 0.5373, "dead_parameters_pct": 5.2},
    }})
    row = _row(payload, "dcnet")
    assert row["metrics"]["r10"] == pytest.approx(0.2793), row["metrics"]
    assert row["unit_source"] == "inferred_from_magnitude", row
    assert row["unit"] == "fraction", row
    assert row["per_metric_units"] == {"r10": "", "r50": ""}, row


def test_a_fraction_suffix_of_the_metric_itself_still_declares(tmp_path: Path) -> None:
    """The parser is not simply "ignore suffixes": its own metric's `_fraction` still declares."""
    payload = _collect(tmp_path, {"checkpoint_selection/fraction.json": {
        "model_id": "fx_fraction", "protocol_id": "protocol_A",
        "aggregate": {"macro_r10": 0.6, "macro_r10_fraction": 0.6},
    }})
    row = _row(payload, "fx_fraction")
    assert row["unit"] == "fraction", row
    assert row["unit_source"] == "sibling_key", row


def test_the_report_path_and_the_row_path_agree(tmp_path: Path) -> None:
    """One parser, two callers: a block must parse the same way whichever path reads it."""
    block = {"macro_r10": 0.2793, "macro_r10_percent": 27.93, "dead_parameters_pct": 5.2}
    payload = _collect(tmp_path, {"checkpoint_selection/dcnet.json": {
        "model_id": "dcnet", "protocol_id": "protocol_A", "aggregate": dict(block)}})
    row = _row(payload, "dcnet")
    values, unit, unit_source, per_metric = nm.served_metrics(dict(block))
    assert values == row["metrics"], (values, row["metrics"])
    assert (unit, unit_source) == (row["unit"], row["unit_source"]), (unit, unit_source, row)
    assert per_metric == row["per_metric_units"], (per_metric, row["per_metric_units"])
    assert nm.DECLARED_UNIT_SOURCES == ("declared", "sibling_key"), nm.DECLARED_UNIT_SOURCES


# ------------------------------------------------------------------------------- R08

def test_equal_metrics_from_two_checkpoints_are_two_measurements(tmp_path: Path) -> None:
    """R08 BEFORE: `checkpoint_B` became `counted=false, duplicate_of=A.json` on a rounded value.

    Equal numbers from different checkpoints are not the same measurement. Both count; the equality
    is recorded informationally.
    """
    payload = _collect(tmp_path, {
        "checkpoint_selection/A.json": ROW,
        "checkpoint_selection/B.json": {**ROW, "checkpoint_id": "checkpoint_B"},
    })
    first = _row(payload, "dcnet", "A.json")
    second = _row(payload, "dcnet", "B.json")
    for row in (first, second):
        assert row["counted"] is True, row
        assert row["duplicate_of"] is None, row
        assert row["metrics"]["r10"] == pytest.approx(12.0), row
    assert first["same_metrics_as"].endswith("B.json"), first
    assert second["same_metrics_as"].endswith("A.json"), second
    assert payload["counts"]["duplicate_rows"] == 0, payload["counts"]


def test_equal_metrics_from_two_files_of_one_checkpoint_also_both_count(tmp_path: Path) -> None:
    """Same checkpoint, two files, no shared backing artifact: still two statements, both counted."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/one.json": dict(ROW),
        "checkpoint_selection/two.json": dict(ROW),
    })
    rows = [row for row in payload["measurements"] if row.get("metrics")]
    assert len(rows) == 2, rows
    assert all(row["counted"] for row in rows), rows
    assert all(row["duplicate_of"] is None for row in rows), rows


def test_a_rollup_restating_an_artifact_is_still_not_counted_twice(tmp_path: Path) -> None:
    """The ancestry rule survives, and it is decided by source KIND, not by equal numbers."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/artifact.json": {"model_id": "fx", "protocol_id": "protocol_A",
                                               "aggregate": {"macro_r10": 30.0}},
        "checkpoint_selection/all_measured_summary.json": {
            "fx": {"macro": {"macro_r10": 30.0}}},
    })
    artifact = _row(payload, "fx", "artifact.json")
    rollup = _row(payload, "fx", "all_measured_summary.json")
    assert artifact["counted"] is True and artifact["duplicate_of"] is None, artifact
    assert rollup["rollup"] is True, rollup
    assert rollup["counted"] is False, rollup
    assert rollup["duplicate_of"].endswith("artifact.json"), rollup


def test_two_exports_of_one_checkpoint_are_one_measurement(tmp_path: Path) -> None:
    """Ancestry the FILE states — same weights digest — is a duplicate; equal scores are not."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/one.json": {"model_id": "fx", "protocol_id": "protocol_A",
                                          "aggregate": {"macro_r10": 30.0},
                                          "checkpoint_sha256": "a" * 64},
        "checkpoint_selection/two.json": {"model_id": "fx", "protocol_id": "protocol_A",
                                          "aggregate": {"macro_r10": 30.0},
                                          "checkpoint_sha256": "a" * 64},
    })
    rows = [row for row in payload["measurements"] if row.get("metrics")]
    assert len(rows) == 2, rows
    assert len([row for row in rows if row["counted"]]) == 1, rows
    assert len([row for row in rows if row["duplicate_of"]]) == 1, rows


def test_two_checkpoints_with_different_digests_and_equal_scores_both_count(
        tmp_path: Path) -> None:
    """The R08 case with the digests present: different weights ⇒ two measurements."""
    payload = _collect(tmp_path, {
        "checkpoint_selection/a.json": {"model_id": "fx", "protocol_id": "protocol_A",
                                        "aggregate": {"macro_r10": 30.0},
                                        "checkpoint_sha256": "a" * 64},
        "checkpoint_selection/b.json": {"model_id": "fx", "protocol_id": "protocol_A",
                                        "aggregate": {"macro_r10": 30.0},
                                        "checkpoint_sha256": "b" * 64},
    })
    rows = [row for row in payload["measurements"] if row.get("metrics")]
    assert all(row["counted"] for row in rows), rows
    assert all(row["duplicate_of"] is None for row in rows), rows


def test_the_real_tree_keeps_equal_metrics_rows_counted() -> None:
    """Snapshot of the real catalog: airknow has two runs at two precisions, and `hint`/`tme` each
    ship two files with the same numbers. None of them may be demoted on value equality."""
    payload = nm.list_native_measurements(ARTIFACTS)
    rows = [row for row in payload["measurements"] if row.get("metrics")]
    counted_airknow = [row for row in rows
                       if row["model_id"] == "airknow" and row["counted"]
                       and not row["rollup"]]
    # R16 changed this from 2 to 3: the ingested `airknow_run3cat_gpu_run1.json` is a THIRD counted
    # measurement (a third run of the `fiq_n05` config, distinct from the two in-tree artifacts), and
    # its three siblings are marked `duplicate_of` rather than counted — see
    # `test_the_airknow_repeats_add_exactly_one_counted_row`.
    assert len(counted_airknow) == 3, counted_airknow
    # `hint` and `tme` each ship two exports of ONE checkpoint (same `checkpoint_sha256`), so
    # exactly one of each pair counts and the other names it as its ancestor.
    for model in ("hint", "tme"):
        pair = [row for row in rows if row["model_id"] == model and not row["rollup"]]
        assert len(pair) == 2, pair
        assert len([row for row in pair if row["counted"]]) == 1, pair
        assert len([row for row in pair if row["duplicate_of"]]) == 1, pair


def test_the_airknow_repeats_add_exactly_one_counted_row() -> None:
    """R16: the four `airknow_run3cat_gpu_run{1..4}` files are draws of ONE config, not four models.

    BEFORE: they were un-ingested (no `replay_type`, wrong directory), so `airknow` served only the
    `_run1` artifacts of two checkpoints. Ingesting them naively would have counted 2 + 4 = 6
    measurements of one model. The repeats carry a draw index in `determinism.repeat_count`, so one
    of them counts and the other three are `duplicate_of` it with a reason.
    """
    payload = nm.list_native_measurements(ARTIFACTS)
    rows = [row for row in payload["measurements"]
            if row["model_id"] == "airknow" and row.get("metrics")]
    repeats = [row for row in rows if row.get("review_repeat_index") is not None]
    assert len(repeats) == 4, repeats
    assert len([row for row in repeats if row["counted"]]) == 1, repeats
    keeper = [row for row in repeats if row["counted"]][0]
    assert keeper["review_repeat_index"] == 1, keeper
    assert keeper["metrics"]["r10"] == pytest.approx(52.924086629607395), keeper
    assert keeper["variance_group"]["n"] == 4, keeper["variance_group"]
    assert keeper["variance_group"]["spread"] == pytest.approx(0.6252866852209422), keeper
    demoted = [row for row in repeats if not row["counted"]]
    assert len(demoted) == 3, demoted
    for row in demoted:
        assert row["duplicate_of"] == keeper["source"]["path"], row
        assert "repeat run" in row["duplicate_reason"], row
        # The number is never lost: it stays readable on the demoted row.
        assert row["metrics"]["r10"] > 0, row
    # One model, three counted rows: two checkpoints of the in-tree tree plus the new run.
    counted = [row for row in rows if row["counted"] and not row["rollup"]]
    assert len(counted) == 3, counted
