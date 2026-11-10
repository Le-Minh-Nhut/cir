"""D1: the native-measurement census must read every shape the tree actually uses.

The first version of `native_measurements.py` accepted only `aggregate.{macro_r10,macro_r50,macro_mean}`
under two globs, and that is the census-v1 mistake: it reported a subset of the measurements and called
it the census. This file pins every shape found in the tree, each with the assertion that would have
failed before, plus the two rules that keep a roll-up from masquerading as a measurement.

The shapes are taken from the files, not invented: `macro.macro_r10` (pair_compat measurements),
`macro_metrics` (ptha_transfer_run1), top-level `macro_r10` (csmcir/measurement.json), and
`<variant>.macro_r10` roll-ups (clip4cir_parsed, all_measured_summary, selection_decisions).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend import native_measurements as nm

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _catalog() -> dict:
    return nm.list_native_measurements(ARTIFACTS)


def _rows_with_metrics(payload: dict) -> list[dict]:
    return [row for row in payload["measurements"] if row.get("metrics")]


def _for_model(payload: dict, model: str) -> list[dict]:
    return [row for row in _rows_with_metrics(payload) if row["model_id"] == model]


# --------------------------------------------------------------------------- each shape

def test_aggregate_block_is_read() -> None:
    payload = _catalog()
    conesep = _for_model(payload, "conesep")
    assert conesep, "no conesep row: the `aggregate.macro_r10` shape is not read"
    row = conesep[0]
    assert row["metric_path"] == "aggregate", row
    assert row["metrics"]["r10"] == pytest.approx(54.76130615726771)
    assert row["protocol_id"] == "fashioniq_full_gallery_ref_excluded", row
    assert row["rollup"] is False, row


def test_macro_block_is_read() -> None:
    """`macro.macro_r10` — the pair_compat measurements state the triple under `macro`."""
    payload = _catalog()
    pair = _for_model(payload, "pair")
    assert pair, "no pair row: the `macro.macro_r10` shape is not read"
    assert {row["metric_path"] for row in pair} == {"macro"}, pair
    assert pair[0]["metrics"]["r10"] == pytest.approx(34.03503453772374), pair[0]
    assert pair[0]["protocol_id"] == "fashioniq_val_split", pair[0]


def test_macro_metrics_block_is_read() -> None:
    """`macro_metrics.{r10,r50,mean}` — bare names under a macro-shaped key.

    This is the shape that a top-level-only reader misses AND a permissive reader over-reads: the same
    permissiveness makes `all_measured_summary`'s per-category blocks look like macro measurements.
    """
    payload = _catalog()
    ptha = [row for row in _for_model(payload, "ptha_mtst") if row["rollup"] is False]
    assert len(ptha) == 1, ptha
    assert ptha[0]["metric_path"] == "macro_metrics", ptha[0]
    assert ptha[0]["metrics"]["r10"] == pytest.approx(37.897844115893044), ptha[0]
    assert ptha[0]["protocol_id"] == "fashioniq_original_split", ptha[0]
    assert ptha[0]["source"]["path"].endswith("ptha_transfer_run1.json"), ptha[0]


def test_top_level_macro_is_read() -> None:
    """`macro_r10` at the document root — csmcir/measurement.json (57.0507)."""
    payload = _catalog()
    rows = [row for row in _for_model(payload, "csmcir") if row["metrics"]]
    assert rows, "no csmcir row with metrics: the top-level shape is not read"
    row = rows[0]
    assert row["metric_path"] == "(top level)", row
    assert row["metrics"]["r10"] == pytest.approx(57.05073873202006), row
    assert row["source"]["path"].endswith("csmcir/measurement.json"), row


def test_a_rollup_is_served_as_a_rollup_not_as_one_model() -> None:
    """`clip4cir_parsed.json` carries two variants in one file; they must not collapse into one."""
    payload = _catalog()
    noft = _for_model(payload, "clip4cir_noft")
    fullft = _for_model(payload, "clip4cir_fullft")
    assert noft and fullft, "clip4cir variants are missing"
    assert noft[0]["metrics"]["r10"] == pytest.approx(35.14068226019541), noft[0]
    assert fullft[0]["metrics"]["r10"] == pytest.approx(43.81821850935618), fullft[0]
    assert noft[0]["rollup"] and fullft[0]["rollup"], (noft[0], fullft[0])
    assert noft[0]["rollup_entry"] == "clip4cir_noft", noft[0]
    assert fullft[0]["rollup_entry"] == "clip4cir_fullft", fullft[0]
    # Both name the same file, and neither is counted as a per-model measurement.
    assert {row["source"]["path"] for row in noft + fullft} == {
        "checkpoint_selection/20261106T002617+0700/measurements/clip4cir_parsed.json"}
    per_model_models = {row["model_id"] for row in _rows_with_metrics(payload)
                        if not row["rollup"]}
    assert "clip4cir_fullft" not in per_model_models, per_model_models


def test_a_rollup_is_exposed_and_named_as_one() -> None:
    payload = _catalog()
    assert payload["rollup_sources"], "no roll-up source was identified"
    for name in ("all_measured_summary.json", "selection_decisions.json",
                 "clip4cir_parsed.json", "pair_compat/measurements/summary.json"):
        assert any(path.endswith(name) for path in payload["rollup_sources"]), (name,
                                                                               payload["rollup_sources"])
    assert payload["counts"]["rollup_rows"] > 0, payload["counts"]


def test_a_rollup_entry_is_not_double_counted_as_a_second_measurement() -> None:
    """`all_measured_summary.json` repeats conesep/clvc/tgcir numbers other files state.

    Those files are individual measurements; the summary is a roll-up. Neither the counts nor the
    per-model set may treat the pair as two measurements of one model.
    """
    payload = _catalog()
    direct = [row for row in _rows_with_metrics(payload) if not row["rollup"]]
    rollups = [row for row in _rows_with_metrics(payload) if row["rollup"]]
    assert payload["counts"]["with_metrics"] == len(direct), payload["counts"]
    assert payload["counts"]["rollup_rows"] == len(rollups), payload["counts"]
    # Every roll-up row either repeats a number a direct row already states (roll-ups round to four
    # decimals, hence the tolerance) or names a model that has no direct file at all. Nothing may be a
    # third case: a roll-up is not a second measurement of a model that already has one.
    direct_pairs = {(row["model_id"], row["metrics"]["r10"]) for row in direct}
    direct_models = {model for model, _value in direct_pairs}
    for row in rollups:
        overlaps = any(model == row["model_id"] and abs(row["metrics"]["r10"] - value) < 5e-3
                       for model, value in direct_pairs)
        assert overlaps or row["model_id"] not in direct_models, row


# --------------------------------------------------------------------------- the census itself

def test_every_active_model_with_a_macro_anywhere_has_a_row() -> None:
    """The five models the consolidator named, each reachable with its number and its provenance."""
    payload = _catalog()
    models = {row["model_id"] for row in _rows_with_metrics(payload)}
    for required in ("ptha_mtst", "pair", "clip4cir_rn50x4_fullft", "combiner_rn50x4_noft", "csmcir"):
        assert required in models, f"{required} has a macro somewhere but no row: {sorted(models)}"
    for required in ("clip4cir_rn50x4_fullft", "combiner_rn50x4_noft"):
        row = _for_model(payload, required)[0]
        # These two exist ONLY in a roll-up; that is exactly why the roll-up is served.
        assert row["rollup"] is True, row
        assert row["metrics"]["r10"] > 0, row
    # `pair` and `ptha_mtst` have their own files, and those are NOT roll-ups.
    assert all(row["rollup"] is False for row in _for_model(payload, "pair")), _for_model(payload, "pair")
    assert all(row["rollup"] is False for row in _for_model(payload, "ptha_mtst")), _for_model(payload, "ptha_mtst")


def test_every_row_is_unverified_with_a_path_and_a_digest() -> None:
    payload = _catalog()
    assert payload["verified"] is False
    assert payload["counts"]["verified"] == 0, payload["counts"]
    for row in payload["measurements"]:
        assert row["verified"] is False, row
        assert row["verification"]["certified"] is False, row
        assert row["verification"]["is_research_evidence"] is False, row
        assert row["verification"]["state"] == "UNVERIFIED_NATIVE_MEASUREMENT", row
        assert row["source"]["path"], row
        assert row["source"]["sha256"] is None or len(row["source"]["sha256"]) == 64, row
    for row in _rows_with_metrics(payload):
        assert len(row["source"]["sha256"]) == 64, row


def test_the_same_model_can_hold_more_than_one_measurement_and_both_survive() -> None:
    payload = _catalog()
    for model in ("airknow", "hint", "tme", "pair"):
        rows = _for_model(payload, model)
        assert len(rows) >= 2, (model, rows)


def test_a_paper_reference_block_is_not_a_measurement() -> None:
    """`dcnet_run1.paper` and `pair_*.paper_reference` are the publication's numbers."""
    payload = _catalog()
    for row in _rows_with_metrics(payload):
        lowered = row["metric_path"].lower()
        assert "paper" not in lowered, row
        assert "reported" not in lowered, row


def test_per_category_blocks_are_not_macro_measurements() -> None:
    """`all_measured_summary.json` has `dress`/`shirt`/`toptee` r10s; none is a macro."""
    payload = _catalog()
    for row in _rows_with_metrics(payload):
        if row["source"]["path"].endswith("all_measured_summary.json"):
            assert row["metric_path"].endswith(".macro"), row
            assert row["metrics"]["r10"] > 20, row


def test_an_unreadable_file_is_a_named_absence_not_a_zero() -> None:
    payload = _catalog()
    unreadable = [row for row in payload["measurements"] if row.get("unreadable")]
    assert unreadable, "the empty measurement files are no longer reported"
    for row in unreadable:
        assert row["metrics"] is None, row
        assert "JSONDecodeError" in row["unreadable"], row


def test_a_blocker_file_is_listed_and_is_not_a_number() -> None:
    payload = _catalog()
    blockers = [row for row in payload["measurements"]
                if row.get("blocker") and not row.get("metrics")]
    names = {row["model_id"] for row in blockers}
    assert {"csmcir", "pair", "ptha_mtst", "tme"} <= names, names


def test_the_catalog_never_reaches_outside_the_artifacts_root() -> None:
    payload = _catalog()
    for row in payload["measurements"]:
        path = row["source"]["path"]
        assert not path.startswith("/"), row
        assert "site-packages" not in path and "/downloads/" not in path, row
        assert path.startswith("checkpoint_selection/"), row


def test_the_endpoint_serves_the_whole_catalog(tmp_path: Path) -> None:
    """The route is a pass-through with no filtering of its own, so it inherits every guarantee."""
    from fastapi.testclient import TestClient

    from workbench.backend.main import app

    with TestClient(app) as client:
        response = client.get("/api/native-measurements")
    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is False
    models = {row["model_id"] for row in payload["measurements"]
              if row.get("metrics")}
    for required in ("ptha_mtst", "pair", "clip4cir_rn50x4_fullft", "combiner_rn50x4_noft", "csmcir"):
        assert required in models, sorted(models)
    assert payload["counts"]["verified"] == 0
    # A settlement check: the JSON round-trips, so nothing un-serializable is in a row.
    assert json.loads(response.text)["counts"] == payload["counts"]
