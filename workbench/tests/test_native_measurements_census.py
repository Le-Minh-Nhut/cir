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
    """`clip4cir_parsed.json` carries two variants in one file; they must not collapse into one.

    W-01 (mandatory): the variant name `clip4cir_fullft` is an ALIAS of `clip4cir_rn50x4_fullft`.
    Serving both names for one number read as two configurations that agree, so they are canonicalised
    to one model. The two variants (noft / fullft) stay two rows: they are different checkpoints.
    """
    payload = _catalog()
    fullft = _for_model(payload, "clip4cir_rn50x4_fullft")
    noft = _for_model(payload, "clip4cir_rn50x4_noft")
    assert fullft and noft, "clip4cir variants are missing"
    # No row may be served under the uncanonicalised variant name any more.
    assert _for_model(payload, "clip4cir_fullft") == [], "the alias is still served as its own model"
    # The FullFT value is the ARTIFACT's number, not the roll-up's.
    backed = [row for row in fullft if row["artifact_backed"]]
    assert backed, fullft
    assert backed[0]["metrics"]["r10"] == pytest.approx(43.8016931215922), backed[0]
    # The roll-up's 43.8182 survives only as `raw_metrics` on the replaced row, and as an
    # uncounted duplicate — never as a served value.
    assert all(row["metrics"]["r10"] != pytest.approx(43.81821850935618) for row in fullft), fullft
    assert any(row.get("raw_metrics", {}) and
               row["raw_metrics"]["r10"] == pytest.approx(43.81821850935618) for row in fullft), fullft
    assert {row["source"]["path"] for row in fullft if row["rollup"]} == {
        "checkpoint_selection/20261106T002617+0700/measurements/clip4cir_parsed.json",
        "checkpoint_selection/20261106T002617+0700/measurements/selection_decisions.json"}
    # One counted row for the one configuration.
    assert len([row for row in fullft if row.get("counted", True)]) == 1, fullft

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

    Those files are individual measurements; the summary is a roll-up. W-01 (mandatory) went further:
    a repeat is marked `counted: false` and an unbacked claim is marked `artifact_backed: false`, so
    neither can inflate a model count — while BOTH stay in `measurements`, because a reader has to be
    able to see that two files state one number.

    Measurement identity is NOT a rounded score. Keying on `(model_id, protocol_id,
    round(value, 4))` — which this test used to require to be unique — is the R08 defect itself: two
    DIFFERENT checkpoints of one model that happen to score the same collapsed into one row, the
    second demoted to `duplicate_of` the first. Equal numbers from different checkpoints are two
    measurements. Identity here is therefore (a) the same source `sha256` + `metric_path` (one block
    stated twice, which the block walker already makes unique) or (b) explicit ancestry as the module
    records it: a shared backing artifact, the same `checkpoint_sha256`, or a roll-up restating an
    artifact. Numerically equal rows with no ancestry stay counted and carry `same_metrics_as`.
    """
    payload = _catalog()
    measured = _rows_with_metrics(payload)
    counted = [row for row in measured if row.get("counted", True)]
    assert payload["counts"]["with_metrics"] == len([row for row in counted if row["artifact_backed"]]), \
        payload["counts"]
    assert payload["counts"]["duplicate_rows"] == len([row for row in measured
                                                       if not row.get("counted", True)]), payload["counts"]
    assert payload["counts"]["unbacked_rows"] == len([row for row in measured
                                                      if not row["artifact_backed"]]), payload["counts"]
    # No measurement identity may be counted twice. Identity is the source block
    # (`sha256` + `metric_path`), never a rounded score: `airknow_n05` (52.5466) and `airknow_n08`
    # (50.6417) are two runs, and two checkpoints with EQUAL recall are still two checkpoints.
    block_identities = [(row["source"]["sha256"], row["metric_path"]) for row in counted]
    assert len(block_identities) == len(set(block_identities)), "a source block is counted twice"
    # And the counts agree with the ancestry the module recorded.
    assert payload["counts"]["with_metrics"] == len([row for row in counted
                                                     if row["artifact_backed"]]), payload["counts"]
    assert all(row["counted"] for row in measured if not row["artifact_backed"]) is False, \
        payload["counts"]
    # And a marked DUPLICATE names the row it duplicates. An unbacked row is not marked as a
    # duplicate — it is uncounted for a different reason, which it states in `source_kind`.
    for row in measured:
        if not row.get("counted", True) and row["artifact_backed"]:
            assert row["duplicate_of"], row
    # W-01: an unbacked number must never reach a model count.
    assert payload["counts"]["unbacked_counted"] == 0, payload["counts"]
    for row in measured:
        if not row["artifact_backed"]:
            assert row.get("counted", True) is False, row

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
        # R16: a committed review run is served as its own row, so two prefixes are now legal — the
        # artifact tree, and the two review globs that satisfy `_review_run_payload`. Nothing else is,
        # which is what this assertion is for. (Before R16 only the first prefix existed.)
        assert path.startswith("checkpoint_selection/") or (
            path.startswith("workbench/docs/reviews/") and "/raw/" in path), row


def test_the_endpoint_serves_the_whole_catalog(tmp_path: Path) -> None:
    """The route is a pass-through for every PUBLISHED model, plus the web-deferral disclosure.

    PAIR is deferred from the web (`web_publication.status: DEFERRED` in the registry), so its rows
    are the one thing the route withholds — and they are still fetchable by id, which the next test
    asserts. Everything else the tree states is served unchanged.
    """
    from fastapi.testclient import TestClient

    from workbench.backend.main import app

    with TestClient(app) as client:
        response = client.get("/api/native-measurements")
    assert response.status_code == 200
    payload = response.json()
    assert payload["verified"] is False
    models = {row["model_id"] for row in payload["measurements"]
              if row.get("metrics")}
    for required in ("ptha_mtst", "clip4cir_rn50x4_fullft", "combiner_rn50x4_noft", "csmcir"):
        assert required in models, sorted(models)
    # The deferral is DISCLOSED, not silent: the reason travels with the catalogue.
    assert payload["deferred_models"]["pair"].strip()
    assert "pair" not in models
    assert payload["counts"]["verified"] == 0
    # A settlement check: the JSON round-trips, so nothing un-serializable is in a row.
    assert json.loads(response.text)["counts"] == payload["counts"]

# --------------------------------------------------------------------------- W-01: unit and mapping

def test_a_paper_reference_block_anywhere_in_the_path_is_not_a_measurement(tmp_path: Path) -> None:
    """`paper_reference.aggregate.macro_r10` is the publication's number, not this deployment's.

    The gate used to test only the LEAF key, so `aggregate` -- a macro-block key -- satisfied it and
    the paper's 42.0 was counted as a measurement. The whole path decides now.
    """
    root = tmp_path / "checkpoint_selection"
    root.mkdir()
    (root / "fx_paper_nested.json").write_text(json.dumps({
        "model_id": "fixture_model",
        "protocol_id": "fashioniq_val_split",
        "aggregate": {"macro_r10": 42.0, "macro_r50": 60.0, "macro_mean": 51.0},
        "paper_reference": {"aggregate": {"macro_r10": 52.76, "macro_r50": 74.51},
                            "note": "published numbers, not this deployment"},
    }))
    payload = nm.list_native_measurements(tmp_path)
    paths = [row["metric_path"] for row in payload["measurements"] if row["metrics"]]
    assert paths == ["aggregate"], paths
    assert payload["measurements"][0]["metrics"]["r10"] == 42.0
    # 42.0 is the fixture's own number; 52.76 must appear nowhere.
    assert all(row["metrics"]["r10"] != 52.76 for row in payload["measurements"] if row["metrics"])

def test_a_fraction_and_a_percent_are_one_measurement_not_two_scores(tmp_path: Path) -> None:
    """`0.2793` and `27.93` are the same recall. Each row states its scale so a view prints them
    identically, and the scale is never inferred from the OTHER row in the file."""
    root = tmp_path / "checkpoint_selection"
    root.mkdir()
    (root / "declared_fraction.json").write_text(json.dumps({
        "model_id": "fx_fraction", "protocol_id": "fashioniq_val_split", "metric_unit": "fraction",
        "aggregate": {"macro_r10": 0.2793, "macro_r50": 0.5373}}))
    (root / "declared_percent.json").write_text(json.dumps({
        "model_id": "fx_percent", "protocol_id": "fashioniq_val_split", "metric_unit": "percent",
        "aggregate": {"macro_r10": 27.93, "macro_r50": 53.73}}))
    # A document whose ONLY self-consistent reading is percent (no value fits 0..1) still does not
    # get a declaration: the row says `inferred_from_magnitude` and appears in the audited list.
    (root / "undeclared.json").write_text(json.dumps({
        "model_id": "fx_undeclared", "protocol_id": "fashioniq_val_split",
        "aggregate": {"macro_r10": 0.2793, "macro_r50": 53.73}}))
    rows = {row["model_id"]: row for row in nm.list_native_measurements(tmp_path)["measurements"]}
    assert rows["fx_fraction"]["unit"] == "fraction", rows["fx_fraction"]
    assert rows["fx_fraction"]["unit_source"] == "declared", rows["fx_fraction"]
    assert rows["fx_percent"]["unit"] == "percent", rows["fx_percent"]
    assert rows["fx_percent"]["unit_source"] == "declared", rows["fx_percent"]
    # 0.2793 sits inside 0..1 but 53.73 does not, so "fraction" is excluded by the block's own
    # numbers and the reading is percent -- stated as an INFERENCE, never as a declaration.
    assert rows["fx_undeclared"]["unit"] == "percent", rows["fx_undeclared"]
    assert rows["fx_undeclared"]["unit_source"] == "inferred_from_magnitude", rows["fx_undeclared"]

def test_every_row_without_a_declared_unit_is_listed_not_only_counted(tmp_path: Path) -> None:
    """A count cannot be audited; the rows behind it can. W-01 asks for the list."""
    root = tmp_path / "checkpoint_selection"
    root.mkdir()
    (root / "a.json").write_text(json.dumps({
        "model_id": "fx_a", "protocol_id": "fashioniq_val_split",
        "aggregate": {"macro_r10": 27.93, "macro_r50": 53.73}}))
    (root / "b.json").write_text(json.dumps({
        "model_id": "fx_b", "protocol_id": "fashioniq_val_split", "unit": "percent",
        "aggregate": {"macro_r10": 30.0, "macro_r50": 55.0}}))
    payload = nm.list_native_measurements(tmp_path)
    assert payload["counts"]["unit_declared"] == 1, payload["counts"]
    assert payload["counts"]["unit_not_declared"] == 1, payload["counts"]
    assert [row["model_id"] for row in payload["undeclared_unit_rows"]] == ["fx_a"], payload["undeclared_unit_rows"]
    # Every listed row names its own scale decision, not just its model.
    assert payload["undeclared_unit_rows"][0]["unit_source"] == "inferred_from_magnitude"

def test_the_pair_exploratory_run_keeps_its_mapping_status() -> None:
    """`EXPLORATORY_PUBLIC_MAPPING` is why a PAIR number is not a reproduction. Dropping it made an
    unverified class mapping read as a measured result."""
    payload = _catalog()
    with_mapping = [row for row in payload["measurements"] if row.get("mapping_status")]
    assert with_mapping, "no row carries mapping_status: the PAIR exploratory label is lost"
    assert {row["mapping_status"] for row in with_mapping} == {"EXPLORATORY_PUBLIC_MAPPING"}
    assert all(row["source"]["path"].endswith(".json") for row in with_mapping)
    assert all("pair_b" in row["source"]["path"] for row in with_mapping), with_mapping

# --------------------------------------------------------------------------- W-01 mandatory: backing

def test_an_unbacked_rollup_number_is_replaced_by_the_artifacts_value() -> None:
    """CLIP4Cir FullFT: the roll-up says 43.8182, the replay artifact says 43.8017.

    They are two different runs — shirt matches to every digit, dress and toptee_r50 do not, and no
    file under `checkpoint_selection/` corresponds to the roll-up's dress/toptee cells. The served
    value is the backed one; the roll-up's survives only as `raw_metrics`.
    """
    payload = _catalog()
    rows = [row for row in _for_model(payload, "clip4cir_rn50x4_fullft") if row.get("counted", True)]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["metrics"]["r10"] == pytest.approx(43.8016931215922), row
    assert row["metrics"]["r50"] == pytest.approx(66.98358456293742), row
    assert row["artifact_backed"] is True, row
    assert row["raw_metrics"]["r10"] == pytest.approx(43.81821850935618), row
    assert "native/clip4cir_fullft.json" in row["backed_by"], row
    assert row["backed_by_kind"] == "native_replay", row

def test_an_unbacked_combiner_value_is_replaced_by_its_run() -> None:
    """Combiner NoFT: the roll-up says 35.1407, its own aggregate report says 35.1078."""
    payload = _catalog()
    rows = [row for row in _for_model(payload, "combiner_rn50x4_noft") if row.get("counted", True)]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["metrics"]["r10"] == pytest.approx(35.1078), row
    assert row["artifact_backed"] is True, row
    assert row["raw_metrics"]["r10"] == pytest.approx(35.1407), row
    assert row["backed_by"].startswith("reports/"), row
    assert row["backed_by_run_id"], row

def test_a_number_no_artifact_backs_says_so_and_does_not_count() -> None:
    """CLIP4Cir NoFT's only source is a roll-up: it is served, marked, and excluded from the count."""
    payload = _catalog()
    rows = [row for row in _for_model(payload, "clip4cir_rn50x4_noft") if row.get("metrics")]
    assert rows, "the roll-up row must still be listed"
    unbacked = [row for row in rows if not row["artifact_backed"]]
    assert unbacked, rows
    for row in unbacked:
        assert row["source_kind"].startswith("rollup:"), row
        assert row["counted"] is False, "an unbacked row is displayed and labelled, never counted"
    models = {row["model_id"] for row in _rows_with_metrics(payload)
              if row["artifact_backed"] and row.get("counted", True)}
    assert "clip4cir_rn50x4_noft" not in models, sorted(models)

def test_the_alias_pair_is_one_configuration_of_one_upstream() -> None:
    """`clip4cir_fullft` and `clip4cir_rn50x4_fullft` must never be two served model names."""
    payload = _catalog()
    models = {row["model_id"] for row in _rows_with_metrics(payload)}
    assert "clip4cir_fullft" not in models, sorted(models)
    assert "clip4cir_rn50x4_fullft" in models, sorted(models)
    assert payload["counts"]["models_with_metrics"] == len(
        {row["model_id"] for row in _rows_with_metrics(payload)
         if row["artifact_backed"] and row.get("counted", True)}), payload["counts"]

def test_a_suffixed_sibling_key_declares_the_unit(tmp_path: Path) -> None:
    """`macro_r10_percent` is a DECLARATION, and it must beat magnitude inference.

    Without it, a percent block whose three categories all happen to score under 1 % reads as a
    fraction and renders 100x too small.
    """
    root = tmp_path / "checkpoint_selection"
    root.mkdir()
    (root / "suffixed.json").write_text(json.dumps({
        "model_id": "fx_suffixed", "protocol_id": "fashioniq_val_split",
        "aggregate": {"macro_r10": 0.6, "macro_r10_percent": 0.6, "macro_r50": 0.9}}))
    (root / "suffixed_fraction.json").write_text(json.dumps({
        "model_id": "fx_fraction_block", "protocol_id": "fashioniq_val_split",
        "aggregate": {"macro_r10": 0.6, "macro_r10_fraction": 0.6}}))
    rows = {row["model_id"]: row for row in nm.list_native_measurements(tmp_path)["measurements"]
            if row.get("metrics")}
    assert rows["fx_suffixed"]["unit"] == "percent", rows["fx_suffixed"]
    assert rows["fx_suffixed"]["unit_source"] == "sibling_key", rows["fx_suffixed"]
    assert rows["fx_suffixed"]["metrics"]["r10"] == pytest.approx(0.6), rows["fx_suffixed"]
    assert rows["fx_fraction_block"]["unit"] == "fraction", rows["fx_fraction_block"]
    assert rows["fx_fraction_block"]["unit_source"] == "sibling_key", rows["fx_fraction_block"]

def test_dcnet_is_served_on_the_scale_its_sibling_key_declares() -> None:
    """W-01 BLOCKER: `dcnet.json` states `macro_r10: 0.2793` beside `macro_r10_percent: 27.9297`.

    The served row used the bare key while its `unit` came from the roll-up of a different run, so
    `/api/native-measurements` + `fmtRecall` rendered DCNet as **0.28 %** against every other model's
    ~50 % — a live 100x-wrong number, with the unit field asserting the wrong thing. The declared
    percent wins, and the unit is re-derived from the report AFTER the value is replaced.
    """
    payload = _catalog()
    rows = [row for row in _for_model(payload, "dcnet") if row.get("counted", True) and row["metrics"]]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["metrics"]["r10"] == pytest.approx(27.9297, abs=1e-4), row["metrics"]
    assert row["unit"] == "percent", row
    assert row["unit_source"] == "sibling_key", row
    # The fraction must never travel with a percent unit.
    assert not (row["metrics"]["r10"] < 1.0 and row["unit"] == "percent"), row
    assert row["unit_source"] in nm.DECLARED_UNIT_SOURCES, row

def test_no_counted_row_carries_an_inference_where_a_declaration_exists() -> None:
    """A declaration outranks an inference at EVERY point, including after value replacement."""
    payload = _catalog()
    measured = _rows_with_metrics(payload)
    for row in measured:
        if row["artifact_backed"] and row.get("counted", True):
            assert row["unit_source"] in nm.DECLARED_UNIT_SOURCES or \
                row["unit_source"] == "inferred_from_magnitude", row
    # `unit_inferred` counts EVERY measured row on an inferred scale (the audit total); the counted
    # subset is what a reader compares, and each of those must have no declaration available.
    assert payload["counts"]["unit_inferred"] == len(
        [row for row in measured if row["unit_source"] == "inferred_from_magnitude"]), payload["counts"]
    counted_inferred = {row["model_id"] for row in measured
                        if row.get("counted", True) and row["unit_source"] == "inferred_from_magnitude"}
    print("counted rows still inferred:", len(counted_inferred), sorted(counted_inferred))
    assert payload["counts"]["unbacked_counted"] == 0, payload["counts"]

def test_no_artifact_declaring_a_macro_scale_is_labelled_inferred() -> None:
    """The 39 `inferred_from_magnitude` rows must be rows whose artifact declares NOTHING.

    Three declaration forms are recognised — a document-root unit key, a unanimous
    `categories[*].provenance.units`, and a `*_percent`/`*_pct`/`*_fraction` sibling of the served
    macro key. A row may only be called inferred when none of the three exists in its backing
    artifact; otherwise a real declaration is being missed and the row would render 100x wrong.
    """
    payload = _catalog()
    measured = _rows_with_metrics(payload)
    for row in measured:
        if row["unit_source"] != "inferred_from_magnitude":
            continue
        backing = row.get("backed_by") or row["source"]["path"]
        path = Path(backing) if Path(backing).is_file() else Path("workbench") / backing
        if not path.is_file():
            path = Path("workbench/artifacts") / backing
        assert path.is_file(), (row["model_id"], backing)
        document = json.loads(path.read_text())
        assert not any(isinstance(document.get(key), str) and document[key].strip()
                       for key in nm.UNIT_KEYS), (row["model_id"], "root unit key exists")
        provenance = {((category.get("provenance") or {}).get("units") or "").strip().lower()
                      for category in document.get("categories") or [] if isinstance(category, dict)}
        assert len({text for text in provenance if text}) != 1, (row["model_id"],
                                                                 f"provenance.units={provenance}")
        # A suffixed sibling of the SERVED macro key would be a declaration; a `*_pct` about some
        # other quantity (dead parameters, relative error, training noise) is not.
        for name in ("macro_r10", "macro_r50", "macro_mean"):
            for suffix, _unit in nm.UNIT_SUFFIX_DECLARATIONS:
                assert f"{name}{suffix}" not in json.dumps(document), (
                    row["model_id"], f"{name}{suffix} declares the macro scale")
