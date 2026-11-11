"""R16: the committed native GPU runs under `docs/reviews/` are ingested, and only those.

Seven real runs existed on disk and served nothing, for two independent reasons: they sit one level
above `raw/native/` and carry no `replay_type`. Fixing the glob alone would have been wrong twice
over — `airknow_run3cat_gpu_run{1..4}.json` are four DRAWS of one configuration, so ingesting them
naively counts one model four times (the R08 defect with the sign flipped), and the numbers must
survive the metadata migration untouched.

Every assertion below is against the real artifacts tree, plus two negative controls on a synthetic
one, so a future glob change cannot ingest a file that merely happens to be JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from workbench.backend import native_measurements as nm

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RAW = Path(__file__).resolve().parents[1] / "docs" / "reviews" / "20261111-master-audit" / "raw"

#: The three models this change makes reachable, with the number their own artifact states.
EXPECTED = {
    "tgcir": (49.720772235486656, 71.10231942031965),
    "clvc_net": (30.02079703540087, 58.67189400437203),
    "airknow": (52.924086629607395, 73.0524441812501),
}

def _catalog() -> dict:
    return nm.list_native_measurements(ARTIFACTS)

def _model_rows(payload: dict, model: str) -> list[dict]:
    return [row for row in payload["measurements"]
            if row["model_id"] == model and row.get("metrics")]

def _leaves(node, trail=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, trail + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, trail + (index,))
    else:
        yield trail, node

# ------------------------------------------------------------------ the three models are served

@pytest.mark.parametrize("model,expected", sorted(EXPECTED.items()))
def test_each_previously_unreachable_model_is_served_with_its_real_macro(model, expected) -> None:
    """BEFORE: `_backing_reports` globbed only `raw/native/` and refused a missing `replay_type`, so
    these three served no number even though the runs are on disk."""
    payload = _catalog()
    counted = [row for row in _model_rows(payload, model) if row["counted"]]
    assert counted, f"{model} is still not served: {_model_rows(payload, model)}"
    served = {round(row["metrics"]["r10"], 9) for row in counted}
    assert round(expected[0], 9) in served, (model, sorted(served))
    r50s = {round(row["metrics"]["r50"], 9) for row in counted}
    assert round(expected[1], 9) in r50s, (model, sorted(r50s))
    for row in counted:
        assert row["artifact_backed"] is True, row
        assert row["source"]["sha256"] and len(row["source"]["sha256"]) == 64, row

def test_the_ingested_rows_say_which_glob_they_came_from() -> None:
    """A reader must be able to tell a `raw/` ingest from a `raw/native/` one."""
    payload = _catalog()
    kinds = {row["source"]["path"]: row["source_kind"]
             for row in payload["measurements"] if row["source_kind"] == "review_run"}
    assert kinds, "no row is marked as a review-run ingest"
    for path in kinds:
        assert path.startswith("workbench/docs/reviews/") and "/raw/" in path, path
        assert "/raw/native/" not in path, path
    # The backing-report path still names its own kind, and it is a different one.
    backed = [row for row in payload["measurements"]
              if row.get("backed_by") and row["backed_by"] != row["source"]["path"]]
    assert any(row.get("backed_by_kind") == "native_replay" for row in backed), backed

# ------------------------------------------------------------ airknow: four files, one measurement

def test_airknow_contributes_exactly_one_counted_row_despite_four_files() -> None:
    """BEFORE: the four repeats were invisible. Ingested naively they would be four measurements of
    ONE config, inflating `models_with_metrics` — so one counts and three name it."""
    payload = _catalog()
    repeats = [row for row in _model_rows(payload, "airknow")
               if row.get("review_repeat_index") is not None]
    assert len(repeats) == 4, repeats
    counted = [row for row in repeats if row["counted"]]
    assert len(counted) == 1, counted
    keeper = counted[0]
    assert keeper["review_repeat_index"] == 1, keeper
    assert keeper["variance_group"]["n"] == 4, keeper["variance_group"]
    assert keeper["variance_group"]["mean"] == pytest.approx(52.54676006631348), keeper
    assert keeper["variance_group"]["sd"] == pytest.approx(0.26728186433775697), keeper
    assert keeper["variance_group"]["spread"] == pytest.approx(0.6252866852209422), keeper
    for row in repeats:
        if row is keeper:
            continue
        assert row["counted"] is False, row
        assert row["duplicate_of"] == keeper["source"]["path"], row
        assert "repeat run" in row["duplicate_reason"], row
        assert "nondeterministic" in row["duplicate_reason"], row
        # The numbers of a demoted draw are KEPT, not erased: evidence is not thrown away.
        assert row["metrics"]["r10"] > 0, row

def test_the_airknow_row_takes_its_checkpoint_from_the_depth_the_file_states_it() -> None:
    """The file states `fiq_n05` per category. Reading only the document root left the row saying
    "no checkpoint", which is what would let a `fiq_n08` report vouch for it (the R01 bypass)."""
    payload = _catalog()
    repeats = [row for row in _model_rows(payload, "airknow")
               if row.get("review_repeat_index") is not None]
    assert {row["checkpoint_id"] for row in repeats} == {"fiq_n05"}, repeats

# ------------------------------------------------------------------ the negative controls

def test_a_raw_run_without_replay_type_or_without_aggregate_is_not_ingested(tmp_path: Path) -> None:
    """The predicate, both halves. Being JSON is not ingestion."""
    raw = tmp_path / "docs" / "reviews" / "R1" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "checkpoint_selection").mkdir()
    (raw / "no_replay_type.json").write_text(json.dumps({
        "model_id": "fx", "protocol": "fashioniq_val_split",
        "aggregate": {"macro_r10": 11.0, "macro_r50": 22.0}}))
    (raw / "no_aggregate.json").write_text(json.dumps({
        "model_id": "fx", "protocol": "fashioniq_val_split",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "results": [{"category": "dress", "r10": 11.0}]}))
    (raw / "not_a_measurement.json").write_text(json.dumps({"verdict": "BLOCKED"}))
    payload = nm.list_native_measurements(tmp_path)
    assert payload["measurements"] == [], payload["measurements"]
    assert payload["counts"]["with_metrics"] == 0, payload["counts"]

def test_a_raw_run_satisfying_the_contract_is_ingested_as_its_own_row(tmp_path: Path) -> None:
    """The positive control for the same predicate, so the test above is not passing for the wrong
    reason (e.g. a path that never gets scanned at all)."""
    raw = tmp_path / "docs" / "reviews" / "R1" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "checkpoint_selection").mkdir()
    (raw / "good.json").write_text(json.dumps({
        "model_id": "fx", "protocol": "fashioniq_val_split",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "aggregate": {"macro_r10": 11.0, "macro_r50": 22.0},
        "recorded_at": "2026-11-10T00:00:00+00:00"}))
    payload = nm.list_native_measurements(tmp_path)
    rows = [row for row in payload["measurements"] if row.get("metrics")]
    assert len(rows) == 1, rows
    assert rows[0]["source_kind"] == "review_run", rows[0]
    assert rows[0]["metrics"]["r10"] == pytest.approx(11.0), rows[0]
    assert payload["counts"]["with_metrics"] == 1, payload["counts"]

def test_a_file_that_states_it_reemits_another_is_not_a_second_measurement(tmp_path: Path) -> None:
    """`clip4cir_fullft.migrated.json` declares `migrated_from.artifact_sha256`, and that digest is
    already on disk: it relabels one run, it does not add one."""
    selection = tmp_path / "checkpoint_selection"
    selection.mkdir()
    raw = tmp_path / "docs" / "reviews" / "R1" / "raw"
    original = raw / "native" / "original.json"
    original.parent.mkdir(parents=True)
    document = {"model_id": "fx", "protocol": "fashioniq_val_split",
                "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
                "aggregate": {"macro_r10": 11.0, "macro_r50": 22.0}}
    original.write_text(json.dumps(document))
    (selection / "fx.json").write_text(json.dumps({
        "model_id": "fx", "protocol": "fashioniq_val_split",
        "aggregate": {"macro_r10": 11.0, "macro_r50": 22.0}}))
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    (raw / "relabelled.json").write_text(json.dumps({
        **document, "protocol": "fashioniq_original_split",
        "migrated_from": {"artifact_sha256": digest}}))
    payload = nm.list_native_measurements(tmp_path)
    paths = [row["source"]["path"] for row in payload["measurements"] if row.get("metrics")]
    # One row — the in-tree artifact the replay backs — not two.
    assert len(paths) == 1, paths
    assert paths[0].endswith("checkpoint_selection/fx.json"), paths

# --------------------------------------------------- the migration changed metadata, not numbers

@pytest.mark.parametrize("name", sorted(p.name[:-len(".pre-r16")] for p in RAW.glob("*.pre-r16")))
def test_the_migrated_artifact_keeps_every_number_byte_identical(name) -> None:
    """The pre-r16 copy is the evidence. Every r1/r10/r50 leaf and the whole aggregate must be equal
    to it, and the ONLY leaves allowed to differ are the metadata keys this change added."""
    before = json.loads((RAW / f"{name}.pre-r16").read_text())
    after = json.loads((RAW / name).read_text())
    first, second = dict(_leaves(before)), dict(_leaves(after))
    changed = [trail for trail in set(first) & set(second) if first[trail] != second[trail]]
    assert changed == [], (name, changed)
    added = {trail for trail in set(second) - set(first)}
    assert added <= {("replay_type",), ("checkpoint_id",)}, (name, sorted(added))
    # The aggregate is compared as a whole, so a reshaped aggregate cannot hide behind equal leaves.
    assert before.get("aggregate") == after.get("aggregate"), name
    for key in ("results", "categories"):
        for old, new in zip(before.get(key) or [], after.get(key) or []):
            for field in ("r1", "r10", "r50"):
                assert old.get(field) == new.get(field), (name, key, field)
