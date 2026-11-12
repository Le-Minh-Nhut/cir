"""A CIRCUS composition cohort must save, reopen and export -- and refuse what it cannot describe.

Before this, `PUT /api/cohorts` answered `run_not_found` for every run the UI lists: cohorts were
describable only in the per-query DuckDB index, and the CIRCUS release runs that this deployment
actually serves live in `artifacts/composition`, not in that index. So the one dataset with real
numbers was the one whose cohorts could not be saved
(`raw/verify/100-api-cache-and-ui-blocked.md` §9).

A composition selection is spelled `c:<run_id>`. The prefix is the disambiguator, so every existing
(bare-id, DuckDB) cohort keeps loading through the original arm -- asserted here, not assumed.

Runs against the real artifacts on this host, skipping cleanly when they are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend import main
from workbench.backend import storage

TGCR_FULL = "7b1fa85007bffdab931f7f19"


def _available(run_id: str) -> bool:
    try:
        storage._composition_rank_artifact(f"c:{run_id}")
    except Exception:
        return False
    return True


needs_tgcir = pytest.mark.skipif(not _available(TGCR_FULL),
                                 reason="the tgcir Full rank artifact is not on this host")


@pytest.fixture()
def cohorts_file(tmp_path, monkeypatch):
    path = tmp_path / "cohorts.json"
    monkeypatch.setattr(main, "COHORTS_PATH", path)
    return path


@needs_tgcir
def test_a_composition_cohort_round_trips_save_reopen_export(cohorts_file: Path):
    """save -> list -> export, with the selection echoed back exactly.

    The provenance is the part that makes this a cohort rather than a query list: run id, protocol,
    the subsets the qids came from, the conditions the run ranked, and the artifact digest the
    selection was validated against.
    """
    client = main.app  # exercised through the module's own handlers below
    from fastapi.testclient import TestClient

    api = TestClient(client)
    body = {"cohort_id": "rt-composition", "protocol_id": "circus_fashioniq_release_v1",
            "query_ids": ["7:1", "7:2", "7:603"], "run_ids": [f"c:{TGCR_FULL}"],
            "definition": {"preset": "all", "k": 10, "subset": "full"},
            "notes": "round-trip probe"}
    saved = api.put("/api/cohorts", json=body)
    assert saved.status_code == 200, saved.text
    record = saved.json()

    provenance = record["definition"]["composition"]
    assert provenance["protocol_id"] == "circus_fashioniq_release_v1"
    assert provenance["family"] == "composition"
    # The qids are released in all three subsets; the COHORT's scope is the caller's
    # `definition.subset`, asserted below and round-tripped from the export.
    assert provenance["qid_subsets"] == ["full", "sf", "validated"], provenance["qid_subsets"]
    assert set(provenance["conditions"]) == {"multimodal", "text_only_black_image",
                                             "image_only_empty_text"}
    assert provenance["runs"][0]["run_id"] == TGCR_FULL
    assert len(provenance["runs"][0]["rank_artifact_sha256"]) == 64

    listed = api.get("/api/cohorts")
    assert listed.status_code == 200
    row = next(item for item in listed.json() if item["cohort_id"] == "rt-composition")
    assert row["run_ids"] == [f"c:{TGCR_FULL}"]
    assert row["protocol_id"] == "circus_fashioniq_release_v1"
    assert row["query_ids"] == ["7:1", "7:2", "7:603"]
    assert row["definition"]["subset"] == "full"

    exported = api.get("/api/cohorts/rt-composition/export/json")
    assert exported.status_code == 200, exported.text
    payload = exported.json()
    # Identity round-trips exactly, and the numbers come from the SAME artifact that was validated.
    assert payload["run_ids"] == [f"c:{TGCR_FULL}"]
    assert payload["protocol_id"] == "circus_fashioniq_release_v1"
    assert payload["query_ids"] == ["7:1", "7:2", "7:603"]
    metrics = payload["metrics"][0]
    assert metrics["run_id"] == TGCR_FULL
    assert metrics["query_count"] == 3
    assert metrics["scored_count"] == 3
    # `7:1` is rank 23, `7:603` and `7:2` are worse: R@10 must be exactly one of three.
    assert metrics["r10"] == pytest.approx(100 / 3), metrics

    # The other two formats render the same numbers rather than a second computation.
    csv_text = json.loads(api.get("/api/cohorts/rt-composition/export/csv").text)["content"]
    assert csv_text.splitlines()[1].startswith(f"{TGCR_FULL},3,3,"), csv_text
    markdown = json.loads(api.get("/api/cohorts/rt-composition/export/markdown").text)["content"]
    assert f"| {TGCR_FULL} | 3 |" in markdown, markdown


@needs_tgcir
def test_a_changed_artifact_refuses_the_saved_cohort_instead_of_rescoring(cohorts_file: Path):
    """The saved digest is a claim about a table; a table that changed invalidates the claim."""
    from fastapi.testclient import TestClient

    api = TestClient(main.app)
    api.put("/api/cohorts", json={
        "cohort_id": "rt-stale", "protocol_id": "circus_fashioniq_release_v1",
        "query_ids": ["7:1"], "run_ids": [f"c:{TGCR_FULL}"], "definition": {}, "notes": ""})
    records = json.loads(cohorts_file.read_text(encoding="utf-8"))
    for record in records:
        if record["cohort_id"] == "rt-stale":
            record["definition"]["composition"]["runs"][0]["rank_artifact_sha256"] = "0" * 64
    cohorts_file.write_text(json.dumps(records), encoding="utf-8")

    refused = api.get("/api/cohorts/rt-stale/export/json")
    assert refused.status_code == 400, refused.text
    body = refused.json()
    assert body["error"] == "cohort_artifact_changed", body
    assert body["details"]["saved_rank_artifact_sha256"] == "0" * 64
    assert len(body["details"]["current_rank_artifact_sha256"]) == 64


def test_an_unknown_run_and_an_unranked_qid_are_refused_by_name(cohorts_file: Path):
    """A cohort may not be validated against a table that does not exist, or carry a qid the run
    never ranked; the `run_not_found` this arm used to produce for EVERY composition run must now
    name only a run that is genuinely absent."""
    from fastapi.testclient import TestClient

    api = TestClient(main.app)
    unknown_run = api.put("/api/cohorts", json={
        "cohort_id": "rt-unknown", "protocol_id": "circus_fashioniq_release_v1",
        "query_ids": ["7:1"], "run_ids": ["c:deadbeefdeadbeefdeadbeef"],
        "definition": {}, "notes": ""})
    assert unknown_run.status_code == 400, unknown_run.text
    body = unknown_run.json()
    assert body["error"] == "run_not_found"
    assert body["details"]["run_id"] == "deadbeefdeadbeefdeadbeef"

    if not _available(TGCR_FULL):
        pytest.skip("the tgcir Full rank artifact is not on this host")
    unranked = api.put("/api/cohorts", json={
        "cohort_id": "rt-qid", "protocol_id": "circus_fashioniq_release_v1",
        "query_ids": ["9999:1"], "run_ids": [f"c:{TGCR_FULL}"], "definition": {}, "notes": ""})
    assert unranked.status_code == 400, unranked.text
    assert unranked.json()["details"]["unknown_query_ids"] == ["9999:1"]


def test_a_mixed_selection_is_refused_and_names_both_families(cohorts_file: Path):
    """An index top-K and a CIRCUS rank are different quantities; one cohort cannot hold both."""
    from fastapi.testclient import TestClient

    api = TestClient(main.app)
    response = api.put("/api/cohorts", json={
        "cohort_id": "rt-mixed", "protocol_id": "circus_fashioniq_release_v1",
        "query_ids": ["7:1"], "run_ids": [f"c:{TGCR_FULL}", "some-index-run"],
        "definition": {}, "notes": ""})
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"] == "mixed_data_kind"
    assert body["details"]["index_run_ids"] == ["some-index-run"]
    assert body["details"]["composition_references"] == [f"c:{TGCR_FULL}"]


def test_a_cohort_saved_before_the_composition_arm_still_loads(cohorts_file: Path, monkeypatch):
    """The old shape has no `definition.composition`, so it falls through to the DuckDB arm.

    Nobody has to migrate a cohort file: a record written before this change is read by exactly the
    code path it was written for.
    """
    from fastapi.testclient import TestClient

    cohorts_file.write_text(json.dumps([{
        "cohort_id": "legacy-duckdb", "protocol_id": "fashioniq_original_split",
        "run_ids": ["index-run-1"], "query_ids": ["dress:0:ref:target"],
        "definition": {"category": "dress"}, "notes": "", "created_at": "2026-11-01T00:00:00Z",
    }]), encoding="utf-8")

    calls: list[tuple[list[str], list[str]]] = []

    def fake_cohort_metrics(run_ids, query_ids, database_path=None):
        calls.append((list(run_ids), list(query_ids)))
        return [{"run_id": run_ids[0], "query_count": len(query_ids), "scored_count": 1,
                 "r1": 100.0, "r5": 100.0, "r10": 100.0, "r20": 100.0, "r50": 100.0,
                 "r100": 100.0, "mean_target_rank": 1.0, "median_target_rank": 1.0}]

    monkeypatch.setattr(main, "cohort_metrics", fake_cohort_metrics)
    api = TestClient(main.app)
    exported = api.get("/api/cohorts/legacy-duckdb/export/json")
    assert exported.status_code == 200, exported.text
    assert calls == [(["index-run-1"], ["dress:0:ref:target"])], calls
    assert exported.json()["metrics"][0]["r1"] == 100.0


def test_a_missing_cohort_is_still_a_404(cohorts_file: Path):
    from fastapi.testclient import TestClient

    api = TestClient(main.app)
    assert api.get("/api/cohorts/nope/export/json").status_code == 404
