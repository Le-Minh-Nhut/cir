"""Fixtures, smoke slices and imported artifacts stay out of default research results.

Every check is provenance-based: a row is excluded because of what it declares about itself
(``is_synthetic``, ``smoke.limited``, ``provenance_source``, ``reproduction_status``), never because
its model name looks familiar. That is the difference between a rule and a coincidence.
"""
from __future__ import annotations

import json
from pathlib import Path

from workbench.backend.composition import (
    is_fixture_summary,
    is_imported_summary,
    is_smoke_summary,
    list_composition_summaries,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
_METRICS = {"ndcg_multimodal": 0.3, "coverage": 1.0, "gap_cohort_count": 4,
            "multimodal": {"ndcg_full": 0.3, "recall_at_10": 30.0, "recall_at_50": 40.0,
                           "mrr": 0.2, "query_count": 4, "scored_count": 4, "coverage": 1.0}}


def _write(root: Path, run_dir: str, **overrides) -> dict:
    payload = {"schema_version": 1, "model_id": "m", "checkpoint_id": "c",
               "run_id": run_dir, "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
               "conditions": list(CONDITIONS), "gallery_count": 10, "query_count": 4,
               "execution_status": "COMPLETED", "metrics": dict(_METRICS),
               "smoke": {"limited": False, "query_limit": None},
               "is_synthetic": False, "synthetic_fixture": False}
    payload.update(overrides)
    path = root / "composition" / "summary" / run_dir / "full.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_default_list_contains_only_measured_rows(tmp_path: Path):
    _write(tmp_path, "measured")
    _write(tmp_path, "fixture", is_synthetic=True, synthetic_fixture=True)
    _write(tmp_path, "smoke", smoke={"limited": True, "query_limit": 1})
    _write(tmp_path, "imported", provenance_source="released_audit_retriever",
           reproduction_status="RELEASED_ARTIFACT_IMPORT")
    ids = [row["run_id"] for row in list_composition_summaries(
        artifacts_root=tmp_path, subset="full", include_fixtures=False, include_smoke=False)]
    assert ids == ["measured"]

def test_imported_rows_are_opt_in_and_sort_last(tmp_path: Path):
    _write(tmp_path, "measured")
    _write(tmp_path, "fixture", is_synthetic=True, synthetic_fixture=True)
    _write(tmp_path, "imported", provenance_source="released_audit_retriever")
    ids = [row["run_id"] for row in list_composition_summaries(
        artifacts_root=tmp_path, subset="full", include_fixtures=True, include_smoke=True,
        include_imported=True)]
    assert ids == ["measured", "fixture", "imported"]

def test_imported_detection_is_provenance_based_not_name_based(tmp_path: Path):
    """A model that happens to be called "released_audit" is still a measurement."""
    declared_here = _write(tmp_path, "looks_imported_but_is_ours")
    assert is_imported_summary(declared_here) is False
    imported = _write(tmp_path, "anything", provenance_source="released_audit_retriever")
    assert is_imported_summary(imported) is True
    by_status = _write(tmp_path, "another", reproduction_status="RELEASED_ARTIFACT_IMPORT")
    assert is_imported_summary(by_status) is True
    by_source = _write(tmp_path, "third", provenance_source="external_artifact")
    assert is_imported_summary(by_source) is True

def test_fixture_and_smoke_detection_precedence(tmp_path: Path):
    fixture = _write(tmp_path, "f", is_synthetic=True, smoke={"limited": True})
    assert is_fixture_summary(fixture) is True
    # A fixture is not a smoke slice: it never measured anything to limit.
    assert is_smoke_summary(fixture) is False
    smoke = _write(tmp_path, "s", smoke={"limited": True})
    assert is_smoke_summary(smoke) is True
    assert is_fixture_summary(smoke) is False

def test_fixture_cannot_sort_above_a_measurement(tmp_path: Path):
    _write(tmp_path, "zzz_measured")
    _write(tmp_path, "aaa_fixture", is_synthetic=True, synthetic_fixture=True)
    rows = list_composition_summaries(artifacts_root=tmp_path, subset="full",
                                      include_fixtures=True, include_smoke=True)
    assert [row["run_id"] for row in rows] == ["zzz_measured", "aaa_fixture"]
