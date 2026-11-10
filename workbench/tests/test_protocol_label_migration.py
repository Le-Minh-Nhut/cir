"""Tests for the protocol-label migration: it must refuse, preserve, and record.

Prefix `test_protocol_` per campaign ownership. The properties pinned here are the ones a wrong
migration would violate: a label-only fix must not touch any measurement, must refuse when the
artifact's own gallery counts contradict the requested cohort, and must refuse an evidence-free
call (a bare label is not provenance).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.scripts.migrate_protocol_label import migrate, sha256


def _artifact(tmp_path: Path, gallery_counts: dict[str, int]) -> Path:
    document = {
        "model_id": "clip4cir_rn50x4_fullft",
        "protocol": "fashioniq_full_gallery_ref_excluded",
        "protocol_source": "third_party/CLIP4Cir/src/validate.py::compute_fiq_val_metrics",
        "complete_benchmark": True,
        "categories": [{"category": name, "gallery_count": count, "query_count": 1, "r10": 1.0, "r50": 2.0}
                       for name, count in gallery_counts.items()],
        "aggregate": {"macro_r10": 43.8016931215922, "macro_r50": 66.98358456293742},
        "checkpoint": {"sha256": {"fiq_comb.pt": "a" * 64}},
    }
    path = tmp_path / "clip4cir_fullft.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


EVIDENCE = {
    "citations": ["workbench/third_party/CLIP4Cir/src/validate.py:162-166 mask is CIRR-only"],
    "gallery_counts": {"dress": 3817, "shirt": 6346, "toptee": 5373},
}


def test_migration_rewrites_only_the_label_and_records_history(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    before = json.loads(path.read_text(encoding="utf-8"))
    digest_before = sha256(path)

    result = migrate(path, "fashioniq_original_split", evidence=EVIDENCE, dry_run=False)

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["protocol"] == "fashioniq_original_split"
    assert after["aggregate"] == before["aggregate"]
    assert after["categories"] == before["categories"]
    assert after["checkpoint"] == before["checkpoint"]
    assert after["protocol_source"] == before["protocol_source"]
    assert after["migrated_from"]["protocol_label"] == "fashioniq_full_gallery_ref_excluded"
    assert after["migrated_from"]["artifact_sha256"] == digest_before
    assert result["sha256_before"] == digest_before
    assert result["sha256_after"] == hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["from_label"] == "fashioniq_full_gallery_ref_excluded"
    assert sidecar["to_label"] == "fashioniq_original_split"
    assert sidecar["artifact_sha256_before"] == digest_before


def test_migration_refuses_when_gallery_counts_contradict_the_cohort(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 2628, "shirt": 3089, "toptee": 2902})
    with pytest.raises(SystemExit) as raised:
        migrate(path, "fashioniq_original_split", evidence=EVIDENCE, dry_run=False)
    assert "contradict" in str(raised.value)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol"] == "fashioniq_full_gallery_ref_excluded"


def test_migration_refuses_evidence_without_citations(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    with pytest.raises(SystemExit) as raised:
        migrate(path, "fashioniq_original_split", evidence={"gallery_counts": {}}, dry_run=False)
    assert "citation" in str(raised.value)


def test_migration_refuses_a_noop_relabel(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    with pytest.raises(SystemExit) as raised:
        migrate(path, "fashioniq_full_gallery_ref_excluded", evidence=EVIDENCE, dry_run=False)
    assert "already declares" in str(raised.value)


def test_dry_run_leaves_the_artifact_untouched(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    digest = sha256(path)
    result = migrate(path, "fashioniq_original_split", evidence=EVIDENCE, dry_run=True)
    assert result["dry_run"] is True
    assert sha256(path) == digest
