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
        # The superseded text: the false claim a frozen field would have preserved forever.
        "protocol_source": ("third_party/CLIP4Cir/src/validate.py::compute_fiq_val_metrics — "
                            "index_features over the whole split.{category}.val.json gallery; the "
                            "reference is excluded by masking its own index entry rather than "
                            "removing it from the gallery"),
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


VETTED_SOURCE = ("third_party/CLIP4Cir/src/validate.py::compute_fiq_val_metrics — index_features "
                 "over the whole split.{category}.val.json gallery (the full released gallery, no "
                 "category filter, no reference mask); recall@10/@50 counted directly from the sorted "
                 "index. The reference-mask branch cited by the superseded text exists only in "
                 "compute_cirr_val_metrics (validate.py:162-166), which the FashionIQ branch never "
                 "calls")

CORRECTION = {
    "protocol_source": VETTED_SOURCE,
    "reason": ("the superseded citation in third_party/CLIP4Cir/src/validate.py claims the reference "
               "is excluded by masking its own index entry; compute_fiq_val_metrics has no such mask"),
    "citations": [
        "workbench/third_party/CLIP4Cir/src/validate.py:21-57 compute_fiq_val_metrics: normalize "
        "(:41), distances = 1 - predicted @ index.T (:44), argsort (:45), labels vs targets only "
        "(:49-50) — no reference write anywhere in the function",
        "workbench/third_party/CLIP4Cir/src/validate.py:162-166 the only reference_mask sits inside "
        "compute_cirr_val_metrics, which computes CIRR metrics and is never called by the FashionIQ "
        "branch",
    ],
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
    # No correction supplied => the citation is left alone. Silence changes nothing.
    assert after["protocol_source"] == before["protocol_source"]
    assert "superseded_protocol_source" not in after
    assert "protocol_source_correction" not in after
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


def test_a_cited_correction_rewrites_the_false_citation_and_keeps_the_original(
        tmp_path: Path) -> None:
    """R1: freezing a wrong citation is not history preservation, it is propagating the error."""
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    before = json.loads(path.read_text(encoding="utf-8"))
    false_citation = before["protocol_source"]
    assert "masking its own index entry" in false_citation

    result = migrate(path, "fashioniq_original_split",
                     evidence={**EVIDENCE, "protocol_source_correction": CORRECTION},
                     dry_run=False)

    after = json.loads(path.read_text(encoding="utf-8"))
    # The label migration is unchanged in behaviour.
    assert after["protocol"] == "fashioniq_original_split"
    assert after["aggregate"] == before["aggregate"]
    assert after["categories"] == before["categories"]
    assert after["checkpoint"] == before["checkpoint"]
    # The old string survives verbatim, under its own field.
    assert after["superseded_protocol_source"] == false_citation
    # The active citation no longer claims a FashionIQ mask.
    assert after["protocol_source"] == VETTED_SOURCE
    assert "masking its own index entry" not in after["protocol_source"]
    assert "162-166" in after["protocol_source"]
    # The disproof travels with the artifact.
    block = after["protocol_source_correction"]
    assert block["field"] == "protocol_source"
    assert block["superseded_field"] == "superseded_protocol_source"
    assert block["citations"] and any("21-57" in cite for cite in block["citations"])
    assert "validate.py" in block["reason"]
    # The sidecar records that a citation moved, and hashes what it superseded.
    sidecar = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["protocol_source_corrected"] is True
    assert sidecar["superseded_protocol_source_sha256"] == hashlib.sha256(
        false_citation.encode("utf-8")).hexdigest()


def test_a_correction_without_a_citation_is_refused(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    digest = sha256(path)
    with pytest.raises(SystemExit, match="no citation"):
        migrate(path, "fashioniq_original_split",
                evidence={**EVIDENCE, "protocol_source_correction":
                          {"protocol_source": VETTED_SOURCE, "reason": "trust me"}},
                dry_run=False)
    assert sha256(path) == digest


def test_a_correction_without_new_text_is_refused(tmp_path: Path) -> None:
    path = _artifact(tmp_path, {"dress": 3817, "shirt": 6346, "toptee": 5373})
    digest = sha256(path)
    with pytest.raises(SystemExit, match="no corrected protocol_source"):
        migrate(path, "fashioniq_original_split",
                evidence={**EVIDENCE, "protocol_source_correction":
                          {"citations": CORRECTION["citations"]}},
                dry_run=False)
    assert sha256(path) == digest
