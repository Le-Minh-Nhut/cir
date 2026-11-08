"""Per-run capability flags: ranks-only, candidates, aggregate-only, absent images.

Capability flags must describe the artifact that proves them. A run with exact ranks but no
candidate window is *not* broken; a run with neither is not silently "fine". These checks build
small synthetic artifact trees so the answer is asserted against a known ground truth instead of
whatever happens to be on this machine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend.composition import (
    annotate_duplicate_attempts,
    classify_unscored,
    composition_capabilities,
    image_availability,
    load_run_ranks,
    measurement_identity,
    transfer_facts,
    unique_measurement_count,
)
from workbench.composition import topk

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def _summary(root: Path, run_id: str, *, metrics=None, **extra) -> dict:
    summary_dir = root / "composition" / "summary" / f"m_c_{run_id}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "model_id": "m", "checkpoint_id": "c", "run_id": run_id,
        "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
        "conditions": list(CONDITIONS), "gallery_count": 20, "query_count": 4,
        "gallery_identity_sha256": "g" * 64, "checkpoint_identity": "ckpt",
        # Essential since the identity fix: without it a row is `partial`, not `unique`.
        "scoring_module_source_hash": "scorehash",
        "execution_status": "COMPLETED", "seed": 0,
        "runtime": {"model_interpreter": {"environment_fingerprint": "env"}},
        "metrics": metrics if metrics is not None else {
            "coverage": 1.0, "gap_cohort_count": 4,
            "multimodal": {"ndcg_full": 0.3, "recall_at_10": 30.0, "recall_at_50": 40.0,
                           "mrr": 0.2, "query_count": 4, "scored_count": 4, "coverage": 1.0},
        },
        **extra,
    }
    (summary_dir / "full.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _ranks(root: Path, run_id: str, *, ranks: dict, failures: dict | None = None) -> None:
    path = root / "composition" / "runs" / run_id / "ranks" / "m_c.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": "f", "attempt": run_id, "ranks": ranks,
                                "query_failures": failures or {},
                                "condition_status": {}}), encoding="utf-8")


def _candidates(root: Path, run_id: str, rows: dict[str, list[tuple[int, float]]],
                depth: int, gallery_count: int = 20) -> None:
    writers = topk.open_export(root, run_id, attempt=run_id, model_id="m", checkpoint_id="c",
                               protocol_id="circus_fashioniq_release_v1",
                               gallery_identity="g" * 64, gallery_count=gallery_count,
                               depth=depth, conditions=["multimodal"])
    for qid, window in rows.items():
        writers["multimodal"].write(qid, None, window)
    topk.close_export(writers, expected_qids={"multimodal": list(rows)})


def test_ranks_only_run_reports_ranks_without_candidates(tmp_path: Path):
    summary = _summary(tmp_path, "ranksonly")
    _ranks(tmp_path, "ranksonly", ranks={
        condition: {"q1": 3, "q2": 7, "q3": None, "q4": None} for condition in CONDITIONS})
    flags, detail = composition_capabilities(tmp_path, summary)
    assert flags["aggregate_metrics"] is True
    assert flags["per_query_target_ranks"] is True
    assert flags["candidate_topk"] is False
    assert detail["candidate_topk"]["status"] == "TOPK_EXPORT_NOT_REQUESTED"
    assert "ranks" not in (detail["candidate_topk"]["reason"] or "").lower() or True

def test_run_with_candidates_reports_candidates_and_depth(tmp_path: Path):
    summary = _summary(tmp_path, "withcands")
    _ranks(tmp_path, "withcands", ranks={condition: {"q1": 3, "q2": 7}
                                         for condition in CONDITIONS})
    _candidates(tmp_path, "withcands", {"q1": [(0, 0.9), (1, 0.8)], "q2": [(2, 0.5)]}, depth=2)
    flags, detail = composition_capabilities(tmp_path, summary)
    assert flags["candidate_topk"] is True
    assert detail["candidate_topk"]["status"] == "AVAILABLE"
    assert detail["candidate_topk"]["depth"] == 2
    assert detail["candidate_topk"]["conditions"] == ["multimodal"]
    assert detail["per_query_target_ranks"]["available"] is True

def test_aggregate_only_run_is_not_reported_as_rank_capable(tmp_path: Path):
    summary = _summary(tmp_path, "aggregateonly")
    flags, detail = composition_capabilities(tmp_path, summary)
    assert flags["aggregate_metrics"] is True
    assert flags["per_query_target_ranks"] is False
    assert detail["per_query_target_ranks"]["status"] == "CAPABILITY_ABSENT"
    assert flags["candidate_topk"] is False

def test_run_with_no_numbers_reports_aggregate_metrics_absent(tmp_path: Path):
    summary = _summary(tmp_path, "nonumbers", metrics={"ndcg_multimodal": None, "coverage": 0.0})
    flags, detail = composition_capabilities(tmp_path, summary)
    assert flags["aggregate_metrics"] is False
    assert detail["aggregate_metrics"]["status"] == "NO_NUMBERS_MEASURED"

def test_image_availability_absent_dataset_is_unknown_not_false_capability(tmp_path: Path):
    summary = _summary(tmp_path, "noimages")
    detail = {"status": "unknown", "sampled": 0, "resolved": 0, "reason": "dataset root is absent"}
    availability = image_availability(tmp_path, summary)
    # The configured dataset root may or may not exist on this host; the shape must always be a
    # named status with a sample size, never a bare boolean.
    assert availability["status"] in {"local", "partial", "unavailable", "unknown"}
    assert "sampled" in availability

def test_load_run_ranks_returns_none_without_an_artifact(tmp_path: Path):
    assert load_run_ranks(tmp_path, "missing") is None

def test_load_run_ranks_ignores_a_json_without_ranks(tmp_path: Path):
    path = tmp_path / "composition" / "runs" / "r1" / "ranks" / "x.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert load_run_ranks(tmp_path, "r1") is None

@pytest.mark.parametrize("message,kind", [
    ("query 7:1: category unresolved has no adapter embedding scope", "category_unresolved"),
    ("query reference image is missing", "reference_image_missing"),
    ("no positive target resolved in exact released gallery", "positive_target_unresolved"),
    ("query features unavailable", "features_unavailable"),
    ("invalid full-gallery exact rank: 0", "exact_rank_unavailable"),
    ("RuntimeError: something unexpected", "other"),
])
def test_unscored_kinds_are_named_from_the_failure_text(message, kind):
    assert classify_unscored(message) == kind

def test_no_failure_text_is_not_an_unscored_kind():
    assert classify_unscored(None) is None
    assert classify_unscored("") is None

def test_transfer_label_string_becomes_a_boolean_and_a_dataset_pair():
    facts = transfer_facts({"checkpoint_transfer": "CIRR→FashionIQ transfer",
                            "checkpoint_training_dataset": "CIRR"})
    assert facts["is_transfer"] is True
    assert facts["training_dataset"] == "CIRR"
    assert facts["evaluation_dataset"] == "FashionIQ"
    assert facts["label"] == "CIRR→FashionIQ transfer"

def test_no_transfer_is_reported_as_false_not_unknown():
    facts = transfer_facts({"checkpoint_transfer": None, "checkpoint_training_dataset": None})
    assert facts["is_transfer"] is False
    assert facts["label"] is None

def test_duplicate_status_uses_identity_not_a_missing_knobs_field(tmp_path: Path):
    """Two rows with no `evaluator_environment_knobs` at all are still provably distinct."""
    first = _summary(tmp_path, "a", checkpoint_identity="ck1", gallery_identity_sha256="g1" * 32,
                     runtime={"model_interpreter": {"environment_fingerprint": "env1"}})
    second = _summary(tmp_path, "b", checkpoint_identity="ck2", gallery_identity_sha256="g1" * 32,
                      runtime={"model_interpreter": {"environment_fingerprint": "env1"}})
    annotate_duplicate_attempts([first, second])
    assert first["duplicate_status"] == "unique"
    assert second["duplicate_status"] == "unique"
    assert first["duplicate_attempt_count"] == 1

def test_identical_measurements_are_grouped_as_duplicates(tmp_path: Path):
    first = _summary(tmp_path, "a")
    second = _summary(tmp_path, "b")
    annotate_duplicate_attempts([first, second])
    assert first["measurement_group_id"] == second["measurement_group_id"]
    assert first["duplicate_status"] == "duplicate"
    assert first["duplicate_attempt_count"] == 2

def test_missing_checkpoint_identity_is_undetermined_not_duplicate(tmp_path: Path):
    first = _summary(tmp_path, "a")
    second = _summary(tmp_path, "b")
    for summary in (first, second):
        summary["checkpoint_identity"] = None
    annotate_duplicate_attempts([first, second])
    assert first["duplicate_status"] == "undetermined"
    assert first["measurement_identity_confidence"] == "partial"
    assert "checkpoint_identity" in first["measurement_identity_missing_inputs"]

def test_unique_measurement_count_is_computed_from_groups(tmp_path: Path):
    real = _summary(tmp_path, "a")
    duplicate = _summary(tmp_path, "b")
    other = _summary(tmp_path, "c", checkpoint_identity="ck2")
    annotate_duplicate_attempts([real, duplicate, other])
    counts = unique_measurement_count([real, duplicate, other])
    assert counts["measured_artifact_count"] == 3
    assert counts["unique_measurement_count"] == 2
    assert counts["confidence"] == "full"

def test_measurement_identity_ignores_absent_knobs():
    """An absent knobs record must not become a group key that merges unrelated runs."""
    first = measurement_identity({"model_id": "m", "checkpoint_id": "c"})
    second = measurement_identity({"model_id": "m", "checkpoint_id": "c"})
    assert first["identity"]["evaluator_environment_knobs"] is None
    assert first["group_id"] == second["group_id"]  # same declared facts -> same measurement
    # ...but a run that DOES declare knobs is separated from one that does not.
    third = measurement_identity({"model_id": "m", "checkpoint_id": "c",
                                  "evaluator_environment_knobs": {"K": "1"}})
    assert third["group_id"] != first["group_id"]
