"""R5 / R6 / R8 — the LIMN diagnostic record must be one validated answer, not three.

Measured before this file existed, on HEAD `2e2e2d2`, against the live endpoint:

* **R5** every one of the twelve summary rows recorded `target_did = 7:42304`, the positive id of
  the OTHER query. `7:3038`'s own positive is `7:3416` (`B004D8QCN4`). The producer read a shared
  loop variable (`t2_dual_scope.py:273`). Rank, score, gallery position, target asset and candidate
  ids were all correct; only the recorded identity was wrong.
* **R6** `depth=0` and `depth=-5` were absorbed into a one-row window with HTTP 200 and nothing in
  the body saying so; `depth=1000` silently returned 100 rows. The summary and the candidates
  endpoint parsed the files independently and could disagree about the same combination.
* **R8** the candidates response carried no image URL, no category, no per-scope label and no
  separate target object: the target was only visible when it happened to fall inside the requested
  window, so the rank-6891 finding could not be rendered at all.

Fixtures are synthetic and exercise the reader's contract; `test_shipped_*` runs the real artifact
when this checkout has it, and skips otherwise (it is an external 74,381-image measurement).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import limn_diagnostic
from workbench.backend.limn_diagnostic import (
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_EVIDENCE_DEFECT,
    DiagnosticEvidenceError,
    load_candidates,
    load_diagnostic,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
SCOPES = ("shirt", "toptee")
QIDS = ("7:3038", "7:3790")
CHECKPOINT = "a" * 64


def _candidates(rank_of_target: int | None, *, duplicates: bool = False,
                scores_descending: bool = True, target_position: int = 100,
                target_asset: str = "TGT0") -> list[dict]:
    """A coherent 4-row window whose flagged row sits where the summary says it does."""
    rows = []
    for rank in range(1, 5):
        is_target = rank_of_target == rank
        rows.append({
            "rank": rank,
            "gallery_position": target_position if is_target else 1000 + rank,
            "gallery_id": ("mbeir_images/fashioniq_images/DUP.jpg" if duplicates
                           else f"mbeir_images/fashioniq_images/{target_asset}.jpg" if is_target
                           else f"mbeir_images/fashioniq_images/IMG{rank}.jpg"),
            "image_path": "mbeir_images/fashioniq_images/IMG.jpg",
            "score": (0.5 if is_target else
                      (0.9 - 0.1 * rank) if scores_descending else (0.1 * rank)),
            "is_target": is_target,
        })
    return rows


def _payload(*, target_did_leak: bool = False, mismatch_checkpoint: bool = False,
             contradiction: bool = False) -> dict:
    queries = {}
    for offset, qid in enumerate(QIDS):
        own_did = f"7:{3416 + offset * 38888}"  # 7:3416 for 7:3038, 7:42304 for 7:3790
        asset = f"mbeir_images/fashioniq_images/TGT{offset}.jpg"
        scopes = {}
        for scope in SCOPES:
            scopes[scope] = {}
            for index, condition in enumerate(CONDITIONS):
                rank = 3 if (offset == 0 and condition == "multimodal") else (6891 if index == 1 else 40 + index)
                scopes[scope][condition] = {
                    "target_rank": rank,
                    "target_score": 0.5,
                    "score_at_rank_1": 0.9,
                    "target_margin_vs_rank1": 0.4,
                    "target_gallery_position": 100 + offset,
                    "target_did": "7:42304" if target_did_leak else own_did,
                    "gallery_size_scored": 74381,
                    "model_text": f"text {qid}",
                    "top100_export": f"limn_dual_scope.{scope}.{condition}.jsonl",
                }
        queries[qid] = {
            "qid": qid,
            "release": {"query_txt": f"text {qid}", "reference_id": f"REF{offset}",
                        "positive_ids": [own_did], "loader_category": None},
            "target_gallery_id": asset,
            "target_position_in_gallery": [100 + offset],
            "scopes": scopes,
        }
    return {
        "artifact_type": "limn_dual_scope_diagnostic",
        "schema_version": 1,
        "canonical_status": {"status": "PARTIAL", "scored_count": 6001, "unscored_count": 2,
                             "unscored_qids": list(QIDS), "note": "diagnostic is not canonical"},
        "gallery": {"count": 74381, "protocol_id": "circus_fashioniq_release_v1", "note": "full"},
        "conditions": list(CONDITIONS),
        "scopes": list(SCOPES),
        "checkpoints": {scope: {"checkpoint_id": f"base_iter0_{scope}",
                                "sha256": ("b" * 64 if mismatch_checkpoint else CHECKPOINT),
                                "filename": f"0_{scope}_best_model.pt"} for scope in SCOPES},
        "rank_convention_crosscheck": {"checked": 36, "all_agree": True, "method": "recompute"},
        "canonical_cohort": {"subset": "full", "query_count": 6003,
                             "query_count_per_subset": {"full": 6003, "sf": 4069, "validated": 586},
                             "scored_count": 6001, "unscored_count": 2},
        "runtime": {"seconds": 60.5},
        "queries": queries,
        "_fixture": {"contradiction": contradiction},
    }


def _root(tmp_path: Path, payload: dict | None, *, target_flags: dict | None = None,
          duplicates: bool = False, scores_descending: bool = True, depth: int | None = None,
          export_patch: dict | None = None) -> Path:
    """The evidence directory at the REAL repo-relative location, under a temp repo root.

    ``depth`` defaults to the number of rows the fixture actually stores: a fixture that declares a
    full window while holding a short one is itself the truncation defect this reader now reports,
    so the positive cases must not accidentally be defect cases.
    """
    root = tmp_path.joinpath(*limn_diagnostic.EVIDENCE_RELATIVE)
    root.mkdir(parents=True)
    if payload is not None:
        (root / "limn_dual_scope.json").write_text(json.dumps(payload), encoding="utf-8")
    target_flags = target_flags or {}
    for scope in SCOPES:
        for condition in CONDITIONS:
            rows = []
            for offset, qid in enumerate(QIDS):
                declared = payload["queries"][qid]["scopes"][scope][condition] if payload else None
                rank = declared["target_rank"] if declared else 3
                # The stored window is 4 rows, so a target ranked 5+ is outside it.
                in_window = 1 <= rank <= 4
                candidates = _candidates(
                    rank if in_window else None,
                    duplicates=duplicates, scores_descending=scores_descending,
                    target_position=(declared["target_gallery_position"] if declared else 100),
                    target_asset=Path(str(payload["queries"][qid]["target_gallery_id"])).stem
                    if payload else "TGT0")
                override = (target_flags.get((qid, scope, condition)) if target_flags else None)
                if override is not None:
                    candidates = _candidates(override, duplicates=duplicates)
                body = {
                    "qid": qid, "scope": scope, "condition": condition,
                    "model_text": f"text {qid}",
                    "query_image_path": f"queries/{qid}.png",
                    "checkpoint_sha256": CHECKPOINT,
                    "gallery_size": 74381,
                    "target_rank": rank,
                    "target_score": 0.5,
                    "depth": len(candidates) if depth is None else depth,
                    "candidates": candidates,
                }
                # A caller that means to contradict the summary edits the export row, so the row is
                # exactly what the summary claims unless the test says otherwise.
                row_patch = (export_patch or {}).get((qid, scope, condition))
                if row_patch:
                    body.update(row_patch)
                rows.append(body)
            (root / f"limn_dual_scope.{scope}.{condition}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return root


def _config(root: Path) -> SimpleNamespace:
    depth = len(limn_diagnostic.EVIDENCE_RELATIVE)
    return SimpleNamespace(CIR_REPO_ROOT=root.parents[depth - 1])


@pytest.fixture(autouse=True)
def _clear_record_cache():
    limn_diagnostic._RECORD_CACHE.clear()
    yield
    limn_diagnostic._RECORD_CACHE.clear()


# ---------------------------------------------------------------- R5: the target identity

def test_served_target_did_is_the_query_s_own_positive_even_when_the_record_leaked_one(
        tmp_path: Path) -> None:
    """R5's exact shape: every row recorded `7:42304`, so `7:3038` read as the other query's target."""
    root = _root(tmp_path, _payload(target_did_leak=True))
    result = load_diagnostic(_config(root))
    assert result["queries"]["7:3038"]["target_did"] == "7:3416"
    assert result["queries"]["7:3790"]["target_did"] == "7:42304"
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                reading = result["queries"][qid]["scopes"][scope][condition]
                assert reading["target_did"] == result["queries"][qid]["target_did"], (
                    f"{qid}/{scope}/{condition} served a target id from another query")


def test_a_leaked_target_did_is_named_as_a_defect_not_repeated(tmp_path: Path) -> None:
    """Repairing the value silently would hide the producer bug that wrote it."""
    root = _root(tmp_path, _payload(target_did_leak=True))
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert result["status"] == DIAGNOSTIC_EVIDENCE_DEFECT
    codes = {defect["code"] for defect in result["defects"]}
    assert "target_did_not_own_query" in codes
    leaked = [defect for defect in result["defects"] if defect["code"] == "target_did_not_own_query"]
    # Six rows of one query carry the other query's id; the correct query carries none.
    assert {defect["qid"] for defect in leaked} == {"7:3038"}


def test_a_consistent_record_is_complete_with_no_defects(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    result = load_diagnostic(_config(root))
    assert result["complete"] is True
    assert result["status"] == DIAGNOSTIC_COMPLETE
    assert result["defects"] == []
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                reading = result["queries"][qid]["scopes"][scope][condition]
                assert reading["label"].startswith("diagnostic")
                assert reading["target_did"]


# ---------------------------------------------------------------- R6: malformed input

def test_mismatched_checkpoint_is_a_defect_not_a_silent_comparison(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload(mismatch_checkpoint=True))
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "checkpoint_summary_vs_export" for defect in result["defects"])


def test_rank_contradicting_the_export_is_a_defect(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload(),
                 export_patch={("7:3038", "shirt", "multimodal"): {"target_rank": 2}})
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    codes = {defect["code"] for defect in result["defects"]}
    assert "rank_summary_vs_export" in codes


def test_score_contradicting_the_export_is_a_defect(tmp_path: Path) -> None:
    payload = _payload()
    payload["queries"]["7:3790"]["scopes"]["toptee"]["multimodal"]["target_score"] = 0.123
    root = _root(tmp_path, payload)
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "score_summary_vs_export" for defect in result["defects"])


def test_duplicate_candidate_ids_are_a_defect(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload(), duplicates=True)
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] in {"export_duplicate_candidate", "export_duplicate_position"}
               for defect in result["defects"])


def test_scores_that_do_not_descend_are_a_defect(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload(), scores_descending=False)
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "export_scores_not_descending" for defect in result["defects"])


def test_broken_depth_is_a_defect(tmp_path: Path) -> None:
    """A window longer than its own declared depth cannot be reproduced from the file."""
    root = _root(tmp_path, _payload(), depth=2)
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "export_depth_short" for defect in result["defects"])


def test_empty_payload_details_are_named_not_empty(tmp_path: Path) -> None:
    root = _root(tmp_path, None)
    result = load_diagnostic(_config(root))
    assert result["available"] is False
    assert result["status"] == "DIAGNOSTIC_NOT_PUBLISHED"
    assert result["reason"]
    assert result["complete"] is False


@pytest.mark.parametrize("qid, scope, condition", [
    ("7:9999", "shirt", "multimodal"),
    ("7:3038", "dress", "multimodal"),
    ("7:3038", "shirt", "audio_only"),
    ("", "shirt", "multimodal"),
])
def test_wrong_identity_components_are_refused_with_a_reason(
        tmp_path: Path, qid: str, scope: str, condition: str) -> None:
    root = _root(tmp_path, _payload())
    with pytest.raises(DiagnosticEvidenceError) as raised:
        load_candidates(qid, scope, condition, config=_config(root))
    assert str(raised.value), "a refusal must say which component was wrong"


@pytest.mark.parametrize("depth", [0, -1, -5, 201, 10_000])
def test_out_of_range_depth_is_refused_not_clamped(tmp_path: Path, depth: int) -> None:
    """Before the fix, depth=0/-5 returned a ONE-row window with HTTP 200 and no notice."""
    root = _root(tmp_path, _payload())
    with pytest.raises(DiagnosticEvidenceError, match="depth"):
        load_candidates("7:3038", "shirt", "multimodal", depth=depth, config=_config(root))


def test_depth_beyond_the_stored_window_reports_the_truncation(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3038", "shirt", "multimodal", depth=200, config=_config(root))
    assert window["depth_requested"] == 200
    # Bounded by the stored depth AND the rows the file actually holds: both are real limits.
    assert window["depth_served"] == min(200, window["depth_stored"], 4)
    assert window["depth_truncated"] is True


def test_summary_and_candidates_describe_the_same_combination(tmp_path: Path) -> None:
    """Both endpoints now read one validated record; they cannot drift apart."""
    root = _root(tmp_path, _payload())
    config = _config(root)
    summary = load_diagnostic(config)
    digest = summary["manifest"]["digest"]
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                declared = summary["queries"][qid]["scopes"][scope][condition]
                window = load_candidates(qid, scope, condition, config=config)
                assert window["target_rank"] == declared["target_rank"]
                assert window["target_score"] == declared["target_score"]
                assert window["target_did"] == declared["target_did"]
                assert window["identity"]["qid"] == qid
                assert window["target"]["rank"] == declared["target_rank"]
                assert window["checkpoint_sha256"] == summary["checkpoints"][scope]["sha256"]
                assert window["identity"]["manifest_digest"] == digest
                assert (window["identity"]["qid"], window["identity"]["scope"],
                        window["identity"]["condition"]) == (qid, scope, condition)


def test_the_validation_is_cached_by_content_identity(tmp_path: Path) -> None:
    """A cached check must never answer for different bytes."""
    root = _root(tmp_path, _payload())
    config = _config(root)
    first = load_diagnostic(config)["manifest"]["digest"]
    load_diagnostic(config)
    assert len(limn_diagnostic._RECORD_CACHE) == 1
    # Rewrite one export with a contradicting rank: a new digest, so a new validation.
    path = root / "limn_dual_scope.shirt.multimodal.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["target_rank"] = 2
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    second = load_diagnostic(config)
    assert second["manifest"]["digest"] != first
    assert second["complete"] is False


# ---------------------------------------------------------------- R8: the gallery

def test_the_target_is_returned_separately_from_the_window(tmp_path: Path) -> None:
    """The rank-6891 target is never inside a 100-row window, and must still be servable."""
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3038", "shirt", "text_only_black_image", config=_config(root))
    assert window["target_rank"] == 6891
    assert window["target"]["in_window"] is False
    assert window["target"]["rank"] == 6891
    assert all(not candidate["is_target"] for candidate in window["candidates"])


def test_a_target_inside_the_window_is_flagged_once(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3038", "shirt", "multimodal", config=_config(root))
    flagged = [candidate for candidate in window["candidates"] if candidate["is_target"]]
    assert len(flagged) == 1
    assert flagged[0]["rank"] == window["target_rank"]
    assert window["target"]["in_window"] is True


def test_every_candidate_carries_an_image_identity(tmp_path: Path) -> None:
    """R8: the response carried no image URL, so no gallery could render it."""
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3038", "shirt", "multimodal", config=_config(root))
    assert window["candidates"], window
    for candidate in window["candidates"]:
        assert candidate["gallery_id"]
        assert candidate["image_stem"]
        assert "image_url" in candidate
    assert window["target"]["image_stem"]


def test_k_follows_the_export_depth_not_a_hardcoded_constant(tmp_path: Path) -> None:
    """The client's `depth=20` came from a literal; the stored depth is what bounds the window."""
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3038", "shirt", "multimodal", depth=20, config=_config(root))
    assert window["depth_served"] == min(20, window["depth_stored"], 4)
    assert len(window["candidates"]) == window["depth_served"]


def test_scope_and_condition_are_echoed_as_request_identity(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    window = load_candidates("7:3790", "toptee", "image_only_empty_text", config=_config(root))
    assert window["identity"]["scope"] == "toptee"
    assert window["identity"]["condition"] == "image_only_empty_text"
    assert window["label"].startswith("diagnostic")
    assert "toptee" in window["label"]


# ---------------------------------------------------------------- the real artifact

def test_shipped_evidence_reconciles_if_present() -> None:
    """The real artifact: every combination's target is its own query's positive id."""
    real = limn_diagnostic.evidence_root()
    if not (real / "limn_dual_scope.json").is_file():
        pytest.skip("dual-scope evidence is not present on this host (external artifact)")
    result = load_diagnostic()
    assert result["available"] is True
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                reading = result["queries"][qid]["scopes"][scope][condition]
                window = load_candidates(qid, scope, condition)
                assert window["target_rank"] == reading["target_rank"]
                assert window["target"]["gallery_id"]
                assert reading["target_did"], f"{qid}/{scope}/{condition} has no target id"
    assert result["manifest"]["digest"]


# ---------------------------------------------------------- M-03: truncation and duplicate rows

def test_a_window_shorter_than_its_declared_depth_is_a_defect(tmp_path: Path) -> None:
    """The producer's own failure mode: declare depth=100, store fewer rows.

    Measured on HEAD before this test existed: exporting `candidates[:1]` from a row that still
    declared `depth=100` left the record COMPLETE with zero defects, so a truncated export read as
    a complete window.
    """
    root = _root(tmp_path, _payload(), depth=100)
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "export_depth_truncated" for defect in result["defects"])


def test_a_duplicate_export_row_for_one_qid_is_a_defect(tmp_path: Path) -> None:
    """Two rows for one qid are two claims about one query; the reader took the first silently."""
    root = _root(tmp_path, _payload())
    path = root / "limn_dual_scope.shirt.multimodal.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clone = dict(rows[0], target_rank=rows[0]["target_rank"] + 1000)
    path.write_text("\n".join(json.dumps(row) for row in rows + [clone]) + "\n", encoding="utf-8")
    result = load_diagnostic(_config(root))
    assert result["complete"] is False
    assert any(defect["code"] == "export_duplicate_qid" for defect in result["defects"])
