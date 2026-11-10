"""The LIMN dual-scope diagnostic must never read as a canonical result.

`7:3038` and `7:3790` cannot be routed to a checkpoint: their reference image and normalized caption
pair are filed under BOTH `shirt` and `toptee`. The canonical evaluation therefore leaves them
unscored (6001/6003) and the adapter refuses instead of guessing.

The diagnostic that shows both scopes is a new production surface, so these tests pin the three
things that make it safe:

1. the canonical status travels WITH the comparison, so a view cannot render one without the other;
2. a missing or one-sided scope refuses rather than returning a comparison that would read as a
   resolved category;
3. each scope's ranks are read only from that scope's own file (no cross-scope mixing, no best-of).

Fixtures are synthetic: they exercise the reader's contract, not the 74,381-image measurement, which
lives in `limn_dual_scope.json` under the review directory.
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
    DiagnosticEvidenceError,
    load_candidates,
    load_diagnostic,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
SCOPES = ("shirt", "toptee")
QIDS = ("7:3038", "7:3790")


def _payload(*, scopes: tuple[str, ...] = SCOPES, queries: tuple[str, ...] = QIDS) -> dict:
    evidence = {}
    for offset, qid in enumerate(queries):
        evidence[qid] = {
            "qid": qid,
            "release": {"query_txt": f"text {qid}", "reference_id": f"REF{offset}",
                        "positive_ids": [f"TGT{offset}"], "loader_category": None},
            "target_gallery_id": f"TGT{offset}",
            "target_position_in_gallery": [100 + offset],
            "scopes": {
                scope: {
                    condition: {
                        "target_rank": 10 * (index + 1) + offset, "target_score": 0.5,
                        "score_at_rank_1": 0.9, "target_margin_vs_rank1": 0.4,
                        "gallery_size_scored": 74381, "model_text": f"text {qid}",
                        "top100_export": f"limn_dual_scope.{scope}.{condition}.jsonl",
                    } for index, condition in enumerate(CONDITIONS)
                } for scope in scopes
            },
        }
    return {
        "artifact_type": "limn_dual_scope_diagnostic",
        "schema_version": 1,
        "canonical_status": {"status": "PARTIAL", "scored_count": 6001, "unscored_count": 2,
                             "unscored_qids": list(QIDS), "note": "diagnostic is not canonical"},
        "label": "ambiguous category — diagnostic comparison",
        "reason": "both scopes carry the same caption pair",
        "gallery": {"count": 74381, "protocol_id": "circus_fashioniq_release_v1", "note": "full gallery"},
        "conditions": list(CONDITIONS),
        "scopes": list(SCOPES),
        # The export rows below carry `"a"*64`; a fixture whose summary and export state different
        # checkpoints is a DEFECT fixture, so the consistent one must agree with its own exports.
        "checkpoints": {scope: {"checkpoint_id": f"base_iter0_{scope}", "sha256": "a" * 64,
                                "filename": f"0_{scope}_best_model.pt"} for scope in SCOPES},
        "rank_convention_crosscheck": {"checked": 36, "all_agree": True, "method": "recompute"},
        "canonical_cohort": {"subset": "full", "query_count": 6003,
                             "query_count_per_subset": {"full": 6003, "sf": 4069,
                                                        "validated": 586},
                             "scored_count": 6001, "unscored_count": 2},
        "runtime": {"seconds": 60.5},
        "queries": evidence,
    }


def _root(tmp_path: Path, payload: dict | None) -> Path:
    """Build the evidence directory at the REAL repo-relative location under a temp repo root."""
    root = tmp_path.joinpath(*limn_diagnostic.EVIDENCE_RELATIVE)
    root.mkdir(parents=True)
    if payload is not None:
        (root / "limn_dual_scope.json").write_text(json.dumps(payload), encoding="utf-8")
    for scope in SCOPES:
        for condition in CONDITIONS:
            rows = [
                {"qid": qid, "scope": scope, "condition": condition,
                 "target_rank": 10 * (CONDITIONS.index(condition) + 1) + offset,
                 "target_score": 0.5, "checkpoint_sha256": "a" * 64, "gallery_size": 74381,
                 "depth": 1, "model_text": f"text {qid}",
                 "candidates": [{"rank": 1, "gallery_position": 0, "gallery_id": "IMG0",
                                 "image_path": "mbeir_images/fashioniq_images/IMG0.jpg",
                                 "score": 0.9, "is_target": False}],
                 "query_image_path": f"queries/{qid}.png"}
                for offset, qid in enumerate(QIDS)
            ]
            (root / f"limn_dual_scope.{scope}.{condition}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return root


def _config(root: Path) -> SimpleNamespace:
    # The reader anchors the fixed repo-relative evidence path on CIR_REPO_ROOT.
    return SimpleNamespace(CIR_REPO_ROOT=root.parents[len(limn_diagnostic.EVIDENCE_RELATIVE) - 1])


def test_canonical_status_is_returned_with_the_comparison(tmp_path: Path) -> None:
    payload = _payload()
    root = _root(tmp_path, payload)
    result = load_diagnostic(_config(root))
    assert result["available"] is True
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    # The canonical state is not optional metadata: it is in the same payload, with its own counts.
    assert result["canonical_status"]["scored_count"] == 6001
    assert result["canonical_status"]["unscored_count"] == 2
    assert sorted(result["canonical_status"]["unscored_qids"]) == sorted(QIDS)
    assert "diagnostic" in result["label"].lower()
    assert set(result["queries"]) == set(QIDS)


def test_one_scope_alone_is_refused(tmp_path: Path) -> None:
    """A single-scope payload would read as "the category was resolved to this scope"."""
    root = _root(tmp_path, _payload(scopes=("shirt",)))
    with pytest.raises(DiagnosticEvidenceError, match="missing scope toptee"):
        load_diagnostic(_config(root))


def test_missing_condition_is_refused(tmp_path: Path) -> None:
    payload = _payload()
    del payload["queries"]["7:3038"]["scopes"]["toptee"]["image_only_empty_text"]
    root = _root(tmp_path, payload)
    with pytest.raises(DiagnosticEvidenceError, match="missing condition"):
        load_diagnostic(_config(root))


def test_canonical_unscored_set_must_match_the_two_queries(tmp_path: Path) -> None:
    payload = _payload()
    payload["canonical_status"]["unscored_qids"] = ["7:3038"]
    root = _root(tmp_path, payload)
    with pytest.raises(DiagnosticEvidenceError, match="unscored queries"):
        load_diagnostic(_config(root))


def test_absent_evidence_is_a_named_absence_not_an_empty_comparison(tmp_path: Path) -> None:
    root = _root(tmp_path, None)
    result = load_diagnostic(_config(root))
    assert result["available"] is False
    assert result["status"] == "DIAGNOSTIC_NOT_PUBLISHED"
    assert result["queries"] == {}


def test_wrong_artifact_type_is_refused(tmp_path: Path) -> None:
    payload = _payload()
    payload["artifact_type"] = "something_else"
    root = _root(tmp_path, payload)
    with pytest.raises(DiagnosticEvidenceError, match="artifact type"):
        load_diagnostic(_config(root))


def test_candidates_come_from_the_requested_scope_only(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    shirt = load_candidates("7:3038", "shirt", "multimodal", config=_config(root))
    toptee = load_candidates("7:3038", "toptee", "multimodal", config=_config(root))
    assert shirt["scope"] == "shirt" and toptee["scope"] == "toptee"
    # Same rank in this fixture, but each answer is read from its own file and carries that scope's
    # checkpoint digest -- the reader has no code path that could mix them.
    assert shirt["checkpoint_sha256"] == toptee["checkpoint_sha256"] == "a" * 64
    assert shirt["target_rank"] == toptee["target_rank"] == 10


def test_unknown_scope_and_condition_are_refused(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    for scope, condition in (("dress", "multimodal"), ("shirt", "audio_only"),
                             ("shirt", "multimodal")):
        if (scope, condition) == ("shirt", "multimodal"):
            continue
        with pytest.raises(DiagnosticEvidenceError):
            load_candidates("7:3038", scope, condition, config=_config(root))


def test_unknown_query_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path, _payload())
    with pytest.raises(DiagnosticEvidenceError, match="not one of the ambiguous"):
        load_candidates("7:1", "shirt", "multimodal", config=_config(root))


def test_shipped_evidence_is_consistent_if_present() -> None:
    """The real artifact, when this checkout has it: both scopes, both queries, both conditions."""
    real = limn_diagnostic.evidence_root()
    if not (real / "limn_dual_scope.json").is_file():
        pytest.skip("dual-scope evidence is not present on this host (external artifact)")
    result = load_diagnostic()
    assert result["available"] is True
    assert sorted(result["queries"]) == sorted(QIDS)
    for qid in QIDS:
        assert sorted(result["queries"][qid]["scopes"]) == sorted(SCOPES)
        for scope in SCOPES:
            assert sorted(result["queries"][qid]["scopes"][scope]) == sorted(CONDITIONS)
    # The diagnostic never rewrites the canonical count it exists beside.
    assert result["canonical_status"]["scored_count"] == 6001
    assert result["canonical_status"]["unscored_count"] == 2
    for qid in QIDS:
        for scope in SCOPES:
            window = load_candidates(qid, scope, "multimodal")
            assert window["qid"] == qid and window["scope"] == scope
            assert window["target_rank"] >= 1
            assert len(window["candidates"]) == 100
            ranks = [row["rank"] for row in window["candidates"]]
            assert ranks == list(range(1, 101))


def test_missing_per_scope_export_is_refused(tmp_path: Path) -> None:
    """A summary without its six per-scope exports must not be served as a complete comparison.

    Measured before this test existed: with the summary present and 0 of 6 JSONL files on disk,
    `load_diagnostic` returned `available=True`, `status=DIAGNOSTIC_COMPLETE`, served all 12 ranks,
    and reported `evidence_files.per_scope == []`. The panel then advertised six per-scope files
    while every cell's evidence link 404'd -- a comparison that cannot be checked is not evidence.
    The module docstring already declared this refusal; the code did not implement it.
    """
    root = _root(tmp_path, _payload())
    for path in root.glob("limn_dual_scope.*.jsonl"):
        path.unlink()
    with pytest.raises(DiagnosticEvidenceError, match="per-scope export"):
        load_diagnostic(_config(root))


def test_payload_declares_the_canonical_cohort(tmp_path: Path) -> None:
    """The canonical 6001/6003 must name the cohort it belongs to.

    Measured on the live app: the `sf` subset page showed a 4,000/4,069 row beside a panel badge
    reading "6,001/6,003", and the `validated` page showed 581/586 beside the same badge. The payload
    carried no subset at all, so the panel could not tell the reader that the number belongs to a
    different cohort. A count that does not say which population it counts is the exact failure this
    diagnostic exists to prevent.
    """
    root = _root(tmp_path, _payload())
    result = load_diagnostic(_config(root))
    cohort = result.get("canonical_cohort")
    assert isinstance(cohort, dict), "the payload must carry its canonical cohort"
    assert cohort.get("subset"), "the cohort must name its subset"
    assert cohort.get("query_count") == 6003, cohort
    # The per-subset denominators are what let a page say 'this page is sf, the number is full'.
    per_subset = cohort.get("query_count_per_subset") or {}
    assert per_subset.get("sf") and per_subset.get("validated"), per_subset

def _deep_window(*, target_rank: int = 43, depth: int = 100) -> dict:
    """An export whose target sits at `target_rank` and which stores `depth` rows.

    The real finding: LIMN `7:3038/toptee/MM` has its target at rank 43 inside a 100-row export.
    A depth=10 or depth=20 request serves ten or twenty rows that do NOT contain it, and the
    response used to answer `in_window: true` because presence was decided against the stored
    window instead of the served one.
    """
    payload = _payload()
    for qid in QIDS:
        offset = list(QIDS).index(qid)
        row = payload["queries"][qid]
        row["scopes"]["toptee"]["multimodal"].update({
            "target_rank": target_rank, "target_score": 0.56,
            "top100_export": "limn_dual_scope.toptee.multimodal.jsonl"})
        payload["checkpoints"]["toptee"]["sha256"] = "a" * 64
        del offset
    return payload

def _deep_root(tmp_path: Path, *, depth: int = 100, target_rank: int = 43) -> Path:
    payload = _deep_window(target_rank=target_rank, depth=depth)
    root = tmp_path.joinpath(*limn_diagnostic.EVIDENCE_RELATIVE)
    root.mkdir(parents=True)
    (root / "limn_dual_scope.json").write_text(json.dumps(payload), encoding="utf-8")
    for scope in SCOPES:
        for condition in CONDITIONS:
            rows = []
            for offset, qid in enumerate(QIDS):
                target = target_rank if scope == "toptee" and condition == "multimodal" else 10 + offset
                candidates = [
                    {"rank": rank, "gallery_position": rank - 1,
                     "gallery_id": f"IMG{rank}", "image_path": f"mbeir_images/fashioniq_images/IMG{rank}.jpg",
                     "score": 0.9 - rank / 1000, "is_target": rank == target}
                    for rank in range(1, depth + 1)
                ]
                rows.append({"qid": qid, "scope": scope, "condition": condition,
                             "target_rank": target, "target_score": 0.56,
                             "checkpoint_sha256": "a" * 64, "gallery_size": 74381,
                             "depth": depth, "model_text": f"text {qid}",
                             "candidates": candidates,
                             "query_image_path": f"queries/{qid}.png"})
            (root / f"limn_dual_scope.{scope}.{condition}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return root

@pytest.mark.parametrize("depth,expect_in_window", [(10, False), (20, False), (43, True), (100, True)])
def test_presence_is_decided_on_the_window_that_is_served(tmp_path: Path, depth: int,
                                                          expect_in_window: bool) -> None:
    """W-03: a target at rank 43 must NOT read as in-window for K=10 or K=20."""
    root = _deep_root(tmp_path)
    payload = load_candidates("7:3038", "toptee", "multimodal", depth=depth, config=_config(root))
    served_ranks = {row["rank"] for row in payload["candidates"]}
    assert payload["target"]["in_window"] is expect_in_window, (
        f"depth={depth}: in_window={payload['target']['in_window']} but the target rank "
        f"{payload['target']['rank']} is {'in' if expect_in_window else 'not in'} "
        f"{sorted(served_ranks)[:3]}…{sorted(served_ranks)[-3:]}")
    # The claim must agree with the rows actually returned, which is the only thing a view shows.
    assert payload["target"]["in_window"] == (payload["target"]["rank"] in served_ranks)
    # Both depths are still reported, so the difference is visible rather than implied.
    assert payload["depth_served"] == depth
    assert payload["depth_stored"] == 100
    assert payload["target"]["in_stored_window"] is True

def test_the_stored_window_is_reported_even_when_the_served_one_excludes_the_target(tmp_path: Path) -> None:
    """A target outside K but inside the stored export is a different fact from one outside both."""
    root = _deep_root(tmp_path)
    narrow = load_candidates("7:3038", "toptee", "multimodal", depth=10, config=_config(root))
    assert narrow["target"]["in_window"] is False
    assert narrow["target"]["in_stored_window"] is True
    assert narrow["target"]["window_served"] == 10
    assert narrow["target"]["window_stored"] == 100
    # The target's own identity travels with it, so a view can render it without the candidate list.
    # `image_url` is None here because this fixture config declares no FASHIONIQ_ROOT: the honest
    # answer when no root is configured, not an invented path.
    assert narrow["target"]["image_url"] is None, narrow["target"]
    assert narrow["target"]["image_stem"] == "IMG43", narrow["target"]
    assert narrow["target"]["gallery_id"] == "IMG43", narrow["target"]
