"""The API surface: capabilities, candidate windows, and unscored-vs-failure separation.

These go through a real dependency root (a temporary ``artifacts`` tree) and the real reader, so a
claimed capability is proven against a file the reader can open, not against a dict the test built.
"""
from __future__ import annotations

import json
from pathlib import Path

from workbench.backend.composition import get_composition_run_detail, get_composition_run_queries
from workbench.composition import topk

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def _root(tmp_path: Path, run_id: str, *, metrics=None) -> Path:
    summary = tmp_path / "composition" / "summary" / f"m_c_{run_id}" / "full.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "schema_version": 1, "model_id": "m", "checkpoint_id": "c", "run_id": run_id,
        "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
        "conditions": list(CONDITIONS), "query_count": 3, "gallery_count": 5,
        "checkpoint_identity": "ck", "gallery_identity_sha256": "g" * 64,
        "execution_status": "COMPLETED", "seed": 0,
        "runtime": {"model_interpreter": {"environment_fingerprint": "env"}},
        "metrics": metrics or {"coverage": 2 / 3, "gap_cohort_count": 3, "multimodal": {
            "ndcg_full": 0.3, "recall_at_10": 33.0, "recall_at_50": 66.0, "mrr": 0.4,
            "query_count": 3, "scored_count": 2, "coverage": 2 / 3}},
    }), encoding="utf-8")
    return tmp_path


def _ranks(root: Path, run_id: str, *, ranks: dict, failures: dict | None = None) -> None:
    path = root / "composition" / "runs" / run_id / "ranks" / "m_c.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": "f", "attempt": run_id, "ranks": ranks,
                                "query_failures": failures or {}}), encoding="utf-8")


def _candidates(root: Path, run_id: str, rows: dict[str, list[tuple[int, float | None]]],
                depth: int, gallery_count: int = 5) -> None:
    writers = topk.open_export(root, run_id, attempt=run_id, model_id="m", checkpoint_id="c",
                               protocol_id="circus_fashioniq_release_v1",
                               gallery_identity=gallery_identity(), gallery_count=gallery_count,
                               depth=depth, conditions=["multimodal"])
    for qid, window in rows.items():
        writers["multimodal"].write(qid, None, window)
    topk.close_export(writers, expected_qids={"multimodal": list(rows)})


def test_detail_reports_capabilities_from_the_artifacts(tmp_path: Path):
    root = _root(tmp_path, "run_a")
    _ranks(root, "run_a", ranks={"multimodal": {"7:1": 2, "7:2": None, "7:3": 40},
                                 "text_only_black_image": {"7:1": 3, "7:2": None, "7:3": 40},
                                 "image_only_empty_text": {"7:1": 4, "7:2": None, "7:3": 40}},
           failures={c: {"7:2": "query features unavailable"} for c in CONDITIONS})
    detail = get_composition_run_detail("run_a", subset="full", artifacts_root=root)
    assert detail is not None
    assert detail["capabilities"] == {"aggregate_metrics": True, "per_query_target_ranks": True,
                                      "candidate_topk": False,
                                      "image_availability": detail["capabilities"]["image_availability"]}
    assert detail["capability_detail"]["candidate_topk"]["status"] == "TOPK_EXPORT_NOT_REQUESTED"
    assert detail["measurement"]["sample_count"] == 3
    assert detail["measurement"]["unscored_query_count"] == 1
    assert detail["measurement"]["unscored_by_kind"] == {"features_unavailable": 1}

def test_query_page_separates_unscored_from_retrieval_failure(tmp_path: Path, monkeypatch):
    """A query with no rank and a query ranked 9 are different facts, not both "failed"."""
    root = _root(tmp_path, "run_b")
    _ranks(root, "run_b", ranks={"multimodal": {"7:1": 2, "7:2": None, "7:3": 40},
                                 "text_only_black_image": {"7:1": 3, "7:2": None, "7:3": 40},
                                 "image_only_empty_text": {"7:1": 4, "7:2": None, "7:3": 40}},
           failures={c: {"7:2": "query 7:2: category unresolved has no adapter embedding scope"}
                     for c in CONDITIONS})
    # The real release supplies the three query identities these qids name.
    release = _fake_release()
    monkeypatch.setattr("workbench.backend.composition.load_cached_release", lambda *a, **k: release)
    page = get_composition_run_queries("run_b", subset="full", artifacts_root=root)
    by_id = {item["query_id"]: item for item in page["items"]}
    assert by_id["7:1"]["query_state"] == "hit"
    assert by_id["7:2"]["query_state"] == "unscored"
    assert by_id["7:2"]["unscored_kind"] == "category_unresolved"
    assert by_id["7:2"]["retrieval_failure"] is False
    assert by_id["7:3"]["query_state"] == "retrieval_failure"
    assert by_id["7:3"]["unscored"] is False
    assert page["counters"] == {"scored": 2, "unscored": 1, "hit_at_k": 1,
                                "retrieval_failure": 1,
                                "unscored_by_kind": {"category_unresolved": 1},
                                "candidates_available": 0}

def test_query_state_filter_selects_one_fact(tmp_path: Path, monkeypatch):
    root = _root(tmp_path, "run_c")
    _ranks(root, "run_c", ranks={c: {"7:1": 2, "7:2": None, "7:3": 40} for c in CONDITIONS},
           failures={c: {"7:2": "query features unavailable"} for c in CONDITIONS})
    monkeypatch.setattr("workbench.backend.composition.load_cached_release", lambda *a, **k: _fake_release())
    only_unscored = get_composition_run_queries("run_c", subset="full", artifacts_root=root,
                                               query_state="unscored")
    assert [item["query_id"] for item in only_unscored["items"]] == ["7:2"]
    only_failures = get_composition_run_queries("run_c", subset="full", artifacts_root=root,
                                                query_state="retrieval_failure")
    assert [item["query_id"] for item in only_failures["items"]] == ["7:3"]

def test_candidate_window_is_served_and_target_rank_can_lie_outside_it(tmp_path: Path, monkeypatch):
    root = _root(tmp_path, "run_d")
    _ranks(root, "run_d", ranks={c: {"7:1": 4, "7:2": None, "7:3": 40} for c in CONDITIONS},
           failures={c: {"7:2": "query features unavailable"} for c in CONDITIONS})
    _candidates(root, "run_d", {"7:1": [(3, 0.9), (0, 0.5)], "7:3": [(1, 0.7)]}, depth=2)
    monkeypatch.setattr("workbench.backend.composition.load_cached_release", lambda *a, **k: _fake_release())

    page = get_composition_run_queries("run_d", subset="full", artifacts_root=root, top_k=10)
    by_id = {item["query_id"]: item for item in page["items"]}
    assert page["candidates"]["available"] is True
    assert page["candidates"]["depth"] == 2
    assert [row["gallery_position"] for row in by_id["7:1"]["top_results"]] == [3, 0]
    # The exact rank (4) is a position in the FULL gallery and survives outside the 2-deep window.
    assert by_id["7:1"]["target_rank"] == 4
    assert by_id["7:1"]["candidate_status"] == "AVAILABLE"
    # A query with no stored window says so; it is not reported as "no candidates at all".
    assert by_id["7:2"]["candidate_status"] == "NO_CANDIDATES_FOR_QUERY"
    assert by_id["7:2"]["top_results"] == []

def test_top_k_parameter_bounds_the_returned_window_only(tmp_path: Path, monkeypatch):
    root = _root(tmp_path, "run_e")
    _ranks(root, "run_e", ranks={c: {"7:1": 1, "7:2": None, "7:3": 9} for c in CONDITIONS},
           failures={c: {"7:2": "query features unavailable"} for c in CONDITIONS})
    _candidates(root, "run_e", {"7:1": [(3, 0.9), (0, 0.5), (1, 0.4)], "7:3": []}, depth=3)
    monkeypatch.setattr("workbench.backend.composition.load_cached_release", lambda *a, **k: _fake_release())
    page = get_composition_run_queries("run_e", subset="full", artifacts_root=root, top_k=2)
    by_id = {item["query_id"]: item for item in page["items"]}
    assert len(by_id["7:1"]["top_results"]) == 2
    assert by_id["7:1"]["candidate_depth"] == 3, "the stored depth is still reported"

def test_gallery_mismatch_refuses_instead_of_resolving_wrong_images(tmp_path: Path, monkeypatch):
    root = _root(tmp_path, "run_f")
    _ranks(root, "run_f", ranks={c: {"7:1": 2, "7:2": None, "7:3": 40} for c in CONDITIONS},
           failures={c: {"7:2": "query features unavailable"} for c in CONDITIONS})
    writers = topk.open_export(root, "run_f", attempt="run_f", model_id="m", checkpoint_id="c",
                               protocol_id="circus_fashioniq_release_v1",
                               gallery_identity="d" * 64,  # deliberately not the loaded gallery
                               gallery_count=5, depth=2, conditions=["multimodal"])
    writers["multimodal"].write("7:1", 2, [(0, 0.5), (1, 0.4)])
    topk.close_export(writers, expected_qids={"multimodal": ["7:1"]})
    monkeypatch.setattr("workbench.backend.composition.load_cached_release", lambda *a, **k: _fake_release())
    page = get_composition_run_queries("run_f", subset="full", artifacts_root=root)
    item = next(entry for entry in page["items"] if entry["query_id"] == "7:1")
    assert item["candidate_status"] == "TOPK_EXPORT_GALLERY_MISMATCH"
    assert item["top_results"] == []


def gallery_identity() -> str:
    """The digest this temporary gallery produces (same recipe the reader recomputes)."""
    return topk.gallery_digest("circus_fashioniq_release_v1",
                               ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"),
                               {name: name for name in
                                ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")})

def _fake_release():
    """A release whose gallery and query identities are exactly what the rank artifacts name."""
    from types import SimpleNamespace

    gallery = SimpleNamespace(protocol_id="circus_fashioniq_release_v1",
                              image_ids=("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"))
    queries = []
    for index, qid in enumerate(("7:1", "7:2", "7:3"), 1):
        queries.append(SimpleNamespace(qid=qid, index=index - 1, query_txt=f"edit {index}",
                                       reference_id="a.jpg", positive_ids=("b.jpg",),
                                       category="dress", subsets=("full", "sf", "validated")))
    names = {"a.jpg": "a.jpg", "b.jpg": "b.jpg", "c.jpg": "c.jpg", "d.jpg": "d.jpg", "e.jpg": "e.jpg"}
    release = SimpleNamespace(gallery=gallery, queries=tuple(queries), candidate_names=names)
    # The digest the reader recomputes must equal the one the writers were opened with.
    digest = topk.gallery_digest(gallery.protocol_id, gallery.image_ids, names)
    assert len(digest) == 64
    return release
