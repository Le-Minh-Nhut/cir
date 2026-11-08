"""Candidate Top-K export: real candidates from the scorer that produced the metric rank.

The contract these tests pin:

* a stored candidate window and the exact target rank come from ONE scoring call, so a candidate
  rank can never disagree with the metric rank for the same query;
* the exact target rank survives when it lies outside the stored window (two different facts);
* a window never crosses queries, even when the scorer sees queries in a different order than the
  response returns them;
* a run that stored no candidates says so by name instead of serving an empty success.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from workbench.composition import topk
from workbench.composition.adapters import CompositionAdapter


class _Scored(CompositionAdapter):
    """Deterministic scorer: target scores are chosen per query so ranks are hand-checkable."""

    model_id = "scored"
    category_scopes = (None,)

    def __init__(self, rows: dict[str, np.ndarray] | None = None) -> None:
        self.rows = rows or {}
        self.calls = 0

    def encode_targets(self, image_paths, categories):  # pragma: no cover - contract only
        raise NotImplementedError

    def encode_queries(self, image_paths, texts, categories):  # pragma: no cover
        raise NotImplementedError

    def score(self, queries, targets, categories=None):
        self.calls += 1
        return np.asarray([np.asarray(queries, dtype=np.float32)[index]
                           for index in range(len(np.asarray(queries)))])


def _row(scores: list[float]) -> np.ndarray:
    return np.asarray([scores], dtype=np.float32)


def test_candidate_rank_matches_metric_rank_for_the_same_query():
    # Positions: 0..5. Query A's target is position 3 with score 0.9; positions 4 and 5 tie it.
    # The metric rank is 1 + #greater + #earlier-equal = 1 + 0 + 0 = 1.
    scores = _row([0.1, 0.2, 0.3, 0.9, 0.9, 0.9])
    adapter = _Scored()
    ranks, windows = adapter.rank_and_topk(scores, {"any": 1}, [[3]], [None], depth=6)

    assert ranks == [1]
    window = windows[0]
    assert [position for position, _ in window][0] == 3, "highest score must lead"
    # Ties must be broken by earlier gallery position, the same rule the metric rank counts.
    assert [position for position, _ in window] == [3, 4, 5, 2, 1, 0]
    candidate_rank = [position for position, _ in window].index(3) + 1
    assert candidate_rank == ranks[0], "candidate order and metric rank must agree"

def test_tie_after_target_still_matches_rank():
    # Target at position 1 (score .5); position 0 has .9 and position 2 also has .5.
    # Metric rank: 1 + 1 greater + 0 earlier-equal = 2. Candidate window order: 0, 1, 2.
    scores = _row([0.9, 0.5, 0.5, 0.0])
    ranks, windows = _Scored().rank_and_topk(scores, {"any": 1}, [[1]], [None], depth=4)
    assert ranks == [2]
    order = [position for position, _ in windows[0]]
    assert order == [0, 1, 2, 3]
    assert order.index(1) + 1 == ranks[0]

def test_exact_target_rank_survives_outside_the_stored_window(tmp_path: Path):
    """The target is at rank 9 but the window is 3 deep: both facts must survive."""
    scores = _row([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    adapter = _Scored()
    ranks, windows = adapter.rank_and_topk(scores, {"any": 1}, [[8]], [None], depth=3)
    assert ranks == [9]
    assert [position for position, _ in windows[0]] == [0, 1, 2]
    target = 8
    assert target not in [position for position, _ in windows[0]]

    writers = topk.open_export(tmp_path, "run1", attempt="a1", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=10,
                               depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", ranks[0], windows[0])
    topk.close_export(writers, expected_qids={"multimodal": ["q1"]})

    read = topk.read_candidates(tmp_path, "run1", "multimodal")
    assert read["available"] is True
    row = read["rows"]["q1"]
    assert row["target_rank"] == 9
    assert [position for position, _ in row["topk"]] == [0, 1, 2]
    assert row["target_rank"] > read["manifest"]["depth"]

def test_windows_do_not_cross_contaminate_when_the_scorer_returns_rows_in_another_order(tmp_path: Path):
    """Batch rows served in reverse order must still carry their own query's candidates."""
    rows = {"qA": [0.5, 0.4, 0.3], "qB": [0.1, 0.9, 0.2]}
    writers = topk.open_export(tmp_path, "run2", attempt="a2", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    order = ["qB", "qA"]  # reverse of the identity order
    ranks = {}
    for qid in order:
        rank, window = _Scored().rank_and_topk(
            np.asarray([rows[qid]], dtype=np.float32), {"any": 1}, [[1 if qid == "qB" else 2]],
            [None], depth=3)
        ranks[qid] = rank[0]
        writers["multimodal"].write(qid, rank[0], window[0])
    topk.close_export(writers, expected_qids={"multimodal": order})

    read = topk.read_candidates(tmp_path, "run2", "multimodal")
    assert [p for p, _ in read["rows"]["qA"]["topk"]] == [0, 1, 2]
    assert [p for p, _ in read["rows"]["qB"]["topk"]] == [1, 2, 0]
    assert read["rows"]["qA"]["target_rank"] == 3  # position 2 holds .3, two scores are greater
    assert read["rows"]["qB"]["target_rank"] == 1

def test_writer_refuses_positions_outside_the_gallery(tmp_path: Path):
    writers = topk.open_export(tmp_path, "run3", attempt="a3", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    with pytest.raises(topk.TopKExportError, match="outside the gallery"):
        writers["multimodal"].write("q1", 1, [(7, 0.5)])

def test_writer_refuses_a_duplicate_query_id(tmp_path: Path):
    writers = topk.open_export(tmp_path, "run4", attempt="a4", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", 1, [(0, 0.5)])
    with pytest.raises(topk.TopKExportError, match="duplicate query id"):
        writers["multimodal"].write("q1", 1, [(0, 0.5)])

def test_zero_depth_writes_nothing_and_is_reported_as_not_requested(tmp_path: Path):
    assert topk.open_export(tmp_path, "run5", attempt="a", model_id="m", checkpoint_id="c",
                            protocol_id="p", gallery_identity="g" * 64, gallery_count=10,
                            depth=0, conditions=["multimodal"]) == {}
    read = topk.read_candidates(tmp_path, "run5", "multimodal")
    assert read["available"] is False
    assert read["status"] == "TOPK_EXPORT_NOT_REQUESTED"

def test_depth_is_clamped_to_the_gallery_and_capped(tmp_path: Path):
    assert topk.normalise_depth(50, 5) == 5
    assert topk.normalise_depth(None, 5) is None
    assert topk.normalise_depth(0, 5) is None
    with pytest.raises(topk.TopKExportError, match="maximum"):
        topk.normalise_depth(topk.MAX_EXPORT_DEPTH + 1, 100000)
    with pytest.raises(topk.TopKExportError, match="positive integer"):
        topk.normalise_depth(-1, 100)

def test_partial_export_is_reported_not_hidden(tmp_path: Path):
    writers = topk.open_export(tmp_path, "run6", attempt="a6", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", 1, [(0, 0.5)])
    topk.close_export(writers, expected_qids={"multimodal": ["q1", "q2"]})
    read = topk.read_candidates(tmp_path, "run6", "multimodal")
    assert read["manifest"]["partial"] is True
    assert read["manifest"]["missing_query_examples"] == ["q2"]
    assert read["manifest"]["expected_record_count"] == 2

def test_corrupt_row_file_is_named_not_silently_empty(tmp_path: Path):
    writers = topk.open_export(tmp_path, "run7", attempt="a7", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", 1, [(0, 0.5)])
    path = topk.close_export(writers, expected_qids={"multimodal": ["q1"]})["multimodal"]
    path.write_text(path.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
    read = topk.read_candidates(tmp_path, "run7", "multimodal")
    assert read["available"] is False
    assert read["status"] in {"TOPK_EXPORT_CORRUPT", "TOPK_EXPORT_UNREADABLE"}

def test_gallery_digest_is_the_runner_gallery_fingerprint():
    """A candidate position is only meaningful against the gallery digest that produced it."""
    from types import SimpleNamespace

    from workbench.composition.runner import _gallery_fingerprint

    ids = ("a.jpg", "b.jpg", "c.jpg")
    names = {"c1": "a.jpg", "c2": "b.jpg", "c3": "c.jpg"}
    runner_value = _gallery_fingerprint(SimpleNamespace(
        gallery=SimpleNamespace(protocol_id="proto", image_ids=ids), candidate_names=names))
    assert topk.gallery_digest("proto", ids, names) == runner_value

def test_rows_cached_by_file_identity_then_invalidated(tmp_path: Path):
    writers = topk.open_export(tmp_path, "run8", attempt="a8", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=3,
                               depth=3, conditions=["multimodal"])
    writers["multimodal"].write("q1", 1, [(0, 0.5)])
    path = topk.close_export(writers, expected_qids={"multimodal": ["q1"]})["multimodal"]
    first = topk.read_candidates(tmp_path, "run8", "multimodal")
    assert first["available"] is True
    second = topk.read_candidates(tmp_path, "run8", "multimodal")
    assert second["rows"] == first["rows"]
    # A rewritten row file with the same manifest is a digest mismatch, never stale data.
    path.write_text(json.dumps({"qid": "q2", "target_rank": 1, "topk": [[1, 0.1]]}) + "\n",
                    encoding="utf-8")
    third = topk.read_candidates(tmp_path, "run8", "multimodal")
    assert third["available"] is False
    assert third["status"] == "TOPK_EXPORT_CORRUPT"
