"""End-to-end: a scored run's candidates agree with its metric ranks, query by query.

This drives the REAL adapter-scoring path (``CompositionAdapter.rank_and_topk`` over a full target
matrix) and the REAL writer, then reads the export back and compares it to the metric ranks the
same scoring call produced. It is CPU-only and uses no model.

The check that matters is the last one: for every query, the metric rank must equal the position of
the best positive inside the stored window whenever the target is inside that window, and the exact
rank must survive when it is not.
"""
from __future__ import annotations

import numpy as np
import pytest

from workbench.composition import topk
from workbench.composition.adapters import CompositionAdapter

GALLERY = 24
QUERIES = 12


class _FullGallery(CompositionAdapter):
    """Cosine scorer over a full gallery; the gallery is the target matrix, as in a real run."""

    model_id = "fullgallery"
    category_scopes = (None,)

    def __init__(self, gallery: np.ndarray) -> None:
        super().__init__(None, None)
        self.gallery = gallery

    def encode_targets(self, image_paths, categories):  # pragma: no cover - contract only
        raise NotImplementedError

    def encode_queries(self, image_paths, texts, categories):  # pragma: no cover
        raise NotImplementedError

    def score(self, queries, targets, categories=None):
        query_values = np.asarray(queries, dtype=np.float32)
        target_values = np.asarray(targets, dtype=np.float32)
        qn = np.linalg.norm(query_values, axis=1, keepdims=True)
        tn = np.linalg.norm(target_values, axis=1, keepdims=True)
        return (query_values / np.maximum(qn, 1e-12)) @ \
               (target_values / np.maximum(tn, 1e-12)).T


def _fixture(seed: int = 7):
    """Quantised features so exact score ties actually occur (the hard case for rank agreement)."""
    rng = np.random.default_rng(seed)
    gallery = np.round(rng.normal(size=(GALLERY, 8)), 1).astype(np.float32)
    queries = np.round(rng.normal(size=(QUERIES, 8)), 1).astype(np.float32)
    positions = [[(index * 3) % GALLERY] for index in range(QUERIES)]
    return gallery, queries, positions


def test_candidates_and_metric_ranks_agree_for_every_query(tmp_path):
    gallery, queries, positions = _fixture()
    adapter = _FullGallery(gallery)

    for depth in (1, 3, 10, GALLERY):
        writers = topk.open_export(tmp_path / f"d{depth}", "run", attempt="a", model_id="m",
                                   checkpoint_id="c", protocol_id="p",
                                   gallery_identity="g" * 64, gallery_count=GALLERY,
                                   depth=depth, conditions=["multimodal"],)
        ranks, windows = adapter.rank_and_topk(queries, gallery, positions, [None] * QUERIES,
                                               depth)
        topk.write_batch(writers, "multimodal", [f"q{i}" for i in range(QUERIES)], ranks, windows)
        topk.close_export(writers, expected_qids={"multimodal": [f"q{i}" for i in range(QUERIES)]})

        export = topk.read_candidates(tmp_path / f"d{depth}", "run", "multimodal")
        assert export["available"] is True, export["status"]

        for index, qid in enumerate(f"q{i}" for i in range(QUERIES)):
            row = export["rows"][qid]
            stored = [position for position, _score in row["topk"]]
            assert row["target_rank"] == ranks[index], (
                f"{qid}: stored target rank {row['target_rank']} != metric rank {ranks[index]}")
            if ranks[index] <= len(stored):
                assert stored[ranks[index] - 1] in positions[index], (
                    f"{qid}: candidate at the metric rank is not the positive")
                # And nothing above it is a positive either, or the metric rank is wrong.
                assert not any(position in positions[index] for position in stored[:ranks[index] - 1])
            else:
                assert positions[index][0] not in stored, (
                    f"{qid}: target is outside depth {depth} but appears in the window")

def test_window_is_sorted_by_the_scorers_own_order(tmp_path):
    gallery, queries, positions = _fixture(seed=11)
    adapter = _FullGallery(gallery)
    scores = adapter.score(queries, gallery)
    _, windows = adapter.rank_and_topk(queries, gallery, positions, [None] * QUERIES, GALLERY)
    for index, window in enumerate(windows):
        stored_scores = [score for _position, score in window]
        assert stored_scores == sorted(stored_scores, reverse=True), (
            f"q{index}: window is not in descending score order")
        for position, score in window:
            assert score == pytest.approx(float(scores[index][position]))

def test_no_duplicate_positions_in_a_window(tmp_path):
    gallery, queries, positions = _fixture(seed=3)
    adapter = _FullGallery(gallery)
    _, windows = adapter.rank_and_topk(queries, gallery, positions, [None] * QUERIES, GALLERY)
    for index, window in enumerate(windows):
        stored = [position for position, _score in window]
        assert len(set(stored)) == len(stored), f"q{index} repeats a gallery position"
        assert sorted(stored) == list(range(GALLERY)), "a full-depth window must cover the gallery"

def test_writer_refuses_a_batch_that_does_not_line_up(tmp_path):
    """A length mismatch must fail loudly: pairing a query with another query's window is the bug."""
    writers = topk.open_export(tmp_path, "run", attempt="a", model_id="m", checkpoint_id="c",
                               protocol_id="p", gallery_identity="g" * 64, gallery_count=4,
                               depth=2, conditions=["multimodal"])
    with pytest.raises(topk.TopKExportError, match="shape mismatch"):
        topk.write_batch(writers, "multimodal", ["q1", "q2"], [1], [[(0, 0.5)]])
    topk.abort_export(writers)
