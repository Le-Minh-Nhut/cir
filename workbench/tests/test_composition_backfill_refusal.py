"""S-01: a resumed run must not attach a Top-K window computed by a DIFFERENT pass.

The old audit reproduced two halves of the same defect: the backfill path did not share the scoring
context (BLAS policy) with the normal rank pass, and a batch fallback changed the shape of the
batch. The BLAS half is covered by `test_review_defects_evaluator.py`; what nothing covered is the
CONSEQUENCE -- the backfill writes a candidate window for a rank the original pass never measured.

`_backfill_export` now takes `original_ranks` and raises `BackfillRankDisagreement` instead of
writing such a window. These tests pin that refusal, including the case where the caller supplies no
record at all (a window with nothing to reconcile against must NOT be silently published as the
window of that measurement).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.composition import runner


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, backfilled_rank: int,
             seen: dict | None = None) -> dict:
    work_root = tmp_path / "work"
    features = work_root / "features"
    features.mkdir(parents=True)
    np.save(features / "gallery_global.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(features / "query_multimodal_global.npy", np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(runner, "_cached_features",
                        lambda *a, **k: ({None: np.zeros((3, 4), dtype=np.float32)},
                                         {None: np.ones((1, 4), dtype=np.float32)}))
    written: dict = {}

    def fake_write_batch(writers, condition, qids, ranks, windows):
        written.setdefault("rows", []).extend(zip(qids, ranks))

    monkeypatch.setattr("workbench.composition.topk.write_batch", fake_write_batch)

    class Adapter:
        depth = 2

        def rank_and_topk(self, vectors, targets, positions, categories, depth):
            if seen is not None:
                seen["called"] = True
            return [backfilled_rank], [{("gallery", 0): (1.0, [0, 1])}]

    return {"work_root": work_root, "adapter": Adapter(), "written": written}


def _call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, backfilled_rank: int,
          original_ranks: dict | None, name: str = "one") -> dict:
    fixture = _fixture(tmp_path / name, monkeypatch, backfilled_rank=backfilled_rank)
    writer = SimpleNamespace(depth=2, write=lambda *a, **k: None)
    query = SimpleNamespace(qid="q0")
    runner._backfill_export(fixture["adapter"], [query], [None], [None], {"q0": (0,)}, {},
                            fixture["work_root"], "fp", ["multimodal"],
                            {"multimodal": writer}, 1, original_ranks=original_ranks)
    return fixture["written"]


def test_a_backfilled_rank_that_disagrees_with_the_recorded_rank_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(runner.BackfillRankDisagreement) as raised:
        _call(tmp_path, monkeypatch, backfilled_rank=2,
              original_ranks={"multimodal": {"q0": 1}})
    assert "disagree" in str(raised.value)
    assert "q0" in str(raised.value), "the refusal must name the query it could not reconcile"


def test_a_backfilled_rank_that_agrees_is_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written = _call(tmp_path, monkeypatch, backfilled_rank=1,
                    original_ranks={"multimodal": {"q0": 1}})
    assert written["rows"] == [("q0", 1)]


def test_an_unscored_query_in_the_record_is_a_disagreement_not_a_match(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The original pass recorded `None` for this query: it has no rank to reconcile against."""
    with pytest.raises(runner.BackfillRankDisagreement):
        _call(tmp_path, monkeypatch, backfilled_rank=1,
              original_ranks={"multimodal": {"q0": None}})


def test_a_condition_missing_from_the_record_is_a_disagreement(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(runner.BackfillRankDisagreement):
        _call(tmp_path, monkeypatch, backfilled_rank=1,
              original_ranks={"other_condition": {"q0": 1}})


def test_an_absent_rank_record_is_refused_rather_than_reconciled_away(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No record means no reconciliation possible. `None` means the caller has nothing to compare
    (and says so); an EMPTY mapping is a record that lacks this condition, which is a disagreement."""
    written = _call(tmp_path, monkeypatch, backfilled_rank=1, original_ranks=None, name="absent")
    assert written["rows"] == [("q0", 1)], (
        "with no record at all there is nothing to disagree with, so the window is written")
    with pytest.raises(runner.BackfillRankDisagreement):
        _call(tmp_path, monkeypatch, backfilled_rank=1, original_ranks={}, name="empty")
