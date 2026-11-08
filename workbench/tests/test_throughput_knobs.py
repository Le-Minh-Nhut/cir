"""Throughput knobs must not change a single feature or rank.

These pin the one property the speed work depends on: the parallel decode produces bit-identical
pixels in the same order, so no encoded feature can move.

The scoring rewrite is deliberately absent. It was measured, gave ~9-10x, and was REJECTED because
it changed ranks (see `workbench/docs/reviews/20261108T-throughput/THROUGHPUT.vi.md`); a regression
test for a change that is not shipped would document the wrong contract.
"""
from __future__ import annotations

import os

import pytest

from workbench.composition.adapters import (
    BATCH_SIZE_ENV, DECODE_WORKERS_ENV, prefetch_rows, require_batch_fits,
    resolved_batch_size, resolved_decode_workers,
    worker_environment,
)
from workbench.composition.runner import (
    SCORE_THREADS_ENV, rank_in_query_chunks, resolved_score_threads, score_in_query_chunks,
)

class _FakeScorer:
    """A scorer whose rows are independent of batch composition, like the real contraction."""

    def __init__(self):
        self.calls = 0

    def score(self, queries, targets, categories=None):
        import numpy as np
        self.calls += 1
        q = np.asarray(queries, dtype=np.float32)
        width = min(q.shape[1], np.asarray(targets, dtype=np.float32).shape[0])
        scaled = q[:, :width] * np.asarray(targets, dtype=np.float32)[:width][None, :]
        return scaled.sum(axis=1, keepdims=True)

    def rank(self, queries, targets, positions, categories=None):
        """Rank = 1 + the first positive's position, mirroring the real rule's shape."""
        return [1 + int(positive[0]) for positive in positions]

    def rank_and_topk(self, queries, targets, positions, categories, depth):
        return self.rank(queries, targets, positions, categories), [[] for _ in positions]

def test_score_threads_default_is_conservative(monkeypatch):
    monkeypatch.delenv(SCORE_THREADS_ENV, raising=False)
    assert 1 <= resolved_score_threads() <= 8

@pytest.mark.parametrize("value", ["0", "-2", "lots"])
def test_invalid_score_threads_refused(monkeypatch, value):
    monkeypatch.setenv(SCORE_THREADS_ENV, value)
    with pytest.raises(ValueError):
        resolved_score_threads()

def test_query_parallel_ranking_is_identical_to_serial(monkeypatch):
    """The contract: concurrency across queries must not move a single rank."""
    import numpy as np
    rng = np.random.default_rng(5)
    scorer = _FakeScorer()
    vectors = rng.random((40, 16)).astype(np.float32)
    targets = rng.random(200).astype(np.float32)
    positions = [[int(rng.integers(0, 200))] for _ in range(40)]
    monkeypatch.setenv(SCORE_THREADS_ENV, "1")
    serial, _ = rank_in_query_chunks(scorer, vectors, targets, positions, None, 0)
    for threads in (2, 4, 8):
        monkeypatch.setenv(SCORE_THREADS_ENV, str(threads))
        parallel, _ = rank_in_query_chunks(scorer, vectors, targets, positions, None, 0)
        assert parallel == serial

def test_query_parallel_scoring_is_identical_to_serial(monkeypatch):
    import numpy as np
    rng = np.random.default_rng(6)
    scorer = _FakeScorer()
    queries = rng.random((50, 8)).astype(np.float32)
    targets = rng.random(8).astype(np.float32)
    monkeypatch.setenv(SCORE_THREADS_ENV, "1")
    serial = score_in_query_chunks(scorer, queries, targets, 8)
    for threads in (2, 4, 8):
        monkeypatch.setenv(SCORE_THREADS_ENV, str(threads))
        assert np.array_equal(score_in_query_chunks(scorer, queries, targets, 8), serial)

def test_decode_workers_default_is_half_the_box(monkeypatch):
    monkeypatch.delenv(DECODE_WORKERS_ENV, raising=False)
    assert resolved_decode_workers() == min(10, (os.cpu_count() or 8) // 2)

def test_overrides_are_read(monkeypatch):
    monkeypatch.setenv(DECODE_WORKERS_ENV, "4")
    assert resolved_decode_workers() == 4
    monkeypatch.setenv(BATCH_SIZE_ENV, "128")
    assert resolved_batch_size() == 128

@pytest.mark.parametrize("variable", [DECODE_WORKERS_ENV, BATCH_SIZE_ENV])
@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_invalid_override_is_refused(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError):
        resolved_decode_workers() if variable == DECODE_WORKERS_ENV else resolved_batch_size()

def test_unset_batch_size_keeps_the_recorded_default(monkeypatch):
    """The default must stay 32: it is written into a run's identity as `adapter_batch_size`."""
    monkeypatch.delenv(BATCH_SIZE_ENV, raising=False)
    assert resolved_batch_size() == 32

def test_parallel_decode_preserves_order_and_values():
    """Order is what keeps feature rows aligned with the released gallery."""
    def decode(chunk):
        return [value * 2 for value in chunk]

    batches = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10]]
    serial = [decode(batch) for batch in batches]
    for workers in (1, 2, 4, 8):
        assert list(prefetch_rows(decode, batches, workers)) == serial

def test_parallel_decode_is_bounded():
    """A full 74381-image gallery must never be buffered; at most workers*2 batches are in flight."""
    live = {"peak": 0, "now": 0}

    def decode(chunk):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        live["now"] -= 1
        return chunk

    batches = [[index] for index in range(64)]
    seen = list(prefetch_rows(decode, batches, 4))
    assert [item for batch in seen for item in batch] == list(range(64))

def test_batch_guard_refuses_before_the_run_not_during_it():
    total = 16303 * 1024 * 1024
    require_batch_fits(32, 14 * 1024 * 1024, total)
    with pytest.raises(ValueError, match="VRAM"):
        require_batch_fits(4096, 14 * 1024 * 1024, total)

def test_worker_environment_pins_blas_threads(monkeypatch):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(DECODE_WORKERS_ENV, "5")
    environment = worker_environment()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        assert environment[variable] == "5"
