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
    BLAS_THREADS_PINNED, SCORE_THREADS_ENV, _blas_thread_limit, _digest,
    _rank_groups_concurrently, rank_in_query_chunks, resolved_score_threads,
    score_in_query_chunks,
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
    """A full 74381-image gallery must never be buffered; at most workers*2 batches outstanding.

    Counts batches that have been SUBMITTED but not yet consumed, which is what occupies memory:
    a completed future still holds its decoded tensor. `ThreadPoolExecutor.map` submits everything
    up front and would show a peak equal to the batch count, so this test fails on that shape.
    """
    import threading
    import time

    workers = 4
    live = {"outstanding": 0, "peak": 0}
    lock = threading.Lock()

    def decode(chunk):
        with lock:
            live["outstanding"] += 1
            live["peak"] = max(live["peak"], live["outstanding"])
        time.sleep(0.001)
        return chunk

    batches = [[index] for index in range(64)]
    seen = []
    for batch in prefetch_rows(decode, batches, workers):
        seen.append(batch)
        # Consuming one batch retires exactly one outstanding slot.
        with lock:
            live["outstanding"] -= 1
    assert [item for batch in seen for item in batch] == list(range(64))
    assert live["peak"] <= workers * 2, (
        f"prefetch_rows held {live['peak']} batches in flight for {len(batches)} batches; "
        f"a bounded pipeline must never exceed workers*2 = {workers * 2}")

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


def test_group_size_does_not_depend_on_thread_count(monkeypatch):
    """D2: how many queries share one score() call is arithmetic, not a performance knob.

    The chunk size must come from the caller, so changing WORKBENCH_SCORE_THREADS can never change
    the group shape an adapter sees (the BLAS matrix M dimension) for an unchanged run id.
    """
    seen = []

    class Recorder(_FakeScorer):
        def rank(self, queries, targets, positions, categories=None):
            seen.append(len(queries))
            return super().rank(queries, targets, positions, categories)

    recorder = Recorder()
    queries = [[i] for i in range(64)]
    targets = [1.0]
    positions = [[0] for _ in range(64)]
    for threads in (1, 2, 4, 8):
        monkeypatch.setenv(SCORE_THREADS_ENV, str(threads))
        seen.clear()
        rank_in_query_chunks(recorder, queries, targets, positions, None, 0,
                            chunk_size=16)
        assert set(seen) == {16}, f"threads={threads} produced group sizes {sorted(set(seen))}"

def test_concurrent_groups_are_identical_to_serial_for_matmul(monkeypatch):
    """D3: an @-based scorer must not move a rank when groups run concurrently.

    Multi-threaded BLAS x multiple group threads is the collision the single BLAS toggle removes;
    a per-call toggle (the first attempt) raced and was measured to move ranks at 4 and 8 threads.
    """
    import numpy as np

    class MatmulScorer(_FakeScorer):
        def rank(self, queries, targets, positions, categories=None):
            q = np.asarray(queries, dtype=np.float32)
            t = np.asarray(targets, dtype=np.float32)
            scores = q @ t.T
            return [1 + int(np.count_nonzero(scores[row] > scores[row][positive[0]]))
                    for row, positive in enumerate(positions)]

    rng = np.random.default_rng(11)
    scorer = MatmulScorer()
    queries = rng.random((128, 16)).astype(np.float32)
    targets = rng.random((256, 16)).astype(np.float32)
    positions = [[int(rng.integers(0, 256))] for _ in range(128)]
    groups = [(list(range(s, s + 32)), positions[s:s + 32], queries[s:s + 32], None)
              for s in range(0, 128, 32)]

    def rank_group(group):
        indices, group_positions, vectors, _ = group
        return scorer.rank(vectors, targets, group_positions), []

    serial = []
    for outcome in _rank_groups_concurrently(rank_group, groups, 1):
        serial.extend(outcome[0])
    for threads in (2, 4, 8):
        parallel = []
        for outcome in _rank_groups_concurrently(rank_group, groups, threads):
            parallel.extend(outcome[0])
        assert parallel == serial, f"threads={threads} changed ranks"

def test_concurrent_group_failure_does_not_abort_siblings():
    """A raised group returns in its own slot so the caller's per-batch fallback still runs."""
    def rank_group(group):
        if group == "bad":
            raise RuntimeError("boom")
        return ([1], [])

    outcomes = _rank_groups_concurrently(rank_group, ["ok", "bad", "ok"], 2)
    assert outcomes[0] == ([1], [])
    assert isinstance(outcomes[1], RuntimeError)
    assert outcomes[2] == ([1], [])


def test_both_serial_and_concurrent_branches_pin_blas(monkeypatch):
    """F1-regression: the BLAS pin must cover the SERIAL branch too.

    BLAS thread count is not numerically neutral (measured score delta 3.576e-07 at 1 thread vs
    2.384e-07 at 10). If only the concurrent branch pinned, WORKBENCH_SCORE_THREADS=1 and =8 would
    compute under different BLAS threading and the knob could move published numbers.
    """
    import workbench.composition.runner as runner_module

    calls = []

    def fake_limit(threads):
        calls.append(int(threads))
        return 7  # a distinctive "previous" value

    monkeypatch.setattr(runner_module, "_blas_thread_limit", fake_limit)
    groups = [("g0",), ("g1",), ("g2",), ("g3",)]

    for threads, expected_parallel in ((1, False), (4, True)):
        calls.clear()
        runner_module._rank_groups_concurrently(lambda group: ([1], []), groups, threads)
        assert 1 in calls, f"threads={threads} never pinned BLAS to 1: {calls}"
        assert calls[-1] == 7, (
            f"threads={threads} did not restore the previous BLAS count: {calls}")
        # exactly one pin and one restore for the whole call, never per worker
        assert calls.count(1) == 1, (
            f"threads={threads} pinned BLAS {calls.count(1)} times; must toggle once per call")

def test_blas_pin_applies_with_a_single_group():
    """One group takes the serial branch; it must still pin and restore."""
    import workbench.composition.runner as runner_module

    calls = []

    def fake_limit(threads):
        calls.append(int(threads))
        return 3

    original = runner_module._blas_thread_limit
    runner_module._blas_thread_limit = fake_limit
    try:
        runner_module._rank_groups_concurrently(lambda group: ([1], []), [("only",)], 8)
    finally:
        runner_module._blas_thread_limit = original
    assert calls[0] == 1 and calls[-1] == 3, calls


def test_blas_policy_is_a_recorded_identity_field():
    """The pin is an ARITHMETIC input: it must partition the run id, or --resume reuses a rank
    artifact scored under a different policy. `runner.py` is not covered by any source hash, so an
    explicit identity field is the only thing that can encode it."""
    import inspect

    source = inspect.getsource(__import__("workbench.composition.runner", fromlist=["x"]))
    assert '"blas_threads_pinned"' in source, (
        "identity must record blas_threads_pinned; the helper lives in runner.py, which "
        "_module_source_hash never covers, so without this field a pin change keeps the fingerprint")
    assert isinstance(BLAS_THREADS_PINNED, int) and BLAS_THREADS_PINNED >= 1

def test_changing_the_blas_policy_changes_the_fingerprint():
    base = {"model_id": "encoder", "scoring_module_source_hash": "abc"}
    pinned = dict(base, blas_threads_pinned=1)
    unpinned = dict(base)                       # a pre-policy artifact carries no such field
    other_policy = dict(base, blas_threads_pinned=10)
    assert _digest(pinned) != _digest(unpinned), "an old artifact could still be resumed"
    assert _digest(pinned) != _digest(other_policy), "two policies must not share a run id"

def test_both_branches_execute_under_the_pinned_value(monkeypatch):
    """Serial and concurrent must both run at BLAS_THREADS_PINNED, not merely both pin something."""
    import workbench.composition.runner as runner_module

    seen = []

    def fake_limit(threads):
        seen.append((int(threads), "pin"
                     if int(threads) == BLAS_THREADS_PINNED else "restore"))
        return 4

    monkeypatch.setattr(runner_module, "_blas_thread_limit", fake_limit)
    groups = [("g",)] * 4
    for threads in (1, 8):
        seen.clear()
        runner_module._rank_groups_concurrently(lambda g: ([1], []), groups, threads)
        pins = [value for value, kind in seen if kind == "pin"]
        assert pins and set(pins) == {BLAS_THREADS_PINNED}, (
            f"threads={threads} pinned {pins}, expected {BLAS_THREADS_PINNED}")
        assert seen[-1][1] == "restore", f"threads={threads} did not restore: {seen}"
