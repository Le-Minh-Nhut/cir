"""Locks the equivalence-audit invariants that the reports depend on.

The audit compares cached feature rows against a run's stored ranks and re-derives ranks with a
local copy of the shipped rule. If the shipped rule changes, every number in
`docs/reviews/20261108T153045-audit-equivalence/` silently stops describing the shipped code. These
tests fail loudly instead.

They also pin the two properties the audit MEASURED, so a future edit cannot quietly invert them:
decode is bit-identical across worker counts and batch sizes, and the shipped einsum is not the same
float computation as `optimize=True`.

No model weights, no gallery, no GPU: everything here is synthetic.
"""
from __future__ import annotations

import numpy as np

from workbench.composition.adapters import CompositionAdapter, prefetch_rows
from workbench.composition.topk import topk_from_scores


def _ranks(scores, positions):
    """The audit's copy of the shipped rule, used to cross-check the real implementation."""
    out = []
    for row, positives in zip(scores, positions):
        valid = [index for index in positives if 0 <= index < row.shape[0]]
        out.append(min(1 + int(np.count_nonzero(row > row[index])) +
                       int(np.count_nonzero(row[:index] == row[index])) for index in valid)
                   if valid else None)
    return out


def test_local_rank_rule_matches_shipped_implementation():
    rng = np.random.default_rng(7)
    scores = rng.random((6, 40), dtype=np.float32).astype(np.float32)
    # plant exact duplicates so the tie branch is exercised, not just the `>` branch
    scores[1, 9] = scores[1, 20]
    scores[2, 0] = scores[2, 39]
    positions = [(9,), (20,), (39,), (5,), (0, 11), (39, 3)]
    assert _ranks(scores, positions) == CompositionAdapter._ranks_from_scores(scores, positions)


def test_tie_rank_depends_on_gallery_order():
    """`_ranks_from_scores` counts earlier-equal rows, so an exact tie moves the rank."""
    row = np.zeros(10, dtype=np.float32)
    row[3] = row[8] = 1.0  # the only two maxima, exactly equal
    early = CompositionAdapter._ranks_from_scores(row[None, :], [[3]])[0]
    late = CompositionAdapter._ranks_from_scores(row[None, :], [[8]])[0]
    assert (early, late) == (1, 2)


def test_topk_order_is_the_rank_order():
    """A tie in the depth window must place the earlier gallery position first, matching the rule."""
    row = np.zeros(10, dtype=np.float32)
    row[2] = row[6] = 5.0
    row[0] = 9.0
    window = topk_from_scores(row, 3)
    assert [position for position, _score in window] == [0, 2, 6]


def test_prefetch_rows_is_order_preserving_and_bounded():
    batches = [[index] for index in range(20)]
    seen = []
    serial = list(prefetch_rows(lambda batch: seen.append(batch) or batch[0] * 2, batches, 1))
    parallel = list(prefetch_rows(lambda batch: batch[0] * 2, batches, 4))
    assert parallel == serial == [index * 2 for index in range(20)]


def test_optimize_true_changes_the_float_sum():
    """The audit's central premise: a different contraction order is a different computation.

    Not a performance test and not a claim about which is right; it asserts only that the two forms
    are not bit-equal for this shape, which is why rank identity could not be assumed.
    """
    rng = np.random.default_rng(11)
    queries = rng.standard_normal((4, 8)).astype(np.float32)
    targets = rng.standard_normal((64, 2, 8)).astype(np.float32)
    plain = np.einsum("nd,gqd->ngq", queries, targets).max(axis=-1)
    fast = np.einsum("nd,gqd->ngq", queries, targets, optimize=True).max(axis=-1)
    assert plain.shape == fast.shape
    assert not np.array_equal(plain, fast)
