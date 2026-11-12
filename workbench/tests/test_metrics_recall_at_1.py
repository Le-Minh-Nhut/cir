"""R@1 is the exact-rank hit at k=1 on the SAME scored cohort as R@10/R@50.

The boundary that matters: a rank of exactly 1 hits, a rank of 2 does not, and an unscored query
removes itself from the denominator's numerator only through `coverage` -- it is never counted as a
miss and never turns R@1 into 0.0. A rank the producer reached through a TIE is a hit, because the
tie policy belongs to the producer (`CompositionAdapter._ranks_from_scores`: descending score, ties
by earlier gallery position); this module consumes ranks and re-ties nothing.
"""
from __future__ import annotations

import pytest

from workbench.composition.metrics import compose, compute_condition_metrics, hit_at_k


def test_boundary_rank_1_hits_and_rank_2_misses() -> None:
    metrics = compute_condition_metrics("multimodal", {"a": 1, "b": 4, "c": None})
    assert metrics.recall_at_1 == 50.0
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.scored_count == 2
    assert metrics.query_count == 3
    # the same scored denominator as R@1: 2 rows; both ranks (1 and 4) are inside k=10 and k=50
    assert metrics.recall_at_10 == 100.0
    assert metrics.recall_at_50 == 100.0
    assert hit_at_k(1, 1) == 1
    assert hit_at_k(2, 1) == 0


def test_rank_1_from_a_producer_tie_is_a_hit_and_is_not_re_tied() -> None:
    """The producer resolved the tie to 1; the metric must not second-guess it."""
    metrics = compute_condition_metrics("multimodal", {"exact_tie": 1, "behind_tie": 2})
    assert metrics.recall_at_1 == 50.0
    assert metrics.recall_at_10 == 100.0


@pytest.mark.parametrize("ranks, reason", [
    ({}, "empty query cohort"),
    ({"a": None, "b": None}, "no queries have an exact rank"),
])
def test_no_scorable_ranks_is_null_with_a_reason_never_zero(ranks, reason) -> None:
    metrics = compute_condition_metrics("image_only_empty_text", ranks)
    assert metrics.recall_at_1 is None
    assert metrics.unavailable_reason == reason
    assert metrics.scored_count == 0


def test_condition_absent_from_the_rank_table_is_null_in_compose() -> None:
    result = compose({"multimodal": {"a": 1}, "text_only_black_image": {"a": 1}})
    assert result.image_only.recall_at_1 is None
    assert result.image_only.unavailable_reason == "no queries have an exact rank"
    assert result.multimodal.recall_at_1 == 100.0
    assert result.text_only.recall_at_1 == 100.0


def test_partial_rank_table_keeps_its_own_denominator() -> None:
    """An incomplete table (some qids missing) is unscored, not a miss, in every recall."""
    result = compose({
        "multimodal": {"a": 1, "b": 2},
        "text_only_black_image": {"a": 1},
        "image_only_empty_text": {"a": 3, "b": 3},
    }, query_count=3)
    assert result.multimodal.recall_at_1 == 50.0
    assert result.multimodal.scored_count == 2
    assert result.multimodal.coverage == pytest.approx(2 / 3)
    assert result.text_only.recall_at_1 == 100.0
    assert result.text_only.coverage == pytest.approx(1 / 3)
    assert result.image_only.recall_at_1 == 0.0
    assert result.image_only.coverage == pytest.approx(2 / 3)


def test_recall_at_1_is_never_above_recall_at_10_or_50() -> None:
    ranks = {f"q{index}": rank for index, rank in enumerate([1, 1, 2, 3, 9, 10, 11, 50, 51, 74381])}
    metrics = compute_condition_metrics("multimodal", ranks)
    assert metrics.recall_at_1 <= metrics.recall_at_10 <= metrics.recall_at_50
    assert metrics.recall_at_1 == 20.0
    assert metrics.recall_at_10 == 60.0
    assert metrics.recall_at_50 == 80.0
