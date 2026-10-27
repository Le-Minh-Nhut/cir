from __future__ import annotations

import math

import pytest

from workbench.composition import metrics
from workbench.composition.metrics import (
    SHORTCUT_K,
    aggregate_subsets,
    compose,
    compute_condition_metrics,
    hit_at_k,
    ndcg_full,
    paired_bootstrap_gap,
    reciprocal_rank,
    shortcut_outcomes,
)


def test_exact_ranks_are_one_based_and_not_top_k_limited() -> None:
    assert hit_at_k(1, 1) == 1
    assert hit_at_k(11, 10) == 0
    assert ndcg_full(1) == 1.0
    assert ndcg_full(31987) == 1.0 / math.log2(31988)
    assert reciprocal_rank(1) == 1.0
    for rank in (0, -1):
        with pytest.raises(ValueError, match="positive 1-based"):
            ndcg_full(rank)


def test_gap_averages_before_ratio_and_retains_negative() -> None:
    ranks = {
        "multimodal": {"a": 1, "b": 10},
        "text_only_black_image": {"a": 2, "b": 1},
        "image_only_empty_text": {"a": 50, "b": 50},
    }
    result = compose(ranks)
    mm = (ndcg_full(1) + ndcg_full(10)) / 2
    text = (ndcg_full(2) + ndcg_full(1)) / 2
    assert result.composition_gap_ndcg == pytest.approx(-0.26520382860036595)
    assert result.composition_gap_ndcg == pytest.approx(1 - max(text, ndcg_full(50)) / mm)
    ratio_of_means = ((1 - max(ndcg_full(2), ndcg_full(50)) / ndcg_full(1)
                       + 1 - max(ndcg_full(1), ndcg_full(50)) / ndcg_full(10)) / 2)
    assert ratio_of_means == pytest.approx(-1.045180686104377)
    assert result.composition_gap_ndcg != pytest.approx(ratio_of_means)
    negative = compose({
        "multimodal": {"a": 10},
        "text_only_black_image": {"a": 1},
        "image_only_empty_text": {"a": 10},
    })
    assert negative.composition_gap_ndcg < 0


def test_unavailable_gaps_have_explicit_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = compose({"multimodal": {}, "text_only_black_image": {}, "image_only_empty_text": {}})
    assert empty.composition_gap_ndcg is None
    assert "empty" in empty.gap_unavailable_reason

    missing = compose({"multimodal": {"q": 1}, "text_only_black_image": {"q": 1}})
    assert missing.composition_gap_ndcg is None
    assert "missing condition" in missing.gap_unavailable_reason

    monkeypatch.setattr(metrics, "ndcg_full", lambda rank: 0.0)
    zero = compose({
        "multimodal": {"q": 1},
        "text_only_black_image": {"q": 1},
        "image_only_empty_text": {"q": 1},
    })
    assert zero.composition_gap_ndcg is None
    assert "zero" in zero.gap_unavailable_reason


def test_unscored_query_reduces_coverage_and_is_not_a_miss() -> None:
    condition = compute_condition_metrics("multimodal", {"scored": 4, "unscored": None})
    assert condition.query_count == 2
    assert condition.scored_count == 1
    assert condition.coverage == 0.5
    assert condition.recall_at_10 == 100.0
    assert condition.ndcg_full == ndcg_full(4)
    assert condition.unavailable_reason is None

    result = compose({
        "multimodal": {"q1": 2, "q2": None},
        "text_only_black_image": {"q1": 5, "q2": 1},
        "image_only_empty_text": {"q1": 6, "q2": 1},
    })
    assert result.multimodal.coverage == 0.5
    assert result.shortcut_outcomes["neither_scored"] == 1


def test_gap_values_share_common_cohort_and_extra_members_only_reduce_coverage() -> None:
    ranks = {
        "multimodal": {"a": 1, "b": 10},
        "text_only_black_image": {"a": 2, "b": None},
        "image_only_empty_text": {"a": 5, "b": 50},
    }
    result = compose(ranks, query_count=3)
    common_gap = 1 - max(result.gap_text_only_ndcg, result.gap_image_only_ndcg) / result.gap_multimodal_ndcg
    naive_gap = 1 - max(result.text_only.ndcg_full, result.image_only.ndcg_full) / result.multimodal.ndcg_full
    assert result.gap_cohort_count == 1
    assert result.composition_gap_ndcg == pytest.approx(common_gap)
    assert result.composition_gap_ndcg != pytest.approx(naive_gap)
    assert result.multimodal.query_count == result.text_only.query_count == 3
    assert result.multimodal.coverage == pytest.approx(2 / 3)
    assert result.text_only.coverage == pytest.approx(1 / 3)
    assert result.gap_multimodal_ndcg == ndcg_full(1)
    assert result.gap_text_only_ndcg == ndcg_full(2)
    assert result.gap_image_only_ndcg == ndcg_full(5)
    without_extra = compose(ranks, query_count=2)
    assert result.multimodal.ndcg_full == without_extra.multimodal.ndcg_full
    assert result.text_only.ndcg_full == without_extra.text_only.ndcg_full
    assert result.multimodal.coverage < without_extra.multimodal.coverage
    assert result.multimodal.query_count == 3

    with pytest.raises(ValueError, match="must not be negative"):
        compose({"multimodal": {"a": 1}}, query_count=-1)


def test_shortcut_outcomes_are_mutually_exclusive() -> None:
    ranks = {
        "multimodal": {"text": 1, "image": 1, "both": 1, "required": 1, "fail": 99, "unscored": None},
        "text_only_black_image": {"text": 1, "image": 99, "both": 1, "required": 99, "fail": 99, "unscored": 1},
        "image_only_empty_text": {"text": 99, "image": 1, "both": 1, "required": 99, "fail": 99, "unscored": 1},
    }
    outcomes = shortcut_outcomes(ranks)
    assert outcomes == {"text_only": 1, "image_only": 1, "both": 1,
                        "composition_required": 1, "unresolved": 1, "neither_scored": 1}
    assert SHORTCUT_K == 10


def test_subsets_reuse_one_rank_table_and_keep_distinct_counts() -> None:
    ranks = {
        "multimodal": {f"q{i}": i + 1 for i in range(6)},
        "text_only_black_image": {f"q{i}": i + 2 for i in range(6)},
        "image_only_empty_text": {f"q{i}": i + 3 for i in range(6)},
    }
    result = aggregate_subsets(ranks, {"full": [f"q{i}" for i in range(6)],
                                       "sf": ["q0", "q1", "q2"], "validated": ["q0"]})
    assert result["full"].multimodal.query_count == 6
    assert result["sf"].multimodal.query_count == 3
    assert result["validated"].multimodal.query_count == 1
    assert result["full"].multimodal.ndcg_full != result["validated"].multimodal.ndcg_full


def test_bootstrap_is_deterministic_for_seed() -> None:
    deltas = [0.1, -0.2, 0.3, 0.05]
    assert paired_bootstrap_gap(deltas, seed=42) == paired_bootstrap_gap(deltas, seed=42)
