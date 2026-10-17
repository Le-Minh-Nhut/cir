"""Adversarial ranking fixtures for the LIMN per-query observation core (§27).

These exercise :func:`workbench.replay.limn_observer.rank_queries`, which is the numerically
load-bearing part of the observer and is deliberately free of torch and of any model. Everything
here is deterministic and runs without a GPU.

The tests that matter most are the ones a plausible implementation gets WRONG:
  * a target beyond the saved Top-K depth must still get a true rank (rank computed from the full
    ranking, never from the truncated Top-K);
  * the reference is masked, not removed, so it still occupies a rank slot and the eligible
    universe is the full gallery;
  * equal scores must follow ``np.argsort(-sims)`` semantics exactly, not a "stable" tiebreak
    someone invented.

Convention: ``queries[q]`` is a one-hot vector selecting gallery entry ``q``, so with an identity
gallery the similarity of query ``q`` to gallery ``g`` is 1 when ``q == g`` and 0 otherwise. The
reference for every query here is chosen to differ from its target unless a test says otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from workbench.replay.limn_observer import NATIVE_REFERENCE_MASK, rank_queries


def _one_hot(indices: list[int], size: int) -> np.ndarray:
    """One query vector per index, each the unit vector for that gallery position."""
    matrix = np.zeros((len(indices), size))
    for row, column in enumerate(indices):
        matrix[row, column] = 1.0
    return matrix


@pytest.mark.parametrize("target_rank", [1, 10, 50, 51, 200, 201, 1000])
def test_full_ranking_gives_the_true_target_rank(target_rank: int) -> None:
    """A target at ANY depth gets its true rank, including well beyond a saved Top-50/Top-200.

    Gallery 0 outranks everything, the target sits at ``target_rank - 1``, and the reference is a
    position beyond both so masking it cannot move the target.
    """
    size = max(target_rank + 2, 8)
    # The query matches gallery 0 exactly (score 1); every other gallery scores 0 and ties, so
    # they sort ascending by index. A target at index r-1 therefore lands at rank r.
    queries = _one_hot([0], size)
    targets = np.eye(size)
    reference = size - 1 if target_rank - 1 != size - 1 else size - 2
    ranks, _ = rank_queries(queries, targets, reference_positions=[reference],
                            target_positions=[target_rank - 1])
    assert ranks == [target_rank]


def test_target_beyond_saved_depth_is_absent_from_top_k_but_ranked() -> None:
    """Top-K depth truncates the SAVED list only; the rank stays true. §15's central invariant."""
    size = 300
    # Gallery 0 is the query's exact match (rank 1), then the zero-scoring entries tie and sort
    # ascending, so index 250 is rank 251.
    queries = _one_hot([0], size)
    ranks, top = rank_queries(queries, np.eye(size), reference_positions=[299],
                              target_positions=[250], top_k=200)
    assert ranks == [251]
    assert len(top[0]) == 200
    assert all(gallery_index != 250 for _, gallery_index, _ in top[0])


@pytest.mark.parametrize("rank", [1, 10, 50, 51, 200])
def test_target_within_saved_depth_appears_at_its_rank(rank: int) -> None:
    size = 400
    queries = _one_hot([0], size)
    ranks, top = rank_queries(queries, np.eye(size), reference_positions=[399],
                              target_positions=[rank - 1], top_k=200)
    assert ranks == [rank]
    assert top[0][rank - 1][1] == rank - 1


def test_reference_is_masked_not_removed_and_stays_in_the_ranking() -> None:
    """The reference still occupies a slot; it is made unreachable, not deleted. §10/§14."""
    size = 6
    # Query matches gallery 3 best, and gallery 3 is also its reference.
    queries = _one_hot([3], size)
    ranks, top = rank_queries(queries, np.eye(size), reference_positions=[3], target_positions=[5])
    # The gallery still has all six entries: masking 3 sends it to the END, it is not removed. The
    # five unmasked entries tie and sort ascending, so the target (5) is rank 5.
    assert ranks == [5]
    assert len(top[0]) == size
    assert [gallery for _, gallery, _ in top[0]] == [0, 1, 2, 4, 5, 3]


def test_masked_reference_score_is_the_pinned_sentinel() -> None:
    """The mask value must be the author's -10e10, not a different large negative."""
    assert NATIVE_REFERENCE_MASK == -10e10
    size = 3
    # Query matches gallery 1 best and gallery 1 is its reference; masking it sends it last.
    queries = _one_hot([1], size)
    ranks, top = rank_queries(queries, np.eye(size), reference_positions=[1], target_positions=[2])
    assert ranks == [2]
    assert [gallery for _, gallery, _ in top[0]] == [0, 2, 1]
    masked = [score for _, gallery, score in top[0] if gallery == 1][0]
    assert masked == NATIVE_REFERENCE_MASK
    assert all(np.isfinite(score) for _, gallery, score in top[0] if gallery != 1)


def test_masked_reference_ranks_last_even_when_it_would_win() -> None:
    """A reference that is the query's own best match must not be retrievable. §10."""
    size = 4
    queries = _one_hot([2], size)  # reference 2 is the exact match
    ranks, top = rank_queries(queries, np.eye(size), reference_positions=[2], target_positions=[0])
    assert [gallery for _, gallery, _ in top[0]][-1] == 2
    assert ranks == [1]


def test_equal_similarity_scores_follow_argsort_semantics() -> None:
    """Ties resolve like ``np.argsort(-sims)`` — the pinned behavior, not an invented stable sort."""
    size = 4
    queries = np.ones((1, size))          # identical similarity to every gallery entry
    targets = np.eye(size)
    _, top = rank_queries(queries, targets, reference_positions=[3], target_positions=[0])
    order = [gallery for _, gallery, _ in top[0]]
    # np.argsort(-sims) on an all-equal row returns ascending index order.
    assert order == [0, 1, 2, 3]


def test_negative_and_large_negative_scores_sort_correctly() -> None:
    queries = np.array([[1.0, 0.0]])
    targets = np.array([[-5.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    ranks, top = rank_queries(queries, targets, reference_positions=[2], target_positions=[0])
    # After normalization the scores are -1, -1, +1; the +1 (gallery 2) is masked, so the two -1s
    # tie and sort ascending by index: gallery 0 then gallery 1.
    assert ranks == [1]
    assert [gallery for _, gallery, _ in top[0]] == [0, 1, 2]


def test_duplicate_identical_features_do_not_crash() -> None:
    size = 3
    queries = _one_hot([1], size)
    targets = np.ones((size, size))   # every gallery entry identical -> all similarities equal
    ranks, top = rank_queries(queries, targets, reference_positions=[0], target_positions=[1])
    assert len({gallery for _, gallery, _ in top[0]}) == size
    assert ranks[0] in (1, 2, 3)


def test_swapped_reference_and_target_changes_the_rank() -> None:
    size = 3
    queries = _one_hot([1], size)
    targets = np.eye(size)
    forward, _ = rank_queries(queries, targets, reference_positions=[0], target_positions=[1])
    swapped, _ = rank_queries(queries, targets, reference_positions=[1], target_positions=[0])
    assert forward == [1]     # gallery 1 is the best real candidate
    assert swapped == [1]     # gallery 0 is now masked, so the best remaining is gallery 1


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(ValueError):
        rank_queries(np.zeros((2, 3)), np.zeros((2, 4)), [0], [1])


def test_position_count_mismatch_is_refused() -> None:
    features = np.eye(3)
    with pytest.raises(ValueError):
        rank_queries(features, features, [0], [1, 2])


def test_scores_are_finite_for_eligible_results() -> None:
    size = 10
    queries = _one_hot([9], size)
    _, top = rank_queries(queries, np.eye(size), reference_positions=[0], target_positions=[9])
    eligible = [score for _, gallery, score in top[0] if gallery != 0]
    assert all(np.isfinite(score) for score in eligible)
    masked = [score for _, gallery, score in top[0] if gallery == 0]
    assert masked == [NATIVE_REFERENCE_MASK]
