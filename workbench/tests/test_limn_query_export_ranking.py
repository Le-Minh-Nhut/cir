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


# ----------------------------------------------------------------- F-16: non-finite refusal

def test_zero_norm_feature_is_refused_not_ranked() -> None:
    """A zero-norm row would normalize to NaN, and the sort would then be decided by the mask.

    NumPy produces a ranking here (the masked reference's negated score is a large finite number,
    which `np.argsort` places ahead of every NaN), so without a guard the observer would report the
    reference at rank 1 and the target below it -- a plausible ranking built from nothing.
    """
    from workbench.replay.limn_observer import NonFiniteRanking

    queries = np.array([[0.0, 0.0]])
    targets = np.eye(2)
    with pytest.raises(NonFiniteRanking, match="zero norm"):
        rank_queries(queries, targets, reference_positions=[1], target_positions=[0])


def test_nan_feature_is_refused() -> None:
    from workbench.replay.limn_observer import NonFiniteRanking

    with pytest.raises(NonFiniteRanking, match="NaN or infinity"):
        rank_queries(np.array([[np.nan, 1.0]]), np.eye(2), reference_positions=[1], target_positions=[0])


def test_infinite_feature_is_refused() -> None:
    from workbench.replay.limn_observer import NonFiniteRanking

    with pytest.raises(NonFiniteRanking, match="NaN or infinity"):
        rank_queries(np.array([[np.inf, 1.0]]), np.eye(2), reference_positions=[1], target_positions=[0])


def test_non_finite_similarity_check_is_defensive_and_documented_as_such() -> None:
    """The similarity guard is unreachable through this code path, and saying so is the honest test.

    A `1e308` raw feature normalizes to `1.0` (L2 normalization divides by the magnitude), so
    finite inputs cannot produce an infinite dot product here. The guard stays as defence in depth
    for a caller that passes already-normalized-but-huge data; what this test pins is the fact that
    enormous magnitudes are handled by normalization, not by the guard.
    """
    huge = np.array([[1e308, 1e308]])
    ranks, top = rank_queries(huge, huge.copy(), reference_positions=[0], target_positions=[0])
    assert ranks == [1]
    assert all(np.isfinite(score) for _, _, score in top[0])


def test_valid_negative_and_positive_embeddings_are_still_accepted() -> None:
    """POSITIVE CONTROL: the refusal must not reject legitimate signed embeddings."""
    queries = np.array([[-1.0, -1.0], [1.0, -1.0]])
    targets = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    ranks, top = rank_queries(queries, targets, reference_positions=[2, 2], target_positions=[0, 1])
    # Query 0 matches gallery 0 exactly (rank 1). Query 1 ties gallery 0 and 1 at 0.0, and the tie
    # resolves ascending by index, so its target (gallery 1) is rank 2. Gallery 2 is the reference.
    assert ranks == [1, 2]
    assert all(np.isfinite(score) for _, _, score in top[0])


# ----------------------------------------------------------------- F-15: native vs eligible

def test_the_masked_reference_always_sorts_last_for_normalized_features() -> None:
    """F-15, resolved: the mask cannot leak into the Top-K for realistic inputs.

    The auditor's differing case requires a similarity at or below -1e11. LIMN normalizes every
    feature row to unit length before `dot`, so each similarity is bounded by 1 in absolute value --
    the mask (-1e11) therefore always sorts last, and the reference can never appear as a candidate.
    The auditor's own trigger (scores <= -1e11) is unreachable through this code path; what IS
    reachable is the version where the pre-normalization magnitude is huge, and that collapses to
    -1 as shown here.

    This is asserted rather than assumed because the equivalence between "masked ranking" and
    "eligible ranking" is exactly what the export's eligibility claim rests on.
    """
    queries = np.array([[1.0, 0.0]])
    # Magnitudes chosen to be enormous in the RAW vector; normalization collapses them to +/-1.
    targets = np.array([[-2e11, 0.0], [1.0, 0.0], [-3e11, 0.0]])
    _, top = rank_queries(queries, targets, reference_positions=[1], target_positions=[0], top_k=3)
    assert [gallery for _, gallery, _ in top[0]] == [0, 2, 1], "the masked reference must sort last"
    assert all(abs(score) <= 1.0 + 1e-9 for _, gallery, score in top[0] if gallery != 1)


def test_a_tied_candidate_set_cannot_be_reordered_by_the_masked_column() -> None:
    """Ties are where a retained column could change the sort; with the mask last, they cannot.

    The auditor's second case (unstable quicksort reordering ties because the reference occupies an
    index) is only reachable if the reference participates in the comparison range. Since the mask
    sorts last for normalized features, the candidates' relative order is decided among themselves.
    """
    size = 10
    queries = np.ones((1, 2))
    targets = np.ones((size, 2))          # every candidate ties
    _, top = rank_queries(queries, targets, reference_positions=[0], target_positions=[1],
                          top_k=size)
    order = [gallery for _, gallery, _ in top[0]]
    assert order[-1] == 0, "the masked reference is last even in a fully tied field"
    assert sorted(order[:-1]) == sorted(range(1, size)), "all real candidates remain eligible"


def test_eligible_ranking_excludes_the_reference_and_reports_the_true_candidate_universe() -> None:
    """The observer's export must not offer the reference as a retrievable candidate (F-15).

    `observe_category` records `eligible_count = gallery_size - 1` under an excluded protocol, and
    the schema bounds `target_rank` by that count, so a rank that only exists because the masked
    reference occupies a slot cannot be published as a valid candidate position.
    """
    from workbench.backend.schemas.results import ResultRun, admission  # noqa: F401

    # The schema-level guarantee is asserted in the publication tests; here the ranking primitive
    # is checked: for realistic (finite, bounded) scores the mask always sorts last.
    queries = np.array([[1.0, 0.0]])
    targets = np.array([[0.5, 0.0], [1.0, 0.0], [0.25, 0.0]])
    _, top = rank_queries(queries, targets, reference_positions=[1], target_positions=[0], top_k=3)
    assert [gallery for _, gallery, _ in top[0]] == [0, 2, 1], "reference must sort last"


def test_saved_depth_never_exceeds_the_eligible_candidate_count() -> None:
    """F-15: a protocol that excludes the reference must not save it as a retrieved image.

    With `top_k >= gallery_size` the native column for the reference is still present, so a naive
    implementation would return it as the last "retrieved" candidate. The export bounds the saved
    depth by `gallery_size - 1` instead, which is the true candidate universe.
    """
    size = 5
    queries = np.ones((1, size))
    targets = np.eye(size)
    _, top = rank_queries(queries, targets, reference_positions=[0], target_positions=[1],
                          top_k=size - 1)
    assert len(top[0]) == size - 1, "save exactly the eligible candidates"
    assert all(gallery != 0 for _, gallery, _ in top[0]), "the reference must not be offered"

    # The native call with the raw size still returns it -- which is why the bound lives in the
    # exporter, not in the ranking primitive that must stay native-faithful.
    _, unbounded = rank_queries(queries, targets, reference_positions=[0], target_positions=[1],
                                top_k=size)
    assert len(unbounded[0]) == size
    assert unbounded[0][-1][1] == 0, "unbounded, the masked reference is the last entry"


# ------------------------------------------------- query identity and cache freshness

def _dataset(tmp_path: Path, category: str = "dress") -> tuple[list[str], list[dict], Path]:
    import json as _json

    images = [f"img{i}" for i in range(4)]
    annotations = [
        {"candidate": "img0", "target": "img1", "captions": ["a red shirt", "is blue"]},
        {"candidate": "img2", "target": "img3", "captions": ["a small bag", "with straps"]},
    ]
    (tmp_path / "image_splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "captions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "image_splits" / f"split.{category}.val.json").write_text(_json.dumps(images))
    (tmp_path / "captions" / f"cap.{category}.val.json").write_text(_json.dumps(annotations))
    return images, annotations, tmp_path


def _native_queries(images: list[str], annotations: list[dict], correction: dict | None = None) -> list[dict]:
    from workbench.replay.limn_observer import expected_model_input

    correction = correction or {}
    return [
        {"source_img_id": images.index(record["candidate"]),
         "target_img_id": images.index(record["target"]),
         "mod": {"str": expected_model_input(record["captions"], correction)}}
        for record in annotations
    ]


def test_aligned_native_queries_are_accepted(tmp_path: Path) -> None:
    """POSITIVE CONTROL: a correctly cached query set must pass."""
    from workbench.replay.limn_observer import assert_alignment

    images, annotations, root = _dataset(tmp_path)
    assert_alignment("dress", _native_queries(images, annotations), images, annotations, root)


def test_a_shuffled_cache_is_detected(tmp_path: Path) -> None:
    """A reordered cache keeps counts equal but mislabels every query -- the dangerous case."""
    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    queries = list(reversed(_native_queries(images, annotations)))
    with pytest.raises(QueryAlignmentError, match="not aligned"):
        assert_alignment("dress", queries, images, annotations, root)


def test_a_truncated_cache_is_detected(tmp_path: Path) -> None:
    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    queries = _native_queries(images, annotations)[:1]
    with pytest.raises(QueryAlignmentError, match="does not match this dataset"):
        assert_alignment("dress", queries, images, annotations, root)


def test_a_swapped_reference_and_target_is_detected(tmp_path: Path) -> None:
    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    queries = _native_queries(images, annotations)
    queries[0]["source_img_id"], queries[0]["target_img_id"] = (
        queries[0]["target_img_id"], queries[0]["source_img_id"])
    with pytest.raises(QueryAlignmentError, match="reference is"):
        assert_alignment("dress", queries, images, annotations, root)


def test_a_cache_built_from_other_captions_is_detected(tmp_path: Path) -> None:
    """Ids can still resolve while the text came from different captions; composition catches it."""
    import json as _json

    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    (root / "captions" / "correction_dict_dress.json").write_text(_json.dumps({}))
    queries = _native_queries(images, annotations)
    queries[0]["mod"]["str"] = "something else entirely"
    with pytest.raises(QueryAlignmentError, match="model input text"):
        assert_alignment("dress", queries, images, annotations, root)


def test_without_an_on_disk_dictionary_composition_cannot_be_checked(tmp_path: Path) -> None:
    """The honest limit of the check: ids are verified, text is only verifiable when the dictionary
    in force is on disk. Documented here so the guarantee is not overstated."""
    from workbench.replay.limn_observer import assert_alignment

    images, annotations, root = _dataset(tmp_path)
    queries = _native_queries(images, annotations)
    queries[0]["mod"]["str"] = "unverifiable but non-empty"
    (root / "captions" / "correction_dict_dress.json").unlink(missing_ok=True)
    assert_alignment("dress", queries, images, annotations, root)


def test_a_stale_correction_dictionary_is_detected(tmp_path: Path) -> None:
    """A different correction dictionary changes every query's text while ids stay aligned."""
    import json as _json

    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    (root / "captions" / "correction_dict_dress.json").write_text(_json.dumps({"shirt": "dress"}))
    # Cached with NO correction applied, while the on-disk dictionary would have rewritten "shirt".
    queries = _native_queries(images, annotations)
    with pytest.raises(QueryAlignmentError, match="correction dictionary"):
        assert_alignment("dress", queries, images, annotations, root)


def test_an_out_of_range_image_position_is_detected(tmp_path: Path) -> None:
    from workbench.replay.limn_observer import QueryAlignmentError, assert_alignment

    images, annotations, root = _dataset(tmp_path)
    queries = _native_queries(images, annotations)
    queries[0]["source_img_id"] = 99
    with pytest.raises(QueryAlignmentError, match="outside split"):
        assert_alignment("dress", queries, images, annotations, root)


def test_expected_model_input_matches_the_pinned_composition() -> None:
    """The composition rule itself: lowercase, punctuation stripped, dict applied, joined by ' and '."""
    from workbench.replay.limn_observer import expected_model_input

    assert expected_model_input(["A Red Shirt!", "IS blue."], {}) == "a red shirt and is blue"
    assert expected_model_input(["a shirt", "is long"], {"shirt": "dress"}) == "a dress and is long"
    # Punctuation becomes whitespace, then tokens are rejoined with single spaces.
    assert expected_model_input(["a,b", "c-d"], {}) == "a b and c d"
