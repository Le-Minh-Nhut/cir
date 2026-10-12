"""`evaluate_fashioniq()` end-to-end protocol dispatch, over a deterministic fake model.

The helper-level tests in `test_fashioniq_protocols.py` pin the ranking rules. These tests drive
the *production* entry point, because the defect they guard against lived in the dispatch: which
protocol the caller named decided whether the reference was excluded, and `fashioniq_val_split`
was left reference-eligible even though its cohort excludes it. A helper test cannot catch that.

The fake model scores by image name, so the same score set works for whichever gallery a protocol
resolves to. Every expected rank is derived from the test's own arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cache.features import TextFeatureCache
from datasets.common import CIRBatch
from datasets.fashioniq import FashionIQAnnotation
from evaluation.fashioniq import FASHIONIQ_PROTOCOLS, evaluate_fashioniq

CATEGORY = "dress"


class _FakeModel:
    """One score row per query, built from per-image scores over the resolved gallery."""

    def __init__(self, scores_by_name: dict[str, float], gallery_order: list[str]) -> None:
        self.scores_by_name = scores_by_name
        self.gallery_order = gallery_order
        self.seen_gallery: list[str] | None = None

    def retrieve(self, *, reference_features, teacher_reference_features, text_states,
                 teacher_text_states, text_attention_mask, text_content_mask, gallery_features):
        assert gallery_features.shape[0] == len(self.gallery_order)
        self.seen_gallery = list(self.gallery_order)
        row = torch.tensor([self.scores_by_name[name] for name in self.gallery_order],
                           dtype=torch.float32)
        return {"scores": row.unsqueeze(0).repeat(reference_features.shape[0], 1)}


# The pair-union gallery is only as wide as the endpoints the annotations name, and Recall@50
# needs 50 candidates, so the fixture declares enough pairs to make both metrics meaningful.
PAIR_COUNT = 27
PAIR_REFS = [f"pref{i:02d}" for i in range(PAIR_COUNT)]
PAIR_TGTS = [f"ptgt{i:02d}" for i in range(PAIR_COUNT)]
PAIR_UNION = [name for pair in zip(PAIR_REFS, PAIR_TGTS) for name in pair]   # 54 endpoints
REFERENCE = PAIR_REFS[0]
TARGET = PAIR_TGTS[0]
# The full-split gallery is a superset of the pair union with extras, so the two protocols are
# genuinely distinguishable in the fixtures as well as in production.
EXTRA = [f"img{i:02d}" for i in range(6)]
FULL_SPLIT = [*PAIR_UNION, *EXTRA]
OTHERS = [name for name in FULL_SPLIT if name not in (REFERENCE, TARGET)]


def _annotations() -> list[FashionIQAnnotation]:
    return [
        FashionIQAnnotation(ref, tgt, ("shorter", "sleeves"), CATEGORY, index)
        for index, (ref, tgt) in enumerate(zip(PAIR_REFS, PAIR_TGTS))
    ]


def _batch() -> CIRBatch:
    return CIRBatch(
        sample_ids=["s0"], benchmark_ids=["b0"], reference_ids=[REFERENCE],
        target_ids=[TARGET], modification_texts=["shorter and sleeves"],
        categories=[CATEGORY], group_members=(), ground_truth_ids=())


def _retrieval_features(ids: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    """One vector per image, as the retrieval cache holds them."""
    return (torch.arange(len(ids) * 2, dtype=torch.float32).reshape(len(ids), 2),
            {name: i for i, name in enumerate(ids)})


def _native_features(ids: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    """Native features have an extra leading axis: `reference_native[:, 0, :]` is read."""
    return (torch.arange(len(ids) * 6, dtype=torch.float32).reshape(len(ids), 2, 3),
            {name: i for i, name in enumerate(ids)})


def _text_cache() -> TextFeatureCache:
    return TextFeatureCache(
        states=torch.zeros(1, 2), teacher_states=torch.zeros(1, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        content_mask=torch.ones(1, 2, dtype=torch.bool),
        sample_to_idx={"s0": 0}, captions={"s0": "shorter and sleeves"}, manifest={})


def _split_root(tmp_path: Path, ids: list[str] | None = None) -> Path:
    (tmp_path / f"split.{CATEGORY}.val.json").write_text(json.dumps(ids or FULL_SPLIT))
    return tmp_path


def _run(tmp_path: Path, protocol: str, scores: dict[str, float],
         gallery_order: list[str] | None = None,
         ids: list[str] | None = None) -> dict[str, float]:
    """Drive the production evaluator with one query and deterministic per-image scores."""
    root = _split_root(tmp_path, ids)
    order = gallery_order if gallery_order is not None else (ids or FULL_SPLIT)
    retrieval, retrieval_idx = _retrieval_features(order)
    native, native_idx = _native_features(order)
    model = _FakeModel(scores, order)
    return evaluate_fashioniq(
        model,
        {CATEGORY: [_batch()]}, {CATEGORY: _annotations()},
        protocol=protocol,
        split_root=root,
        split="val",
        retrieval_features=retrieval,
        native_features=native,
        retrieval_name_to_idx=retrieval_idx,
        native_name_to_idx=native_idx,
        device=torch.device("cpu"),
        text_cache=_text_cache(),
    )


def _scores_with_reference_first(target_rank: int) -> dict[str, float]:
    """Reference wins; the target sits at `target_rank` among the non-reference images.

    `target_rank` is therefore exactly the target's rank once the reference is excluded.
    """
    others = [name for name in FULL_SPLIT if name not in (REFERENCE, TARGET)]
    others.remove(TARGET) if TARGET in others else None
    ordered = [TARGET, *[name for name in others if name != TARGET]]
    ordered.remove(TARGET)
    ordered.insert(target_rank - 1, TARGET)
    scores = {REFERENCE: 1.0}
    for offset, name in enumerate(ordered):
        scores[name] = 1.0 - 0.001 * (offset + 1)
    return scores


# --------------------------------------------------------------- the defect this pass closes

def test_reference_exclusion_changes_recall_through_the_production_entry_point(
        tmp_path: Path) -> None:
    """The cohort's reference rule must decide the ranking rule, not a caller argument.

    One score matrix, two cohorts. The reference wins outright and the target sits immediately
    behind it, so a reference-eligible run misses at k=10 while an excluding run hits — the
    difference is the protocol, nothing else.
    """
    # The target is 10th among the non-reference images. With the reference eligible it is rank
    # 11 and misses the top ten; with the reference excluded it is rank 10 and is retrieved. So
    # one matrix separates the two cohorts at k=10.
    scores = _scores_with_reference_first(target_rank=10)
    assert scores[REFERENCE] > scores[TARGET]

    eligible_original = _run(tmp_path, "fashioniq_original_split", scores)
    assert eligible_original["recall_at_10"] == 0.0, eligible_original
    assert eligible_original["recall_at_50"] == 100.0, eligible_original

    # Same scores, excluding cohort: the reference is dropped, so the target moves to rank 10.
    excluded_full = _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores)
    assert excluded_full["recall_at_10"] == 100.0, excluded_full
    assert excluded_full["recall_at_50"] == 100.0, excluded_full

    # `fashioniq_val_split` is an excluding cohort too, so it must behave like the exclusion
    # family and not like the eligible one. Its gallery is the pair union, so the target's rank
    # there is 1 and the assertion is that the reference is not counted against it.
    pair_scores = _scores_with_reference_first(target_rank=1)
    pair_scores = {name: pair_scores[name] for name in PAIR_UNION}
    excluded_val = _run(tmp_path, "fashioniq_val_split", pair_scores, gallery_order=PAIR_UNION)
    assert excluded_val["recall_at_10"] == 100.0, excluded_val

    # The accepted alias resolves to the same cohort rule.
    assert _run(tmp_path, "fashioniq_val", pair_scores,
                gallery_order=PAIR_UNION) == excluded_val


def test_val_split_resolves_to_an_excluding_pair_union_protocol() -> None:
    """The registry name and its alias are one protocol, not two independently coded paths."""
    resolved = FASHIONIQ_PROTOCOLS["fashioniq_val_split"]
    assert resolved.reference_excluded is True
    assert resolved.gallery == "pair_union"
    assert "fashioniq_val" in resolved.aliases


# ------------------------------------------------------------------- rank boundaries, prod path

@pytest.mark.parametrize("target_rank,expected_10,expected_50", [
    (10, 100.0, 100.0),   # inside the first ten
    (11, 0.0, 100.0),     # just outside ten, inside fifty
    (50, 0.0, 100.0),     # last of fifty
    (51, 0.0, 0.0),       # just outside fifty
])
def test_exclusion_rank_boundaries_at_the_production_entry_point(
        tmp_path: Path, target_rank: int, expected_10: float, expected_50: float) -> None:
    """Recall@10 and Recall@50 must cut exactly at their boundary once exclusion is applied."""
    scores = _scores_with_reference_first(target_rank=target_rank)
    result = _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores)
    assert result["recall_at_10"] == expected_10, (target_rank, result)
    assert result["recall_at_50"] == expected_50, (target_rank, result)


def test_full_gallery_protocols_share_the_candidate_universe(tmp_path: Path) -> None:
    """Exclusion changes the ranking rule, never the gallery the model scores against."""
    from evaluation.fashioniq import build_fashioniq_gallery
    root = _split_root(tmp_path)
    original = build_fashioniq_gallery(
        "fashioniq_original_split", root, CATEGORY, _annotations(), "val")
    excluded = build_fashioniq_gallery(
        "fashioniq_full_gallery_ref_excluded", root, CATEGORY, _annotations(), "val")
    assert original == excluded == FULL_SPLIT
    assert len(original) == len(FULL_SPLIT) > len(PAIR_UNION)


def test_val_split_gallery_is_the_pair_union_not_the_split(tmp_path: Path) -> None:
    from evaluation.fashioniq import build_fashioniq_gallery
    root = _split_root(tmp_path)
    gallery = build_fashioniq_gallery("fashioniq_val_split", root, CATEGORY, _annotations(), "val")
    assert gallery == PAIR_UNION
    assert len(gallery) == 2 * PAIR_COUNT
    assert set(gallery).isdisjoint(EXTRA), "the full-split extras are not endpoints"


def test_the_model_scores_the_gallery_the_protocol_resolved(tmp_path: Path) -> None:
    """The candidate universe handed to the model must match the resolved protocol."""
    # The fake model asserts the width it is handed, so a mismatch fails the run itself.
    _run(tmp_path, "fashioniq_original_split", {name: 0.5 for name in FULL_SPLIT})
    _run(tmp_path, "fashioniq_val_split", {name: 0.5 for name in PAIR_UNION},
         gallery_order=PAIR_UNION)


# -------------------------------------------------------------------------------- refusals

def test_an_unknown_protocol_is_refused_before_any_scoring(tmp_path: Path) -> None:
    """A misspelled protocol must fail, never fall back to reference-eligible evaluation."""
    with pytest.raises(ValueError, match="Unsupported FashionIQ protocol"):
        _run(tmp_path, "fashioniq_original_splt", _scores_with_reference_first(1))


def test_a_missing_reference_in_the_gallery_is_refused(tmp_path: Path) -> None:
    """An excluding cohort cannot exclude a reference it cannot locate."""
    ids = [TARGET, *OTHERS]
    scores = {name: 1.0 for name in ids}
    with pytest.raises(KeyError):
        _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores, ids=ids)


def test_a_missing_target_in_the_gallery_is_refused(tmp_path: Path) -> None:
    ids = [REFERENCE, *OTHERS]
    scores = {name: 1.0 for name in ids}
    with pytest.raises(AssertionError, match="missing from gallery"):
        _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores, ids=ids)


def test_non_finite_scores_are_refused(tmp_path: Path) -> None:
    """A NaN score makes the ranking meaningless; it must not silently produce a number."""
    scores = {name: float("nan") for name in FULL_SPLIT}
    with pytest.raises(AssertionError, match="non-finite"):
        _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores)


def test_duplicate_gallery_ids_are_refused(tmp_path: Path) -> None:
    duplicated = [REFERENCE, REFERENCE, TARGET, *OTHERS]
    scores = {name: 1.0 for name in duplicated}
    with pytest.raises(AssertionError):
        _run(tmp_path, "fashioniq_full_gallery_ref_excluded", scores, ids=duplicated)


# ------------------------------------------------------------------------ category handling

def test_categories_are_evaluated_independently_and_macroed_equally(tmp_path: Path) -> None:
    """Each category is its own cohort; the macro is their equal-weight mean.

    Category isolation matters because a query from one category must never contribute to
    another's gallery or denominator, and because the registry reports a macro of per-category
    recalls rather than a pooled query average.
    """
    categories = ("dress", "shirt", "toptee")
    for name in categories:
        (tmp_path / f"split.{name}.val.json").write_text(json.dumps(PAIR_UNION))

    win = _scores_with_reference_first(target_rank=1)
    win = {name: win[name] for name in PAIR_UNION}
    lose = {name: 0.5 for name in PAIR_UNION}
    lose[REFERENCE] = 1.0
    lose[TARGET] = 0.0

    class _PerCategoryModel:
        def __init__(self, scores: dict[str, float]) -> None:
            self.scores = scores

        def retrieve(self, *, reference_features, teacher_reference_features, text_states,
                     teacher_text_states, text_attention_mask, text_content_mask,
                     gallery_features):
            row = torch.tensor([self.scores[name] for name in PAIR_UNION], dtype=torch.float32)
            return {"scores": row.unsqueeze(0).repeat(reference_features.shape[0], 1)}

    retrieval, retrieval_idx = _retrieval_features(PAIR_UNION)
    native, native_idx = _native_features(PAIR_UNION)
    category_results = {}
    for name in categories:
        scores = win if name != "toptee" else lose
        # The pair-union gallery is built from the annotations, so each category must declare
        # its full set of pairs for Recall@10/@50 to have a boundary inside the gallery.
        metrics = evaluate_fashioniq(
            _PerCategoryModel(scores), {name: [_batch()]},
            {name: _annotations()},
            protocol="fashioniq_val_split", split_root=tmp_path, split="val",
            retrieval_features=retrieval, native_features=native,
            retrieval_name_to_idx=retrieval_idx, native_name_to_idx=native_idx,
            device=torch.device("cpu"), text_cache=_text_cache())
        category_results[name] = {"recall_at_10": metrics["recall_at_10"],
                                  "recall_at_50": metrics["recall_at_50"]}

    # Category isolation: dress and shirt retrieve the target, toptee cannot.
    assert category_results["dress"]["recall_at_10"] == 100.0
    assert category_results["shirt"]["recall_at_10"] == 100.0
    # toptee's target is scored last, so it misses both cuts: the failure is category-local and
    # must not leak into the other two.
    assert category_results["toptee"]["recall_at_10"] == 0.0
    assert category_results["toptee"]["recall_at_50"] == 0.0
    assert category_results["dress"]["recall_at_50"] == 100.0

    from evaluation.fashioniq import macro_average_fashioniq
    averaged = macro_average_fashioniq(category_results)
    # Equal weight across categories, not weighted by query or gallery size: two of three
    # categories retrieve, so the macro is two thirds, not the pooled query rate.
    assert averaged["recall_at_10"] == pytest.approx(200.0 / 3)
    assert averaged["recall_at_50"] == pytest.approx(200.0 / 3)


def test_an_empty_category_set_is_refused() -> None:
    """A macro over no categories is not a metric."""
    from evaluation.fashioniq import macro_average_fashioniq
    with pytest.raises(AssertionError):
        macro_average_fashioniq({})
