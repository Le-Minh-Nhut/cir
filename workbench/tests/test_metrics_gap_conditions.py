"""`compose` is parameterised over the condition triple; the default triple is unchanged.

Two separate claims, both asserted against values that are NOT recomputed from the implementation:

1. **The default call is unchanged.** The expected numbers below were captured from
   `git show 88dd3a1:workbench/composition/metrics.py` (the pre-change function) on the same rank
   table and pasted here as literals, so this is a regression test and not a tautology. The only
   addition is the new `gap_conditions` field, which is asserted separately.
2. **A variant triple uses the same rule.** A crafted cohort whose per-condition nDCG means are
   exactly 0.5 / 0.25 / 0.125 gives `gap == 0.5` by hand, over an explicitly substituted image role.

A missing condition and a qid unscored in one condition keep the existing reason strings and the
existing cohort semantics.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from workbench.composition.metrics import compose, ndcg_full, shortcut_outcomes

PROTOCOL = ("multimodal", "text_only_black_image", "image_only_empty_text")
VARIANT = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")

#: Captured from the pre-change `compose` at 88dd3a1 for the table below, every field, as literals.
BEFORE_TABLE = {
    "multimodal": {"a": 1, "b": 10, "c": None, "d": 400},
    "text_only_black_image": {"a": 2, "b": 1, "c": 30, "d": 400},
    "image_only_empty_text": {"a": 50, "b": 50, "c": None, "d": None},
}
BEFORE_FIXED = {
    "composition_gap_ndcg": -0.26520382860036595,
    "gap_cohort_count": 2,
    "gap_image_only_ndcg": 0.17629143438888212,
    "gap_multimodal_ndcg": 0.6445324131589439,
    "gap_text_only_ndcg": 0.8154648767857288,
    "mrr_gap": -0.19999999999999996,
    "shortcut_outcomes": {"both": 0, "composition_required": 0, "image_only": 0,
                          "neither_scored": 2, "text_only": 2, "unresolved": 0},
    "multimodal": {"coverage": 0.75, "mrr": 0.3675, "ndcg_full": 0.46823524691863566,
                   "query_count": 4, "recall_at_1": 33.333333333333336,
                   "recall_at_10": 66.66666666666667, "recall_at_50": 66.66666666666667,
                   "scored_count": 3},
    "text_only": {"coverage": 1.0, "mrr": 0.38395833333333335, "ndcg_full": 0.4871049386478941,
                  "query_count": 4, "recall_at_1": 25.0, "recall_at_10": 50.0,
                  "recall_at_50": 75.0, "scored_count": 4},
    "image_only": {"coverage": 0.5, "mrr": 0.02, "ndcg_full": 0.17629143438888212,
                   "query_count": 4, "recall_at_1": 0.0, "recall_at_10": 0.0,
                   "recall_at_50": 100.0, "scored_count": 2},
}

def test_default_triple_is_bit_identical_to_the_pre_change_function() -> None:
    result = compose(BEFORE_TABLE)
    payload = asdict(result)
    for field in ("composition_gap_ndcg", "gap_cohort_count", "gap_image_only_ndcg",
                  "gap_multimodal_ndcg", "gap_text_only_ndcg", "mrr_gap"):
        assert payload[field] == BEFORE_FIXED[field], field
    assert payload["gap_unavailable_reason"] is None
    assert payload["shortcut_outcomes"] == BEFORE_FIXED["shortcut_outcomes"]
    for role in ("multimodal", "text_only", "image_only"):
        actual = payload[role]
        for field, expected in BEFORE_FIXED[role].items():
            assert actual[field] == expected, f"{role}.{field}"
        assert actual["unavailable_reason"] is None
    # The ONE new field, and it names the protocol triple.
    assert payload["gap_conditions"] == PROTOCOL


def test_explicit_default_triple_matches_the_implicit_one() -> None:
    assert asdict(compose(BEFORE_TABLE)) == asdict(compose(BEFORE_TABLE, conditions=PROTOCOL))


def test_variant_gap_is_half_on_a_crafted_cohort() -> None:
    """mm nDCG 0.5, text 0.25, image 0.125 -> 1 - max(.25,.125)/.5 = 0.5, by hand.

    Ranks chosen so each condition's mean single-positive nDCG is exact:
    nDCG(7) = 1/log2(8) = 1/3, so [1,7,7,7] means (1 + 3*(1/3))/4 = 0.5;
    nDCG(15) = 1/log2(16) = 1/4, so four rank-15 rows mean 0.25;
    nDCG(255) = 1/log2(256) = 1/8, so four rank-255 rows mean 0.125.
    """
    qids = ["a", "b", "c", "d"]
    mm_ranks = {"a": 1, "b": 7, "c": 7, "d": 7}
    text_ranks = {qid: 15 for qid in qids}
    image_ranks = {qid: 255 for qid in qids}
    for ranks, target in ((mm_ranks, 0.5), (text_ranks, 0.25), (image_ranks, 0.125)):
        assert sum(ndcg_full(rank) for rank in ranks.values()) / len(ranks) == pytest.approx(target)

    result = compose({"multimodal": mm_ranks, "text_only_black_image": text_ranks,
                      "clvc_image_encoder_only": image_ranks}, conditions=VARIANT)
    assert result.gap_conditions == VARIANT
    assert result.gap_cohort_count == 4
    assert result.gap_multimodal_ndcg == pytest.approx(0.5)
    assert result.gap_text_only_ndcg == pytest.approx(0.25)
    assert result.gap_image_only_ndcg == pytest.approx(0.125)
    assert result.composition_gap_ndcg == pytest.approx(0.5)
    assert result.gap_unavailable_reason is None
    assert result.image_only.condition == "clvc_image_encoder_only"


def test_variant_gap_reports_the_same_reasons_as_the_protocol_gap() -> None:
    missing = compose({"multimodal": {"q": 1}, "text_only_black_image": {"q": 1}},
                      conditions=VARIANT)
    assert missing.composition_gap_ndcg is None
    assert missing.gap_unavailable_reason == \
        "missing condition(s): clvc_image_encoder_only"
    assert missing.gap_conditions == VARIANT

    empty = compose({"multimodal": {}, "text_only_black_image": {},
                     "clvc_image_encoder_only": {}}, conditions=VARIANT)
    assert empty.composition_gap_ndcg is None
    assert empty.gap_unavailable_reason == "empty query cohort"
    assert empty.gap_cohort_count == 0


def test_a_qid_unscored_in_any_variant_condition_leaves_the_cohort() -> None:
    ranks = {
        "multimodal": {"a": 1, "b": 2, "c": 4},
        "text_only_black_image": {"a": 2, "b": 4, "c": 8},
        "clvc_image_encoder_only": {"a": None, "b": 2, "c": 2},
    }
    result = compose(ranks, conditions=VARIANT)
    assert result.gap_cohort_count == 2, "only b and c carry a rank in all three"
    expected_mm = (ndcg_full(2) + ndcg_full(4)) / 2
    assert result.gap_multimodal_ndcg == pytest.approx(expected_mm)
    # The per-condition metrics still use their OWN scored cohort, so `a` is not lost from them.
    assert result.multimodal.scored_count == 3
    assert result.image_only.scored_count == 2
    assert result.image_only.condition == "clvc_image_encoder_only"
    assert result.image_only.unavailable_reason is None


def test_shortcut_outcomes_follow_the_triple_they_are_given() -> None:
    """Bins must describe the cohort actually scored, never a hard-coded module tuple."""
    ranks = {
        "multimodal": {"a": 1, "b": 50},
        "text_only_black_image": {"a": 50, "b": 50},
        "clvc_image_encoder_only": {"a": 2, "b": 50},
    }
    variant = compose(ranks, conditions=VARIANT)
    assert variant.shortcut_outcomes["image_only"] == 1, "a: image hits, text does not"
    assert variant.shortcut_outcomes["unresolved"] == 1, "b: nothing in the top 10"
    assert variant.shortcut_outcomes["neither_scored"] == 0

    # The same table against the protocol triple: `image_only_empty_text` has no rows at all, so
    # every qid is excluded instead of being read as a miss.
    protocol = compose(ranks)
    assert protocol.shortcut_outcomes["neither_scored"] == 2
    assert protocol.shortcut_outcomes["image_only"] == 0


def test_compose_refuses_a_triple_that_is_not_three_distinct_names() -> None:
    for bad in (("multimodal", "multimodal", "image_only_empty_text"), ("multimodal", "text_only_black_image")):
        with pytest.raises(ValueError, match="three distinct names"):
            compose({"multimodal": {"q": 1}}, conditions=bad)
    with pytest.raises(ValueError, match="three distinct names"):
        shortcut_outcomes({"multimodal": {"q": 1}}, conditions=("a", "b", "c", "d"))
