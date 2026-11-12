"""The gap triple is resolved from the conditions actually scored, by ROLE, and it is recorded.

The defect this closes: `compose` was parameterised, but the runner still passed the protocol triple,
so a CLVC Full run under the new profile would publish `composition_gap_ndcg: null` again — its image
role is `image_only_empty_text`, which stays `ABLATION_UNSUPPORTED` for that model. A ~30-minute GPU
run would then be spent re-shipping the defect.

Two directions are asserted:
* CLVC-shaped (ablation carries ranks, protocol image condition absent) → gap is NOT null, and the
  value equals the hand-computed one over that cohort;
* TG-CIR-shaped (protocol triple scored) → the resolved triple is the default and the gap is
  unchanged, bit for bit, from the pre-change function.

CPU only: no GPU, no model, no composition run.
"""
from __future__ import annotations

import pytest

from workbench.composition import runner
from workbench.composition.metrics import compose, ndcg_full

PROTOCOL = ("multimodal", "text_only_black_image", "image_only_empty_text")
IMAGE_ONLY = "clvc_image_encoder_only"
PROTOCOL_TRIPLE = PROTOCOL

@pytest.mark.parametrize(("scored", "expected", "reason_contains"), [
    # TG-CIR today, and any run that scored the protocol triple: unchanged.
    (["multimodal", "text_only_black_image", "image_only_empty_text"], PROTOCOL_TRIPLE, None),
    # CLVC under the new profile: the ablation fills the image role.
    (["multimodal", "text_only_black_image", IMAGE_ONLY],
     ("multimodal", "text_only_black_image", IMAGE_ONLY), "image role filled by"),
    # CLVC's OLD profile: no image role at all, protocol triple kept, gap will be null with compose's
    # own reason. The resolution says which role is missing.
    (["multimodal", "text_only_black_image"], PROTOCOL_TRIPLE, "role 'image' has no scored condition"),
    # Both image conditions scored: the ABLATION NEVER REPLACES a protocol number.
    (["multimodal", "text_only_black_image", "image_only_empty_text", IMAGE_ONLY],
     PROTOCOL_TRIPLE, None),
    # No text role: still resolved, and the reason names the missing role.
    (["multimodal", IMAGE_ONLY],
     ("multimodal", "text_only_black_image", IMAGE_ONLY), "role 'text' has no scored condition"),
    ([], PROTOCOL_TRIPLE, "role 'multimodal' has no scored condition"),
])
def test_resolved_triple_matrix(scored, expected, reason_contains) -> None:
    triple, reason = runner.resolve_gap_conditions(scored)
    assert triple == expected
    if reason_contains is None:
        assert reason is None
    else:
        assert reason_contains in (reason or "")


def test_ablation_never_replaces_a_scored_protocol_image_condition() -> None:
    """Order of the caller's list must not decide the role; a scored protocol condition wins."""
    forward, _ = runner.resolve_gap_conditions(["multimodal", "text_only_black_image",
                                                "image_only_empty_text", IMAGE_ONLY])
    reverse, _ = runner.resolve_gap_conditions([IMAGE_ONLY, "image_only_empty_text",
                                                "text_only_black_image", "multimodal"])
    assert forward == reverse == PROTOCOL_TRIPLE


def test_clvc_shaped_table_produces_a_real_gap_with_a_hand_computed_value() -> None:
    """Three conditions scored, `image_only_empty_text` absent, gap must NOT be null."""
    ranks = {
        "multimodal": {"a": 1, "b": 7, "c": 7, "d": 7},          # mean nDCG 0.5
        "text_only_black_image": {"a": 15, "b": 15, "c": 15, "d": 15},   # 0.25
        IMAGE_ONLY: {"a": 255, "b": 255, "c": 255, "d": 255},     # 0.125
    }
    scored = [condition for condition in ranks if ranks[condition]]
    triple, reason = runner.resolve_gap_conditions(scored)
    assert triple == ("multimodal", "text_only_black_image", IMAGE_ONLY)
    assert "image role filled by" in reason

    result = compose(ranks, query_count=4, conditions=triple)
    assert result.composition_gap_ndcg is not None, "the CLVC Full run must not publish a null gap"
    assert result.composition_gap_ndcg == pytest.approx(0.5)
    assert result.gap_cohort_count == 4
    assert result.gap_conditions == triple
    # nDCG means, by hand: 0.5 / 0.25 / 0.125
    assert result.gap_multimodal_ndcg == pytest.approx(0.5)
    assert result.gap_text_only_ndcg == pytest.approx(0.25)
    assert result.gap_image_only_ndcg == pytest.approx(0.125)
    # MRR gap over the same cohort: mm (1 + 3/7)/4, text 1/15, image 1/255.
    mm_mrr = (1.0 + 3 * (1.0 / 7)) / 4
    assert result.mrr_gap == pytest.approx(mm_mrr - max(1.0 / 15, 1.0 / 255))


def test_tgcir_shaped_table_keeps_the_protocol_gap_bit_for_bit() -> None:
    """The inverse direction: today's TG-CIR case must not move by even one bit."""
    ranks = {
        "multimodal": {"a": 1, "b": 10, "c": None, "d": 400},
        "text_only_black_image": {"a": 2, "b": 1, "c": 30, "d": 400},
        "image_only_empty_text": {"a": 50, "b": 50, "c": None, "d": None},
    }
    triple, reason = runner.resolve_gap_conditions(list(ranks))
    assert triple == PROTOCOL_TRIPLE
    assert reason is None
    resolved = compose(ranks, conditions=triple)
    default = compose(ranks)
    assert resolved.composition_gap_ndcg == default.composition_gap_ndcg == -0.26520382860036595
    assert resolved.gap_cohort_count == default.gap_cohort_count == 2
    assert resolved.shortcut_outcomes == default.shortcut_outcomes
    assert resolved.gap_conditions == default.gap_conditions == PROTOCOL_TRIPLE


def test_only_conditions_with_a_rank_count_as_scored() -> None:
    """A condition present in the table but ALL-null is not scored, so it cannot fill a role."""
    triple, reason = runner.resolve_gap_conditions(["multimodal", "text_only_black_image"])
    assert triple[2] == "image_only_empty_text"
    assert "role 'image' has no scored condition" in reason


def test_every_unscored_condition_still_reports_compose_reason() -> None:
    """A role with no scored condition yields the EXISTING reason, not a new one."""
    ranks = {condition: {"a": None} for condition in ("multimodal", "text_only_black_image",
                                                      IMAGE_ONLY)}
    triple, _ = runner.resolve_gap_conditions([])
    result = compose(ranks, conditions=triple)
    assert result.composition_gap_ndcg is None
    assert result.gap_unavailable_reason == "missing condition(s): image_only_empty_text"
    assert ndcg_full(1) == 1.0
