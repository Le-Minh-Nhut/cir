"""Golden caption fixtures for the FashionIQ caption policies the workbench ships.

These are source-derived expectations, not runtime equivalences: the upstream methods build
their captions inside their own dataset classes (and then hand the string to a LAVIS/BLIP caption
processor), so the workbench can only be shown to reproduce the *string construction* stage on
the laptop. The processor and tokenizer stage remains a GPU-host question, recorded in
workbench/docs/audits/PHASE2_SOURCES_*.md.

Every case below names the source behaviour it mirrors:
- `ordered_and` mirrors QuRe / TME evaluation string construction.
- `normalized_ordered_and` mirrors the iLearn lowercasing + punctuation-to-space + token
  dictionary normalisation followed by source-order `" and "` composition.
"""
from __future__ import annotations

import pytest

from datasets.fashioniq import (
    compose_fashioniq_caption,
    normalize_fashioniq_caption,
)


def test_ordered_and_matches_the_qure_tme_string_rule() -> None:
    """cap1 is stripped and capitalised, cap2 is only stripped, joined by ` and `.

    The exclamation mark is *not* removed: `.strip(".?, ")` strips only those characters from
    the ends, which is what the upstream construction does.
    """
    assert compose_fashioniq_caption(
        ("Make it RED!", "Less-long."), "ordered_and") == "Make it red! and Less-long"
    assert compose_fashioniq_caption(
        ("a red shirt.", "with longer sleeves"), "ordered_and") == "A red shirt and with longer sleeves"


def test_ordered_and_collapses_endpoint_punctuation_only() -> None:
    assert compose_fashioniq_caption(("dots...", "commas,,"), "ordered_and") == "Dots and commas"


def test_normalized_ordered_and_lowercases_and_strips_punctuation() -> None:
    """ENCODER-style normalisation: lowercase, punctuation to space, then compose."""
    assert compose_fashioniq_caption(
        ("Make it RED!", "Less-long."),
        "normalized_ordered_and") == "make it red and less long"


def test_normalized_ordered_and_applies_the_token_dictionary_exactly() -> None:
    """Dictionary lookup is exact per token, after normalisation — not substring matching."""
    corrections = {"blue": "crimson"}
    assert compose_fashioniq_caption(
        ("A BLUE shirt.", "shorter sleeves"),
        "normalized_ordered_and",
        corrections) == "a crimson shirt and shorter sleeves"
    # `blueish` is a different token and must be left alone.
    assert normalize_fashioniq_caption("blueish BLUE", corrections) == "blueish crimson"


def test_caption_order_is_source_order_and_is_not_sorted() -> None:
    """A reversed pair must produce a reversed string: the policy is order-sensitive."""
    forward = compose_fashioniq_caption(("first part", "second part"), "ordered_and")
    reversed_order = compose_fashioniq_caption(("second part", "first part"), "ordered_and")
    assert forward == "First part and second part"
    assert reversed_order == "Second part and first part"
    assert forward != reversed_order


def test_empty_and_special_captions_do_not_crash() -> None:
    """The corpus contains short and odd captions; the composition must stay total."""
    assert compose_fashioniq_caption(("", "shorter sleeves"), "normalized_ordered_and") == \
        " and shorter sleeves"
    assert compose_fashioniq_caption(("shorter sleeves", ""), "normalized_ordered_and") == \
        "shorter sleeves and "
    # Punctuation-only caption normalises to an empty token stream.
    assert normalize_fashioniq_caption("...!!!") == ""


def test_long_captions_are_truncated_by_the_processor_not_by_composition() -> None:
    """Composition must not silently truncate: the upstream processor owns the token limit."""
    long_caption = " ".join(["word"] * 80)
    composed = compose_fashioniq_caption((long_caption, "extra"), "normalized_ordered_and")
    assert composed.count("word") == 80
    assert composed.endswith(" and extra")


def test_an_unknown_caption_policy_is_refused() -> None:
    with pytest.raises(AssertionError):
        compose_fashioniq_caption(("a", "b"), "some_other_policy")
