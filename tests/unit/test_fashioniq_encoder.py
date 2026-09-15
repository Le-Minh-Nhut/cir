from __future__ import annotations

import torch

from datasets.fashioniq import (
    FashionIQAnnotation,
    build_pair_union_gallery,
    compose_fashioniq_caption,
)
from evaluation.fashioniq_encoder import (
    evaluate_fashioniq_encoder_category,
    fashioniq_encoder_selection_score,
)
from train_fashioniq_encoder import CAPTION_POLICY


def test_fashioniq_encoder_caption_policy_and_reduced_gallery() -> None:
    caption = compose_fashioniq_caption(
        ("Make it RED!", "Less-long."),
        CAPTION_POLICY,
        {"red": "crimson"},
    )
    annotations = [
        FashionIQAnnotation("ref-a", "target-a", ("one", "two"), "dress", 0),
        FashionIQAnnotation("ref-b", "target-a", ("three", "four"), "dress", 1),
    ]

    assert caption == "make it crimson and less long"
    assert build_pair_union_gallery(annotations) == ["ref-a", "target-a", "ref-b"]


def test_fashioniq_encoder_category_recalls_are_independent() -> None:
    gallery = ["reference", "target", "a", "b"]
    reference_ids = ["reference"]
    target_ids = ["target"]

    dress = evaluate_fashioniq_encoder_category(
        torch.tensor([[0.99, 0.90, 0.80, 0.70]]),
        target_ids,
        reference_ids,
        gallery,
        "dress",
    )
    shirt = evaluate_fashioniq_encoder_category(
        torch.tensor([[0.99, 0.80, 0.90, 0.70]]),
        target_ids,
        reference_ids,
        gallery,
        "shirt",
    )
    toptee = evaluate_fashioniq_encoder_category(
        torch.tensor([[0.99, 0.70, 0.90, 0.80]]),
        target_ids,
        reference_ids,
        gallery,
        "toptee",
    )

    assert dress == {"dress_r1": 100.0, "dress_r10": 100.0, "dress_r50": 100.0}
    assert shirt == {"shirt_r1": 0.0, "shirt_r10": 100.0, "shirt_r50": 100.0}
    assert toptee == {"toptee_r1": 0.0, "toptee_r10": 100.0, "toptee_r50": 100.0}
    assert not set(dress) & set(shirt)
    assert not set(shirt) & set(toptee)


def test_fashioniq_encoder_selection_is_sum_of_category_r10() -> None:
    metrics = {
        "dress_r1": 1.0,
        "dress_r10": 10.0,
        "dress_r50": 99.0,
        "shirt_r1": 2.0,
        "shirt_r10": 20.0,
        "shirt_r50": 98.0,
        "toptee_r1": 3.0,
        "toptee_r10": 30.0,
        "toptee_r50": 97.0,
    }

    assert fashioniq_encoder_selection_score(metrics) == 60.0
