from __future__ import annotations

import random

import torch

from datasets.fashioniq import (
    compose_fashioniq_caption,
    normalize_fashioniq_caption,
    validate_correction_policy,
)
from evaluation.fashioniq import evaluate_fashioniq_recall, macro_average_fashioniq


def test_validated_caption_order_and_correction_semantics() -> None:
    captions = ("Is BRIGT, red.", "Has sleeves?")
    corrections = {"brigt": "bright"}
    assert compose_fashioniq_caption(captions, "ordered_and") == "Is brigt, red and Has sleeves"
    assert normalize_fashioniq_caption(captions[0], corrections) == "is bright red"
    assert (
        compose_fashioniq_caption(captions, "normalized_ordered_and", corrections)
        == "is bright red and has sleeves"
    )
    # Determinism is controlled by the dataset's seeded local RNG; branch semantics remain exact.
    assert compose_fashioniq_caption(captions, "randomized_four_way", rng=random.Random(1)) == "Is brigt, red and Has sleeves"
    assert validate_correction_policy("fashioniq") == "fashioniq"
    assert validate_correction_policy("none") == "none"


def test_fashioniq_hand_ranking_and_macro_metrics() -> None:
    gallery_ids = tuple(f"image-{index}" for index in range(60))
    scores = torch.zeros(2, 60)
    scores[0, 7] = 10
    scores[1, 55] = 10
    scores[1, :49] = torch.arange(49, dtype=torch.float32) + 20
    metrics = evaluate_fashioniq_recall(scores, ("image-7", "image-55"), gallery_ids)
    assert metrics == {"recall_at_10": 50.0, "recall_at_50": 100.0}
    macro = macro_average_fashioniq({"a": metrics, "b": {"recall_at_10": 100.0, "recall_at_50": 100.0}})
    assert macro == {"recall_at_10": 75.0, "recall_at_50": 100.0, "mean_recall": 87.5}
