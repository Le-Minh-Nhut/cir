from __future__ import annotations

import random
import json

import pytest
import torch

from datasets.fashioniq import (
    compose_fashioniq_caption,
    normalize_fashioniq_caption,
    resolve_fashioniq_correction_dicts,
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


def test_correction_resolver_fails_clearly_without_silent_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        resolve_fashioniq_correction_dicts(tmp_path, "fashioniq")
    message = str(error.value)
    assert "correction_dict_dress.json" in message
    assert "correction_dict_shirt.json" in message
    assert "correction_dict_toptee.json" in message
    assert str(tmp_path.resolve()) in message
    assert "correction_policy=none" in message
    assert resolve_fashioniq_correction_dicts(tmp_path, "none") is None


def test_correction_resolver_loads_all_audited_files(tmp_path) -> None:
    for category in ("dress", "shirt", "toptee"):
        (tmp_path / f"correction_dict_{category}.json").write_text(
            json.dumps({"brigt": "bright"}), encoding="utf-8"
        )
    resolved = resolve_fashioniq_correction_dicts(tmp_path, "fashioniq")
    assert resolved == {category: {"brigt": "bright"} for category in ("dress", "shirt", "toptee")}
