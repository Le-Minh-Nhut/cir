from __future__ import annotations

import torch

from evaluation.cirr import evaluate_cirr_metrics as evaluate_cirr_legacy_metrics
from evaluation.cirr_encoder import evaluate_cirr_encoder_metrics
from evaluation.fashioniq import evaluate_fashioniq_category
from evaluation.fashioniq_encoder import evaluate_fashioniq_encoder_category


def test_encoder_evaluators_do_not_replace_legacy_evaluators() -> None:
    assert evaluate_cirr_legacy_metrics is not evaluate_cirr_encoder_metrics
    assert evaluate_cirr_legacy_metrics.__module__ == "evaluation.cirr"
    assert evaluate_cirr_encoder_metrics.__module__ == "evaluation.cirr_encoder"
    assert evaluate_fashioniq_category is not evaluate_fashioniq_encoder_category
    assert evaluate_fashioniq_category.__module__ == "evaluation.fashioniq"
    assert evaluate_fashioniq_encoder_category.__module__ == (
        "evaluation.fashioniq_encoder"
    )

    gallery = ["ref", "target", "a", "b", "c", "d"]
    scores = torch.tensor([[0.99, 0.80, 0.90, 0.70, 0.60, 0.50]])
    groups = [["ref", "target", "a", "b", "c", "d"]]
    legacy = evaluate_cirr_legacy_metrics(
        scores, ["target"], ["ref"], groups, gallery
    )
    encoder = evaluate_cirr_encoder_metrics(
        scores, ["target"], ["ref"], groups, gallery
    )

    assert "mean_recall" in legacy and "selection_score" not in legacy
    assert "selection_score" in encoder and "mean_recall" in encoder