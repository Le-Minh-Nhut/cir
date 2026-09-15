from __future__ import annotations

import torch

from evaluation.cirr import evaluate_cirr_metrics, global_recall_at_k, subset_recall_at_k


def test_cirr_global_recall_removes_reference() -> None:
    gallery = ["ref", "a", "target", "other"]
    scores = torch.tensor([[100.0, 8.0, 9.0, 7.0]])

    assert global_recall_at_k(scores, ["target"], ["ref"], gallery, 1) == 100.0


def test_cirr_subset_recall_uses_only_group_without_reference() -> None:
    gallery = ["ref", "a", "b", "target", "c", "d", "outside"]
    scores = torch.tensor([[100.0, 9.0, 7.0, 8.0, 6.0, 5.0, 99.0]])
    group = [["ref", "a", "b", "target", "c", "d"]]

    assert subset_recall_at_k(scores, ["target"], ["ref"], group, gallery, 1) == 0.0
    assert subset_recall_at_k(scores, ["target"], ["ref"], group, gallery, 2) == 100.0
    assert subset_recall_at_k(scores, ["target"], ["ref"], group, gallery, 3) == 100.0

    metrics = evaluate_cirr_metrics(scores, ["target"], ["ref"], group, gallery)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 100.0
    assert metrics["recall_subset_at_1"] == 0.0
    assert metrics["recall_subset_at_2"] == 100.0
    assert metrics["recall_subset_at_3"] == 100.0
