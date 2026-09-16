from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from evaluation.cirr_encoder import (
    cosine_similarity_scores,
    evaluate_cirr_encoder_metrics,
    rank_reference_excluded_gallery,
)
from models.taper import TAPER


def test_cirr_encoder_reference_exclusion_global_subset_and_selection() -> None:
    gallery = ["reference", "target", "a", "b", "c", "d", "outside"]
    scores = torch.tensor([[0.99, 0.80, 0.90, 0.70, 0.60, 0.50, 0.95]])
    group = [["reference", "target", "a", "b", "c", "d"]]

    ranking = rank_reference_excluded_gallery(scores, ["reference"], gallery)
    ranked_ids = [gallery[index] for index in ranking[0].tolist()]
    metrics = evaluate_cirr_encoder_metrics(
        scores,
        ["target"],
        ["reference"],
        group,
        gallery,
    )

    assert ranked_ids == ["outside", "a", "target", "b", "c", "d"]
    assert "reference" not in ranked_ids
    assert metrics == {
        "recall_at_1": 0.0,
        "recall_at_5": 100.0,
        "recall_at_10": 100.0,
        "recall_at_50": 100.0,
        "recall_subset_at_1": 0.0,
        "recall_subset_at_2": 100.0,
        "recall_subset_at_3": 100.0,
        "selection_score": 500.0,
        "mean_recall": 50.0,
    }


def test_cirr_encoder_cosine_similarity_normalizes_both_sides() -> None:
    queries = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    gallery = torch.tensor([[6.0, 8.0], [-4.0, 3.0]])

    scores = cosine_similarity_scores(queries, gallery)

    assert torch.allclose(scores, torch.tensor([[1.0, 0.0], [0.8, 0.6]]), atol=1e-6)


def test_existing_taper_scores_are_cosine_equivalent_for_vector_gallery() -> None:
    raw_queries = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    gallery = torch.tensor([[6.0, 8.0], [-4.0, 3.0]])
    query_harness = SimpleNamespace(query_head=lambda states: states)
    score_harness = SimpleNamespace(query_dim=2)

    taper_queries = TAPER.make_query(query_harness, raw_queries)
    taper_scores = TAPER._retrieval_scores(
        score_harness, taper_queries, gallery.unsqueeze(1)
    )

    assert torch.allclose(
        taper_scores,
        cosine_similarity_scores(raw_queries, gallery),
        atol=1e-6,
    )


def _encoder_reference_metrics(
    scores: np.ndarray,
    target: str,
    reference: str,
    group: list[str],
    gallery: list[str],
) -> dict[str, float]:
    gallery_index = {image_id: index for index, image_id in enumerate(gallery)}
    ranking = np.argsort(-scores[0], kind="stable")
    ranking = ranking[ranking != gallery_index[reference]]
    subset_columns = {
        gallery_index[image_id] for image_id in group if image_id != reference
    }
    subset_ranking = [column for column in ranking if column in subset_columns]
    metrics = {
        f"recall_at_{k}": 100.0 * float(gallery_index[target] in ranking[:k])
        for k in (1, 5, 10, 50)
    }
    metrics.update(
        {
            f"recall_subset_at_{k}": 100.0
            * float(gallery_index[target] in subset_ranking[:k])
            for k in (1, 2, 3)
        }
    )
    metrics["selection_score"] = sum(metrics.values())
    metrics["mean_recall"] = 0.5 * (
        metrics["recall_at_5"] + metrics["recall_subset_at_1"]
    )
    return metrics


def test_cirr_encoder_matches_official_ranking_logic_on_synthetic_case() -> None:
    gallery = ["reference", "target", "a", "b", "c", "d", "outside"]
    group = ["reference", "target", "a", "b", "c", "d"]
    scores = torch.tensor([[0.99, 0.80, 0.90, 0.70, 0.60, 0.50, 0.95]])

    ours = evaluate_cirr_encoder_metrics(
        scores, ["target"], ["reference"], [group], gallery
    )
    reference = _encoder_reference_metrics(
        scores.numpy(), "target", "reference", group, gallery
    )

    assert ours == reference
