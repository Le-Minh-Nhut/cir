from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from cache.features import (
    TextFeatureCache,
    get_features_by_ids,
    get_text_features_by_sample_ids,
)
from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery


FASHIONIQ_ENCODER_CATEGORIES = ("dress", "shirt", "toptee")
FASHIONIQ_ENCODER_RECALL_KS = (1, 10, 50)


def fashioniq_encoder_recall_at_k(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    k: int,
) -> float:
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(gallery_ids)):
        raise ValueError("scores must have shape [queries, gallery]")
    if not target_ids or len(reference_ids) != len(target_ids):
        raise ValueError("FashionIQ query IDs must be non-empty and aligned")
    if k < 1:
        raise ValueError("k must be positive")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("FashionIQ gallery IDs must be unique")

    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    hits = []
    for row, (target_id, reference_id) in enumerate(
        zip(target_ids, reference_ids, strict=True)
    ):
        if target_id not in gallery_index or reference_id not in gallery_index:
            raise ValueError("FashionIQ target/reference is missing from the ENCODER gallery")
        ranking = torch.argsort(scores[row], descending=True, stable=True)
        ranking = ranking[ranking != gallery_index[reference_id]]
        hits.append(
            bool(
                ranking[: min(k, len(ranking))]
                .eq(gallery_index[target_id])
                .any()
            )
        )
    return 100.0 * sum(hits) / len(hits)


def evaluate_fashioniq_encoder_category(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    category: str,
) -> dict[str, float]:
    if category not in FASHIONIQ_ENCODER_CATEGORIES:
        raise ValueError(f"Unsupported FashionIQ category: {category}")
    return {
        f"{category}_r{k}": fashioniq_encoder_recall_at_k(
            scores, target_ids, reference_ids, gallery_ids, k
        )
        for k in FASHIONIQ_ENCODER_RECALL_KS
    }


def fashioniq_encoder_selection_score(metrics: dict[str, float]) -> float:
    selection_metrics = [f"{category}_r10" for category in FASHIONIQ_ENCODER_CATEGORIES]
    missing = [name for name in selection_metrics if name not in metrics]
    if missing:
        raise ValueError(f"FashionIQ ENCODER selection metrics are missing: {missing}")
    return sum(metrics[name] for name in selection_metrics)


@torch.no_grad()
def evaluate_fashioniq_encoder(
    model,
    val_loaders: dict[str, DataLoader],
    val_annotations: dict[str, Sequence[FashionIQAnnotation]],
    *,
    retrieval_features: Tensor,
    native_features: Tensor,
    retrieval_name_to_idx: dict[str, int],
    native_name_to_idx: dict[str, int],
    text_cache: TextFeatureCache,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}

    for category in FASHIONIQ_ENCODER_CATEGORIES:
        val_loader = val_loaders[category]
        gallery_ids = build_pair_union_gallery(val_annotations[category])
        gallery_features = get_features_by_ids(
            gallery_ids, retrieval_features, retrieval_name_to_idx
        ).to(device)
        score_batches: list[Tensor] = []
        target_ids: list[str] = []
        reference_ids: list[str] = []

        for batch in val_loader:
            reference_native = get_features_by_ids(
                batch.reference_ids, native_features, native_name_to_idx
            ).to(device=device, dtype=torch.float32)
            text_states, teacher_text_states, attention_mask, content_mask = (
                get_text_features_by_sample_ids(
                    batch.sample_ids, batch.modification_texts, text_cache
                )
            )
            output = model.retrieve(
                reference_features=reference_native[:, 0, :],
                teacher_reference_features=reference_native,
                text_states=text_states.to(device=device, dtype=torch.float32),
                teacher_text_states=teacher_text_states.to(
                    device=device, dtype=torch.float32
                ),
                text_attention_mask=attention_mask.to(device=device, dtype=torch.bool),
                text_content_mask=content_mask.to(device=device, dtype=torch.bool),
                gallery_features=gallery_features,
            )
            # TAPER produces normalized queries and cosine-equivalent scores
            # against normalized cached gallery tokens.
            score_batches.append(output["scores"].cpu())
            for target_id in batch.target_ids:
                if target_id is None:
                    raise ValueError("FashionIQ validation sample is missing target_id")
                target_ids.append(target_id)
            reference_ids.extend(batch.reference_ids)

        if not score_batches:
            raise RuntimeError(f"FashionIQ ENCODER validation loader is empty: {category}")
        metrics.update(
            evaluate_fashioniq_encoder_category(
                torch.cat(score_batches),
                target_ids,
                reference_ids,
                gallery_ids,
                category,
            )
        )

    metrics["selection_score"] = fashioniq_encoder_selection_score(metrics)
    return metrics
