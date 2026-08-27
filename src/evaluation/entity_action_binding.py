from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader

from cache.features import (
    DenseImageFeatureCache,
    TextFeatureCache,
    get_dense_features_by_ids,
    get_features_by_ids,
    get_text_features_with_global_by_sample_ids,
)
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    macro_average_fashioniq,
)
from models.entity_action_binding import EntityActionBindingCIR, RelationVariant


@torch.no_grad()
def encode_image_ids(
    model: EntityActionBindingCIR,
    image_ids: list[str],
    *,
    global_features: torch.Tensor,
    global_name_to_idx: dict[str, int],
    dense_cache: DenseImageFeatureCache,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        image_global = get_features_by_ids(ids, global_features, global_name_to_idx)
        if image_global.ndim != 3 or image_global.shape[1] != 1:
            raise ValueError("Global image cache must be [N,1,D]")
        dense, mask = get_dense_features_by_ids(ids, dense_cache)
        result = model.encode_image(
            image_global[:, 0].to(device=device, dtype=torch.float32),
            dense.to(device=device),
            mask.to(device=device),
        )
        encoded.append(result["embedding"].cpu())
    return torch.cat(encoded, dim=0)


@torch.no_grad()
def evaluate_entity_action_binding(
    model: EntityActionBindingCIR,
    val_loaders: Mapping[str, DataLoader],
    val_annotations: Mapping[str, list],
    *,
    protocol: str,
    split_root,
    split: str,
    global_features: torch.Tensor,
    global_name_to_idx: dict[str, int],
    dense_cache: DenseImageFeatureCache,
    text_cache: TextFeatureCache,
    device: torch.device,
    gallery_batch_size: int,
    variant: RelationVariant = "full",
    relation_index: int | None = None,
) -> dict[str, float]:
    category_results: dict[str, dict[str, float]] = {}
    for category, loader in val_loaders.items():
        gallery_ids = build_fashioniq_gallery(
            protocol, split_root, category, val_annotations[category], split
        )
        gallery = encode_image_ids(
            model,
            gallery_ids,
            global_features=global_features,
            global_name_to_idx=global_name_to_idx,
            dense_cache=dense_cache,
            device=device,
            batch_size=gallery_batch_size,
        ).to(device)
        score_batches: list[torch.Tensor] = []
        target_ids: list[str] = []
        for batch in loader:
            reference_global = get_features_by_ids(
                batch.reference_ids, global_features, global_name_to_idx
            )
            reference_dense, reference_mask = get_dense_features_by_ids(
                batch.reference_ids, dense_cache
            )
            text_states, _, content_mask, text_global = (
                get_text_features_with_global_by_sample_ids(
                    batch.sample_ids, batch.modification_texts, text_cache
                )
            )
            output = model(
                reference_global=reference_global[:, 0].to(device=device, dtype=torch.float32),
                reference_dense=reference_dense.to(device=device),
                reference_dense_mask=reference_mask.to(device=device),
                text_global=text_global.to(device=device, dtype=torch.float32),
                text_states=text_states.to(device=device, dtype=torch.float32),
                text_content_mask=content_mask.to(device=device, dtype=torch.bool),
                variant=variant,
                relation_index=relation_index,
            )
            score_batches.append((output["query"] @ gallery.T).cpu())
            if any(target_id is None for target_id in batch.target_ids):
                raise ValueError("Validation sample is missing target_id")
            target_ids.extend(str(target_id) for target_id in batch.target_ids)
        category_results[category] = evaluate_fashioniq_category(
            torch.cat(score_batches), target_ids, gallery_ids
        )
    average = macro_average_fashioniq(category_results)
    metrics = dict(average)
    for category, result in category_results.items():
        metrics[f"{category}_recall_at_10"] = result["recall_at_10"]
        metrics[f"{category}_recall_at_50"] = result["recall_at_50"]
    return metrics
