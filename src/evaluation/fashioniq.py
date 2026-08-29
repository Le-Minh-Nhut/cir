from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.images import ImageBatch, ImageIdDataset, collate_image_ids
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery, load_fashioniq_split_ids
from models.iag_srme.model import IAGSRME


def recall_at_k(
    scores: Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    k: int,
    reference_ids: Sequence[str] | None = None,
) -> float:
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(gallery_ids)):
        raise ValueError("scores must be [queries,gallery]")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("gallery IDs must be unique")
    if not 0 < k <= len(gallery_ids):
        raise ValueError("k outside gallery size")
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if any(target_id not in gallery_index for target_id in target_ids):
        raise ValueError("target missing from gallery")
    filtered_scores = scores.clone()
    if reference_ids is not None:
        if len(reference_ids) != len(target_ids):
            raise ValueError("reference_ids length mismatch")
        for row, reference_id in enumerate(reference_ids):
            if reference_id in gallery_index and reference_id != target_ids[row]:
                filtered_scores[row, gallery_index[reference_id]] = -torch.inf
    targets = torch.tensor(
        [gallery_index[target_id] for target_id in target_ids], device=scores.device
    )[:, None]
    return filtered_scores.topk(k, dim=1).indices.eq(targets).any(dim=1).float().mean().item() * 100.0


def evaluate_fashioniq_recall(
    scores: Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    return {
        "recall_at_10": recall_at_k(scores, target_ids, gallery_ids, 10, reference_ids),
        "recall_at_50": recall_at_k(scores, target_ids, gallery_ids, 50, reference_ids),
    }


def build_fashioniq_gallery(
    protocol: str,
    split_root: str | Path,
    category: str,
    annotations: Sequence[FashionIQAnnotation],
    split: str,
) -> list[str]:
    if protocol == "fashioniq_original":
        result = load_fashioniq_split_ids(split_root, split, category)
    elif protocol == "fashioniq_val":
        result = build_pair_union_gallery(annotations)
    else:
        raise ValueError(f"unsupported FashionIQ protocol: {protocol}")
    if len(set(result)) != len(result):
        raise ValueError("gallery must contain unique image IDs")
    return result


def macro_average_fashioniq(results: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not results:
        raise ValueError("category results must not be empty")
    recall_10 = sum(item["recall_at_10"] for item in results.values()) / len(results)
    recall_50 = sum(item["recall_at_50"] for item in results.values()) / len(results)
    return {"recall_at_10": recall_10, "recall_at_50": recall_50, "mean_recall": 0.5 * (recall_10 + recall_50)}


@torch.no_grad()
def encode_gallery(
    model: IAGSRME,
    image_ids: list[str],
    image_store: DirectoryImageStore,
    image_processor: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tensor:
    loader = DataLoader(
        ImageIdDataset(image_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(
            collate_image_ids, image_store=image_store, image_processor=image_processor
        ),
    )
    features: list[Tensor] = []
    ordered_ids: list[str] = []
    for batch_ids, pixels in loader:
        ordered_ids.extend(batch_ids)
        features.append(model.encode_gallery(pixels.to(device)).cpu())
    if ordered_ids != image_ids:
        raise AssertionError("gallery encoding order changed")
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_fashioniq(
    model: IAGSRME,
    val_loaders: Mapping[str, DataLoader[ImageBatch]],
    val_annotations: Mapping[str, Sequence[FashionIQAnnotation]],
    *,
    protocol: str,
    split_root: str | Path,
    split: str,
    image_store: DirectoryImageStore,
    image_processor: Any,
    device: torch.device,
    gallery_batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    model.eval()
    category_results: dict[str, dict[str, float]] = {}
    for category, loader in val_loaders.items():
        gallery_ids = build_fashioniq_gallery(
            protocol, split_root, category, val_annotations[category], split
        )
        gallery = encode_gallery(
            model,
            gallery_ids,
            image_store,
            image_processor,
            device,
            gallery_batch_size,
            num_workers,
        ).to(device)
        queries: list[Tensor] = []
        target_ids: list[str] = []
        reference_ids: list[str] = []
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(
                batch.reference_pixels,
                batch.input_ids,
                batch.attention_mask,
                batch.content_mask,
            )
            queries.append(output.final_query)
            reference_ids.extend(batch.reference_ids)
            if any(target_id is None for target_id in batch.target_ids):
                raise ValueError("validation target missing")
            target_ids.extend(str(target_id) for target_id in batch.target_ids)
        scores = F.normalize(torch.cat(queries), dim=-1) @ F.normalize(gallery, dim=-1).T
        category_results[category] = evaluate_fashioniq_recall(
            scores, target_ids, gallery_ids, reference_ids
        )
    average = macro_average_fashioniq(category_results)
    metrics = dict(average)
    for category, result in category_results.items():
        metrics[f"{category}_recall_at_10"] = result["recall_at_10"]
        metrics[f"{category}_recall_at_50"] = result["recall_at_50"]
    return metrics
