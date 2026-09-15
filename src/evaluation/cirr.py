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
from datasets.common import CIRBatch


def _validate_scores(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
) -> dict[str, int]:
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(gallery_ids)):
        raise ValueError("scores must have shape [queries, gallery]")
    if not target_ids:
        raise ValueError("CIRR metrics require at least one query")
    if len(reference_ids) != len(target_ids):
        raise ValueError("reference IDs must align with target IDs")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("CIRR gallery IDs must be unique")

    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    missing_targets = [target_id for target_id in target_ids if target_id not in gallery_index]
    missing_references = [
        reference_id for reference_id in reference_ids if reference_id not in gallery_index
    ]
    if missing_targets:
        raise ValueError(f"CIRR targets missing from gallery: {missing_targets[:5]}")
    if missing_references:
        raise ValueError(f"CIRR references missing from gallery: {missing_references[:5]}")
    return gallery_index


def global_recall_at_k(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    gallery_index = _validate_scores(scores, target_ids, reference_ids, gallery_ids)
    if len(gallery_ids) < 2:
        raise ValueError("CIRR retrieval requires at least two gallery images")

    hits: list[bool] = []
    for row, (target_id, reference_id) in enumerate(
        zip(target_ids, reference_ids, strict=True)
    ):
        reference_column = gallery_index[reference_id]
        ranking = torch.argsort(scores[row], descending=True)
        ranking = ranking[ranking != reference_column]
        target_column = gallery_index[target_id]
        hits.append(bool(ranking[: min(k, len(ranking))].eq(target_column).any()))

    return float(torch.tensor(hits, dtype=torch.float32).mean().item() * 100.0)


def subset_recall_at_k(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    group_members: Sequence[Sequence[str]],
    gallery_ids: Sequence[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    gallery_index = _validate_scores(scores, target_ids, reference_ids, gallery_ids)
    if len(group_members) != len(target_ids):
        raise ValueError("CIRR groups must align with score rows")

    hits: list[bool] = []
    for row, (target_id, reference_id, members) in enumerate(
        zip(target_ids, reference_ids, group_members, strict=True)
    ):
        if len(set(members)) != len(members):
            raise ValueError(f"CIRR group for row {row} contains duplicate IDs")
        if reference_id not in members:
            raise ValueError(f"CIRR group for row {row} does not contain its reference")

        candidates = [member for member in members if member != reference_id]
        if target_id not in candidates:
            raise ValueError(f"CIRR group for row {row} does not contain its target")
        missing = [member for member in candidates if member not in gallery_index]
        if missing:
            raise ValueError(f"CIRR group images missing from gallery: {missing[:5]}")

        columns = torch.tensor(
            [gallery_index[member] for member in candidates], device=scores.device
        )
        ranking = torch.argsort(scores[row, columns], descending=True)
        target_position = candidates.index(target_id)
        hits.append(bool(ranking[: min(k, len(candidates))].eq(target_position).any()))

    return float(torch.tensor(hits, dtype=torch.float32).mean().item() * 100.0)


def evaluate_cirr_metrics(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    group_members: Sequence[Sequence[str]],
    gallery_ids: Sequence[str],
    *,
    global_ks: Sequence[int] = (1, 5, 10, 50),
    subset_ks: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    metrics = {
        f"recall_at_{k}": global_recall_at_k(
            scores, target_ids, reference_ids, gallery_ids, k
        )
        for k in global_ks
    }
    metrics.update(
        {
            f"recall_subset_at_{k}": subset_recall_at_k(
                scores, target_ids, reference_ids, group_members, gallery_ids, k
            )
            for k in subset_ks
        }
    )
    if "recall_at_5" in metrics and "recall_subset_at_1" in metrics:
        metrics["mean_recall"] = 0.5 * (
            metrics["recall_at_5"] + metrics["recall_subset_at_1"]
        )
    return metrics


@torch.no_grad()
def evaluate_cirr(
    model,
    val_loader: DataLoader[CIRBatch],
    *,
    gallery_ids: Sequence[str],
    retrieval_features: Tensor,
    native_features: Tensor,
    retrieval_name_to_idx: dict[str, int],
    native_name_to_idx: dict[str, int],
    text_cache: TextFeatureCache,
    device: torch.device,
    global_ks: Sequence[int] = (1, 5, 10, 50),
    subset_ks: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    model.eval()
    gallery_features = get_features_by_ids(
        gallery_ids, retrieval_features, retrieval_name_to_idx
    ).to(device)
    score_batches: list[Tensor] = []
    target_ids: list[str] = []
    reference_ids: list[str] = []
    groups: list[tuple[str, ...]] = []

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
            teacher_text_states=teacher_text_states.to(device=device, dtype=torch.float32),
            text_attention_mask=attention_mask.to(device=device, dtype=torch.bool),
            text_content_mask=content_mask.to(device=device, dtype=torch.bool),
            gallery_features=gallery_features,
        )
        score_batches.append(output["scores"].cpu())

        for target_id in batch.target_ids:
            if target_id is None:
                raise ValueError("CIRR validation requires target_hard")
            target_ids.append(target_id)
        reference_ids.extend(batch.reference_ids)
        groups.extend(batch.group_members)

    if not score_batches:
        raise RuntimeError("CIRR validation loader is empty")

    return evaluate_cirr_metrics(
        torch.cat(score_batches),
        target_ids,
        reference_ids,
        groups,
        gallery_ids,
        global_ks=global_ks,
        subset_ks=subset_ks,
    )
