from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from cache.features import (
    TextFeatureCache,
    get_features_by_ids,
    get_text_features_by_sample_ids,
)
from datasets.common import CIRBatch


CIRR_ENCODER_GLOBAL_KS = (1, 5, 10, 50)
CIRR_ENCODER_SUBSET_KS = (1, 2, 3)
CIRR_ENCODER_SELECTION_METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "recall_subset_at_1",
    "recall_subset_at_2",
    "recall_subset_at_3",
)


def cosine_similarity_scores(query_features: Tensor, gallery_features: Tensor) -> Tensor:
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("query and gallery features must both be two-dimensional")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("query and gallery feature dimensions must match")
    return F.normalize(query_features, dim=-1) @ F.normalize(
        gallery_features, dim=-1
    ).T


def _validate_inputs(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
) -> dict[str, int]:
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(gallery_ids)):
        raise ValueError("scores must have shape [queries, gallery]")
    if not target_ids:
        raise ValueError("CIRR ENCODER metrics require at least one query")
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


def rank_reference_excluded_gallery(
    scores: Tensor,
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
) -> Tensor:
    if scores.ndim != 2 or scores.shape != (len(reference_ids), len(gallery_ids)):
        raise ValueError("scores must have shape [queries, gallery]")
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    missing = [reference_id for reference_id in reference_ids if reference_id not in gallery_index]
    if missing:
        raise ValueError(f"CIRR references missing from gallery: {missing[:5]}")

    rankings = torch.argsort(scores, dim=1, descending=True, stable=True)
    filtered_rows = []
    for row, reference_id in enumerate(reference_ids):
        reference_column = gallery_index[reference_id]
        filtered_rows.append(rankings[row][rankings[row] != reference_column])
    return torch.stack(filtered_rows)


def _recall_at_k(
    rankings: Sequence[Tensor],
    target_ids: Sequence[str],
    gallery_index: dict[str, int],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    hits = []
    for ranking, target_id in zip(rankings, target_ids, strict=True):
        target_column = gallery_index[target_id]
        hits.append(bool(ranking[: min(k, len(ranking))].eq(target_column).any()))
    return 100.0 * sum(hits) / len(hits)


def _filter_global_rankings_to_subsets(
    rankings: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    group_members: Sequence[Sequence[str]],
    gallery_index: dict[str, int],
) -> list[Tensor]:
    subset_rankings = []
    for row, (target_id, reference_id, members) in enumerate(
        zip(target_ids, reference_ids, group_members, strict=True)
    ):
        if len(set(members)) != len(members):
            raise ValueError(f"CIRR group for row {row} contains duplicate IDs")
        if reference_id not in members:
            raise ValueError(f"CIRR group for row {row} does not contain its reference")
        candidates = [member for member in members if member != reference_id]
        if target_id not in candidates:
            raise ValueError(f"CIRR group for row {row} does not contain target_hard")
        missing = [member for member in candidates if member not in gallery_index]
        if missing:
            raise ValueError(f"CIRR group images missing from gallery: {missing[:5]}")

        subset_columns = torch.tensor(
            [gallery_index[member] for member in candidates],
            device=rankings.device,
        )
        membership = rankings[row].unsqueeze(1).eq(subset_columns).any(dim=1)
        subset_rankings.append(rankings[row][membership])
    return subset_rankings


def evaluate_cirr_encoder_metrics(
    scores: Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    group_members: Sequence[Sequence[str]],
    gallery_ids: Sequence[str],
) -> dict[str, float]:
    gallery_index = _validate_inputs(scores, target_ids, reference_ids, gallery_ids)
    if len(group_members) != len(target_ids):
        raise ValueError("CIRR groups must align with score rows")

    rankings = rank_reference_excluded_gallery(scores, reference_ids, gallery_ids)
    subset_rankings = _filter_global_rankings_to_subsets(
        rankings,
        target_ids,
        reference_ids,
        group_members,
        gallery_index,
    )
    metrics = {
        f"recall_at_{k}": _recall_at_k(rankings, target_ids, gallery_index, k)
        for k in CIRR_ENCODER_GLOBAL_KS
    }
    metrics.update(
        {
            f"recall_subset_at_{k}": _recall_at_k(
                subset_rankings, target_ids, gallery_index, k
            )
            for k in CIRR_ENCODER_SUBSET_KS
        }
    )
    metrics["selection_score"] = sum(
        metrics[name] for name in CIRR_ENCODER_SELECTION_METRICS
    )
    return metrics


@torch.no_grad()
def evaluate_cirr_encoder(
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
        # TAPER q0 and cached gallery tokens already use L2-normalized similarity.
        # The max over CSMCIR gallery tokens is part of the existing TAPER model.
        score_batches.append(output["scores"].cpu())

        for target_id in batch.target_ids:
            if target_id is None:
                raise ValueError("CIRR validation requires target_hard")
            target_ids.append(target_id)
        reference_ids.extend(batch.reference_ids)
        groups.extend(batch.group_members)

    if not score_batches:
        raise RuntimeError("CIRR ENCODER validation loader is empty")
    return evaluate_cirr_encoder_metrics(
        torch.cat(score_batches),
        target_ids,
        reference_ids,
        groups,
        gallery_ids,
    )
