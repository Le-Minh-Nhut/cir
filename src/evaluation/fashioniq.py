from collections.abc import Sequence
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from cache.features import get_features_by_ids, TextFeatureCache, get_text_features_by_sample_ids
from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery, load_fashioniq_split_ids


"""
             A     B     C     D     E

query 0     0.1   0.9   0.3   0.2   0.4
query 1     0.8   0.2   0.1   0.7   0.3
query 2     0.1   0.2   0.95  0.3   0.4

annotations của VAL
      ↓
build_fashioniq_gallery(...)
      ↓
gallery_ids
      ↓
load/encode các ảnh gallery
      ↓
gallery_features

VAL queries
      ↓
encode query
      ↓
query_features

query_features × gallery_features
      ↓
scores [Q, G]
      ↓
evaluate_fashioniq_category(
    scores,
    target_ids,
    gallery_ids,
)
      ↓
R@10, R@50
"""

def recall_at_k(scores: torch.Tensor, target_ids: Sequence[str], gallery_ids: Sequence[str], k: int) -> float:
    assert scores.ndim == 2
    query_count, gallery_count = scores.shape

    assert query_count == len(target_ids)
    assert gallery_count == len(gallery_ids)
    assert 0 < k <= gallery_count
    assert len(set(gallery_ids)) == gallery_count

    gallery_index = {}

    for index, image_id in enumerate(gallery_ids):
        gallery_index[image_id] = index

    target_indices = []

    for target_id in target_ids:
        assert target_id in gallery_index
        target_indices.append(gallery_index[target_id])
    # dim=1 -> sort theo chiều cột từng hàng 
    # argsort nó trả ra 1 mảng index được sắp xếp theo score
    rankings = torch.argsort(scores, dim=1, descending=True,)
    top_k = rankings[:, :k] # lấy hết các hàng và chỉ lấy k cột
    # unsqueeze thêm 1 chiều tại vị trí 1 
    target_tensor = torch.tensor(target_indices, device=rankings.device,).unsqueeze(1) 
    hits_per_query = top_k.eq(target_tensor).any(dim=1) # dim=1 thì cho phép nó coi từng hàng để ra True False duyêt theo cột bất kì nào True 

    return hits_per_query.float().mean().item() * 100.0

def mask_fashioniq_reference_scores(
    scores: torch.Tensor,
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    target_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Clone scores and exclude each query's own reference from its row only."""
    if scores.ndim != 2 or scores.shape != (len(reference_ids), len(gallery_ids)):
        raise ValueError("scores must align with reference_ids and gallery_ids")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("FashionIQ gallery IDs must be unique")
    if target_ids is not None:
        if len(target_ids) != len(reference_ids):
            raise ValueError("target_ids must align with reference_ids")
        identical = [
            reference
            for reference, target in zip(reference_ids, target_ids, strict=True)
            if reference == target
        ]
        if identical:
            raise ValueError(
                "FashionIQ modification query has reference_id == target_id; "
                "refusing to mask the target"
            )
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    missing = sorted({image_id for image_id in reference_ids if image_id not in gallery_index})
    if missing:
        raise ValueError(f"FashionIQ references absent from gallery: {missing[:5]}")
    adjusted = scores.clone()
    rows = torch.arange(len(reference_ids), device=scores.device)
    columns = torch.tensor(
        [gallery_index[image_id] for image_id in reference_ids],
        device=scores.device,
    )
    adjusted[rows, columns] = -torch.inf
    return adjusted


def apply_fashioniq_protocol_mask(
    protocol: str,
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Apply only the ranking adjustment explicitly owned by a protocol.

    FashionIQ VAL-split follows the ENCODER/OFFSET/MELT-style convention:
    pair-union validation gallery with per-query reference exclusion.
    """
    if protocol == "fashioniq_original":
        return scores.clone()
    if protocol == "fashioniq_val":
        if reference_ids is None:
            raise ValueError("fashioniq_val evaluation requires reference_ids")
        return mask_fashioniq_reference_scores(
            scores,
            reference_ids,
            gallery_ids,
            target_ids=target_ids,
        )
    raise ValueError(f"Unsupported FashionIQ protocol: {protocol}")


def evaluate_fashioniq_recall(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    protocol: str = "fashioniq_original",
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    adjusted = apply_fashioniq_protocol_mask(
        protocol, scores, target_ids, gallery_ids, reference_ids
    )
    return {
        "recall_at_1": recall_at_k(
            scores=adjusted,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            k=1,
        ),
        "recall_at_10": recall_at_k(
            scores=adjusted,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            k=10,
        ),
        "recall_at_50": recall_at_k(
            scores=adjusted,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            k=50,
        ),
    }


def fashioniq_target_ranks(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    protocol: str = "fashioniq_original",
    reference_ids: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return deterministic one-indexed target ranks for a fixed gallery."""
    scores = apply_fashioniq_protocol_mask(
        protocol, scores, target_ids, gallery_ids, reference_ids
    )
    if scores.ndim != 2 or scores.shape != (len(target_ids), len(gallery_ids)):
        raise ValueError("scores must align with target_ids and gallery_ids")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("FashionIQ gallery IDs must be unique")
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    missing = sorted({target_id for target_id in target_ids if target_id not in gallery_index})
    if missing:
        raise ValueError(f"FashionIQ targets absent from gallery: {missing[:5]}")
    target_index = torch.tensor(
        [gallery_index[target_id] for target_id in target_ids],
        device=scores.device,
    )
    ordering = scores.argsort(dim=-1, descending=True)
    positions = ordering.eq(target_index[:, None]).to(torch.int64).argmax(dim=-1)
    return positions + 1


def evaluate_fashioniq_ranking(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    protocol: str = "fashioniq_original",
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """FashionIQ recalls plus target-rank diagnostics on the same gallery."""
    recalls = evaluate_fashioniq_recall(
        scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    )
    ranks = fashioniq_target_ranks(
        scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    ).float()
    return {
        **recalls,
        "mean_target_rank": float(ranks.mean()),
        "median_target_rank": float(ranks.median()),
        "mrr": float(ranks.reciprocal().mean()),
    }


def compare_fashioniq_rankings(
    dynamic_scores: torch.Tensor,
    frozen_scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    protocol: str = "fashioniq_original",
    reference_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Paired retrieval/rank comparison on one identical FashionIQ gallery."""
    dynamic = evaluate_fashioniq_ranking(
        dynamic_scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    )
    frozen = evaluate_fashioniq_ranking(
        frozen_scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    )
    dynamic_ranks = fashioniq_target_ranks(
        dynamic_scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    ).float()
    frozen_ranks = fashioniq_target_ranks(
        frozen_scores,
        target_ids,
        gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    ).float()
    return {
        "dynamic": dynamic,
        "frozen": frozen,
        "delta": {
            key: dynamic[key] - frozen[key]
            for key in ("recall_at_10", "recall_at_50")
        }
        | {
            "mean_recall": (
                dynamic["recall_at_10"]
                + dynamic["recall_at_50"]
                - frozen["recall_at_10"]
                - frozen["recall_at_50"]
            )
            / 2.0,
            "mean_target_rank": dynamic["mean_target_rank"]
            - frozen["mean_target_rank"],
            "median_target_rank": dynamic["median_target_rank"]
            - frozen["median_target_rank"],
            "mrr": dynamic["mrr"] - frozen["mrr"],
        },
        "target_rank_improved_fraction": float(
            (dynamic_ranks < frozen_ranks).float().mean()
        ),
        "target_rank_worsened_fraction": float(
            (dynamic_ranks > frozen_ranks).float().mean()
        ),
        "same_gallery": True,
        "gallery_size": len(gallery_ids),
    }

def build_original_gallery(split_root: str | Path, category: str, split: str) -> list[str]:
    gallery_ids = load_fashioniq_split_ids(
        split_root=split_root,
        split=split,
        category=category,
    )

    assert len(set(gallery_ids)) == len(gallery_ids)
    return gallery_ids


def build_val_gallery(annotations: Sequence[FashionIQAnnotation]) -> list[str]:
    gallery_ids = build_pair_union_gallery(annotations=annotations)
    assert len(set(gallery_ids)) == len(gallery_ids)
    return gallery_ids

def build_fashioniq_gallery(protocol: str, split_root: str | Path, category: str, annotations: Sequence[FashionIQAnnotation], split: str,) -> list[str]:
    if protocol == "fashioniq_original":
        return build_original_gallery(split_root=split_root, category=category, split=split)

    if protocol == "fashioniq_val":
        return build_val_gallery(annotations)

    raise ValueError(f"Unsupported FashionIQ protocol: {protocol}")

def evaluate_fashioniq_category(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    *,
    protocol: str = "fashioniq_original",
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    return evaluate_fashioniq_recall(
        scores=scores,
        target_ids=target_ids,
        gallery_ids=gallery_ids,
        protocol=protocol,
        reference_ids=reference_ids,
    )

def macro_average_fashioniq(category_results: dict[str, dict[str, float]]) -> dict[str, float]:
    assert len(category_results) > 0

    recall_at_1 = sum(
        result["recall_at_1"]
        for result in category_results.values()
    ) / len(category_results)

    recall_at_10 = sum(
        result["recall_at_10"]
        for result in category_results.values()
    ) / len(category_results)

    recall_at_50 = sum(
        result["recall_at_50"]
        for result in category_results.values()
    ) / len(category_results)

    return {
        "recall_at_1": recall_at_1,
        "recall_at_10": recall_at_10,
        "recall_at_50": recall_at_50,
        "mean_recall": (recall_at_10 + recall_at_50) / 2,
    }

def evaluate_fashioniq(
    model,
    val_loaders,
    val_annotations,
    *,
    protocol,
    split_root,
    split,
    retrieval_features,
    native_features,
    retrieval_name_to_idx,
    native_name_to_idx,
    device,
    text_cache: TextFeatureCache,
):
    category_results = {}

    for category, val_loader in val_loaders.items():
        annotations = val_annotations[category]

        gallery_ids = build_fashioniq_gallery(
            protocol=protocol,
            split_root=split_root,
            split=split,
            category=category,
            annotations=annotations,
        )

        gallery_features = get_features_by_ids(gallery_ids, retrieval_features, retrieval_name_to_idx).to(device)
        score_batches = []
        target_ids = []
        reference_ids = []

        for batch in val_loader:
            reference_native = get_features_by_ids(batch.reference_ids, native_features, native_name_to_idx,).to(device)
            reference_features = reference_native[:, 0, :]

            (text_states, teacher_text_states, attention_mask, content_mask) = get_text_features_by_sample_ids(batch.sample_ids, batch.modification_texts, text_cache)
            text_states = text_states.to(device=device, dtype=torch.float32)
            teacher_text_states = teacher_text_states.to(device=device, dtype=torch.float32)
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            content_mask = content_mask.to(device=device, dtype=torch.bool)

            output = model.retrieve(
                reference_features=reference_features,
                teacher_reference_features=reference_native,
                text_states=text_states,
                teacher_text_states=teacher_text_states,
                text_attention_mask=attention_mask,
                text_content_mask=content_mask,
                gallery_features=gallery_features,
            )

            score_batches.append(output["scores"].cpu())
            reference_ids.extend(str(reference_id) for reference_id in batch.reference_ids)

            for target_id in batch.target_ids:
                if target_id is None:
                    raise ValueError("Validation sample is missing target_id")
                target_ids.append(target_id)

        scores = torch.cat(score_batches)
        category_results[category] = evaluate_fashioniq_category(
            scores=scores,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            protocol=protocol,
            reference_ids=reference_ids,
        )

    average = macro_average_fashioniq(category_results)

    metrics = {
        "recall_at_1": average["recall_at_1"],
        "recall_at_10": average["recall_at_10"],
        "recall_at_50": average["recall_at_50"],
        "mean_recall": average["mean_recall"],
    }

    for category, result in category_results.items():
        metrics[f"{category}_recall_at_1"] = result["recall_at_1"]
        metrics[f"{category}_recall_at_10"] = result["recall_at_10"]
        metrics[f"{category}_recall_at_50"] = result["recall_at_50"]

    return metrics
