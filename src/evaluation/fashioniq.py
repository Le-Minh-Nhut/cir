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

def evaluate_fashioniq_recall(scores: torch.Tensor, target_ids: Sequence[str], gallery_ids: Sequence[str]) -> dict[str, float]:
    return {
        "recall_at_10": recall_at_k(
            scores=scores,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            k=10,
        ),
        "recall_at_50": recall_at_k(
            scores=scores,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            k=50,
        ),
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

def evaluate_fashioniq_category(scores: torch.Tensor, target_ids: Sequence[str], gallery_ids: Sequence[str]) -> dict[str, float]:
    return evaluate_fashioniq_recall(
        scores=scores,
        target_ids=target_ids,
        gallery_ids=gallery_ids,
    )

def macro_average_fashioniq(category_results: dict[str, dict[str, float]]) -> dict[str, float]:
    assert len(category_results) > 0

    recall_at_10 = sum(
        result["recall_at_10"]
        for result in category_results.values()
    ) / len(category_results)

    recall_at_50 = sum(
        result["recall_at_50"]
        for result in category_results.values()
    ) / len(category_results)

    return {
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

            for target_id in batch.target_ids:
                if target_id is None:
                    raise ValueError("Validation sample is missing target_id")
                target_ids.append(target_id)

        scores = torch.cat(score_batches)
        category_results[category] = evaluate_fashioniq_category(scores=scores, target_ids=target_ids, gallery_ids=gallery_ids)

    average = macro_average_fashioniq(category_results)

    metrics = {
        "recall_at_10": average["recall_at_10"],
        "recall_at_50": average["recall_at_50"],
        "mean_recall": average["mean_recall"],
    }

    for category, result in category_results.items():
        metrics[f"{category}_recall_at_10"] = result["recall_at_10"]
        metrics[f"{category}_recall_at_50"] = result["recall_at_50"]

    return metrics