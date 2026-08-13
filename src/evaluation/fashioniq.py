from collections.abc import Sequence
from pathlib import Path
import torch

from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery, load_fashioniq_split_ids


"""
             A     B     C     D     E

query 0     0.1   0.9   0.3   0.2   0.4
query 1     0.8   0.2   0.1   0.7   0.3
query 2     0.1   0.2   0.95  0.3   0.4
"""

""""
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

    return hits_per_query.float().mean().item()

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

def build_original_gallery(split_root: str | Path, category: str) -> list[str]:
    gallery_ids = load_fashioniq_split_ids(
        split_root=split_root,
        split="val",
        category=category,
    )

    assert len(set(gallery_ids)) == len(gallery_ids)
    return gallery_ids


def build_val_gallery(annotations: Sequence[FashionIQAnnotation]) -> list[str]:
    gallery_ids = build_pair_union_gallery(annotations=annotations)
    assert len(set(gallery_ids)) == len(gallery_ids)
    return gallery_ids

def build_fashioniq_gallery(protocol: str, split_root: str | Path, category: str, annotations: Sequence[FashionIQAnnotation]) -> list[str]:
    if protocol == "fashioniq_original":
        return build_original_gallery(split_root=split_root, category=category,)

    if protocol == "fashioniq_val":
        return build_val_gallery(annotations=annotations,)

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
    }