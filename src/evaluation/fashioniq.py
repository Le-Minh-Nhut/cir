from collections.abc import Sequence
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import math
from collections import defaultdict
import torch.nn.functional as F

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

class ScalarAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name: str, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)

        finite = torch.isfinite(values)
        values = values[finite]

        if values.numel() == 0:
            return

        self.sums[name] += values.sum().item()
        self.counts[name] += values.numel()

    def mean(self, name: str) -> float:
        if self.counts[name] == 0:
            return float("nan")

        return self.sums[name] / self.counts[name]

def active_pair_cosine_per_sample(x: torch.Tensor, active_slots: torch.Tensor) -> torch.Tensor:
    """
    x:            [B, L, D]
    active_slots: [B, L] bool

    Return:
        cosine trung bình giữa các cặp active slot
        cho từng sample có >= 2 active slots.

        Shape [M], M <= B.
    """

    if x.ndim != 3:
        raise ValueError("x must be [B,L,D]")

    if active_slots.shape != x.shape[:2]:
        raise ValueError("active_slots must match x[:2]")

    _, num_slots, _ = x.shape
    normalized = F.normalize(x.float(), dim=-1, eps=1e-6)
    similarity = normalized @ normalized.transpose(1, 2)
    upper = torch.triu(torch.ones(num_slots, num_slots, dtype=torch.bool, device=x.device,), diagonal=1)
    pair_valid = (active_slots[:, :, None] & active_slots[:, None, :] & upper[None, :, :])
    pair_count = pair_valid.sum(dim=(1, 2))
    pair_sum = (similarity * pair_valid.to(similarity.dtype)).sum(dim=(1, 2))
    valid_samples = pair_count > 0

    return pair_sum[valid_samples] / pair_count[valid_samples]

def update_a4_slot_diagnostics(
    accumulator: ScalarAccumulator,
    output: dict[str, torch.Tensor],
    content_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    valid = attention_mask.to(torch.bool) & content_mask.to(torch.bool)

    ownership_logits = output["ownership_logits"].float()
    soft = F.softmax(ownership_logits, dim=1)
    num_destinations = soft.shape[1]
    soft_entropy = -(soft.clamp_min(1e-12)* soft.clamp_min(1e-12).log()).sum(dim=1)
    soft_entropy = soft_entropy / math.log(num_destinations)
    top2 = soft.topk(k=2, dim=1).values
    soft_winner_confidence = top2[:, 0, :]
    soft_top1_top2_margin = top2[:, 0, :] - top2[:, 1, :]
    accumulator.add("soft_entropy", soft_entropy[valid])
    accumulator.add("soft_top1_top2_margin", soft_top1_top2_margin[valid],)
    accumulator.add("soft_winner_confidence", soft_winner_confidence[valid])
    slot_mass = output["slot_mass"]
    active_slots = slot_mass > 0.0
    num_slots = slot_mass.shape[1]
    active_count = active_slots.sum(dim=1)
    accumulator.add("multi_active_sample_rate", (active_count >= 2).float())

    for slot_id in range(num_slots):
        accumulator.add(f"slot_{slot_id}_nonempty_rate", active_slots[:, slot_id].float())

    semantic_cos = active_pair_cosine_per_sample(output["slot_semantics"], active_slots)
    effect_cos = active_pair_cosine_per_sample(output["slot_effects"], active_slots)
    edit_slot_cos = active_pair_cosine_per_sample(output["edit_slots"], active_slots)
    accumulator.add("active_pair_semantic_cosine", semantic_cos)
    accumulator.add("active_pair_teacher_effect_cosine", effect_cos)
    accumulator.add("active_pair_edit_slot_cosine", edit_slot_cos)
    slot_gates = output["slot_gates"]
    hard_active = output["hard_active_slot_mask"]
    for slot_id in range(num_slots):
        accumulator.add(f"slot_{slot_id}_gate_mean", slot_gates[:, slot_id])
        accumulator.add(f"slot_{slot_id}_hard_active_rate", hard_active[:, slot_id].float())

    trace_slot_ids = output["trace_slot_ids"]
    trace_valid = output["trace_valid_mask"].to(torch.bool)
    valid_steps_per_sample = trace_valid.sum(dim=1).float()
    accumulator.add("valid_execution_steps_mean", valid_steps_per_sample)
    for slot_id in range(num_slots):
        executed = (trace_slot_ids.eq(slot_id) & trace_valid).any(dim=1).float()
        accumulator.add(f"slot_{slot_id}_execution_rate", executed)

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
    slot_diagnostics = ScalarAccumulator()

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
            update_a4_slot_diagnostics(accumulator=slot_diagnostics, output=output, content_mask=content_mask)

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

    diagnostic_names = [
        "soft_entropy",
        "soft_top1_top2_margin",
        "soft_winner_confidence",
        "multi_active_sample_rate",


        "active_pair_semantic_cosine",
        "active_pair_teacher_effect_cosine",
        "active_pair_edit_slot_cosine",

        "valid_execution_steps_mean",
    ]

    for category, result in category_results.items():
        metrics[f"{category}_recall_at_10"] = result["recall_at_10"]
        metrics[f"{category}_recall_at_50"] = result["recall_at_50"]

    for name in diagnostic_names:
        metrics[f"slot/{name}"] = slot_diagnostics.mean(name)

    for slot_id in range(model.num_slots):
        for metric_name in (
            "nonempty_rate",
            "gate_mean",
            "hard_active_rate",
            "execution_rate",
        ):
            key = f"slot_{slot_id}_{metric_name}"
            metrics[f"slot/{key}"] = slot_diagnostics.mean(key)

    return metrics