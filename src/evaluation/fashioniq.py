from collections.abc import Sequence
from pathlib import Path
import torch

from cache.features import get_features_by_ids, TextFeatureCache, get_text_features_by_sample_ids
from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery, load_fashioniq_split_ids

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
    rankings = torch.argsort(scores, dim=1, descending=True,)
    top_k = rankings[:, :k]
    target_tensor = torch.tensor(target_indices, device=rankings.device,).unsqueeze(1) 
    hits_per_query = top_k.eq(target_tensor).any(dim=1)

    return hits_per_query.float().mean().item() * 100.0

def recall_at_k_reference_excluded(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    reference_ids: Sequence[str],
    gallery_ids: Sequence[str],
    k: int,
) -> float:
    """Recall@k with each query's own reference removed from the ranked gallery.

    This is the iLearn full-gallery protocol. The candidate universe is the whole validation
    split in its declared order, and a query cannot retrieve its own reference.

    The pinned evaluators *mask* the reference score to a very negative value and then sort
    (`sims[i, targets.index(source)] = -10e10` before `argsort`). Removing the column and masking
    it are equivalent for the top-k hit test whenever all other scores exceed the mask, but the
    two are not the same operation, and the mask depends on that bound holding. This helper
    removes the column explicitly, which is what the *cohort* means, and the tests pin the
    behaviour rather than relying on the equivalence.
    """
    assert scores.ndim == 2
    query_count, gallery_count = scores.shape

    assert query_count == len(target_ids) == len(reference_ids)
    assert gallery_count == len(gallery_ids)
    assert 0 < k <= gallery_count
    assert len(set(gallery_ids)) == gallery_count

    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}

    hits: list[bool] = []

    for row, (target_id, reference_id) in enumerate(zip(target_ids, reference_ids)):
        assert target_id in gallery_index, f"target {target_id} missing from gallery"
        assert reference_id in gallery_index, f"reference {reference_id} missing from gallery"

        reference_column = gallery_index[reference_id]
        ranking = torch.argsort(scores[row], descending=True)
        ranking = ranking[ranking != reference_column]
        target_column = gallery_index[target_id]

        hits.append(bool(ranking[: min(k, len(ranking))].eq(target_column).any()))

    return float(torch.tensor(hits, dtype=torch.float32).mean().item() * 100.0)


def evaluate_fashioniq_recall(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Recall@10/@50 for a FashionIQ cohort.

    ``reference_ids`` is required by protocols that exclude the reference from the gallery and
    must be None for protocols that leave it eligible, so a caller cannot silently evaluate an
    exclusion protocol as if it were a plain one (or the reverse).
    """
    if reference_ids is None:
        return {
            "recall_at_10": recall_at_k(
                scores=scores, target_ids=target_ids, gallery_ids=gallery_ids, k=10),
            "recall_at_50": recall_at_k(
                scores=scores, target_ids=target_ids, gallery_ids=gallery_ids, k=50),
        }

    return {
        "recall_at_10": recall_at_k_reference_excluded(
            scores=scores, target_ids=target_ids, reference_ids=reference_ids,
            gallery_ids=gallery_ids, k=10),
        "recall_at_50": recall_at_k_reference_excluded(
            scores=scores, target_ids=target_ids, reference_ids=reference_ids,
            gallery_ids=gallery_ids, k=50),
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
    """Build the candidate universe for one protocol.

    Both spellings are accepted: the registry's protocol ids (`fashioniq_original_split`,
    `fashioniq_full_gallery_ref_excluded`, `fashioniq_val_split`) and the shorter internal names
    earlier callers used (`fashioniq_original`, `fashioniq_val`). A caller holding a registry id
    could not evaluate the two older families at all, which is a gap rather than a naming
    preference; the two spellings of each family are the same cohort, so they share a branch.
    """
    if protocol in ("fashioniq_original", "fashioniq_original_split"):
        return build_original_gallery(split_root=split_root, category=category, split=split)

    if protocol == "fashioniq_full_gallery_ref_excluded":
        # Same candidate universe as the original split — the whole validation split in its
        # declared order — with the reference excluded from ranking rather than included. The
        # exception is applied in `evaluate_fashioniq_recall`, not here, because exclusion is a
        # per-query property: the gallery itself is what the upstream evaluator scores against.
        return build_original_gallery(split_root=split_root, category=category, split=split)

    if protocol in ("fashioniq_val", "fashioniq_val_split"):
        return build_val_gallery(annotations)

    raise ValueError(f"Unsupported FashionIQ protocol: {protocol}")

def evaluate_fashioniq_category(
    scores: torch.Tensor,
    target_ids: Sequence[str],
    gallery_ids: Sequence[str],
    reference_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    return evaluate_fashioniq_recall(
        scores=scores,
        target_ids=target_ids,
        gallery_ids=gallery_ids,
        reference_ids=reference_ids,
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

            for target_id, reference_id in zip(batch.target_ids, batch.reference_ids):
                if target_id is None:
                    raise ValueError("Validation sample is missing target_id")
                target_ids.append(target_id)
                reference_ids.append(reference_id)

        scores = torch.cat(score_batches)
        # The exclusion protocol needs per-query references; the reference-eligible protocol
        # must not receive them, so a protocol mix-up cannot pass unnoticed.
        excluded = protocol == "fashioniq_full_gallery_ref_excluded"
        category_results[category] = evaluate_fashioniq_category(
            scores=scores,
            target_ids=target_ids,
            gallery_ids=gallery_ids,
            reference_ids=reference_ids if excluded else None,
        )

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
