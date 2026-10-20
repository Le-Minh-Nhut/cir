"""Observation-only per-query exporter for the pinned ENCODER evaluation.

ENCODER's native ``test.py::test`` builds a full ranking -- ``np.argsort(-sims[i, :])[:50]`` after
masking the reference with ``-10e10`` -- and keeps only the Recall at 1/10/50. This module recovers
the per-query rankings the native function discards, so a second model can be compared with LIMN on
stable query identity instead of on aggregate numbers alone.

The two models share a protocol (``fashioniq_val_split``) and the same ranking shape, so the ranking
core, the identity alignment and the native cross-check are reused from ``limn_observer`` verbatim
rather than copied. What differs is only what the models genuinely differ in:

* the dataset exposes ``test_queries_<category>``/``test_targets_<category>`` (ENCODER) instead of
  ``test_queries``/``test_targets`` selected by the constructor (LIMN);
* the query and target forward passes take no ``dataset_type`` argument;
* the native mask is applied by position into ``test_targets``, which is exactly the gallery order
  reused below.

As with LIMN, this is an observation and not a second implementation: the authoritative aggregate
still comes from the untouched native ``test()``, and publication is refused unless the recovered
rankings reproduce those hit counts exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from workbench.replay.limn_observer import (
    DEFAULT_TOP_K,
    ObservedCategory,
    ObservedQuery,
    _image_ids,
    _raw_captions,
    assert_alignment,
    cross_check_with_native,
    rank_queries,
)

#: ENCODER's pinned evaluator names its per-category test lists with this suffix
#: (``test.py::test``: ``testset.test_queries_dress``, `..._shirt`, `..._toptee`).
_CATEGORY_SUFFIX = {"dress": "dress", "shirt": "shirt", "toptee": "toptee"}

def _category_lists(dataset: Any, category: str) -> tuple[list, list]:
    """The native per-category query and target lists, exactly as ``test.py`` selects them."""
    suffix = _CATEGORY_SUFFIX.get(category)
    if suffix is None:
        raise ValueError(f"unsupported ENCODER category: {category}")
    queries = getattr(dataset, f"test_queries_{suffix}")
    targets = getattr(dataset, f"test_targets_{suffix}")
    return queries, targets

def observe_category(category: str, model: Any, dataset: Any, dataset_root: Path,
                     batch_size: int = 64, top_k: int = DEFAULT_TOP_K) -> ObservedCategory:
    """Re-derive one category's per-query rankings from the native model and native dataset.

    ``model`` and ``dataset`` must be the objects ``replay.encoder.evaluate_categories`` loaded, and
    the two forward calls below are the same ones ``test.py`` makes with the same arguments.
    """
    import numpy as np
    import torch

    test_queries, test_targets = _category_lists(dataset, category)
    images = _image_ids(dataset_root, category)
    raw_captions = _raw_captions(dataset_root, category)
    # Identity before ranking: checked against the annotations the export claims, never assumed
    # from list lengths.
    assert_alignment(category, test_queries, images, raw_captions, dataset_root)

    gallery_positions = [entry["target_img_id"] for entry in test_targets]
    gallery_ids = [images[position] for position in gallery_positions]
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError(f"{category}: gallery contains duplicate canonical image ids")
    position_of = {position: index for index, position in enumerate(gallery_positions)}

    with torch.no_grad():
        query_features: list["np.ndarray"] = []
        batch_images: list[Any] = []
        batch_texts: list[str] = []
        for entry in test_queries:
            batch_images.append(entry["source_img_data"])
            batch_texts.append(entry["mod"]["str"])
            if len(batch_images) >= batch_size or entry is test_queries[-1]:
                stacked = torch.stack([
                    item if "torch" in str(type(item)) else torch.from_numpy(item).float()
                    for item in batch_images
                ]).float().cuda()
                features = model.extract_retrieval_compose(stacked, batch_texts)
                query_features.append(features.data.cpu().numpy())
                batch_images, batch_texts = [], []

        target_features: list["np.ndarray"] = []
        batch_images = []
        for entry in test_targets:
            batch_images.append(entry["target_img_data"])
            if len(batch_images) >= batch_size or entry is test_targets[-1]:
                stacked = torch.stack([
                    item if "torch" in str(type(item)) else torch.from_numpy(item).float()
                    for item in batch_images
                ]).float().cuda()
                features = model.extract_retrieval_target(stacked).data.cpu().numpy()
                target_features.append(features)
                batch_images = []

    all_queries = np.concatenate(query_features)
    all_imgs = np.concatenate(target_features)

    # The saved depth is bounded by the ELIGIBLE candidate count: the masked reference still occupies
    # a column in the native matrix, so a depth equal to the gallery size would list it as a
    # "retrieved" image in a protocol that excludes it.
    eligible_depth = min(top_k, len(gallery_ids) - 1)
    ranks, top_rankings = rank_queries(
        all_queries, all_imgs,
        [position_of[entry["source_img_id"]] for entry in test_queries],
        [position_of[entry["target_img_id"]] for entry in test_queries],
        top_k=eligible_depth,
    )

    observed = []
    for index, entry in enumerate(test_queries):
        record = raw_captions[index]
        observed.append(ObservedQuery(
            annotation_index=index,
            reference_id=images[entry["source_img_id"]],
            target_id=images[entry["target_img_id"]],
            raw_captions=(record["captions"][0], record["captions"][1]),
            model_input_text=entry["mod"]["str"],
            target_rank=ranks[index],
            top_results=tuple((rank, gallery_ids[gallery], score)
                              for rank, gallery, score in top_rankings[index]),
            category=category,
        ))

    return ObservedCategory(
        category=category,
        gallery_size=len(gallery_ids),
        eligible_count=len(gallery_ids) - 1,
        queries=tuple(observed),
    )

__all__ = ["observe_category", "cross_check_with_native"]
