"""PyTorch Dataset and DataLoader utilities for EDIR and PinPoint.

Works seamlessly across all ~15 CIR repos (Python 3.8–3.12, any torchvision /
CLIP / BLIP-2 image & text transforms).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from workbench.benchmarks.schema import (
    CIRCorpusRecord,
    CIRQueryRecord,
    DEFAULT_SPLIT_TAG,
    SplitBundle,
    load_split_bundle,
)


def _open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as im:
        return im.convert("RGB")


def _apply_transform(obj: Any, transform: Optional[Callable[[Any], Any]]) -> Any:
    return transform(obj) if transform is not None else obj


def seed_worker(worker_id: int) -> None:
    """Deterministic DataLoader worker seeding."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


class CIRTrainingDataset:
    """Unified PyTorch-compatible training Dataset for EDIR and PinPoint.

    Supports three positive-sampling strategies:
      - ``"random"``: 1 item per query; samples 1 random positive per access (great
        when balancing queries equally across single-positive EDIR and multi-positive
        PinPoint).
      - ``"expand"``: expands every ``(query, positive_id)`` pair (yields ~3.5k pairs
        for EDIR train, ~42k pairs for PinPoint train).
      - ``"first"``: deterministic first positive per query (useful for validation loss).

    Optionally samples ``num_hard_negatives`` explicit hard negatives per item.
    """

    def __init__(
        self,
        bundle: SplitBundle,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        text_transform: Optional[Callable[[str], Any]] = None,
        positive_mode: str = "random",
        num_hard_negatives: int = 0,
        load_images: bool = True,
        seed: int = 42,
    ) -> None:
        if positive_mode not in ("random", "expand", "first"):
            raise ValueError(f"Invalid positive_mode={positive_mode!r}; expected 'random', 'expand', or 'first'.")
        if num_hard_negatives < 0:
            raise ValueError("num_hard_negatives must be >= 0")

        self.bundle = bundle
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.positive_mode = positive_mode
        self.num_hard_negatives = num_hard_negatives
        self.load_images = load_images
        self._rng = random.Random(seed)

        # Validate every query has at least 1 positive target
        self._index: List[Tuple[int, Optional[int]]] = []
        for qi, q in enumerate(bundle.queries):
            if not q.positive_ids:
                raise ValueError(f"Query {q.query_id} in {bundle.dataset}/{bundle.split_name} has 0 positive_ids")
            if positive_mode == "expand":
                for pi in range(len(q.positive_ids)):
                    self._index.append((qi, pi))
            else:
                self._index.append((qi, None))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        qi, pi = self._index[idx]
        q: CIRQueryRecord = self.bundle.queries[qi]

        if self.positive_mode == "expand":
            assert pi is not None
            pos_id = q.positive_ids[pi]
        elif self.positive_mode == "first":
            pos_id = q.positive_ids[0]
        else:
            pos_id = self._rng.choice(q.positive_ids)

        ref_path = q.ref_path(self.bundle.data_dir)
        pos_path = self.bundle.resolve_image_path(pos_id)

        item: Dict[str, Any] = {
            "query_id": q.query_id,
            "normalized_query_id": q.normalized_query_id,
            "ref_id": q.ref_id,
            "target_id": pos_id,
            "ref_path": str(ref_path),
            "target_path": str(pos_path),
            "text": _apply_transform(q.text, self.text_transform),
            "raw_text": q.text,
            "category": q.category,
            "is_single": q.is_single,
        }

        if self.load_images:
            item["ref_image"] = _apply_transform(_open_rgb(ref_path), self.image_transform)
            item["target_image"] = _apply_transform(_open_rgb(pos_path), self.image_transform)
            if not q.is_single and q.ref2_image_rel:
                ref2_path = q.ref2_path(self.bundle.data_dir)
                assert ref2_path is not None
                item["ref2_id"] = q.ref2_id
                item["ref2_path"] = str(ref2_path)
                item["ref2_image"] = _apply_transform(_open_rgb(ref2_path), self.image_transform)

        if self.num_hard_negatives > 0:
            pool = list(q.negative_ids)
            if len(pool) >= self.num_hard_negatives:
                chosen_negs = self._rng.sample(pool, self.num_hard_negatives)
            elif pool:
                chosen_negs = [self._rng.choice(pool) for _ in range(self.num_hard_negatives)]
            else:
                chosen_negs = []
            item["hard_neg_ids"] = chosen_negs
            item["hard_neg_paths"] = [str(self.bundle.resolve_image_path(nid)) for nid in chosen_negs]
            if self.load_images:
                item["hard_neg_images"] = [
                    _apply_transform(_open_rgb(self.bundle.resolve_image_path(nid)), self.image_transform)
                    for nid in chosen_negs
                ]

        return item


class CIRQueryEvalDataset:
    """Deterministic evaluation query Dataset for EDIR and PinPoint."""

    def __init__(
        self,
        bundle: SplitBundle,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        text_transform: Optional[Callable[[str], Any]] = None,
        load_images: bool = True,
    ) -> None:
        self.bundle = bundle
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.bundle.queries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q: CIRQueryRecord = self.bundle.queries[idx]
        ref_path = q.ref_path(self.bundle.data_dir)
        item: Dict[str, Any] = {
            "query_id": q.query_id,
            "normalized_query_id": q.normalized_query_id,
            "ref_id": q.ref_id,
            "ref_path": str(ref_path),
            "ref2_id": q.ref2_id,
            "text": _apply_transform(q.text, self.text_transform),
            "raw_text": q.text,
            "category": q.category,
            "is_single": q.is_single,
            "positive_ids": list(q.positive_ids),
            "negative_ids": list(q.negative_ids),
        }
        if self.load_images:
            item["ref_image"] = _apply_transform(_open_rgb(ref_path), self.image_transform)
            if not q.is_single and q.ref2_image_rel:
                ref2_path = q.ref2_path(self.bundle.data_dir)
                assert ref2_path is not None
                item["ref2_path"] = str(ref2_path)
                item["ref2_image"] = _apply_transform(_open_rgb(ref2_path), self.image_transform)
        return item


class CIRCorpusDataset:
    """Deterministic gallery/corpus Dataset for EDIR and PinPoint."""

    def __init__(
        self,
        bundle: SplitBundle,
        image_transform: Optional[Callable[[Image.Image], Any]] = None,
        load_images: bool = True,
    ) -> None:
        self.bundle = bundle
        self.image_transform = image_transform
        self.load_images = load_images

    def __len__(self) -> int:
        return len(self.bundle.corpus)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        c: CIRCorpusRecord = self.bundle.corpus[idx]
        path = c.image_path(self.bundle.data_dir)
        item: Dict[str, Any] = {
            "image_id": c.image_id,
            "image_path": str(path),
        }
        if self.load_images:
            item["image"] = _apply_transform(_open_rgb(path), self.image_transform)
        return item


def _maybe_stack(values: Sequence[Any]) -> Any:
    import torch

    if values and all(isinstance(v, torch.Tensor) for v in values):
        return torch.stack(list(values), dim=0)
    return list(values)


def collate_train_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function safe for Tensor images, PIL images, variable-length lists, and hard negatives."""
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ("ref_image", "target_image", "ref2_image"):
            out[k] = _maybe_stack(vals)
        elif k == "hard_neg_images":
            # Stack per-item hard negatives -> [B, N_neg, C, H, W] if tensors
            import torch

            if vals and all(v and all(isinstance(x, torch.Tensor) for x in v) for v in vals):
                out[k] = torch.stack([torch.stack(v, dim=0) for v in vals], dim=0)
            else:
                out[k] = vals
        elif k == "text":
            out[k] = _maybe_stack(vals)
        else:
            out[k] = vals
    return out


def collate_eval_queries(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for CIRQueryEvalDataset (preserves variable-length positive_ids/negative_ids)."""
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ("ref_image", "ref2_image", "text"):
            out[k] = _maybe_stack(vals)
        else:
            out[k] = vals
    return out


def collate_corpus(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for CIRCorpusDataset."""
    out: Dict[str, Any] = {
        "image_id": [b["image_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }
    if "image" in batch[0]:
        out["image"] = _maybe_stack([b["image"] for b in batch])
    return out


def build_dataloaders(
    dataset: str,
    train_image_transform: Optional[Callable[[Image.Image], Any]] = None,
    eval_image_transform: Optional[Callable[[Image.Image], Any]] = None,
    text_transform: Optional[Callable[[str], Any]] = None,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 4,
    positive_mode: str = "random",
    num_hard_negatives: int = 0,
    corpus_scope: str = "full",
    split_tag: str = DEFAULT_SPLIT_TAG,
    data_root: Optional[Path] = None,
    seed: int = 42,
    pin_memory: bool = True,
    verify_checksums: bool = True,
) -> Dict[str, Any]:
    """One-call factory returning standardized train/val/test DataLoaders + SplitBundles.

    Caps ``num_workers`` at 10 to respect workspace CPU budget.
    """
    import torch
    from torch.utils.data import DataLoader

    effective_workers = max(0, min(int(num_workers), 10))

    train_bundle = load_split_bundle(
        dataset, "train", corpus_scope=corpus_scope, split_tag=split_tag,
        data_root=data_root, verify_checksums=verify_checksums,
    )
    val_bundle = load_split_bundle(
        dataset, "val", corpus_scope=corpus_scope, split_tag=split_tag,
        data_root=data_root, verify_checksums=False,
    )
    test_bundle = load_split_bundle(
        dataset, "test", corpus_scope=corpus_scope, split_tag=split_tag,
        data_root=data_root, verify_checksums=False,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds = CIRTrainingDataset(
        train_bundle,
        image_transform=train_image_transform,
        text_transform=text_transform,
        positive_mode=positive_mode,
        num_hard_negatives=num_hard_negatives,
        seed=seed,
    )
    val_q_ds = CIRQueryEvalDataset(val_bundle, image_transform=eval_image_transform, text_transform=text_transform)
    val_c_ds = CIRCorpusDataset(val_bundle, image_transform=eval_image_transform)
    test_q_ds = CIRQueryEvalDataset(test_bundle, image_transform=eval_image_transform, text_transform=text_transform)
    test_c_ds = CIRCorpusDataset(test_bundle, image_transform=eval_image_transform)

    common_kw: Dict[str, Any] = {
        "num_workers": effective_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if effective_workers > 0 else None,
    }

    return {
        "bundles": {"train": train_bundle, "val": val_bundle, "test": test_bundle},
        "train_loader": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
            collate_fn=collate_train_batch, generator=gen, **common_kw,
        ),
        "val_query_loader": DataLoader(
            val_q_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            collate_fn=collate_eval_queries, **common_kw,
        ),
        "val_corpus_loader": DataLoader(
            val_c_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            collate_fn=collate_corpus, **common_kw,
        ),
        "test_query_loader": DataLoader(
            test_q_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            collate_fn=collate_eval_queries, **common_kw,
        ),
        "test_corpus_loader": DataLoader(
            test_c_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            collate_fn=collate_corpus, **common_kw,
        ),
    }
