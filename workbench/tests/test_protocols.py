from __future__ import annotations

import torch

from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery
from evaluation.fashioniq import build_original_gallery, recall_at_k
from evaluation.fashioniq_encoder import fashioniq_encoder_recall_at_k


def test_original_split_uses_file_order_and_keeps_reference(tmp_path) -> None:
    (tmp_path / "split.dress.val.json").write_text('["reference", "target", "other"]')
    gallery = build_original_gallery(tmp_path, "dress", "val")
    assert gallery == ["reference", "target", "other"]
    assert recall_at_k(torch.tensor([[0.9, 0.8, 0.7]]), ["target"], gallery, 1) == 0.0


def test_val_split_union_order_and_reference_exclusion() -> None:
    annotations = [FashionIQAnnotation("ref-a", "target-a", ("one", "two"), "dress", 0), FashionIQAnnotation("ref-b", "target-a", ("three", "four"), "dress", 1)]
    gallery = build_pair_union_gallery(annotations)
    assert gallery == ["ref-a", "target-a", "ref-b"]
    assert fashioniq_encoder_recall_at_k(torch.tensor([[0.99, 0.90, 0.80]]), ["target-a"], ["ref-a"], gallery, 1) == 100.0
