from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from cache.taper_mag import FrozenVisionCache, ImageCacheManifest, stable_json_hash
from models.taper_mag.contracts import SupervisionBatch
from precompute_taper_mag_vision import fashioniq_cache_scopes
from training.negative_bank import NegativeBank


def _manifest(kind: str, mapping: dict[str, int], feature_dim: int, *, complete: bool = True) -> ImageCacheManifest:
    return ImageCacheManifest(
        schema_version=2,
        cache_kind=kind,
        image_scope="complete_split" if kind == "global" else "reference_only",
        model_id="qihoo360/fg-clip2-base",
        revision="a" * 40,
        processor_config_hash="processor",
        extraction_method="official",
        normalization="L2" if kind == "global" else "none",
        dtype="float32",
        image_id_mapping_hash=stable_json_hash(mapping),
        feature_dim=feature_dim,
        patch_policy="official_dynamic_v1",
        split="train",
        spatial_shapes_present=kind == "dense_reference",
        image_count=len(mapping),
        complete_split=complete,
    )


def _cache(tmp_path) -> FrozenVisionCache:
    tmp_path.mkdir(parents=True, exist_ok=True)
    global_dir = tmp_path / "global"
    dense_dir = tmp_path / "dense_reference"
    global_dir.mkdir()
    dense_dir.mkdir()
    global_mapping = {"ref-a": 0, "target-only": 1, "gallery-only": 2, "ref-b": 3}
    dense_mapping = {"ref-a": 0, "ref-b": 1}
    globals_ = torch.nn.functional.normalize(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4), dim=-1).numpy()
    np.save(global_dir / "global.npy", globals_, allow_pickle=False)
    (global_dir / "name_to_idx.json").write_text(json.dumps(global_mapping), encoding="utf-8")
    _manifest("global", global_mapping, 4).write(global_dir / "manifest.json")
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    np.save(dense_dir / "dense_values.npy", values, allow_pickle=False)
    np.save(dense_dir / "dense_offsets.npy", np.array([0, 4, 5]), allow_pickle=False)
    np.save(dense_dir / "spatial_shapes.npy", np.array([[2, 2], [1, 1]]), allow_pickle=False)
    (dense_dir / "reference_name_to_idx.json").write_text(json.dumps(dense_mapping), encoding="utf-8")
    _manifest("dense_reference", dense_mapping, 4).write(dense_dir / "manifest.json")
    return FrozenVisionCache.load(tmp_path)


def test_split_mmap_cache_scopes_and_batch_retrieval(tmp_path) -> None:
    cache = _cache(tmp_path)
    assert isinstance(cache.global_embeddings, np.memmap)
    assert isinstance(cache.dense_store.values, np.memmap)
    assert "target-only" in cache.name_to_idx
    assert "gallery-only" in cache.name_to_idx
    assert "target-only" not in cache.reference_name_to_idx
    globals_ = cache.global_by_ids(["target-only", "ref-a"])
    assert globals_.shape == (2, 4)
    dense, mask, shapes = cache.dense_by_ids(["ref-b", "ref-a"])
    assert dense.shape == (2, 4, 4)
    assert mask.tolist() == [[True, False, False, False], [True, True, True, True]]
    assert shapes.tolist() == [[1, 1], [2, 2]]


def test_cache_manifest_mismatch_and_partial_debug_rejected(tmp_path) -> None:
    cache = _cache(tmp_path)
    with pytest.raises(RuntimeError, match="feature_dim"):
        FrozenVisionCache.load(
            tmp_path,
            expected_global=replace(cache.global_manifest, feature_dim=8),
        )
    partial = replace(cache.dense_manifest, complete_split=False)
    partial.write(tmp_path / "dense_reference" / "manifest.json")
    with pytest.raises(RuntimeError, match="Partial/debug dense"):
        FrozenVisionCache.load(tmp_path)


def test_cache_validation_rejects_bad_offsets_and_manifest_hash(tmp_path) -> None:
    cache = _cache(tmp_path)
    np.save(tmp_path / "dense_reference" / "dense_offsets.npy", np.array([0, 5, 4]), allow_pickle=False)
    with pytest.raises(RuntimeError, match="span|monotonic"):
        FrozenVisionCache.load(tmp_path)
    _cache(tmp_path / "fresh")
    manifest = cache.global_manifest
    replace(manifest, image_id_mapping_hash="wrong").write(tmp_path / "fresh" / "global" / "manifest.json")
    with pytest.raises(RuntimeError, match="mapping hash"):
        FrozenVisionCache.load(tmp_path / "fresh")


def test_annotation_reference_scope_excludes_target_and_gallery_only(tmp_path) -> None:
    (tmp_path / "captions").mkdir()
    (tmp_path / "image_splits").mkdir()
    categories = ("dress",)
    annotations = [
        {"candidate": "ref-a", "target": "target-only", "captions": ["red", "long"]},
        {"candidate": "ref-a", "target": "target-only", "captions": ["blue", "short"]},
    ]
    (tmp_path / "captions" / "cap.dress.train.json").write_text(json.dumps(annotations), encoding="utf-8")
    (tmp_path / "image_splits" / "split.dress.train.json").write_text(
        json.dumps(["ref-a", "target-only", "gallery-only"]), encoding="utf-8"
    )
    global_ids, reference_ids = fashioniq_cache_scopes(tmp_path, "train", categories)
    assert global_ids == ["ref-a", "target-only", "gallery-only"]
    assert reference_ids == ["ref-a"]


def test_negative_bank_mines_mmap_in_chunks_without_positive_leakage(tmp_path) -> None:
    cache = _cache(tmp_path)
    ids = tuple(
        image_id
        for image_id, _ in sorted(cache.name_to_idx.items(), key=lambda item: item[1])
    )
    bank = NegativeBank(cache.global_embeddings, ids, hard_negatives=2)
    supervision = SupervisionBatch(
        target_embedding=cache.global_by_ids(["target-only", "gallery-only"]),
        target_ids=("target-only", "gallery-only"),
        positive_ids=(("target-only",), ("gallery-only",)),
    )
    negatives = bank.mine_once(cache.global_by_ids(["ref-a", "ref-b"]), supervision)
    assert negatives.embeddings.shape == (2, 2, 4)
    assert torch.isfinite(negatives.embeddings).all()
    assert "target-only" not in negatives.ids[0]
    assert "gallery-only" not in negatives.ids[1]
