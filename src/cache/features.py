import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def load_feature_manifest(feature_dir) -> dict:
    path = Path(feature_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature manifest: {path}")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise TypeError(f"Feature manifest must be a JSON object: {path}")
    return manifest


def validate_feature_manifest(
    manifest: dict,
    *,
    model_id: str,
    revision: str,
    cache_name: str,
    correction_policy: str | None = None,
) -> None:
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"Wrong model/revision in {cache_name} cache manifest: "
            f"model_id={manifest.get('model_id')!r}, "
            f"revision={manifest.get('revision')!r}"
        )
    if correction_policy is not None:
        cached_policy = manifest.get("correction_policy")
        legacy_corrected_cache = cache_name in {"train/text", "val/text"}
        if (
            cached_policy is None
            and correction_policy == "fashioniq"
            and legacy_corrected_cache
        ):
            warnings.warn(
                f"Legacy corrected {cache_name} cache manifest has no "
                "correction_policy; treating it as 'fashioniq'",
                stacklevel=2,
            )
        elif cached_policy != correction_policy:
            raise ValueError(
                f"Wrong correction policy in {cache_name} cache manifest: "
                f"expected={correction_policy!r}, cached={cached_policy!r}"
            )


def validate_text_cache_subdir(text_cache_subdir: str, correction_policy: str) -> str:
    if correction_policy not in {"fashioniq", "none"}:
        raise ValueError(
            f"Unsupported FashionIQ correction policy {correction_policy!r}"
        )
    path = Path(text_cache_subdir)
    if (
        not text_cache_subdir
        or path.name != text_cache_subdir
        or text_cache_subdir in {".", "..", "images"}
    ):
        raise ValueError("text_cache_subdir must be one safe directory name")
    if correction_policy == "none" and text_cache_subdir == "text":
        raise ValueError(
            "correction_policy='none' cannot write to the corrected baseline text/ cache"
        )
    return text_cache_subdir


def load_features(feature_dir):
    feature_dir = Path(feature_dir)

    pt_path = feature_dir / "images.pt"
    npy_path = feature_dir / "images.npy"

    if npy_path.is_file():
        array = np.load(npy_path, mmap_mode="c")
        features = torch.from_numpy(array)
    elif pt_path.is_file():
        features = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    else:
        raise FileNotFoundError(f"No feature cache found in {feature_dir}")

    with (feature_dir / "name_to_idx.json").open("r", encoding="utf-8") as file:
        name_to_idx = json.load(file)

    return features, name_to_idx


def get_features_by_ids(image_ids: Sequence[str], features: torch.Tensor, name_to_idx: dict[str, int]) -> torch.Tensor:
    indices = [name_to_idx[image_id] for image_id in image_ids]
    return features[indices]

@dataclass(frozen=True)
class TextFeatureCache:
    states: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor

    sample_to_idx: dict[str, int]
    captions: dict[str, str]
    manifest: dict
    global_features: torch.Tensor | None = None


def load_text_features(feature_dir) -> TextFeatureCache:
    feature_dir = Path(feature_dir)

    required = (
        "states.npy",
        "attention_mask.npy",
        "content_mask.npy",
        "sample_to_idx.json",
        "captions.json",
        "manifest.json",
    )

    for name in required:
        path = feature_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing text cache file: {path}"
            )

    states = torch.from_numpy(
        np.load(
            feature_dir / "states.npy",
            mmap_mode="c",
        )
    )

    attention_mask = torch.from_numpy(
        np.load(
            feature_dir / "attention_mask.npy",
            mmap_mode="c",
        )
    )

    content_mask = torch.from_numpy(
        np.load(
            feature_dir / "content_mask.npy",
            mmap_mode="c",
        )
    )

    with (
        feature_dir / "sample_to_idx.json"
    ).open("r", encoding="utf-8") as file:
        sample_to_idx = json.load(file)

    with (
        feature_dir / "captions.json"
    ).open("r", encoding="utf-8") as file:
        captions = json.load(file)

    manifest = load_feature_manifest(feature_dir)
    global_path = feature_dir / "global.npy"
    manifest_has_global = bool(manifest.get("has_global_features", global_path.is_file()))
    global_features = (
        torch.from_numpy(np.load(global_path, mmap_mode="c"))
        if manifest_has_global and global_path.is_file()
        else None
    )
    if manifest_has_global and not global_path.is_file():
        raise FileNotFoundError(f"Text manifest requires missing cache file: {global_path}")

    num_samples = len(sample_to_idx)

    if states.ndim != 3:
        raise ValueError(f"states must be [Q,N,D], got {tuple(states.shape)}")
    if attention_mask.ndim != 2 or content_mask.ndim != 2:
        raise ValueError("attention_mask and content_mask must be [Q,N]")

    arrays = {
        "states": states,
        "attention_mask": attention_mask,
        "content_mask": content_mask,
    }

    for name, tensor in arrays.items():
        if tensor.shape[0] != num_samples:
            raise ValueError(
                f"{name} rows ({tensor.shape[0]}) "
                f"!= sample index size ({num_samples})"
            )

    if global_features is not None:
        if global_features.ndim != 2 or global_features.shape[0] != num_samples:
            raise ValueError(
                "global.npy must be [Q,D] and align with the text sample index; "
                f"got {tuple(global_features.shape)}"
            )
        if not torch.isfinite(global_features).all():
            raise FloatingPointError("Text global cache contains NaN or Inf")
        if manifest.get("global_shape") not in (None, list(global_features.shape)):
            raise ValueError("Text manifest global_shape does not match global.npy")

    if set(sample_to_idx) != set(captions):
        raise ValueError(
            "Text-cache sample IDs and caption IDs differ"
        )

    if attention_mask.shape != states.shape[:2]:
        raise ValueError(
            "attention_mask shape mismatch"
        )

    if content_mask.shape != states.shape[:2]:
        raise ValueError(
            "content_mask shape mismatch"
        )

    if (content_mask.to(torch.bool) & ~attention_mask.to(torch.bool)).any():
        raise ValueError("content_mask contains positions outside attention_mask")

    return TextFeatureCache(
        states=states,
        attention_mask=attention_mask,
        content_mask=content_mask,
        sample_to_idx=sample_to_idx,
        captions=captions,
        manifest=manifest,
        global_features=global_features,
    )


def get_text_features_by_sample_ids(sample_ids: Sequence[str], modification_texts: Sequence[str], cache: TextFeatureCache):
    if len(sample_ids) != len(modification_texts):
        raise ValueError("sample_ids and modification_texts length mismatch")

    indices = []

    for sample_id, runtime_caption in zip(sample_ids, modification_texts, strict=True):
        if sample_id not in cache.sample_to_idx:
            raise KeyError(f"Missing sample_id from text cache: {sample_id}")

        cached_caption = cache.captions.get(sample_id)
        if cached_caption is None:
            raise KeyError(f"Missing cached caption for: {sample_id}")

        if runtime_caption != cached_caption:
            raise RuntimeError(
                "Runtime caption does not match cached caption "
                f"for {sample_id}\n"
                f"runtime={runtime_caption!r}\n"
                f"cached ={cached_caption!r}"
            )

        indices.append(cache.sample_to_idx[sample_id])

    return (
        cache.states[indices],
        cache.attention_mask[indices],
        cache.content_mask[indices],
    )


def get_text_features_with_global_by_sample_ids(
    sample_ids: Sequence[str],
    modification_texts: Sequence[str],
    cache: TextFeatureCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A8.0 text lookup that fails rather than falling back without globals."""

    states, attention_mask, content_mask = get_text_features_by_sample_ids(
        sample_ids, modification_texts, cache
    )
    if cache.global_features is None:
        raise FileNotFoundError(
            "A8.0 requires global.npy in the FG-CLIP2 text cache; rerun "
            "src/precompute_fgclip2_text.py with --save-global"
        )
    indices = [cache.sample_to_idx[sample_id] for sample_id in sample_ids]
    return states, attention_mask, content_mask, cache.global_features[indices]


@dataclass(frozen=True)
class DenseImageFeatureCache:
    """Memory-mapped ragged FG-CLIP2 dense image features."""

    values: np.ndarray
    offsets: np.ndarray
    spatial_shapes: np.ndarray
    name_to_idx: dict[str, int]
    manifest: dict


def load_dense_image_features(feature_dir: str | Path) -> DenseImageFeatureCache:
    feature_dir = Path(feature_dir)
    for name in (
        "values.npy",
        "offsets.npy",
        "spatial_shapes.npy",
        "name_to_idx.json",
        "manifest.json",
    ):
        if not (feature_dir / name).is_file():
            raise FileNotFoundError(f"Missing dense image cache file: {feature_dir / name}")

    values = np.load(feature_dir / "values.npy", mmap_mode="r")
    offsets = np.load(feature_dir / "offsets.npy", mmap_mode="r")
    spatial_shapes = np.load(feature_dir / "spatial_shapes.npy", mmap_mode="r")
    with (feature_dir / "name_to_idx.json").open("r", encoding="utf-8") as file:
        name_to_idx = json.load(file)
    manifest = load_feature_manifest(feature_dir)

    num_images = len(name_to_idx)
    if values.ndim != 2:
        raise ValueError(f"Dense values must be [total_tokens,D], got {values.shape}")
    if offsets.shape != (num_images + 1,):
        raise ValueError(f"Dense offsets must be [{num_images + 1}], got {offsets.shape}")
    if spatial_shapes.shape != (num_images, 2):
        raise ValueError(
            f"Dense spatial_shapes must be [{num_images},2], got {spatial_shapes.shape}"
        )
    expected_indices = set(range(num_images))
    if set(name_to_idx.values()) != expected_indices:
        raise ValueError("Dense image name_to_idx must be a contiguous bijection")
    offsets_i64 = np.asarray(offsets, dtype=np.int64)
    if offsets_i64[0] != 0 or offsets_i64[-1] != len(values):
        raise ValueError("Dense offsets endpoints do not match values.npy")
    if np.any(offsets_i64[1:] < offsets_i64[:-1]):
        raise ValueError("Dense offsets are not monotonic")
    token_counts = offsets_i64[1:] - offsets_i64[:-1]
    grid_counts = np.asarray(spatial_shapes, dtype=np.int64).prod(axis=1)
    if np.any(spatial_shapes <= 0) or not np.array_equal(token_counts, grid_counts):
        raise ValueError("Dense offsets do not match spatial_shapes token counts")
    if not np.isfinite(values).all():
        raise FloatingPointError("Dense image cache contains NaN or Inf")
    if manifest.get("total_token_count") != len(values):
        raise ValueError("Dense manifest total_token_count does not match values.npy")
    if manifest.get("num_images") not in (None, num_images):
        raise ValueError("Dense manifest num_images does not match name_to_idx")
    if manifest.get("feature_dim") != values.shape[1]:
        raise ValueError("Dense manifest feature_dim does not match values.npy")
    if manifest.get("storage_dtype") not in (None, values.dtype.name):
        raise ValueError("Dense manifest storage_dtype does not match values.npy")

    return DenseImageFeatureCache(
        values=values,
        offsets=offsets,
        spatial_shapes=spatial_shapes,
        name_to_idx=name_to_idx,
        manifest=manifest,
    )


def get_dense_features_by_ids(
    image_ids: Sequence[str],
    cache: DenseImageFeatureCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ragged rows into CPU float32 tokens and a True=real mask."""

    if not image_ids:
        raise ValueError("image_ids must not be empty")
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for image_id in image_ids:
        if image_id not in cache.name_to_idx:
            raise KeyError(f"Missing image_id from dense cache: {image_id}")
        index = cache.name_to_idx[image_id]
        start = int(cache.offsets[index])
        end = int(cache.offsets[index + 1])
        row = np.asarray(cache.values[start:end])
        rows.append(row)
        lengths.append(end - start)

    dim = int(cache.values.shape[1])
    maximum = max(lengths)
    dense_tokens = torch.zeros(len(rows), maximum, dim, dtype=torch.float32)
    dense_mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for batch_index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        dense_tokens[batch_index, :length].copy_(
            torch.from_numpy(np.asarray(row, dtype=np.float32).copy())
        )
        dense_mask[batch_index, :length] = True
    return dense_tokens, dense_mask
