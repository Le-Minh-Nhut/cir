from pathlib import Path
from collections.abc import Sequence
import torch
import json
import numpy as np


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

from dataclasses import dataclass


@dataclass(frozen=True)
class TextFeatureCache:
    states: torch.Tensor
    teacher_states: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor

    sample_to_idx: dict[str, int]
    captions: dict[str, str]
    manifest: dict


def load_text_features(feature_dir) -> TextFeatureCache:
    feature_dir = Path(feature_dir)

    required = (
        "states.npy",
        "teacher_states.npy",
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

    teacher_states = torch.from_numpy(
        np.load(
            feature_dir / "teacher_states.npy",
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

    with (
        feature_dir / "manifest.json"
    ).open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    num_samples = len(sample_to_idx)

    arrays = {
        "states": states,
        "teacher_states": teacher_states,
        "attention_mask": attention_mask,
        "content_mask": content_mask,
    }

    for name, tensor in arrays.items():
        if tensor.shape[0] != num_samples:
            raise ValueError(
                f"{name} rows ({tensor.shape[0]}) "
                f"!= sample index size ({num_samples})"
            )

    if set(sample_to_idx) != set(captions):
        raise ValueError(
            "Text-cache sample IDs and caption IDs differ"
        )

    if states.shape[:2] != teacher_states.shape[:2]:
        raise ValueError(
            "states and teacher_states [Q,N] mismatch"
        )

    if attention_mask.shape != states.shape[:2]:
        raise ValueError(
            "attention_mask shape mismatch"
        )

    if content_mask.shape != states.shape[:2]:
        raise ValueError(
            "content_mask shape mismatch"
        )

    return TextFeatureCache(
        states=states,
        teacher_states=teacher_states,
        attention_mask=attention_mask,
        content_mask=content_mask,
        sample_to_idx=sample_to_idx,
        captions=captions,
        manifest=manifest,
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
        cache.teacher_states[indices],
        cache.attention_mask[indices],
        cache.content_mask[indices],
    )