from pathlib import Path
from collections.abc import Sequence
import torch
import json


def load_features(feature_dir):
    feature_dir = Path(feature_dir)
    features = torch.load(feature_dir / "images.pt", map_location="cpu", mmap=True, weights_only=True,)

    with (feature_dir / "name_to_idx.json").open("r", encoding="utf-8",) as file:
        name_to_idx = json.load(file)

    return features, name_to_idx

def get_features_by_ids(image_ids: Sequence[str], features: torch.Tensor, name_to_idx: dict[str, int]) -> torch.Tensor:
    indices = [
        name_to_idx[image_id]
        for image_id in image_ids
    ]

    return features[indices]