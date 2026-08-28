from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from datasets.fashioniq import FashionIQDataset


CATEGORIES = ("dress", "shirt", "toptee")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported TAPER-MAG config schema")
    return config


def make_dataset(
    config: dict[str, Any],
    split: str,
    categories: tuple[str, ...],
    correction_dicts: dict[str, dict[str, str]] | None,
) -> FashionIQDataset:
    annotation_root = Path(config["data"]["dataset_root"]) / "captions"
    policy_key = "train_caption_policy" if split == "train" else "validation_caption_policy"
    return FashionIQDataset(
        annotation_root,
        split,
        categories,
        caption_policy=config["data"][policy_key],
        correction_dicts=correction_dicts,
        seed=int(config["seed"]),
    )
