from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evaluation.fashioniq import build_validation_datasets
from models.iag_srme import BackboneBuildSpec, validate_checkpoint_backbone_metadata


def test_evaluation_dataset_uses_explicit_caption_policy(tmp_path) -> None:
    record = [
        {
            "candidate": "reference",
            "target": "target",
            "captions": ["make it red", "add long sleeves"],
        }
    ]
    (tmp_path / "cap.dress.val.json").write_text(json.dumps(record), encoding="utf-8")

    datasets = build_validation_datasets(
        tmp_path,
        ["dress"],
        caption_policy="normalized_ordered_and",
        seed=9,
        correction_dicts={"dress": {}},
    )

    assert datasets["dress"].caption_policy == "normalized_ordered_and"
    assert datasets["dress"][0].modification_text == "make it red and add long sleeves"


def test_canonical_experiments_use_complete_ordered_caption_text() -> None:
    root = Path(__file__).parents[1]
    for name in ("iag_srme_base_full.yaml", "iag_srme_large_text_ft.yaml"):
        config = yaml.safe_load((root / "conf" / "experiment" / name).read_text())
        assert config["train_caption_policy"] == "ordered_and"
        assert config["val_caption_policy"] == "ordered_and"


def test_evaluation_rejects_checkpoint_from_another_backbone_revision() -> None:
    metadata = {"backbone_checkpoint": "qihoo360/fg-clip-base", "backbone_revision": "abc"}
    with pytest.raises(ValueError, match="backbone mismatch"):
        validate_checkpoint_backbone_metadata(
            metadata,
            BackboneBuildSpec(
                backbone_type="fgclip",
                checkpoint="qihoo360/fg-clip-base",
                revision="verified-revision",
                train_vision=True,
                train_text=True,
                train_text_projection=False,
            ),
        )
