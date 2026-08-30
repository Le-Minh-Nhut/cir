from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from evaluate import (
    validate_checkpoint_backbone_metadata,
    validate_checkpoint_model_config,
)
from evaluation.fashioniq import build_validation_datasets
from models.iag_srme import IAGSRMEConfig


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
            metadata, "qihoo360/fg-clip-base", "verified-revision"
        )


def test_evaluation_rejects_non_state_model_config_mismatch() -> None:
    configured = IAGSRMEConfig(query_cap=1000.0, enable_visual_null=True)
    stored = IAGSRMEConfig(query_cap=0.5, enable_visual_null=True)
    with pytest.raises(ValueError, match="model-config mismatch"):
        validate_checkpoint_model_config(
            {"model_config": asdict(stored)},
            configured,
        )


def test_evaluation_allows_legacy_checkpoint_without_model_config() -> None:
    validate_checkpoint_model_config({}, IAGSRMEConfig())
