from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import yaml
from torch import nn
from torch.optim import SGD

from datasets.fashioniq import FashionIQAnnotation
from evaluation.fashioniq import build_fashioniq_gallery
from models.iag_srme import (
    BackboneBuildSpec,
    backbone_spec_from_metadata,
    validate_checkpoint_backbone_metadata,
)
from training.engine import PrecisionPolicy, save_checkpoint


OPENCLIP_METADATA = {
    "backbone_type": "openclip",
    "backbone_checkpoint": "ViT-B-16",
    "backbone_revision": "laion2b_s34b_b88k",
    "backbone_library_version": "3.3.0",
    "backbone_weights_repository": "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "backbone_weights_revision": "7288da5a0d6f0b51c4a2b27c624837a9236d0112",
}


def _expected_openclip_spec() -> BackboneBuildSpec:
    return BackboneBuildSpec(
        backbone_type="openclip",
        checkpoint="ViT-B-16",
        revision="laion2b_s34b_b88k",
        library_version="3.3.0",
        weights_repository="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        weights_revision="7288da5a0d6f0b51c4a2b27c624837a9236d0112",
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )


def test_fashioniq_val_protocol_uses_pair_union_gallery(tmp_path: Path) -> None:
    annotations = [
        FashionIQAnnotation("reference-a", "target-a", ("a", "b"), "dress", 0),
        FashionIQAnnotation("reference-b", "target-a", ("c", "d"), "dress", 1),
    ]
    gallery = build_fashioniq_gallery(
        "fashioniq_val", tmp_path, "dress", annotations, "val"
    )
    assert gallery == ["reference-a", "target-a", "reference-b"]


def test_openclip_ablation_selects_val_protocol() -> None:
    root = Path(__file__).parents[1]
    protocol = yaml.safe_load(
        (root / "conf/protocol/fashioniq_val.yaml").read_text(encoding="utf-8")
    )
    assert protocol == {
        "name": "fashioniq_val",
        "dataset_name": "fashioniq",
        "split": "val",
    }


def test_diagnostic_backbone_factory_recognizes_openclip_metadata() -> None:
    spec = backbone_spec_from_metadata(
        {
            "backbone_type": "openclip",
            "backbone_checkpoint": "ViT-B-16",
            "backbone_revision": "laion2b_s34b_b88k",
            "backbone_library_version": "3.3.0",
            "backbone_weights_repository": "repository",
            "backbone_weights_revision": "immutable-sha",
        },
        train_vision=False,
        train_text=False,
        train_text_projection=False,
    )
    assert spec.backbone_type == "openclip"
    assert spec.checkpoint == "ViT-B-16"
    assert spec.revision == "laion2b_s34b_b88k"
    assert spec.library_version == "3.3.0"
    assert spec.weights_repository == "repository"
    assert spec.weights_revision == "immutable-sha"


def test_correct_openclip_checkpoint_identity_passes() -> None:
    validate_checkpoint_backbone_metadata(OPENCLIP_METADATA, _expected_openclip_spec())


def test_backbone_identity_validation_is_independent_of_evaluation_protocol() -> None:
    metadata = {
        **OPENCLIP_METADATA,
        "evaluation_protocol": "fashioniq_val",
        "selection_protocol": "fashioniq_val",
    }
    validate_checkpoint_backbone_metadata(metadata, _expected_openclip_spec())
    metadata["evaluation_protocol"] = "fashioniq_original"
    metadata["selection_protocol"] = "fashioniq_original"
    validate_checkpoint_backbone_metadata(metadata, _expected_openclip_spec())


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("backbone_weights_revision", "wrong-immutable-sha"),
        ("backbone_library_version", "3.2.0"),
        ("backbone_weights_repository", "wrong/repository"),
    ],
)
def test_openclip_checkpoint_identity_rejects_reproducibility_mismatch(
    field: str, wrong_value: str
) -> None:
    metadata = {**OPENCLIP_METADATA, field: wrong_value}
    with pytest.raises(ValueError, match="backbone mismatch"):
        validate_checkpoint_backbone_metadata(metadata, _expected_openclip_spec())


def test_legacy_fgclip_checkpoint_identity_remains_compatible() -> None:
    metadata = {
        "backbone_checkpoint": "qihoo360/fg-clip-base",
        "backbone_revision": "verified-revision",
    }
    expected = BackboneBuildSpec(
        backbone_type="fgclip",
        checkpoint="qihoo360/fg-clip-base",
        revision="verified-revision",
        train_vision=True,
        train_text=True,
        train_text_projection=False,
        trust_remote_code=True,
    )
    validate_checkpoint_backbone_metadata(metadata, expected)


def test_checkpoint_persists_openclip_and_val_protocol_metadata(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    model.backbone = SimpleNamespace(
        backbone_type="openclip",
        checkpoint="ViT-B-16",
        revision="laion2b_s34b_b88k",
        library="open_clip_torch",
        library_version="3.3.0",
        weights_repository="repository",
        weights_revision="immutable-sha",
    )
    objective = nn.Linear(2, 1)
    optimizer = SGD([*model.parameters(), *objective.parameters()], lr=0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        objective,
        optimizer,
        epoch=3,
        precision=PrecisionPolicy("fp16", True, torch.float16, True),
        validation_metrics={
            "fashioniq_original": {"mean_recall": 40.0},
            "fashioniq_val": {"mean_recall": 42.0},
        },
        selection_protocol="fashioniq_val",
    )
    metadata = torch.load(path, weights_only=True)["metadata"]

    assert metadata["backbone_type"] == "openclip"
    assert metadata["backbone_checkpoint"] == "ViT-B-16"
    assert metadata["backbone_revision"] == "laion2b_s34b_b88k"
    assert metadata["backbone_library_version"] == "3.3.0"
    assert metadata["backbone_weights_revision"] == "immutable-sha"
    assert metadata["evaluation_protocol"] == "fashioniq_val"
    assert metadata["selection_protocol"] == "fashioniq_val"
    assert metadata["validation_metrics"]["fashioniq_original"]["mean_recall"] == 40.0
