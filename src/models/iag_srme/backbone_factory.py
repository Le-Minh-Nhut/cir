from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .backbone import FGCLIPBackbone, FGCLIPRegime
from .openclip_backbone import OpenCLIPBackbone, OpenCLIPRegime


@dataclass(frozen=True, slots=True)
class BackboneBuildSpec:
    backbone_type: str
    checkpoint: str
    revision: str
    train_vision: bool
    train_text: bool
    train_text_projection: bool
    trust_remote_code: bool = False
    library_version: str | None = None
    weights_repository: str | None = None
    weights_revision: str | None = None


def build_backbone(
    spec: BackboneBuildSpec, internal_width: int
) -> tuple[nn.Module, Any, Any]:
    if spec.backbone_type == "fgclip":
        regime = FGCLIPRegime(
            checkpoint=spec.checkpoint,
            revision=spec.revision,
            train_vision=spec.train_vision,
            train_text=spec.train_text,
            train_text_projection=spec.train_text_projection,
            trust_remote_code=spec.trust_remote_code,
        )
        backbone = FGCLIPBackbone.from_pretrained(regime, internal_width)
        tokenizer, processor = FGCLIPBackbone.load_processor(
            regime.checkpoint, regime.revision, regime.trust_remote_code
        )
        return backbone, tokenizer, processor
    if spec.backbone_type == "openclip":
        if spec.library_version is None:
            raise ValueError("OpenCLIP requires a pinned library_version")
        if spec.weights_repository is None or spec.weights_revision is None:
            raise ValueError("OpenCLIP requires a pinned weight repository and revision")
        regime = OpenCLIPRegime(
            model_name=spec.checkpoint,
            pretrained=spec.revision,
            library_version=spec.library_version,
            weights_repository=spec.weights_repository,
            weights_revision=spec.weights_revision,
            train_vision=spec.train_vision,
            train_text=spec.train_text,
            train_text_projection=spec.train_text_projection,
        )
        return OpenCLIPBackbone.from_pretrained(regime, internal_width)
    raise ValueError(f"unsupported backbone type: {spec.backbone_type}")


def backbone_spec_from_metadata(
    metadata: dict[str, Any],
    *,
    train_vision: bool,
    train_text: bool,
    train_text_projection: bool,
) -> BackboneBuildSpec:
    checkpoint = metadata.get("backbone_checkpoint")
    revision = metadata.get("backbone_revision")
    if not isinstance(checkpoint, str) or not isinstance(revision, str):
        raise ValueError("checkpoint backbone checkpoint/revision metadata is incomplete")
    backbone_type = metadata.get("backbone_type", "fgclip")
    if not isinstance(backbone_type, str):
        raise ValueError("checkpoint backbone type metadata is invalid")
    library_version = metadata.get("backbone_library_version")
    if library_version is not None and not isinstance(library_version, str):
        raise ValueError("checkpoint backbone library version metadata is invalid")
    weights_repository = metadata.get("backbone_weights_repository")
    weights_revision = metadata.get("backbone_weights_revision")
    return BackboneBuildSpec(
        backbone_type=backbone_type,
        checkpoint=checkpoint,
        revision=revision,
        train_vision=train_vision,
        train_text=train_text,
        train_text_projection=train_text_projection,
        trust_remote_code=backbone_type == "fgclip",
        library_version=library_version,
        weights_repository=(
            weights_repository if isinstance(weights_repository, str) else None
        ),
        weights_revision=weights_revision if isinstance(weights_revision, str) else None,
    )
