from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from backbones.fgclip2_base import (
    FGCLIP2_BASE_MODEL_ID,
    FGCLIP2_BASE_REVISION,
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
)
from conftest import FakeFGCLIP2, FakeImageProcessor, FakeTokenizer


def make_backbone(text_tuning: TextTuningConfig | None = None) -> FGCLIP2BaseBackbone:
    return FGCLIP2BaseBackbone(
        model=FakeFGCLIP2(),
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
        text_tuning=text_tuning,
        vision_tuning=VisionTuningConfig(),
    )


def test_pinned_contract_and_manifest() -> None:
    backbone = make_backbone()
    assert backbone.model_id == FGCLIP2_BASE_MODEL_ID
    assert backbone.revision == FGCLIP2_BASE_REVISION
    assert backbone.contract.text_dim == 16
    assert backbone.contract.vision_dim == 16
    assert backbone.contract.retrieval_dim == 16
    assert backbone.contract.text_blocks == backbone.contract.vision_blocks == 12
    manifest = backbone.manifest()
    assert len(manifest.sha256) == 64
    assert manifest.text_walk_type == "short"
    assert manifest.max_text_length == 64


def test_last_four_text_blocks_gradient_and_vision_frozen() -> None:
    backbone = make_backbone(
        TextTuningConfig(
            mode="last_n_blocks",
            num_unfrozen_blocks=4,
            train_final_norm=True,
            train_projection=False,
        )
    )
    assert backbone.unfrozen_text_block_names == tuple(
        f"text_model.encoder.layers.{index}" for index in range(8, 12)
    )
    assert not any(parameter.requires_grad for parameter in backbone.model.vision_model.parameters())
    tokenized = backbone.tokenize_texts(["make it bright", "add long sleeves"])
    states = backbone.encode_text_tokens(tokenized)
    states.square().mean().backward()
    for index, block in enumerate(backbone.model.text_model.encoder.layers):
        grads = [parameter.grad for parameter in block.parameters()]
        if index < 8:
            assert all(gradient is None for gradient in grads)
        else:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in grads)
    assert all(parameter.grad is None for parameter in backbone.model.vision_model.parameters())


def test_frozen_and_full_text_modes() -> None:
    frozen = make_backbone(
        TextTuningConfig(
            mode="frozen", num_unfrozen_blocks=0, train_final_norm=False, train_projection=False
        )
    )
    assert not any(parameter.requires_grad for parameter in frozen.model.text_model.parameters())
    full = make_backbone(
        TextTuningConfig(
            mode="full", num_unfrozen_blocks=12, train_final_norm=True, train_projection=True
        )
    )
    assert all(parameter.requires_grad for parameter in full.model.text_model.parameters())


def test_fake_official_image_paths_are_normalized_masked_and_finite() -> None:
    backbone = make_backbone()
    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (48, 32))]
    global_features = backbone.encode_image_global(images)
    assert global_features.shape == (2, 16)
    torch.testing.assert_close(global_features.norm(dim=-1), torch.ones(2))
    dense = backbone.encode_image_dense(images)
    assert dense.tokens.shape == (2, 4, 16)
    assert dense.mask.all()
    assert torch.isfinite(dense.tokens).all()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_FGCLIP2_INTEGRATION") != "1",
    reason="set RUN_FGCLIP2_INTEGRATION=1 to download/inspect the pinned 1.54GB checkpoint",
)
def test_real_pinned_runtime_contract() -> None:
    backbone = FGCLIP2BaseBackbone(
        text_tuning=TextTuningConfig(
            mode="last_n_blocks",
            num_unfrozen_blocks=4,
            train_final_norm=True,
            train_projection=False,
        )
    )
    assert backbone.contract.text_dim == 768
    assert backbone.contract.vision_dim == 768
    assert backbone.contract.retrieval_dim == 768
    assert backbone.contract.text_blocks == backbone.contract.vision_blocks == 12
    assert backbone.unfrozen_text_block_names == tuple(
        f"text_model.encoder.layers.{index}" for index in range(8, 12)
    )
    assert not any(parameter.requires_grad for parameter in backbone.model.vision_model.parameters())
    tokenized = backbone.tokenize_texts(["make it red and sleeveless"])
    states = backbone.encode_text_tokens(tokenized)
    assert states.shape == (1, 64, 768)
    assert torch.isfinite(states).all()
    assert states.requires_grad
    del states
    images = [Image.new("RGB", (32, 48), color=(80, 120, 160))]
    global_features = backbone.encode_image_global(images)
    assert global_features.shape == (1, 768)
    torch.testing.assert_close(global_features.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)
    dense = backbone.encode_image_dense(images)
    assert dense.tokens.ndim == 3 and dense.tokens.shape[0] == 1
    assert dense.tokens.shape[-1] == 768
    assert dense.mask.sum().item() == dense.spatial_shapes.prod(dim=-1).sum().item()
    assert torch.isfinite(dense.tokens).all()
