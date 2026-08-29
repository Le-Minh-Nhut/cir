from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from models.iag_srme.backbone import FGCLIPBackbone


class CountingVisionModel(nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, hidden_size))
        self.post_layernorm = nn.LayerNorm(hidden_size)
        self.calls = 0

    def forward(self, pixel_values, output_hidden_states=False, return_dict=False):
        del output_hidden_states, return_dict
        self.calls += 1
        pooled = pixel_values.mean(dim=(-2, -1)) @ self.weight
        tokens = pooled[:, None, :].expand(-1, 5, -1)
        return SimpleNamespace(hidden_states=(tokens * 0.5, tokens), pooler_output=pooled)


class CountingFGCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            projection_dim=8, text_config=SimpleNamespace(hidden_size=7)
        )
        self.vision_model = CountingVisionModel()
        self.visual_projection = nn.Linear(6, 8, bias=False)
        self.text_model = nn.Linear(7, 7)
        self.text_projection = nn.Linear(7, 8, bias=False)
        self.dense_helper_calls = 0
        self.global_helper_calls = 0

    def forward_without_attn(self, hidden_state):
        return hidden_state

    def get_image_dense_features(self, pixel_values):
        self.dense_helper_calls += 1
        outputs = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        return self.visual_projection(outputs.hidden_states[-2][:, 1:])

    def get_image_features(self, pixel_values):
        self.global_helper_calls += 1
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return self.visual_projection(outputs.pooler_output)


def test_reference_is_one_vision_pass_and_global_path_never_extracts_dense() -> None:
    checkpoint = CountingFGCLIP()
    backbone = FGCLIPBackbone(checkpoint, internal_width=4, train_vision=True)
    pixels = torch.randn(2, 3, 4, 4)

    anchor, reference_global = backbone.encode_reference_images(pixels)
    assert anchor.shape == (2, 4, 4)
    assert reference_global.shape == (2, 8)
    assert checkpoint.vision_model.calls == 1
    assert checkpoint.dense_helper_calls == 0
    assert checkpoint.global_helper_calls == 0

    target_global = backbone.encode_global_images(pixels)
    gallery_global = backbone.encode_global_images(pixels)
    assert target_global.shape == gallery_global.shape == (2, 8)
    assert checkpoint.vision_model.calls == 3
    assert checkpoint.dense_helper_calls == 0
    assert checkpoint.global_helper_calls == 2


def test_frozen_vision_stays_frozen_and_in_eval_mode() -> None:
    checkpoint = CountingFGCLIP()
    backbone = FGCLIPBackbone(checkpoint, internal_width=4, train_vision=False)
    backbone.train()

    assert not checkpoint.vision_model.training
    assert not checkpoint.visual_projection.training
    assert all(not parameter.requires_grad for parameter in checkpoint.vision_model.parameters())
    assert all(
        not parameter.requires_grad for parameter in checkpoint.visual_projection.parameters()
    )
    assert checkpoint.text_model.training
    assert all(parameter.requires_grad for parameter in checkpoint.text_model.parameters())


def test_full_regime_trains_vision_and_projection() -> None:
    checkpoint = CountingFGCLIP()
    backbone = FGCLIPBackbone(checkpoint, internal_width=4, train_vision=True)
    backbone.train()

    assert checkpoint.vision_model.training
    assert checkpoint.visual_projection.training
    assert all(parameter.requires_grad for parameter in checkpoint.vision_model.parameters())
    assert all(parameter.requires_grad for parameter in checkpoint.visual_projection.parameters())
