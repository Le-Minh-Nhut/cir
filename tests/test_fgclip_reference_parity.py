from __future__ import annotations

import copy

import torch

from models.iag_srme.backbone import FGCLIPBackbone
from test_backbone_call_count import CountingFGCLIP


def _parity_objective(
    backbone: FGCLIPBackbone, pixels: torch.Tensor, *, official_helpers: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if official_helpers:
        dense = backbone.model.get_image_dense_features(pixel_values=pixels)
        global_features = backbone.model.get_image_features(pixel_values=pixels)
    else:
        outputs = backbone.model.vision_model(
            pixel_values=pixels, output_hidden_states=True, return_dict=True
        )
        dense, global_features = backbone.reference_features_from_vision_outputs(outputs)
    anchor = backbone.anchor_projection(dense)
    objective = dense.square().mean() + global_features.square().mean() + anchor.square().mean()
    return dense, global_features, objective


def test_mock_one_pass_values_match_official_helpers() -> None:
    torch.manual_seed(71)
    backbone = FGCLIPBackbone(CountingFGCLIP(), internal_width=4, train_vision=True).eval()
    pixels = torch.randn(2, 3, 4, 4)

    official_dense, official_global, _ = _parity_objective(
        backbone, pixels, official_helpers=True
    )
    manual_dense, manual_global, _ = _parity_objective(
        backbone, pixels, official_helpers=False
    )

    assert torch.equal(manual_dense, official_dense)
    assert torch.equal(manual_global, official_global)


def test_mock_one_pass_gradients_match_two_official_helpers() -> None:
    torch.manual_seed(73)
    official = FGCLIPBackbone(CountingFGCLIP(), internal_width=4, train_vision=True).eval()
    manual = copy.deepcopy(official)
    pixels = torch.randn(2, 3, 4, 4)
    official_parameters = (
        official.model.vision_model.weight,
        official.model.visual_projection.weight,
        official.anchor_projection[0].weight,
    )
    manual_parameters = (
        manual.model.vision_model.weight,
        manual.model.visual_projection.weight,
        manual.anchor_projection[0].weight,
    )

    _, _, official_objective = _parity_objective(official, pixels, official_helpers=True)
    _, _, manual_objective = _parity_objective(manual, pixels, official_helpers=False)
    official_gradients = torch.autograd.grad(official_objective, official_parameters)
    manual_gradients = torch.autograd.grad(manual_objective, manual_parameters)

    for expected, actual in zip(official_gradients, manual_gradients, strict=True):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
        assert actual.abs().sum() > 0
