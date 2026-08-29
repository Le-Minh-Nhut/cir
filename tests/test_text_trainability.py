from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from losses.factor import FactorCompletenessLoss
from losses.marginal import MarginalActionLoss
from losses.retrieval import TerminalRetrievalLoss
from models.iag_srme import FGCLIPBackbone, IAGSRME, IAGSRMEConfig, IAGSRMECore
from test_parameter_update import TinyFGCLIP


def _build_model(enable_factor_head: bool) -> IAGSRME:
    torch.manual_seed(151)
    backbone = FGCLIPBackbone(
        TinyFGCLIP(),
        internal_width=8,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=8,
            num_candidates=4,
            max_steps=2,
            num_heads=2,
            retrieval_dim=8,
            selector_gumbel_noise=False,
            enable_factor_head=enable_factor_head,
        )
    )
    return IAGSRME(backbone, core).train()


def _text_parameters(model: IAGSRME) -> dict[str, torch.nn.Parameter]:
    return {
        "fgclip_text_model": model.backbone.model.text_model.embedding.weight,
        "fgclip_text_projection": model.backbone.model.text_projection.weight,
        "iag_text_projection": model.backbone.text_projection[0].weight,
    }


@pytest.mark.parametrize("loss_name", ["terminal", "marginal", "factor"])
def test_text_gradient_contract_is_explicit_per_loss(loss_name: str) -> None:
    model = _build_model(enable_factor_head=loss_name == "factor")
    reference = torch.randn(4, 3, 4, 4)
    target = torch.randn(4, 3, 4, 4)
    input_ids = torch.randint(0, 32, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.long)
    output = model(reference, input_ids, mask, mask.bool())
    positives = torch.eye(4, dtype=torch.bool)
    target_embeddings = model.encode_global_images(target)
    if loss_name == "terminal":
        loss = TerminalRetrievalLoss()(output.final_query, target_embeddings, positives)
    elif loss_name == "marginal":
        loss = MarginalActionLoss()(output.trace, target_embeddings, positives)
    else:
        assert output.factors is not None and output.auxiliary_anchor is not None
        loss, _ = FactorCompletenessLoss()(output.factors, output.auxiliary_anchor)
    loss.backward()

    parameters = _text_parameters(model)
    assert parameters["fgclip_text_model"].requires_grad
    assert parameters["iag_text_projection"].requires_grad
    assert not parameters["fgclip_text_projection"].requires_grad
    assert parameters["fgclip_text_model"].grad is not None
    assert parameters["fgclip_text_model"].grad.abs().sum() > 0
    assert parameters["iag_text_projection"].grad is not None
    assert parameters["iag_text_projection"].grad.abs().sum() > 0
    assert parameters["fgclip_text_projection"].grad is None


def test_canonical_backbones_freeze_unused_fgclip_text_projection() -> None:
    root = Path(__file__).parents[1]
    for filename in ("fgclip_base_full.yaml", "fgclip_large_text_ft.yaml"):
        config = yaml.safe_load((root / "conf" / "backbone" / filename).read_text())
        assert config["train_text"] is True
        assert config["train_text_projection"] is False
