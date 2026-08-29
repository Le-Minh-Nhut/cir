from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from losses.factor import FactorCompletenessLoss
from models.iag_srme import IAGSRMEConfig, IAGSRMECore
from models.iag_srme.factorization import SemanticFullQueryAnchor


def test_semantic_anchor_is_parameter_free_fgclip_space_composition() -> None:
    module = SemanticFullQueryAnchor()
    reference = F.normalize(torch.randn(3, 24), dim=-1)
    text = F.normalize(torch.randn(3, 24), dim=-1)

    actual = module(reference, text)

    assert list(module.parameters()) == []
    assert torch.allclose(actual, F.normalize(reference + text, dim=-1))


def test_factor_loss_trains_factors_but_detaches_semantic_geometry(synthetic_encoded) -> None:
    torch.manual_seed(31)
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_heads=4,
            retrieval_dim=24,
            max_steps=1,
            selector_gumbel_noise=False,
            enable_factor_head=True,
        )
    )
    encoded = copy.copy(synthetic_encoded)
    encoded.reference_global = encoded.reference_global.detach().requires_grad_()
    encoded.text_semantic_global = encoded.text_semantic_global.detach().requires_grad_()
    output = core(encoded)
    assert output.factors is not None and output.auxiliary_anchor is not None

    loss, _ = FactorCompletenessLoss()(output.factors, output.auxiliary_anchor)
    loss.backward()

    factor_weight = core.factor_fuser.network[-1].weight
    assert factor_weight.grad is not None and factor_weight.grad.abs().sum() > 0
    assert encoded.reference_global.grad is None
    assert encoded.text_semantic_global.grad is None


def test_auxiliary_anchor_cannot_change_executor_or_readout(synthetic_encoded) -> None:
    torch.manual_seed(37)
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_heads=4,
            retrieval_dim=24,
            max_steps=2,
            selector_gumbel_noise=False,
            enable_factor_head=True,
        )
    ).eval()
    changed = copy.copy(synthetic_encoded)
    changed.text_semantic_global = F.normalize(
        torch.randn_like(synthetic_encoded.text_semantic_global), dim=-1
    )

    first = core(synthetic_encoded)
    second = core(changed)

    assert not torch.equal(first.auxiliary_anchor, second.auxiliary_anchor)
    assert torch.equal(first.final_query, second.final_query)
    assert torch.equal(first.trace[0].scores, second.trace[0].scores)
    assert torch.equal(first.trace[0].candidate_states, second.trace[0].candidate_states)
