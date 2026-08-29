from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from models.iag_srme import BackboneOutput, IAGSRMEConfig, IAGSRMECore


@pytest.fixture
def synthetic_encoded() -> BackboneOutput:
    torch.manual_seed(7)
    batch, tokens, length, width, retrieval_dim = 3, 13, 8, 32, 24
    return BackboneOutput(
        anchor=torch.randn(batch, tokens, width),
        reference_global=F.normalize(torch.randn(batch, retrieval_dim), dim=-1),
        text_tokens=torch.randn(batch, length, width),
        text_global=torch.randn(batch, width),
        text_content_mask=torch.ones(batch, length, dtype=torch.bool),
    )


@pytest.fixture
def core() -> IAGSRMECore:
    torch.manual_seed(11)
    return IAGSRMECore(
        IAGSRMEConfig(
            width=32,
            num_candidates=4,
            max_steps=3,
            num_heads=4,
            retrieval_dim=24,
            selector_gumbel_noise=False,
        )
    )

