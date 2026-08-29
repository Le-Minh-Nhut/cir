from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StableFactorFuser(nn.Module):
    def __init__(self, width: int = 256, factor_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, factor_dim),
        )

    def forward(self, intents: Tensor, original_evidence: Tensor) -> Tensor:
        if intents.shape != original_evidence.shape:
            raise ValueError("WHAT and WHERE evidence must share [B,K,d]")
        features = torch.cat(
            [
                intents,
                original_evidence,
                intents * original_evidence,
                (intents - original_evidence).abs(),
            ],
            dim=-1,
        )
        return F.normalize(self.network(features), dim=-1)


class AuxiliaryFullQueryAnchor(nn.Module):
    """Target-free, auxiliary-only reference+text relational anchor."""

    def __init__(self, width: int = 256, factor_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, factor_dim),
        )

    def forward(self, anchor: Tensor, text_global: Tensor) -> Tensor:
        anchor_global = anchor.mean(dim=1)
        return F.normalize(self.network(torch.cat([anchor_global, text_global], dim=-1)), dim=-1)
