from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class DynamicApplicabilityGate(nn.Module):
    """Shared state-dependent WHETHER gate over grounded edit contexts."""

    def __init__(self, width: int = 256, initial_applicability: float = 0.98) -> None:
        super().__init__()
        if not 0.0 < initial_applicability < 1.0:
            raise ValueError("initial_applicability must be strictly between zero and one")
        self.initial_applicability = float(initial_applicability)
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, 1)
        nn.init.zeros_(self.projection.weight)
        initial_logit = math.log(initial_applicability / (1.0 - initial_applicability))
        nn.init.constant_(self.projection.bias, initial_logit)

    def forward(self, contexts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if contexts.ndim != 3:
            raise ValueError("contexts must be [B,K,d]")
        logits = self.projection(self.norm(contexts)).squeeze(-1)
        # Sigmoid arithmetic stays in FP32 under AMP so the applicability variable cannot
        # inherit Entmax's exact sparse-support exclusion behavior.
        confidence = torch.sigmoid(logits.float()).to(logits.dtype)
        null_probability = 1.0 - confidence
        return logits, confidence, null_probability
