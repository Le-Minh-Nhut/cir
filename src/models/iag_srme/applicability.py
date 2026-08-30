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
        # Keep the entire WHETHER pathway in FP32. Casting only the sigmoid input to FP32
        # is insufficient because returning confidence to fp16 quantizes small learned
        # changes near the initial c=0.98 operating point.
        with torch.autocast(device_type=contexts.device.type, enabled=False):
            normalized = self.norm(contexts.float())
            logits = self.projection(normalized).squeeze(-1)
            confidence = torch.sigmoid(logits)
            null_probability = 1.0 - confidence
        return logits, confidence, null_probability
