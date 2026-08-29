from __future__ import annotations

import torch
from torch import Tensor, nn


class GroundedEditContext(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(6 * width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.intent_residual = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        intents: Tensor,
        original: Tensor,
        current: Tensor,
        change: Tensor,
    ) -> Tensor:
        if not (intents.shape == original.shape == current.shape == change.shape):
            raise ValueError("all context inputs must be [B,K,d] with identical shape")
        features = torch.cat(
            [intents, original, current, change, intents * current, intents - current], dim=-1
        )
        return self.output_norm(self.fusion(features) + self.intent_residual(intents))

