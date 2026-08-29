from __future__ import annotations

import math

import torch
from entmax import entmax15
from torch import Tensor, nn


class AnchorGrounder(nn.Module):
    """Ground stable intents independently over immutable anchor tokens."""

    def __init__(self, width: int = 256, grounding_width: int | None = None) -> None:
        super().__init__()
        grounding_width = grounding_width or width
        self.intent_projection = nn.Linear(width, grounding_width, bias=False)
        self.anchor_projection = nn.Linear(width, grounding_width, bias=False)
        self.scale = math.sqrt(grounding_width)

    def forward(self, intents: Tensor, anchor: Tensor) -> Tensor:
        if intents.ndim != 3 or anchor.ndim != 3:
            raise ValueError("intents and anchor must be rank 3")
        if intents.shape[0] != anchor.shape[0] or intents.shape[-1] != anchor.shape[-1]:
            raise ValueError("intent/anchor batch and width must match")
        logits = torch.einsum(
            "bkd,bnd->bkn",
            self.intent_projection(intents),
            self.anchor_projection(anchor),
        ) / self.scale
        supports = entmax15(logits, dim=-1)
        if supports.shape != (intents.shape[0], intents.shape[1], anchor.shape[1]):
            raise AssertionError("support shape invariant failed")
        if not torch.allclose(
            supports.sum(dim=-1),
            torch.ones_like(supports[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise AssertionError("each candidate support must normalize over visual tokens")
        return supports

