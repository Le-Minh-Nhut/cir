from __future__ import annotations

import torch
from torch import Tensor, nn


class GroundedStateReader(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.original_projection = nn.Linear(width, width, bias=False)
        self.current_projection = nn.Linear(width, width, bias=False)
        self.change_projection = nn.Linear(width, width, bias=False)

    def forward(self, supports: Tensor, anchor: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if supports.ndim != 3 or anchor.ndim != 3 or state.shape != anchor.shape:
            raise ValueError("supports=[B,K,N] and anchor/state=[B,N,d] are required")
        if supports.shape[0] != anchor.shape[0] or supports.shape[2] != anchor.shape[1]:
            raise ValueError("support visual axis must match anchor")
        original = torch.einsum("bkn,bnd->bkd", supports, self.original_projection(anchor))
        current = torch.einsum("bkn,bnd->bkd", supports, self.current_projection(state))
        change = torch.einsum(
            "bkn,bnd->bkd", supports, self.change_projection(state - anchor)
        )
        return original, current, change
