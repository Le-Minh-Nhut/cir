from __future__ import annotations

import torch
from torch import Tensor, nn


class ConsequenceScorer(nn.Module):
    """Target-free shared scorer of actual counterfactual consequences."""

    def __init__(self, width: int = 256, retrieval_dim: int = 512) -> None:
        super().__init__()
        input_width = width + retrieval_dim + width + 3
        self.score_head = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 1),
        )

    def forward(
        self,
        contexts: Tensor,
        delta_z: Tensor,
        delta_q: Tensor,
        local_change: Tensor,
        supports: Tensor,
    ) -> Tensor:
        if contexts.shape[:2] != supports.shape[:2] or delta_z.shape[:3] != (
            supports.shape[0], supports.shape[1], supports.shape[2]
        ):
            raise ValueError("candidate axes must match")
        support_gate = supports / supports.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled_delta = torch.einsum("bkn,bknd->bkd", support_gate, delta_z)
        effect_norm = delta_q.norm(dim=-1, keepdim=True)
        change_norm = local_change.norm(dim=-1, keepdim=True)
        support_fraction = (supports > 0).to(delta_z.dtype).mean(dim=-1, keepdim=True)
        scalar_features = torch.cat([effect_norm, change_norm, support_fraction], dim=-1)
        features = torch.cat([contexts, delta_q, pooled_delta, scalar_features], dim=-1)
        return self.score_head(features).squeeze(-1)
