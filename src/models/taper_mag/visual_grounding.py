from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.taper_mag.text_reader import PreNormCrossAttentionBlock


@dataclass(frozen=True, slots=True)
class GroundingOutput:
    grounded: Tensor
    attention: Tensor


class GatedEditConditioning(nn.Module):
    def __init__(self, d_model: int = 256, gate_bias: float = -1.0) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.edit_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, query_identities: Tensor, text_reads: Tensor) -> tuple[Tensor, Tensor]:
        if query_identities.ndim == 2:
            query_identities = query_identities.unsqueeze(0).expand(text_reads.shape[0], -1, -1)
        gates = self.gate(torch.cat([query_identities, text_reads], dim=-1)).sigmoid()
        conditioned = self.norm(
            query_identities + gates * self.edit_projection(text_reads)
        )
        return conditioned, gates


class EditAwareVisualGrounding(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = PreNormCrossAttentionBlock(
            d_model, num_heads=num_heads, ffn_multiplier=4, dropout=dropout
        )

    def forward(
        self, conditioned_queries: Tensor, local_tokens: Tensor, local_mask: Tensor
    ) -> GroundingOutput:
        grounded, attention = self.block(
            conditioned_queries, local_tokens, local_mask.bool()
        )
        return GroundingOutput(grounded=grounded, attention=attention)
