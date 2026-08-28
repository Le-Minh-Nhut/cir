from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.taper_mag.text_reader import SharedQueryTextReader
from models.taper_mag.visual_grounding import (
    EditAwareVisualGrounding,
    GatedEditConditioning,
)


@dataclass(frozen=True, slots=True)
class OperatorSet:
    text_reads: Tensor
    visual_reads: Tensor
    operators: Tensor
    conditioned_queries: Tensor
    text_attention: Tensor
    visual_attention: Tensor
    edit_gates: Tensor


class CandidateOperatorGenerator(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_reader = SharedQueryTextReader(
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.edit_conditioning = GatedEditConditioning(d_model=d_model)
        self.grounding = EditAwareVisualGrounding(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(4 * d_model)
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: Tensor,
        text_content_mask: Tensor,
        initial_local: Tensor,
        local_mask: Tensor,
    ) -> OperatorSet:
        text = self.text_reader(text_tokens, text_content_mask)
        query_identities = self.text_reader.queries
        conditioned, gates = self.edit_conditioning(query_identities, text.reads)
        visual = self.grounding(conditioned, initial_local, local_mask)
        fused = torch.cat(
            [
                text.reads,
                visual.grounded,
                text.reads * visual.grounded,
                text.reads - visual.grounded,
            ],
            dim=-1,
        )
        identities = query_identities.unsqueeze(0).expand(text_tokens.shape[0], -1, -1)
        operators = self.output_norm(
            identities + self.fusion(self.fusion_norm(fused))
        )
        return OperatorSet(
            text_reads=text.reads,
            visual_reads=visual.grounded,
            operators=operators,
            conditioned_queries=conditioned,
            text_attention=text.attention,
            visual_attention=visual.attention,
            edit_gates=gates,
        )
