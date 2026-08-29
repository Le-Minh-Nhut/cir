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


class SemanticFullQueryAnchor(nn.Module):
    """Parameter-free, auxiliary-only composition in FG-CLIP retrieval space.

    Both operands are outputs of the checkpoint's trained semantic projection heads.
    Keeping this composition parameter-free avoids turning a detached random head into
    the relational target for the factor losses.
    """

    def forward(self, reference_global: Tensor, text_semantic_global: Tensor) -> Tensor:
        if reference_global.ndim != 2 or reference_global.shape != text_semantic_global.shape:
            raise ValueError("semantic globals must share [B,D]")
        return F.normalize(reference_global + text_semantic_global, dim=-1)
