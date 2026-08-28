from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TextControlConfig:
    text_dim: int = 768
    retrieval_dim: int = 768


def content_pool(text_tokens: Tensor, text_content_mask: Tensor) -> Tensor:
    if text_tokens.ndim != 3 or text_content_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text tokens/mask must be [B,M,D] and [B,M]")
    mask = text_content_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Every modification requires a content token")
    weights = mask.to(text_tokens.dtype)
    return (text_tokens * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=1, keepdim=True
    )


class ReferenceOnlyControl(nn.Module):
    """M0: exact normalized FG-CLIP2 reference retrieval embedding."""

    def forward(self, reference_global: Tensor) -> Tensor:
        if reference_global.ndim != 2:
            raise ValueError("reference_global must be [B,retrieval_dim]")
        query = torch.nn.functional.normalize(reference_global.float(), dim=-1)
        if not torch.isfinite(query).all():
            raise FloatingPointError("M0 query contains NaN/Inf")
        return query


class TextOnlyControl(nn.Module):
    """M1: online contextual modification text with no reference-image input."""

    def __init__(self, config: TextControlConfig | None = None) -> None:
        super().__init__()
        self.config = config or TextControlConfig()
        self.text_readout = nn.Sequential(
            nn.LayerNorm(self.config.text_dim),
            nn.Linear(self.config.text_dim, self.config.retrieval_dim),
        )

    def forward(self, text_tokens: Tensor, text_content_mask: Tensor) -> Tensor:
        pooled = content_pool(text_tokens, text_content_mask)
        query = torch.nn.functional.normalize(self.text_readout(pooled).float(), dim=-1)
        if query.shape != (text_tokens.shape[0], self.config.retrieval_dim):
            raise RuntimeError("M1 retrieval dimension mismatch")
        return query


class SimpleSumControl(nn.Module):
    """M2: normalized reference plus projected text with one tuned scalar gate."""

    def __init__(self, config: TextControlConfig | None = None) -> None:
        super().__init__()
        self.config = config or TextControlConfig()
        self.text_readout = nn.Sequential(
            nn.LayerNorm(self.config.text_dim),
            nn.Linear(self.config.text_dim, self.config.retrieval_dim),
        )
        self.log_alpha = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def forward(
        self,
        reference_global: Tensor,
        text_tokens: Tensor,
        text_content_mask: Tensor,
    ) -> Tensor:
        if reference_global.shape != (text_tokens.shape[0], self.config.retrieval_dim):
            raise ValueError("reference_global must be [B,retrieval_dim]")
        reference = torch.nn.functional.normalize(reference_global.float(), dim=-1)
        text = self.text_readout(content_pool(text_tokens, text_content_mask)).float()
        query = torch.nn.functional.normalize(reference + self.alpha.float() * text, dim=-1)
        if not torch.isfinite(query).all():
            raise FloatingPointError("M2 query contains NaN/Inf")
        return query
