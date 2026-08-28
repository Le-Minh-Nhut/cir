from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OneShotControlConfig:
    text_dim: int = 768
    retrieval_dim: int = 768
    hidden_dim: int = 768
    dropout: float = 0.1


class FGCLIP2OneShotControl(nn.Module):
    """Separate same-backbone reference+text CIR control; never part of TAPER."""

    def __init__(self, config: OneShotControlConfig | None = None) -> None:
        super().__init__()
        self.config = config or OneShotControlConfig()
        cfg = self.config
        self.text_projection = nn.Sequential(
            nn.LayerNorm(cfg.text_dim),
            nn.Linear(cfg.text_dim, cfg.retrieval_dim),
        )
        self.composer = nn.Sequential(
            nn.LayerNorm(4 * cfg.retrieval_dim),
            nn.Linear(4 * cfg.retrieval_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.retrieval_dim),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(2 * cfg.retrieval_dim, cfg.retrieval_dim), nn.Sigmoid()
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        reference_global: Tensor,
        text_tokens: Tensor,
        text_content_mask: Tensor,
    ) -> Tensor:
        batch = reference_global.shape[0]
        if reference_global.shape != (batch, self.config.retrieval_dim):
            raise ValueError("reference_global must be [B,retrieval_dim]")
        if text_tokens.ndim != 3 or text_tokens.shape[0] != batch or text_tokens.shape[-1] != self.config.text_dim:
            raise ValueError("text_tokens must be [B,M,text_dim]")
        if text_content_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_content_mask must be [B,M]")
        mask = text_content_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("Every modification requires a content token")
        weights = mask.to(text_tokens.dtype)
        pooled = (text_tokens * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        )
        text = self.text_projection(pooled)
        reference = torch.nn.functional.normalize(reference_global.float(), dim=-1).to(text.dtype)
        fused = torch.cat(
            [reference, text, reference * text, reference - text], dim=-1
        )
        delta = self.composer(fused)
        gate = self.residual_gate(torch.cat([reference, text], dim=-1))
        query = torch.nn.functional.normalize((reference + gate * delta).float(), dim=-1)
        if query.shape != (batch, self.config.retrieval_dim) or not torch.isfinite(query).all():
            raise RuntimeError("Invalid one-shot retrieval query")
        return query
