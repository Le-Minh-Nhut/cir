from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TextReadOutput:
    reads: Tensor
    attention: Tensor


class PreNormCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, queries: Tensor, memory: Tensor, memory_mask: Tensor) -> tuple[Tensor, Tensor]:
        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError("queries and memory must be rank-3")
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("memory_mask must be bool [B,M]")
        if not memory_mask.any(dim=1).all():
            raise ValueError("Cross-attention received an all-masked sample")
        attended, weights = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = queries + attended
        hidden = self.output_norm(hidden + self.ffn(self.ffn_norm(hidden)))
        # [B,H,Q,M] -> [B,Q,M]. This remains independently normalized over M.
        weights = weights.mean(dim=1)
        weights = weights.masked_fill(~memory_mask[:, None, :], 0.0)
        return hidden, weights


class SharedQueryTextReader(nn.Module):
    """K independent shared queries reading the complete instruction.

    There is intentionally no normalization, ownership, routing, or competition over K.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.orthogonal_(self.queries)
        self.block = PreNormCrossAttentionBlock(
            d_model, num_heads=num_heads, ffn_multiplier=4, dropout=dropout
        )

    def forward(self, text_tokens: Tensor, content_mask: Tensor) -> TextReadOutput:
        batch = text_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        reads, attention = self.block(queries, text_tokens, content_mask.bool())
        return TextReadOutput(reads=reads, attention=attention)
