from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TextIntentEncoder(nn.Module):
    """Four stable query identities independently read every valid text token."""

    def __init__(self, width: int = 256, num_candidates: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.num_candidates = num_candidates
        self.query_bank = nn.Parameter(torch.empty(num_candidates, width))
        nn.init.normal_(self.query_bank, std=width**-0.5)
        self.cross_attention = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, text_tokens: Tensor, content_mask: Tensor) -> Tensor:
        if text_tokens.ndim != 3 or text_tokens.shape[-1] != self.width:
            raise ValueError("text_tokens must be [B,L,d]")
        if content_mask.shape != text_tokens.shape[:2] or content_mask.dtype != torch.bool:
            raise ValueError("content_mask must be boolean [B,L]")
        if not content_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one content token")
        batch_size = text_tokens.shape[0]
        queries = self.query_bank.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            query=queries,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~content_mask,
            need_weights=False,
        )
        residual = queries + attended
        intents = self.output_norm(residual + self.ffn(residual))
        if intents.shape != (batch_size, self.num_candidates, self.width):
            raise AssertionError("intent shape invariant failed")
        return intents


class SemanticClaimHead(nn.Module):
    """Independent sigmoid compatibility; deliberately not the intent attention."""

    def __init__(self, width: int = 256, claim_width: int | None = None) -> None:
        super().__init__()
        claim_width = claim_width or width
        self.query_projection = nn.Linear(width, claim_width, bias=False)
        self.token_projection = nn.Linear(width, claim_width, bias=False)
        self.scale = math.sqrt(claim_width)

    def forward(self, intents: Tensor, text_tokens: Tensor, content_mask: Tensor) -> Tensor:
        if intents.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("intents and text_tokens must be rank 3")
        if intents.shape[0] != text_tokens.shape[0] or intents.shape[-1] != text_tokens.shape[-1]:
            raise ValueError("intent/text batch and width must match")
        if content_mask.shape != text_tokens.shape[:2]:
            raise ValueError("content_mask must be [B,L]")
        logits = torch.einsum(
            "bkd,bld->bkl",
            self.query_projection(intents),
            self.token_projection(text_tokens),
        ) / self.scale
        return logits.masked_fill(~content_mask[:, None, :], torch.finfo(logits.dtype).min)


def masked_text_mean(text_tokens: Tensor, content_mask: Tensor) -> Tensor:
    weights = content_mask.to(text_tokens.dtype)
    return torch.einsum("bl,bld->bd", weights, text_tokens) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

