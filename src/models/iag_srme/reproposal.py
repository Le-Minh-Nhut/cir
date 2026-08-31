from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicIntentReproposal(nn.Module):
    """Shared target-free current-state WHAT residual for persistent candidates.

    Base intents query both the current token state and accumulated change, then use
    that evidence to re-read the original token-level modification text. The final
    projection is zero initialized, so the module initially returns the base intents
    exactly while retaining an immediate gradient path into the output projection.
    """

    def __init__(self, width: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.state_norm = nn.LayerNorm(width)
        self.change_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.state_attention = nn.MultiheadAttention(
            width, num_heads, batch_first=True
        )
        self.change_attention = nn.MultiheadAttention(
            width, num_heads, batch_first=True
        )
        self.state_query_projection = nn.Linear(2 * width, width)
        self.text_attention = nn.MultiheadAttention(
            width, num_heads, batch_first=True
        )
        self.residual_hidden = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, width),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(width, width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        base_intents: Tensor,
        current_state: Tensor,
        anchor: Tensor,
        text_tokens: Tensor,
        text_content_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_intents.ndim != 3 or base_intents.shape[-1] != self.width:
            raise ValueError("base_intents must be [B,K,d]")
        if current_state.ndim != 3 or current_state.shape[-1] != self.width:
            raise ValueError("current_state must be [B,N,d]")
        if anchor.shape != current_state.shape:
            raise ValueError("anchor/current_state shape mismatch")
        if text_tokens.ndim != 3 or text_tokens.shape[-1] != self.width:
            raise ValueError("text_tokens must be [B,L,d]")
        if (
            text_content_mask.shape != text_tokens.shape[:2]
            or text_content_mask.dtype != torch.bool
        ):
            raise ValueError("text_content_mask must be boolean [B,L]")
        if not text_content_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one content token")
        batch_size = base_intents.shape[0]
        if current_state.shape[0] != batch_size or text_tokens.shape[0] != batch_size:
            raise ValueError("reproposal batch dimensions must match")

        normalized_state = self.state_norm(current_state)
        state_evidence, _ = self.state_attention(
            query=base_intents,
            key=normalized_state,
            value=normalized_state,
            need_weights=False,
        )
        accumulated_change = self.change_norm(current_state - anchor)
        change_evidence, _ = self.change_attention(
            query=base_intents,
            key=accumulated_change,
            value=accumulated_change,
            need_weights=False,
        )
        text_query = base_intents + self.state_query_projection(
            torch.cat([state_evidence, change_evidence], dim=-1)
        )
        normalized_text = self.text_norm(text_tokens)
        text_evidence, _ = self.text_attention(
            query=text_query,
            key=normalized_text,
            value=normalized_text,
            key_padding_mask=~text_content_mask,
            need_weights=False,
        )
        hidden = self.residual_hidden(
            torch.cat(
                [base_intents, state_evidence, change_evidence, text_evidence],
                dim=-1,
            )
        )
        residual = self.output_projection(hidden)
        current_intents = base_intents + residual
        if current_intents.shape != base_intents.shape:
            raise AssertionError("dynamic intent shape invariant failed")
        if not torch.isfinite(current_intents).all():
            raise FloatingPointError("dynamic intent reproposal produced non-finite values")
        return current_intents, residual
