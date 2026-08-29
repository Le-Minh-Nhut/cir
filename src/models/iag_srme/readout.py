from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from numerics import fp32_if_low_precision, normalize_fp32


def cap_vector(vector: Tensor, cap: float, epsilon: float = 1e-8) -> Tensor:
    with torch.autocast(device_type=vector.device.type, enabled=False):
        working = fp32_if_low_precision(vector)
        norm = working.norm(dim=-1, keepdim=True)
        bounded = cap * torch.tanh(norm / cap) * working / norm.clamp_min(epsilon)
    return bounded.to(vector.dtype)


class TokenStateReadout(nn.Module):
    """Retrieval displacement can only arise from accumulated token change."""

    def __init__(self, width: int = 256, retrieval_dim: int = 512, query_cap: float = 0.5) -> None:
        super().__init__()
        if query_cap <= 0:
            raise ValueError("query_cap must be positive")
        self.width = width
        self.retrieval_dim = retrieval_dim
        self.query_cap = query_cap
        self.state_norm = nn.LayerNorm(width)
        self.state_projection = nn.Linear(width, width, bias=False)
        self.change_projection = nn.Linear(width, width, bias=False)
        self.text_projection = nn.Linear(width, width, bias=False)
        self.attention_score = nn.Linear(width, 1, bias=False)
        self.output_projection = nn.Linear(width, retrieval_dim, bias=False)

    def _readout_flat(
        self, state: Tensor, anchor: Tensor, text_global: Tensor, reference_global: Tensor
    ) -> Tensor:
        if state.ndim != 3 or anchor.shape != state.shape:
            raise ValueError("state and anchor must be equal [B,N,d]")
        if text_global.shape != (state.shape[0], self.width):
            raise ValueError("text_global must be [B,d]")
        if reference_global.shape != (state.shape[0], self.retrieval_dim):
            raise ValueError("reference_global must be [B,D]")
        displacement = state - anchor
        hidden = F.silu(
            self.state_projection(self.state_norm(state))
            + self.change_projection(displacement)
            + self.text_projection(text_global)[:, None, :]
        )
        weights = self.attention_score(hidden).squeeze(-1).softmax(dim=-1)
        pooled_change = self.output_projection(torch.einsum("bn,bnd->bd", weights, displacement))
        bounded_change = cap_vector(pooled_change, self.query_cap)
        return normalize_fp32(reference_global + bounded_change, dim=-1)

    def forward(
        self, state: Tensor, anchor: Tensor, text_global: Tensor, reference_global: Tensor
    ) -> Tensor:
        if state.ndim == 3:
            return self._readout_flat(state, anchor, text_global, reference_global)
        if state.ndim != 4:
            raise ValueError("state must be [B,N,d] or [B,K,N,d]")
        batch_size, candidates, tokens, width = state.shape
        if anchor.shape != (batch_size, tokens, width):
            raise ValueError("anchor must be [B,N,d]")
        flat_state = state.reshape(batch_size * candidates, tokens, width)
        flat_anchor = anchor[:, None].expand(-1, candidates, -1, -1).reshape_as(flat_state)
        flat_text = (
            text_global[:, None].expand(-1, candidates, -1).reshape(batch_size * candidates, -1)
        )
        flat_reference = (
            reference_global[:, None]
            .expand(-1, candidates, -1)
            .reshape(batch_size * candidates, -1)
        )
        result = self._readout_flat(flat_state, flat_anchor, flat_text, flat_reference)
        return result.reshape(batch_size, candidates, self.retrieval_dim)
