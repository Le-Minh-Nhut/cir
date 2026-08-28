from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.taper_mag.executor import CandidateBatch
from models.taper_mag.state import LocalState


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked.float(), dim=dim).to(logits.dtype)
    return probabilities.masked_fill(~mask, 0.0)


@dataclass(frozen=True, slots=True)
class ReadoutOutput:
    context: Tensor
    internal: Tensor
    change: Tensor
    query: Tensor
    state_attention: Tensor
    change_attention: Tensor


@dataclass(frozen=True, slots=True)
class CandidateReadoutOutput:
    internal: Tensor
    change: Tensor
    query: Tensor
    change_attention: Tensor


class ChangeAwareReadout(nn.Module):
    """Retrieval-space anchor plus a deterministic projected local-state change."""

    def __init__(self, d_model: int = 256, retrieval_dim: int = 768) -> None:
        super().__init__()
        self.d_model = d_model
        self.retrieval_dim = retrieval_dim
        self.reference_query = nn.Linear(d_model, d_model)
        self.state_key = nn.Linear(d_model, d_model)
        self.state_value = nn.Linear(d_model, d_model)
        self.reference_context = nn.Linear(d_model, d_model)
        self.local_context = nn.Linear(d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.state_norm = nn.LayerNorm(d_model)
        self.delta_norm = nn.LayerNorm(d_model)
        self.pool_state = nn.Linear(d_model, d_model)
        self.pool_delta = nn.Linear(d_model, d_model)
        self.pool_context = nn.Linear(d_model, d_model)
        self.pool_score = nn.Linear(d_model, 1)
        # Bias-free residual projections are required for exact q_0 == reference_global.
        self.change_projection = nn.Linear(d_model, d_model, bias=False)
        self.retrieval_projection = nn.Linear(d_model, retrieval_dim, bias=False)

    def state_context(self, state: LocalState) -> tuple[Tensor, Tensor]:
        logits = torch.einsum(
            "bd,bnd->bn",
            self.reference_query(state.reference_anchor),
            self.state_key(state.local),
        ) / math.sqrt(self.d_model)
        weights = _masked_softmax(logits, state.local_mask, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, self.state_value(state.local))
        context = self.context_norm(
            self.reference_context(state.reference_anchor) + self.local_context(pooled)
        )
        return context, weights

    def _change(
        self,
        local: Tensor,
        initial_local: Tensor,
        context: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        delta = local - initial_local
        scores = self.pool_score(
            torch.nn.functional.silu(
                self.pool_state(self.state_norm(local))
                + self.pool_delta(self.delta_norm(delta))
                + self.pool_context(context).unsqueeze(-2)
            )
        ).squeeze(-1)
        weights = _masked_softmax(scores, mask, dim=-1)
        pooled_delta = torch.einsum("...n,...nd->...d", weights, delta)
        change = self.change_projection(pooled_delta)
        internal = context + change
        return internal, change, weights

    def forward(self, state: LocalState) -> ReadoutOutput:
        context, state_attention = self.state_context(state)
        internal, change, change_attention = self._change(
            state.local, state.initial_local, context, state.local_mask
        )
        # The exact identity invariant is preserved at Z_t == Z_0.
        query = F.normalize(
            state.reference_global + self.retrieval_projection(change).to(state.reference_global.dtype),
            dim=-1,
        )
        return ReadoutOutput(
            context, internal, change, query, state_attention, change_attention
        )

    def forward_candidates(
        self, state: LocalState, candidates: CandidateBatch
    ) -> CandidateReadoutOutput:
        batch, queries, tokens, width = candidates.local.shape
        local = candidates.local.reshape(batch * queries, tokens, width)
        initial = (
            state.initial_local[:, None]
            .expand(-1, queries, -1, -1)
            .reshape(batch * queries, tokens, width)
        )
        mask = (
            state.local_mask[:, None]
            .expand(-1, queries, -1)
            .reshape(batch * queries, tokens)
        )
        reference_anchor = (
            state.reference_anchor[:, None]
            .expand(-1, queries, -1)
            .reshape(batch * queries, width)
        )
        reference_global = (
            state.reference_global[:, None]
            .expand(-1, queries, -1)
            .reshape(batch * queries, state.reference_global.shape[-1])
        )
        candidate_state = LocalState(
            local=local,
            initial_local=initial,
            local_mask=mask,
            reference_global=reference_global,
            reference_anchor=reference_anchor,
            alive=state.alive[:, None].expand(-1, queries).reshape(-1),
        )
        context, _ = self.state_context(candidate_state)
        internal, change, attention = self._change(local, initial, context, mask)
        internal = internal.reshape(batch, queries, width)
        change = change.reshape(batch, queries, width)
        attention = attention.reshape(batch, queries, tokens)
        reference = state.reference_global[:, None, :]
        query = F.normalize(
            reference + self.retrieval_projection(change).to(reference.dtype), dim=-1
        )
        return CandidateReadoutOutput(internal, change, query, attention)
