from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.taper_mag.state import LocalState


@dataclass(frozen=True, slots=True)
class StateFeatures:
    hidden: Tensor
    transformed: Tensor
    context: Tensor


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    local: Tensor
    support: Tensor
    delta: Tensor
    support_context: Tensor
    delta_norm: Tensor


class SharedLocalExecutor(nn.Module):
    """Shared factorized gated FiLM local editor for all K candidates."""

    def __init__(
        self,
        d_model: int = 256,
        support_bias: float = -1.3862943611198906,
        support_temperature: float = 1.0,
        layerscale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if support_temperature <= 0:
            raise ValueError("support_temperature must be positive")
        self.d_model = d_model
        self.support_temperature = support_temperature
        self.state_norm = nn.LayerNorm(d_model)
        self.state_hidden = nn.Linear(d_model, d_model)
        self.context_hidden = nn.Linear(d_model, d_model)
        self.shared_transform = nn.Linear(d_model, d_model)
        self.operator_support = nn.Linear(d_model, d_model, bias=False)
        self.state_support = nn.Linear(d_model, d_model, bias=False)
        self.support_bias = nn.Parameter(torch.tensor(float(support_bias)))
        self.film = nn.Linear(d_model, 3 * d_model + 1)
        self.layerscale = nn.Parameter(torch.full((d_model,), layerscale_init))
        nn.init.zeros_(self.film.bias)
        nn.init.normal_(self.film.weight, std=0.01)

    def encode_state(self, state: LocalState, context: Tensor) -> StateFeatures:
        hidden = torch.nn.functional.silu(
            self.state_hidden(self.state_norm(state.local))
            + self.context_hidden(context).unsqueeze(1)
        )
        hidden = hidden.masked_fill(~state.local_mask.unsqueeze(-1), 0.0)
        transformed = self.shared_transform(hidden)
        return StateFeatures(hidden=hidden, transformed=transformed, context=context)

    def enumerate(
        self,
        state: LocalState,
        features: StateFeatures,
        operators: Tensor,
        *,
        delta_scale: float | Tensor = 1.0,
    ) -> CandidateBatch:
        if operators.ndim != 3 or operators.shape[0] != state.local.shape[0]:
            raise ValueError("operators must be [B,K,d_model]")
        operator_keys = self.operator_support(operators)
        state_keys = self.state_support(features.hidden)
        support_logits = torch.einsum("bkd,bnd->bkn", operator_keys, state_keys)
        support_logits = support_logits / math.sqrt(self.d_model) + self.support_bias
        support = torch.sigmoid(support_logits / self.support_temperature)
        support = support * state.local_mask[:, None, :].to(support.dtype)

        film = self.film(operators)
        gamma, beta, direction, rho = torch.split(
            film, [self.d_model, self.d_model, self.d_model, 1], dim=-1
        )
        residual = torch.tanh(
            (1.0 + 0.1 * torch.tanh(gamma)).unsqueeze(2)
            * features.transformed.unsqueeze(1)
            + beta.unsqueeze(2)
        ) * torch.tanh(direction).unsqueeze(2)
        scale = torch.as_tensor(delta_scale, device=residual.device, dtype=residual.dtype)
        delta = (
            support.unsqueeze(-1)
            * torch.sigmoid(rho).unsqueeze(2)
            * self.layerscale.view(1, 1, 1, -1)
            * residual
            * scale
        )
        delta = delta * state.local_mask[:, None, :, None].to(delta.dtype)
        candidates = state.local.unsqueeze(1) + delta
        mass = support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        support_context = torch.einsum("bkn,bnd->bkd", support / mass, state.local)
        delta_norm = delta.float().flatten(2).norm(dim=-1)
        return CandidateBatch(candidates, support, delta, support_context, delta_norm)

    @staticmethod
    def gather_selected(
        state: LocalState, candidates: CandidateBatch, actions: Tensor, execute_mask: Tensor
    ) -> LocalState:
        batch, _, tokens, width = candidates.local.shape
        gather_index = actions.clamp_max(candidates.local.shape[1] - 1).view(batch, 1, 1, 1)
        gather_index = gather_index.expand(-1, 1, tokens, width)
        selected = candidates.local.gather(1, gather_index).squeeze(1)
        local = torch.where(execute_mask[:, None, None], selected, state.local)
        return state.with_local(local)

    def recompute_selected(
        self,
        state: LocalState,
        features: StateFeatures,
        operators: Tensor,
        actions: Tensor,
        execute_mask: Tensor,
    ) -> tuple[LocalState, CandidateBatch]:
        """Re-execute one selected operator per sample from the immutable parent."""
        if actions.shape != (state.local.shape[0],):
            raise ValueError("actions must be [B]")
        batch, candidate_count, width = operators.shape
        if candidate_count <= 0 or width != self.d_model:
            raise ValueError("operators must be non-empty [B,K,d_model]")
        selected_index = actions.clamp(max=candidate_count - 1).view(batch, 1, 1)
        selected_operator = operators.gather(
            1, selected_index.expand(-1, 1, width)
        )
        recomputed = self.enumerate(state, features, selected_operator)
        selected = recomputed.local[:, 0]
        local = torch.where(execute_mask[:, None, None], selected, state.local)
        return state.with_local(local), recomputed
