from __future__ import annotations

import torch
from torch import Tensor, nn


class HardStopSelector(nn.Module):
    """Hard forward, differentiable backward, deterministic evaluation selector."""

    def __init__(self, temperature: float = 1.0, gumbel_noise: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.gumbel_noise = gumbel_noise

    def forward(self, logits: Tensor, live: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != 2 or live.shape != logits.shape[:1] or live.dtype != torch.bool:
            raise ValueError("logits=[B,K+1] and live=bool[B] are required")
        continuous = (logits / self.temperature).softmax(dim=-1)
        decision_logits = logits
        if self.training and self.gumbel_noise:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            decision_logits = logits - torch.log(-torch.log(uniform))
        indices = decision_logits.argmax(dim=-1)
        stop_index = logits.shape[-1] - 1
        indices = torch.where(live, indices, torch.full_like(indices, stop_index))
        hard = torch.nn.functional.one_hot(indices, logits.shape[-1]).to(logits.dtype)
        if self.training:
            action = hard + continuous - continuous.detach()
        else:
            action = hard
        return action, hard


def select_next_state(
    candidate_states: Tensor,
    current_state: Tensor,
    candidate_queries: Tensor,
    current_query: Tensor,
    action: Tensor,
) -> tuple[Tensor, Tensor]:
    if candidate_states.ndim != 4 or candidate_queries.ndim != 3:
        raise ValueError("candidate states/queries must be rank 4/rank 3")
    all_states = torch.cat([candidate_states, current_state[:, None]], dim=1)
    all_queries = torch.cat([candidate_queries, current_query[:, None]], dim=1)
    if action.shape != all_states.shape[:2] or action.shape != all_queries.shape[:2]:
        raise ValueError("action axis must include candidates plus STOP")
    next_state = torch.einsum("bk,bknd->bnd", action, all_states)
    next_query = torch.einsum("bk,bkd->bd", action, all_queries)
    return next_state, next_query
