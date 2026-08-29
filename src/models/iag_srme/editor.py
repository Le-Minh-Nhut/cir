from __future__ import annotations

import torch
from torch import Tensor, nn


class SharedTokenEditor(nn.Module):
    """One bounded executor shared by every candidate and timestep."""

    def __init__(self, width: int = 256, lambda_z: float = 0.1, epsilon: float = 1e-8) -> None:
        super().__init__()
        if not 0.0 < lambda_z <= 1.0:
            raise ValueError("lambda_z must be in (0,1]")
        self.lambda_z = lambda_z
        self.epsilon = epsilon
        self.state_norm = nn.LayerNorm(width)
        self.anchor_norm = nn.LayerNorm(width)
        self.state_projection = nn.Linear(width, width, bias=False)
        self.anchor_projection = nn.Linear(width, width, bias=False)
        self.context_projection = nn.Linear(width, width, bias=False)
        self.change_projection = nn.Linear(width, width, bias=False)
        self.direction = nn.Linear(width, width)

    def forward(
        self, contexts: Tensor, supports: Tensor, anchor: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        if contexts.ndim != 3 or supports.ndim != 3 or anchor.shape != state.shape:
            raise ValueError("contexts/supports rank 3 and equal anchor/state are required")
        batch_size, candidates, tokens = supports.shape
        width = state.shape[-1]
        if contexts.shape != (batch_size, candidates, width):
            raise ValueError("contexts must be [B,K,d]")
        if anchor.shape[:2] != (batch_size, tokens):
            raise ValueError("support token axis must match anchor/state")
        hidden = (
            self.state_projection(self.state_norm(state))[:, None, :, :]
            + self.anchor_projection(self.anchor_norm(anchor))[:, None, :, :]
            + self.context_projection(contexts)[:, :, None, :]
            + self.change_projection(state - anchor)[:, None, :, :]
        )
        direction = torch.tanh(self.direction(torch.nn.functional.silu(hidden)))
        support_gate = supports / supports.amax(dim=-1, keepdim=True).clamp_min(self.epsilon)
        delta_z = self.lambda_z * support_gate[..., None] * direction
        candidate_states = state[:, None, :, :] + delta_z
        if delta_z.shape != (batch_size, candidates, tokens, width):
            raise AssertionError("delta shape invariant failed")
        return delta_z, candidate_states
