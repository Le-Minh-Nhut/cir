from __future__ import annotations

import torch
from torch import Tensor, nn

from .factor import RelationalGeometry, kl_divergence


def leave_one_out_logits(
    log_factor_distribution: Tensor,
    active_weights: Tensor | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """Correct geometric-mean leave-one-out algebra, including optional activity weights."""

    if log_factor_distribution.ndim != 3:
        raise ValueError("log_factor_distribution must be [B,K,G]")
    candidates = log_factor_distribution.shape[1]
    if candidates < 2:
        raise ValueError("leave-one-out needs at least two candidates")
    if active_weights is None:
        total_log = log_factor_distribution.sum(dim=1, keepdim=True)
        return (total_log - log_factor_distribution) / (candidates - 1)
    if active_weights.shape != log_factor_distribution.shape[:2]:
        raise ValueError("active_weights must be [B,K]")
    weights = active_weights.clamp_min(0).to(log_factor_distribution.dtype)
    weighted = weights[..., None] * log_factor_distribution
    total_log = weighted.sum(dim=1, keepdim=True)
    remaining_weight = weights.sum(dim=1, keepdim=True)[:, :, None] - weights[..., None]
    if (remaining_weight <= 0).any():
        raise ValueError("every evaluated leave-one-out set must retain active factor mass")
    return (total_log - weighted) / remaining_weight.clamp_min(epsilon)


class UniqueContributionLoss(nn.Module):
    """Optional necessity loss; callers must define activity for variable-factor data."""

    def __init__(self, margin: float = 0.05) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        self.margin = margin

    def forward(
        self,
        geometry: RelationalGeometry,
        active_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        loo_logits = leave_one_out_logits(
            geometry.log_factor_distribution, active_weights=active_weights
        )
        loo_distribution = loo_logits.softmax(dim=-1)
        full = geometry.full_distribution[:, None, :].expand_as(loo_distribution)
        loo_error = kl_divergence(full, loo_distribution)
        contribution = loo_error - geometry.all_factor_error[:, None]
        hinge = torch.relu(self.margin - contribution)
        if active_weights is None:
            return hinge.mean(), contribution
        weights = active_weights.clamp_min(0).to(hinge.dtype)
        return (hinge * weights).sum() / weights.sum().clamp_min(1.0), contribution
