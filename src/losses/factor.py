from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class RelationalGeometry:
    full_distribution: Tensor  # [B,B]
    factor_distribution: Tensor  # [B,K,B]
    log_factor_distribution: Tensor  # [B,K,B]
    all_factor_distribution: Tensor  # [B,B]
    all_factor_error: Tensor  # [B]


def kl_divergence(probability: Tensor, approximation: Tensor, epsilon: float = 1e-8) -> Tensor:
    return (
        probability
        * (probability.clamp_min(epsilon).log() - approximation.clamp_min(epsilon).log())
    ).sum(dim=-1)


def relational_geometry(
    factors: Tensor,
    auxiliary_anchor: Tensor,
    *,
    anchor_temperature: float,
    factor_temperature: float,
    self_masked: bool = False,
    active_weights: Tensor | None = None,
    epsilon: float = 1e-8,
) -> RelationalGeometry:
    if factors.ndim != 3 or auxiliary_anchor.ndim != 2:
        raise ValueError("factors=[B,K,D] and auxiliary_anchor=[B,D] required")
    if factors.shape[0] != auxiliary_anchor.shape[0] or factors.shape[-1] != auxiliary_anchor.shape[-1]:
        raise ValueError("factor/anchor batch and dimensions must match")
    if anchor_temperature <= 0 or factor_temperature <= 0:
        raise ValueError("temperatures must be positive")
    # First implementation deliberately stops the powerful auxiliary anchor.
    anchors = F.normalize(auxiliary_anchor.detach(), dim=-1)
    normalized_factors = F.normalize(factors, dim=-1)
    full_logits = anchors @ anchors.T / anchor_temperature
    factor_logits = torch.einsum("bkd,jd->bkj", normalized_factors, anchors) / factor_temperature
    if self_masked:
        if factors.shape[0] < 2:
            raise ValueError("self-masked relational geometry requires batch size >= 2")
        diagonal = torch.eye(factors.shape[0], dtype=torch.bool, device=factors.device)
        full_logits = full_logits.masked_fill(diagonal, -torch.inf)
        factor_logits = factor_logits.masked_fill(diagonal[:, None, :], -torch.inf)
    full_distribution = full_logits.softmax(dim=-1)
    log_factor = factor_logits.log_softmax(dim=-1)
    factor_distribution = log_factor.exp()
    if active_weights is None:
        all_logits = log_factor.mean(dim=1)
    else:
        if active_weights.shape != factors.shape[:2]:
            raise ValueError("active_weights must be [B,K]")
        weights = active_weights.clamp_min(0).to(factors.dtype)
        if not weights.sum(dim=1).gt(0).all():
            raise ValueError("each sample needs at least one active factor")
        all_logits = (weights[..., None] * log_factor).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(epsilon)
    all_distribution = all_logits.softmax(dim=-1)
    all_error = kl_divergence(full_distribution, all_distribution, epsilon)
    return RelationalGeometry(
        full_distribution,
        factor_distribution,
        log_factor,
        all_distribution,
        all_error,
    )


class FactorCompletenessLoss(nn.Module):
    def __init__(
        self,
        anchor_temperature: float = 0.1,
        factor_temperature: float = 0.1,
        self_masked: bool = False,
    ) -> None:
        super().__init__()
        self.anchor_temperature = anchor_temperature
        self.factor_temperature = factor_temperature
        self.self_masked = self_masked

    def forward(
        self,
        factors: Tensor,
        auxiliary_anchor: Tensor,
        active_weights: Tensor | None = None,
    ) -> tuple[Tensor, RelationalGeometry]:
        geometry = relational_geometry(
            factors,
            auxiliary_anchor,
            anchor_temperature=self.anchor_temperature,
            factor_temperature=self.factor_temperature,
            self_masked=self.self_masked,
            active_weights=active_weights,
        )
        return geometry.all_factor_error.mean(), geometry

