from __future__ import annotations

import torch
from entmax import entmax15
from torch import Tensor, nn

from models.iag_srme.outputs import RecurrentStepOutput

from .retrieval import retrieval_energy


def detached_marginal_utilities(
    current_query: Tensor,
    candidate_queries: Tensor,
    target_bank: Tensor,
    positive_mask: Tensor,
    retrieval_temperature: float,
) -> Tensor:
    """Target evaluates consequences but cannot construct or update them through this path."""

    if candidate_queries.ndim != 3:
        raise ValueError("candidate_queries must be [B,K,D]")
    batch_size, candidates, width = candidate_queries.shape
    current_energy = retrieval_energy(
        current_query, target_bank, positive_mask, retrieval_temperature
    )
    flat_candidates = candidate_queries.reshape(batch_size * candidates, width)
    expanded_mask = (
        positive_mask[:, None, :]
        .expand(-1, candidates, -1)
        .reshape(batch_size * candidates, target_bank.shape[0])
    )
    candidate_energy = retrieval_energy(
        flat_candidates, target_bank, expanded_mask, retrieval_temperature
    ).reshape(batch_size, candidates)
    gains = current_energy[:, None] - candidate_energy
    stop = torch.zeros(batch_size, 1, dtype=gains.dtype, device=gains.device)
    return torch.cat([gains, stop], dim=-1).detach()


def entmax15_fenchel_young(scores: Tensor, target_distribution: Tensor) -> Tensor:
    """Fenchel–Young loss for an arbitrary detached 1.5-entmax target distribution."""

    if scores.shape != target_distribution.shape:
        raise ValueError("scores and target_distribution must have identical shape")
    prediction = entmax15(scores, dim=-1)

    def omega_entropy(probabilities: Tensor) -> Tensor:
        return (1.0 - (probabilities * probabilities.clamp_min(0).sqrt()).sum(dim=-1)) / 0.75

    # Ω*(s)+Ω(y)-<s,y>, written using p*=entmax(s). It is zero when y=p* and
    # has gradient p*-y with respect to s.
    return (
        omega_entropy(prediction)
        - omega_entropy(target_distribution)
        + ((prediction - target_distribution) * scores).sum(dim=-1)
    )


class MarginalActionLoss(nn.Module):
    def __init__(
        self,
        retrieval_temperature: float = 0.07,
        utility_temperature: float = 1.0,
        score_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if utility_temperature <= 0 or score_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.retrieval_temperature = retrieval_temperature
        self.utility_temperature = utility_temperature
        self.score_temperature = score_temperature

    def forward(
        self,
        trace: tuple[RecurrentStepOutput, ...],
        target_bank: Tensor,
        positive_mask: Tensor,
    ) -> Tensor:
        losses: list[Tensor] = []
        masks: list[Tensor] = []
        for step in trace:
            utilities = detached_marginal_utilities(
                step.current_query,
                step.candidate_queries,
                target_bank.detach(),
                positive_mask,
                self.retrieval_temperature,
            )
            target_distribution = entmax15(utilities / self.utility_temperature, dim=-1)
            losses.append(
                entmax15_fenchel_young(
                    step.logits_with_stop / self.score_temperature,
                    target_distribution,
                )
            )
            masks.append(step.live_before.to(step.scores.dtype))
        if not losses:
            raise ValueError("trace must contain at least one recurrent step")
        loss_values = torch.stack(losses, dim=1)
        live_mask = torch.stack(masks, dim=1)
        return (loss_values * live_mask).sum() / live_mask.sum().clamp_min(1.0)
