from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from models.taper_mag.contracts import SupervisionBatch
from training.negative_bank import CommonNegativeSet


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    raw_gain: Tensor
    net_values: Tensor
    negative_ids: tuple[tuple[str, ...], ...]


class MarginalGainTeacher:
    def __init__(self, retrieval_temperature: float = 0.07) -> None:
        if retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")
        self.retrieval_temperature = retrieval_temperature

    @torch.no_grad()
    def score(
        self,
        current_query: Tensor,
        candidate_queries: Tensor,
        supervision: SupervisionBatch,
        negatives: CommonNegativeSet,
        *,
        step_cost: float = 0.0,
    ) -> TeacherOutput:
        if candidate_queries.ndim != 3 or candidate_queries.shape[0] != current_query.shape[0]:
            raise ValueError("candidate_queries must be [B,K,D]")
        batch, queries, retrieval_dim = candidate_queries.shape
        supervision.validate(batch, retrieval_dim)
        negatives.validate(batch, retrieval_dim)
        current = F.normalize(current_query.detach().float(), dim=-1)
        candidates = F.normalize(candidate_queries.detach().float(), dim=-1)
        positive = F.normalize(supervision.target_embedding.detach().float(), dim=-1)
        negative = F.normalize(negatives.embeddings.detach().float(), dim=-1)

        current_positive = torch.einsum("bd,bd->b", current, positive).unsqueeze(-1)
        current_negative = torch.einsum("bd,bhd->bh", current, negative)
        current_logits = torch.cat([current_positive, current_negative], dim=-1)
        current_loss = -torch.log_softmax(
            current_logits / self.retrieval_temperature, dim=-1
        )[:, 0]

        candidate_positive = torch.einsum("bkd,bd->bk", candidates, positive).unsqueeze(-1)
        candidate_negative = torch.einsum("bkd,bhd->bkh", candidates, negative)
        candidate_logits = torch.cat([candidate_positive, candidate_negative], dim=-1)
        candidate_loss = -torch.log_softmax(
            candidate_logits / self.retrieval_temperature, dim=-1
        )[..., 0]
        raw_gain = current_loss[:, None] - candidate_loss
        stop = torch.zeros(batch, 1, device=raw_gain.device, dtype=raw_gain.dtype)
        net_values = torch.cat([raw_gain - step_cost, stop], dim=-1)
        return TeacherOutput(raw_gain.detach(), net_values.detach(), negatives.ids)
