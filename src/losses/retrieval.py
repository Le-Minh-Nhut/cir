from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from numerics import fp32_if_low_precision


def positive_mask_from_ids(target_ids: Sequence[str], device: torch.device) -> Tensor:
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    return torch.tensor(
        [[left == right for right in target_ids] for left in target_ids],
        dtype=torch.bool,
        device=device,
    )


def retrieval_energy(
    queries: Tensor,
    target_bank: Tensor,
    positive_mask: Tensor,
    temperature: float,
) -> Tensor:
    """Per-query multi-positive contrastive energy against one shared bank."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if queries.ndim != 2 or target_bank.ndim != 2 or queries.shape[-1] != target_bank.shape[-1]:
        raise ValueError("queries and target_bank must be [Q,D] and [G,D]")
    if positive_mask.shape != (queries.shape[0], target_bank.shape[0]):
        raise ValueError("positive_mask must be [Q,G]")
    if not positive_mask.any(dim=-1).all():
        raise ValueError("every query needs at least one positive")
    # Similarity and logsumexp remain FP32 under AMP; gradients still reach both branches.
    with torch.autocast(device_type=queries.device.type, enabled=False):
        query_values = fp32_if_low_precision(queries)
        target_values = fp32_if_low_precision(target_bank)
        logits = (
            F.normalize(query_values, dim=-1)
            @ F.normalize(target_values, dim=-1).T
            / temperature
        )
        positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
        return torch.logsumexp(logits, dim=-1) - torch.logsumexp(
            positive_logits, dim=-1
        )


class TerminalRetrievalLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, bidirectional: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(
        self, queries: Tensor, targets: Tensor, positive_mask: Tensor | None = None
    ) -> Tensor:
        if queries.shape != targets.shape or queries.ndim != 2:
            raise ValueError("in-batch queries and targets must share [B,D]")
        batch_size = queries.shape[0]
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, dtype=torch.bool, device=queries.device)
        forward = retrieval_energy(queries, targets, positive_mask, self.temperature).mean()
        if not self.bidirectional:
            return forward
        reverse = retrieval_energy(targets, queries, positive_mask.T, self.temperature).mean()
        return 0.5 * (forward + reverse)
