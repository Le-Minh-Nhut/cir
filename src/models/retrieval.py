from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def target_positive_mask(
    target_ids: Sequence[str] | Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Return [B,B] positives, including duplicate target IDs."""

    if isinstance(target_ids, Tensor):
        ids = target_ids.reshape(-1)
        if ids.numel() != batch_size:
            raise ValueError("target_ids Tensor must have B elements")
        ids = ids.to(device=device)
        return ids[:, None].eq(ids[None, :])
    if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes)):
        ids = list(target_ids)
        if len(ids) != batch_size:
            raise ValueError("target_ids sequence must have B elements")
        return torch.tensor(
            [[left == right for right in ids] for left in ids],
            dtype=torch.bool,
            device=device,
        )
    raise TypeError("target_ids must be a Tensor or non-string sequence")


def multi_positive_retrieval_loss(
    query: Tensor,
    targets: Tensor,
    target_ids: Sequence[str] | Tensor,
    *,
    temperature: float,
) -> Tensor:
    """A3.2-compatible query-to-target multi-positive contrastive loss."""

    if query.ndim != 2 or targets.ndim != 2 or query.shape != targets.shape:
        raise ValueError("query and targets must share shape [B,D]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(query, dim=-1) @ F.normalize(targets, dim=-1).T
    logits = logits / temperature
    positive = target_positive_mask(target_ids, len(query), query.device)
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
