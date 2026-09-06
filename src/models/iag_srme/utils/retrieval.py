from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def build_teacher_masks(
    target_ids: Sequence[str | None],
    device: torch.device,
    relevance: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build identity-aware positives and false-negative-safe negatives."""

    valid = torch.tensor([value is not None for value in target_ids], device=device)
    positive = torch.tensor(
        [
            [left is not None and right is not None and left == right for right in target_ids]
            for left in target_ids
        ],
        dtype=torch.bool,
        device=device,
    )
    if relevance is not None:
        positive = positive | relevance.to(device=device, dtype=torch.bool)
    positive = positive & valid[:, None] & valid[None, :]
    negative = valid[None, :] & ~positive
    return positive, negative, valid


def teacher_retrieval_loss(
    query: Tensor,
    targets: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
    temperature: float,
) -> Tensor:
    """Multi-positive loss for [B,D] or sibling queries [B,K,D], always in FP32."""

    leading = query.shape[:-1]
    flat = query.reshape(-1, query.shape[-1])
    repeats = flat.shape[0] // positive_mask.shape[0]
    positive = positive_mask[:, None].expand(-1, repeats, -1).reshape(flat.shape[0], -1)
    negative = negative_mask[:, None].expand(-1, repeats, -1).reshape(flat.shape[0], -1)
    denominator = positive | negative
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = F.normalize(flat.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T
        logits = logits / temperature
        positive_lse = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=-1)
        total_lse = torch.logsumexp(logits.masked_fill(~denominator, -torch.inf), dim=-1)
    return (total_lse - positive_lse).reshape(leading)


@torch.no_grad()
def marginal_teacher_utilities(
    current_query: Tensor,
    candidate_queries: Tensor,
    targets: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Detached one-step gain against one common parent/sibling evaluator pool."""

    current_loss = teacher_retrieval_loss(
        current_query.float(),
        targets.float(),
        positive_mask,
        negative_mask,
        temperature,
    )
    candidate_loss = teacher_retrieval_loss(
        candidate_queries.float(),
        targets.float(),
        positive_mask,
        negative_mask,
        temperature,
    )
    valid_rows = positive_mask.any(dim=-1) & negative_mask.any(dim=-1)
    return (current_loss[:, None] - candidate_loss).detach(), valid_rows
