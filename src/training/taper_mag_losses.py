from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def stop_anchored_listwise_utility_loss(
    predicted_gain: Tensor,
    teacher_raw_gain: Tensor,
    *,
    step_cost: float = 0.0,
    teacher_temperature: float = 1.0,
    policy_temperature: float = 1.0,
) -> Tensor:
    if predicted_gain.shape != teacher_raw_gain.shape:
        raise ValueError("predicted and teacher gains must have identical [*,K] shape")
    if teacher_temperature <= 0 or policy_temperature <= 0:
        raise ValueError("Utility temperatures must be positive")
    stop = torch.zeros((*predicted_gain.shape[:-1], 1), device=predicted_gain.device)
    teacher_values = torch.cat(
        [teacher_raw_gain.detach().float() - step_cost, stop.float()], dim=-1
    )
    predicted_values = torch.cat(
        [predicted_gain.float() - step_cost, stop.float()], dim=-1
    )
    targets = torch.softmax(teacher_values / teacher_temperature, dim=-1)
    log_probabilities = torch.log_softmax(predicted_values / policy_temperature, dim=-1)
    return F.kl_div(log_probabilities, targets, reduction="batchmean")


def _positive_mask(
    target_ids: tuple[str, ...], positive_ids: tuple[tuple[str, ...], ...], device: torch.device
) -> Tensor:
    batch = len(target_ids)
    mask = torch.zeros(batch, batch, dtype=torch.bool, device=device)
    for query_index, positives in enumerate(positive_ids):
        allowed = set(positives) | {target_ids[query_index]}
        for target_index, target_id in enumerate(target_ids):
            if target_id in allowed:
                mask[query_index, target_index] = True
    if not mask.any(dim=1).all() or not mask.any(dim=0).all():
        raise ValueError("Every query and target requires at least one positive")
    return mask


def terminal_bidirectional_infonce(
    query: Tensor,
    target: Tensor,
    target_ids: tuple[str, ...],
    positive_ids: tuple[tuple[str, ...], ...],
    *,
    temperature: float = 0.07,
) -> Tensor:
    if query.shape != target.shape or query.ndim != 2:
        raise ValueError("query and target must be matching [B,D] tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = F.normalize(query.float(), dim=-1) @ F.normalize(target.float(), dim=-1).T
    logits = logits / temperature
    mask = _positive_mask(target_ids, positive_ids, logits.device)
    denominator_q = torch.logsumexp(logits, dim=1)
    numerator_q = torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)
    denominator_i = torch.logsumexp(logits, dim=0)
    numerator_i = torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=0)
    return 0.5 * ((denominator_q - numerator_q).mean() + (denominator_i - numerator_i).mean())
