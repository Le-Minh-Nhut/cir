from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_dac_candidate_count(num_candidates: int) -> int:
    """Validate the balanced binary hierarchy required by paper DAC."""

    if num_candidates <= 0 or num_candidates & (num_candidates - 1):
        raise ValueError("DAC candidate count must be a positive power of two")
    return int(math.log2(num_candidates))


def dac_stage(
    global_step: int, split_interval_steps: int, num_candidates: int
) -> int:
    """Return the zero-based DAC stage for completed optimizer updates."""

    max_stage = validate_dac_candidate_count(num_candidates)
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if split_interval_steps <= 0:
        raise ValueError("DAC split interval must be positive")
    return min(global_step // split_interval_steps, max_stage)


def dac_groups(num_candidates: int, stage: int) -> tuple[tuple[int, ...], ...]:
    """Build the fixed contiguous binary partition for a DAC stage."""

    max_stage = validate_dac_candidate_count(num_candidates)
    if stage < 0:
        raise ValueError("DAC stage must be non-negative")
    effective_stage = min(stage, max_stage)
    group_size = num_candidates // (2**effective_stage)
    return tuple(
        tuple(range(start, start + group_size))
        for start in range(0, num_candidates, group_size)
    )


def dac_responsibility_weights(
    candidate_losses: Tensor, stage: int
) -> tuple[Tensor, Tensor]:
    """Return detached uniform weights over each row's winning DAC group."""

    if candidate_losses.ndim < 1 or candidate_losses.shape[-1] == 0:
        raise ValueError("candidate losses must have a non-empty candidate axis")
    num_candidates = candidate_losses.shape[-1]
    groups = dac_groups(num_candidates, stage)
    group_size = len(groups[0])

    detached_losses = candidate_losses.detach().float()
    winner = detached_losses.argmin(dim=-1)
    winning_groups = torch.div(winner, group_size, rounding_mode="floor")
    group_start = winning_groups * group_size
    candidate_index = torch.arange(
        num_candidates, device=candidate_losses.device
    ).view(*([1] * (candidate_losses.ndim - 1)), num_candidates)
    in_group = (candidate_index >= group_start.unsqueeze(-1)) & (
        candidate_index < (group_start + group_size).unsqueeze(-1)
    )
    weights = in_group.to(dtype=torch.float32) / float(group_size)
    return weights.detach(), winning_groups.detach()
