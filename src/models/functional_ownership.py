from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import Tensor


def slot_pairs(num_slots: int, device: torch.device) -> Tensor:
    if num_slots < 1:
        raise ValueError("num_slots must be >= 1")
    pairs = list(combinations(range(num_slots), 2))
    if not pairs:
        return torch.empty(0, 2, dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def coalition_masks(num_slots: int, device: torch.device) -> Tensor:
    """Return all ``2**L`` real-slot coalitions in integer-mask order."""
    if num_slots < 1:
        raise ValueError("num_slots must be >= 1")
    return torch.tensor(
        [
            [bool(mask & (1 << slot_id)) for slot_id in range(num_slots)]
            for mask in range(1 << num_slots)
        ],
        dtype=torch.bool,
        device=device,
    )


def coalition_index(coalition_mask: Tensor) -> Tensor:
    """Convert ``[...,L]`` Boolean coalition masks to integer indices."""
    if coalition_mask.ndim < 1 or coalition_mask.dtype != torch.bool:
        raise TypeError("coalition_mask must be a bool Tensor with shape [...,L]")
    powers = 1 << torch.arange(
        coalition_mask.shape[-1],
        dtype=torch.long,
        device=coalition_mask.device,
    )
    return (coalition_mask.to(torch.long) * powers).sum(dim=-1)


def conditional_phi(
    coalition_loss: Tensor,
    base_coalition: Tensor,
    slot_id: int,
) -> Tensor:
    """Exact ``loss(A) - loss(A union {slot})`` for each batch item."""
    if coalition_loss.ndim != 3:
        raise ValueError("coalition_loss must be [B,2^L,H]")
    batch_size, num_coalitions, _ = coalition_loss.shape
    num_slots = int(math.log2(num_coalitions))
    if (1 << num_slots) != num_coalitions:
        raise ValueError("coalition_loss coalition dimension must be a power of two")
    if base_coalition.shape != (batch_size,):
        raise ValueError("base_coalition must be [B]")
    if base_coalition.dtype != torch.long:
        raise TypeError("base_coalition must be torch.long")
    if slot_id < 0 or slot_id >= num_slots:
        raise ValueError("slot_id out of range")
    if ((base_coalition < 0) | (base_coalition >= num_coalitions)).any():
        raise ValueError("base_coalition index out of range")
    bit = 1 << slot_id
    if ((base_coalition & bit) != 0).any():
        raise ValueError("candidate slot is already present in base coalition")
    batch_ids = torch.arange(batch_size, device=coalition_loss.device)
    return (
        coalition_loss[batch_ids, base_coalition]
        - coalition_loss[batch_ids, base_coalition | bit]
    )


@torch.no_grad()
def build_conditional_residual_plan(
    coalition_loss: Tensor,
    candidate_mask: Tensor,
    task_error_rank: Tensor,
    *,
    rank_gate_enabled: bool = True,
    rank_threshold: float = 0.25,
    mode_capacity: int | None = None,
    allow_unassigned_modes: bool = True,
    pair_lookahead: bool = True,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Build exact finite-coalition residual-credit jobs.

    The planner recomputes mode-wise marginal utility after every accepted
    block.  Its decisions and weights are a detached oracle; differentiable
    step queries are constructed separately by :class:`TAPER`.
    """
    if coalition_loss.ndim != 3:
        raise ValueError("coalition_loss must be [B,2^L,H]")
    batch_size, num_coalitions, num_modes = coalition_loss.shape
    num_slots = int(math.log2(num_coalitions))
    if (1 << num_slots) != num_coalitions:
        raise ValueError("coalition_loss coalition dimension must be a power of two")
    if candidate_mask.shape != (batch_size, num_slots):
        raise ValueError("candidate_mask must be [B,L]")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be bool")
    if task_error_rank.shape != (batch_size,):
        raise ValueError("task_error_rank must be [B]")
    if not torch.isfinite(coalition_loss).all():
        raise ValueError("coalition_loss contains NaN or Inf")
    if not torch.isfinite(task_error_rank).all():
        raise ValueError("task_error_rank contains NaN or Inf")
    if not isinstance(rank_gate_enabled, bool):
        raise TypeError("rank_gate_enabled must be bool")
    if not math.isfinite(rank_threshold) or rank_threshold < 0:
        raise ValueError("rank_threshold must be finite and >= 0")
    if mode_capacity is not None and mode_capacity < 1:
        raise ValueError("mode_capacity must be None or >= 1")
    if allow_unassigned_modes is not True:
        raise ValueError("allow_unassigned_modes must be true")
    if not isinstance(pair_lookahead, bool):
        raise TypeError("pair_lookahead must be bool")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    device = coalition_loss.device
    dtype = coalition_loss.dtype
    max_steps = num_slots
    base_indices = torch.full(
        (batch_size, max_steps), -1, dtype=torch.long, device=device
    )
    next_indices = torch.full_like(base_indices, -1)
    new_block_mask = torch.zeros(
        batch_size, max_steps, num_slots, dtype=torch.bool, device=device
    )
    mode_credit = torch.zeros(
        batch_size, max_steps, num_modes, dtype=dtype, device=device
    )
    accepted_mask = torch.zeros(
        batch_size, max_steps, dtype=torch.bool, device=device
    )
    inferred_k_eff = torch.ones(batch_size, dtype=torch.long, device=device)
    stop_no_gain = torch.zeros(batch_size, dtype=torch.bool, device=device)
    unresolved_modes = torch.zeros(
        batch_size, num_modes, dtype=torch.bool, device=device
    )
    clone_rejected = torch.zeros(batch_size, dtype=dtype, device=device)
    clone_candidates = torch.zeros_like(clone_rejected)

    for batch_id in range(batch_size):
        rank = max(float(task_error_rank[batch_id]), 1.0)
        if not rank_gate_enabled:
            k_eff = min(num_slots, num_modes)
        elif rank <= 1.0 + rank_threshold:
            k_eff = 1
        else:
            k_eff = min(num_slots, num_modes, max(1, math.ceil(rank)))
        inferred_k_eff[batch_id] = k_eff
        capacity = (
            num_modes
            if k_eff == 1
            else (
                math.ceil(num_modes / k_eff)
                if mode_capacity is None
                else mode_capacity
            )
        )

        losses = coalition_loss[batch_id]
        empty = losses[0]
        initial_effect = torch.stack(
            [empty - losses[1 << slot_id] for slot_id in range(num_slots)],
            dim=0,
        ).clamp_min(0.0)
        residual_modes = (
            initial_effect
            * candidate_mask[batch_id, :, None].to(initial_effect.dtype)
        ).amax(dim=0) > eps
        if pair_lookahead:
            candidate_ids = candidate_mask[batch_id].nonzero(
                as_tuple=False
            ).flatten().tolist()
            for first, second in combinations(candidate_ids, 2):
                pair_loss = losses[(1 << first) | (1 << second)]
                pair_gain = (empty - pair_loss).clamp_min(0.0)
                interaction = (
                    losses[1 << first]
                    + losses[1 << second]
                    - pair_loss
                    - empty
                ).clamp_min(0.0)
                residual_modes |= (
                    torch.minimum(pair_gain, interaction) > eps
                )
        available = candidate_mask[batch_id].clone()
        coalition = 0
        used_slot_count = 0
        step = 0

        while step < max_steps and used_slot_count < k_eff:
            if not bool(available.any()) or not bool(residual_modes.any()):
                break
            base = losses[coalition]
            conditional = torch.zeros(
                num_slots, num_modes, dtype=dtype, device=device
            )
            for slot_id in range(num_slots):
                if bool(available[slot_id]):
                    conditional[slot_id] = (
                        base - losses[coalition | (1 << slot_id)]
                    ).clamp_min(0.0)

            if step > 0:
                eligible = available[:, None] & residual_modes[None]
                initially_positive = (initial_effect > eps) & eligible
                clone_candidates[batch_id] += initially_positive.sum()
                clone_rejected[batch_id] += (
                    initially_positive & (conditional <= eps)
                ).sum()

            remaining_capacity = torch.full(
                (num_slots,), capacity, dtype=torch.long, device=device
            )
            proposal = _greedy_mode_assignment(
                conditional,
                residual_modes,
                available,
                remaining_capacity,
                eps=eps,
            )
            owner_score = (
                conditional * proposal.to(conditional.dtype)
            ).sum(dim=-1).masked_fill(~available, -1.0)
            owner = int(owner_score.argmax().item())
            singleton_accepted = float(owner_score[owner]) > eps

            block = torch.zeros(num_slots, dtype=torch.bool, device=device)
            credited_modes = torch.zeros(
                num_modes, dtype=torch.bool, device=device
            )
            credit = torch.zeros(num_modes, dtype=dtype, device=device)

            if singleton_accepted:
                block[owner] = True
                credited_modes = proposal[owner] & residual_modes
                credit[credited_modes] = conditional[owner, credited_modes]
            elif pair_lookahead and (k_eff - used_slot_count) >= 2:
                best_score = eps
                best_pair: tuple[int, int] | None = None
                best_modes: Tensor | None = None
                best_gain: Tensor | None = None
                available_ids = available.nonzero(as_tuple=False).flatten().tolist()
                for first, second in combinations(available_ids, 2):
                    pair_index = coalition | (1 << first) | (1 << second)
                    pair_gain = (base - losses[pair_index]).clamp_min(0.0)
                    interaction = (
                        losses[coalition | (1 << first)]
                        + losses[coalition | (1 << second)]
                        - losses[pair_index]
                        - base
                    ).clamp_min(0.0)
                    pair_utility = torch.minimum(pair_gain, interaction)
                    pair_modes = residual_modes & (pair_utility > eps)
                    pair_capacity = min(num_modes, 2 * capacity)
                    if int(pair_modes.sum()) > pair_capacity:
                        mode_ids = pair_utility.masked_fill(
                            ~pair_modes, -1.0
                        ).topk(pair_capacity).indices
                        limited = torch.zeros_like(pair_modes)
                        limited[mode_ids] = True
                        pair_modes = limited
                    score = float(pair_utility[pair_modes].sum())
                    if score > best_score:
                        best_score = score
                        best_pair = (first, second)
                        best_modes = pair_modes
                        best_gain = pair_gain
                if best_pair is not None and best_modes is not None and best_gain is not None:
                    block[list(best_pair)] = True
                    credited_modes = best_modes
                    credit[credited_modes] = best_gain[credited_modes]

            if not bool(block.any()) or not bool(credited_modes.any()):
                stop_no_gain[batch_id] = bool(residual_modes.any())
                break

            next_coalition = coalition
            for slot_id in block.nonzero(as_tuple=False).flatten().tolist():
                next_coalition |= 1 << slot_id
            base_indices[batch_id, step] = coalition
            next_indices[batch_id, step] = next_coalition
            new_block_mask[batch_id, step] = block
            mode_credit[batch_id, step] = credit
            accepted_mask[batch_id, step] = True

            residual_modes &= ~credited_modes
            available &= ~block
            coalition = next_coalition
            used_slot_count += int(block.sum())
            step += 1

        unresolved_modes[batch_id] = residual_modes

    clone_rejection_fraction = torch.where(
        clone_candidates > 0,
        clone_rejected / clone_candidates.clamp_min(1.0),
        torch.zeros_like(clone_rejected),
    )
    return {
        "base_coalition_index": base_indices,
        "next_coalition_index": next_indices,
        "new_block_mask": new_block_mask,
        "mode_credit": mode_credit,
        "accepted_mask": accepted_mask,
        "inferred_k_eff": inferred_k_eff,
        "unresolved_modes": unresolved_modes,
        "stop_no_gain": stop_no_gain,
        "clone_rejection_fraction": clone_rejection_fraction,
    }


def _validate_effects(effects: Tensor, candidate_mask: Tensor) -> None:
    if effects.ndim != 3:
        raise ValueError("functional_effects must be [B,L,H]")
    if candidate_mask.shape != effects.shape[:2]:
        raise ValueError("candidate_mask must be [B,L]")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("candidate_mask must be bool")
    if effects.shape[-1] < 1:
        raise ValueError("functional_effects must contain at least one mode")
    if not torch.isfinite(effects).all():
        raise ValueError("functional_effects contains NaN or Inf")


def _positive_row_similarity(
    effects: Tensor,
    candidate_mask: Tensor,
    eps: float,
) -> Tensor:
    """Mean cosine between useful candidate rows, zero when fewer than two."""
    batch_size, num_slots, _ = effects.shape
    normalized = F.normalize(effects.clamp_min(0.0), dim=-1, eps=eps)
    similarity = normalized @ normalized.transpose(1, 2)
    row_is_positive = effects.clamp_min(0.0).sum(dim=-1) > eps
    pair_mask = (
        candidate_mask[:, :, None]
        & candidate_mask[:, None, :]
        & row_is_positive[:, :, None]
        & row_is_positive[:, None, :]
    )
    upper = torch.triu(
        torch.ones(
            num_slots,
            num_slots,
            dtype=torch.bool,
            device=effects.device,
        ),
        diagonal=1,
    )
    pair_mask &= upper[None]
    pair_count = pair_mask.sum(dim=(1, 2))
    pair_sum = (similarity * pair_mask.to(similarity.dtype)).sum(dim=(1, 2))
    return torch.where(
        pair_count > 0,
        pair_sum / pair_count.clamp_min(1).to(pair_sum.dtype),
        torch.zeros(batch_size, dtype=effects.dtype, device=effects.device),
    )


def _greedy_mode_assignment(
    utility: Tensor,
    unsolved_modes: Tensor,
    available_slots: Tensor,
    remaining_capacity: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Deterministic max-utility proposal with NULL and upper capacities.

    This is a small non-differentiable functional-credit oracle, not token
    routing. Edges are considered by descending positive utility with stable
    slot/mode tie breaking. Every mode has at most one proposed owner.
    """
    num_slots, num_modes = utility.shape
    assignment = torch.zeros(
        num_slots,
        num_modes,
        dtype=torch.bool,
        device=utility.device,
    )
    edges: list[tuple[float, int, int]] = []
    for slot_id in range(num_slots):
        if not bool(available_slots[slot_id]) or remaining_capacity[slot_id] <= 0:
            continue
        for mode_id in range(num_modes):
            value = float(utility[slot_id, mode_id])
            if bool(unsolved_modes[mode_id]) and value > eps:
                edges.append((-value, slot_id, mode_id))
    edges.sort()

    used = torch.zeros_like(remaining_capacity)
    mode_owned = torch.zeros(
        num_modes,
        dtype=torch.bool,
        device=utility.device,
    )
    for _, slot_id, mode_id in edges:
        if bool(mode_owned[mode_id]):
            continue
        if int(used[slot_id]) >= int(remaining_capacity[slot_id]):
            continue
        assignment[slot_id, mode_id] = True
        mode_owned[mode_id] = True
        used[slot_id] += 1
    return assignment


@torch.no_grad()
def functional_mode_assignment(
    functional_effects: Tensor,
    candidate_mask: Tensor,
    task_error_rank: Tensor,
    *,
    rank_gate_enabled: bool = True,
    rank_threshold: float = 0.25,
    mode_capacity: int | None = None,
    allow_unassigned_modes: bool = True,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Assign residual retrieval modes before block residual pursuit.

    ``B[s,j]`` is recomputed after every credited block. In multi-mode
    examples, an adaptive upper capacity prevents the current global atom from
    owning every mode. The subsequent full-row projection removes exact/span
    clones without turning different mode IDs into fake specialization.
    """
    _validate_effects(functional_effects, candidate_mask)
    batch_size, num_slots, num_modes = functional_effects.shape
    if task_error_rank.shape != (batch_size,):
        raise ValueError("task_error_rank must be [B]")
    if not torch.isfinite(task_error_rank).all():
        raise ValueError("task_error_rank contains NaN or Inf")
    if not isinstance(rank_gate_enabled, bool):
        raise TypeError("rank_gate_enabled must be bool")
    if not torch.isfinite(torch.tensor(rank_threshold)) or rank_threshold < 0:
        raise ValueError("rank_threshold must be finite and >= 0")
    if mode_capacity is not None and mode_capacity < 1:
        raise ValueError("mode_capacity must be None or >= 1")
    if allow_unassigned_modes is not True:
        raise ValueError("allow_unassigned_modes must be true")
    if not torch.isfinite(torch.tensor(eps)) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    effects = functional_effects.float().clamp_min(0.0)
    credit = torch.zeros_like(effects)
    assignment = torch.zeros_like(effects, dtype=torch.bool)
    proposal_assignment = torch.zeros_like(assignment)
    credited_mask = torch.zeros_like(candidate_mask)
    credit_order = torch.full(
        (batch_size, num_slots),
        -1,
        dtype=torch.long,
        device=effects.device,
    )
    inferred_k_eff = torch.ones(
        batch_size,
        dtype=torch.long,
        device=effects.device,
    )
    residual_active_modes = torch.zeros(
        batch_size,
        dtype=effects.dtype,
        device=effects.device,
    )

    for batch_id in range(batch_size):
        rank = max(float(task_error_rank[batch_id]), 1.0)
        if not rank_gate_enabled:
            k_eff = min(num_slots, num_modes)
        elif rank <= 1.0 + rank_threshold:
            k_eff = 1
        else:
            k_eff = min(num_slots, num_modes, max(1, math.ceil(rank)))
        inferred_k_eff[batch_id] = k_eff

        if k_eff == 1:
            capacity = num_modes
        elif mode_capacity is None:
            capacity = math.ceil(num_modes / k_eff)
        else:
            capacity = mode_capacity

        residual = effects[batch_id].clone()
        available = candidate_mask[batch_id].clone()
        unsolved = effects[batch_id].amax(dim=0) > eps
        remaining_capacity = torch.full(
            (num_slots,),
            capacity,
            dtype=torch.long,
            device=effects.device,
        )

        for step in range(k_eff):
            if not bool(unsolved.any()) or not bool(available.any()):
                break

            if k_eff == 1:
                score = (
                    residual.clamp_min(0.0)
                    * unsolved[None].to(residual.dtype)
                ).sum(dim=-1)
                score = score.masked_fill(~available, -1.0)
                owner = int(score.argmax().item())
                if float(score[owner]) <= eps:
                    break
                proposal = torch.zeros_like(assignment[batch_id])
                proposal[owner] = unsolved & (residual[owner] > eps)
            else:
                proposal = _greedy_mode_assignment(
                    residual.clamp_min(0.0),
                    unsolved,
                    available,
                    remaining_capacity,
                    eps=eps,
                )
                owner_score = (
                    residual.clamp_min(0.0)
                    * proposal.to(residual.dtype)
                ).sum(dim=-1)
                owner_score = owner_score.masked_fill(~available, -1.0)
                owner = int(owner_score.argmax().item())
                if float(owner_score[owner]) <= eps:
                    break

            if step == 0:
                # B_train is the first rank-gated global job assignment. Later
                # proposals are recomputed acceptance oracles after residual
                # updates, but cannot silently change the training jobs.
                proposal_assignment[batch_id] = proposal
            owned = proposal[owner] & unsolved & (residual[owner] > eps)
            if not bool(owned.any()):
                available[owner] = False
                continue

            assignment[batch_id, owner, owned] = True
            credit[batch_id, owner, owned] = residual[owner, owned]
            credited_mask[batch_id, owner] = True
            credit_order[batch_id, step] = owner
            remaining_capacity[owner] -= owned.sum()
            available[owner] = False
            unsolved &= ~owned

            direction = residual[owner].clamp_min(0.0)
            unit = direction / direction.norm().clamp_min(eps)
            projection = (residual @ unit).unsqueeze(-1) * unit.unsqueeze(0)
            residual = residual - projection
            residual[owner] = 0.0
            if step == 0:
                remaining = residual.clamp_min(0.0) * available[:, None]
                residual_active_modes[batch_id] = (
                    remaining.amax(dim=0) > eps
                ).to(effects.dtype).sum()

    training_assignment = proposal_assignment
    training_credit = effects * training_assignment.to(effects.dtype)
    owned_modes = training_assignment.any(dim=1)
    unique_owned_modes = assignment.any(dim=1)
    positive_modes = effects.amax(dim=1) > eps
    owned_mode_count = owned_modes.sum(dim=1)
    unowned_positive_mode_count = (positive_modes & ~owned_modes).sum(dim=1)
    modes_per_owner = training_assignment.sum(dim=2)
    max_modes_per_owner = modes_per_owner.max(dim=1).values
    credited_owner_count = credited_mask.sum(dim=1)
    multi_mode = inferred_k_eff > 1
    owner_positive_modes = effects.gt(eps).sum(dim=2)
    sole_owner = credited_mask.to(torch.long).argmax(dim=1)
    sole_owner_positive = owner_positive_modes.gather(
        1,
        sole_owner[:, None],
    ).squeeze(1)
    giant_owner = (
        multi_mode
        & (credited_owner_count == 1)
        & (sole_owner_positive > 1)
    )
    unresolved_multimode = multi_mode & (
        (credited_owner_count < inferred_k_eff)
        | ((positive_modes & ~unique_owned_modes).sum(dim=1) > 0)
    )

    total_positive_mass = (
        effects * candidate_mask[:, :, None].to(effects.dtype)
    ).sum(dim=(1, 2))
    credited_mass = credit.sum(dim=(1, 2))
    redundant_fraction = (
        1.0 - credited_mass / total_positive_mass.clamp_min(eps)
    ).clamp(0.0, 1.0)
    redundant_fraction = torch.where(
        total_positive_mass > eps,
        redundant_fraction,
        torch.zeros_like(redundant_fraction),
    )

    return {
        "assignment": assignment,
        "training_assignment": training_assignment,
        "proposal_assignment": proposal_assignment,
        "credit": credit,
        "training_credit": training_credit,
        "credited_mask": credited_mask,
        "credit_order": credit_order,
        "inferred_k_eff": inferred_k_eff,
        "owned_mode_count": owned_mode_count,
        "unowned_positive_mode_count": unowned_positive_mode_count,
        "max_modes_per_owner": max_modes_per_owner,
        "giant_owner": giant_owner,
        "ownership_row_similarity": _positive_row_similarity(
            effects,
            candidate_mask,
            eps,
        ),
        "unresolved_multimode": unresolved_multimode,
        "residual_active_modes": residual_active_modes,
        "unique_mode_coverage": unique_owned_modes.sum(dim=1).to(
            effects.dtype
        ) / float(num_modes),
        "redundant_credit_fraction": redundant_fraction,
    }


@torch.no_grad()
def block_residual_credit(
    functional_effects: Tensor,
    candidate_mask: Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Greedy block Gram-Schmidt credit over hard-negative error modes.

    Positive rows describe mode-wise loss reduction by a slot. Once a row is
    credited, its span is removed from every remaining row. Selection is
    discrete and detached; the returned residual credits are training weights,
    not differentiable ownership predictions.
    """
    _validate_effects(functional_effects, candidate_mask)
    if not torch.isfinite(torch.tensor(eps)) or eps <= 0:
        raise ValueError("eps must be finite and > 0")

    effects = functional_effects.float().clamp_min(0.0)
    batch_size, num_slots, num_modes = effects.shape
    credit = torch.zeros_like(effects)
    credited_mask = torch.zeros_like(candidate_mask)
    order = torch.full(
        (batch_size, num_slots),
        -1,
        dtype=torch.long,
        device=effects.device,
    )
    residual_active_modes = torch.zeros(
        batch_size,
        dtype=effects.dtype,
        device=effects.device,
    )

    for batch_id in range(batch_size):
        residual = effects[batch_id].clone()
        available = candidate_mask[batch_id].clone()
        for step in range(num_slots):
            score = residual.clamp_min(0.0).sum(dim=-1)
            score = score.masked_fill(~available, -1.0)
            slot_id = int(score.argmax().item())
            if float(score[slot_id]) <= eps:
                break

            direction = residual[slot_id].clamp_min(0.0)
            credit[batch_id, slot_id] = direction
            credited_mask[batch_id, slot_id] = True
            order[batch_id, step] = slot_id
            available[slot_id] = False

            unit = direction / direction.norm().clamp_min(eps)
            projection = (residual @ unit).unsqueeze(-1) * unit.unsqueeze(0)
            residual = residual - projection
            residual[slot_id] = 0.0
            if step == 0:
                remaining = residual.clamp_min(0.0) * available[:, None]
                residual_active_modes[batch_id] = (
                    remaining.amax(dim=0) > eps
                ).to(effects.dtype).sum()

    total_positive_mass = (
        effects * candidate_mask[:, :, None].to(effects.dtype)
    ).sum(dim=(1, 2))
    credited_mass = credit.sum(dim=(1, 2))
    redundant_fraction = (
        1.0 - credited_mass / total_positive_mass.clamp_min(eps)
    ).clamp(0.0, 1.0)
    redundant_fraction = torch.where(
        total_positive_mass > eps,
        redundant_fraction,
        torch.zeros_like(redundant_fraction),
    )
    unique_mode_coverage = (
        credit.amax(dim=1) > eps
    ).to(effects.dtype).sum(dim=1) / float(num_modes)

    return {
        "credit": credit,
        "credited_mask": credited_mask,
        "credit_order": order,
        "residual_active_modes": residual_active_modes,
        "unique_mode_coverage": unique_mode_coverage,
        "redundant_credit_fraction": redundant_fraction,
    }


@torch.no_grad()
def pair_synergy_credit(
    empty_loss: Tensor,
    singleton_loss: Tensor,
    pair_loss: Tensor,
    pairs: Tensor,
    candidate_mask: Tensor,
    *,
    available_mode_mask: Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Keep the strongest positive pair interaction for every sample."""
    if empty_loss.ndim != 2:
        raise ValueError("empty_loss must be [B,H]")
    if singleton_loss.ndim != 3:
        raise ValueError("singleton_loss must be [B,L,H]")
    if pair_loss.ndim != 3:
        raise ValueError("pair_loss must be [B,P,H]")
    batch_size, num_slots, num_modes = singleton_loss.shape
    if empty_loss.shape != (batch_size, num_modes):
        raise ValueError("empty_loss shape mismatch")
    if candidate_mask.shape != (batch_size, num_slots):
        raise ValueError("candidate_mask shape mismatch")
    if pairs.shape != (pair_loss.shape[1], 2):
        raise ValueError("pairs must be [P,2] aligned with pair_loss")
    if pair_loss.shape[0] != batch_size or pair_loss.shape[2] != num_modes:
        raise ValueError("pair_loss shape mismatch")
    if available_mode_mask is None:
        available_mode_mask = torch.ones(
            batch_size,
            num_modes,
            dtype=torch.bool,
            device=pair_loss.device,
        )
    elif available_mode_mask.shape != (batch_size, num_modes):
        raise ValueError("available_mode_mask must be [B,H]")

    if pairs.numel() == 0:
        return {
            "credit": torch.zeros_like(pair_loss),
            "credited_mask": torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=pair_loss.device,
            ),
            "synergy_fraction": pair_loss.new_zeros(()),
        }

    first = pairs[:, 0]
    second = pairs[:, 1]
    positive_synergy = (
        singleton_loss[:, first, :]
        + singleton_loss[:, second, :]
        - pair_loss
        - empty_loss[:, None, :]
    ).float().clamp_min(0.0)
    positive_pair_improvement = (
        empty_loss[:, None, :] - pair_loss
    ).float().clamp_min(0.0)
    # Positive interaction alone is insufficient: a pair that is still worse
    # than EMPTY is orthogonal junk, not useful synergy.
    synergy = torch.minimum(positive_synergy, positive_pair_improvement)
    synergy = synergy * available_mode_mask[:, None].to(synergy.dtype)
    pair_is_candidate = candidate_mask[:, first] & candidate_mask[:, second]
    score = synergy.sum(dim=-1).masked_fill(~pair_is_candidate, -1.0)
    winner = score.argmax(dim=1)
    winner_score = score.gather(1, winner[:, None]).squeeze(1)
    active = winner_score > eps

    credit = torch.zeros_like(synergy)
    credited_mask = torch.zeros_like(pair_is_candidate)
    batch_ids = active.nonzero(as_tuple=False).squeeze(1)
    if batch_ids.numel() > 0:
        pair_ids = winner[batch_ids]
        credit[batch_ids, pair_ids] = synergy[batch_ids, pair_ids]
        credited_mask[batch_ids, pair_ids] = True

    return {
        "credit": credit,
        "credited_mask": credited_mask,
        "synergy_fraction": active.to(synergy.dtype).mean(),
    }


def functional_credit_loss(
    singleton_loss: Tensor,
    singleton_credit: Tensor,
    *,
    pair_loss: Tensor | None = None,
    pair_credit: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Mode-weighted loss with gradients only through credited blocks.

    Credits are detached oracle weights. A future cross-fit implementation can
    pass bank-A-derived credits with bank-B losses without changing this API.
    """
    if singleton_loss.shape != singleton_credit.shape:
        raise ValueError("singleton_loss and singleton_credit must match [B,L,H]")
    if singleton_loss.ndim != 3:
        raise ValueError("singleton_loss must be [B,L,H]")

    credit = singleton_credit.detach().clamp_min(0.0)
    block_mass = credit.sum(dim=-1)
    active = block_mass > eps
    normalized = credit / block_mass.clamp_min(eps).unsqueeze(-1)
    block_loss = (normalized * singleton_loss).sum(dim=-1)
    numerator = (block_loss * active.to(block_loss.dtype)).sum()
    denominator = active.sum().to(block_loss.dtype)

    if (pair_loss is None) != (pair_credit is None):
        raise ValueError("pair_loss and pair_credit must be provided together")
    if pair_loss is not None and pair_credit is not None:
        if pair_loss.shape != pair_credit.shape or pair_loss.ndim != 3:
            raise ValueError("pair loss/credit must match [B,P,H]")
        pair_weight = pair_credit.detach().clamp_min(0.0)
        pair_mass = pair_weight.sum(dim=-1)
        pair_active = pair_mass > eps
        pair_normalized = pair_weight / pair_mass.clamp_min(eps).unsqueeze(-1)
        pair_block_loss = (pair_normalized * pair_loss).sum(dim=-1)
        numerator = numerator + (
            pair_block_loss * pair_active.to(pair_block_loss.dtype)
        ).sum()
        denominator = denominator + pair_active.sum().to(block_loss.dtype)

    return numerator / denominator.clamp_min(1.0)


def pairwise_error_modes(
    queries: Tensor,
    candidates: Tensor,
    *,
    margin: float,
    temperature: float,
) -> Tensor:
    """Return unaggregated positive-vs-hard-negative loss [B,V,H]."""
    if queries.ndim != 3:
        raise ValueError("queries must be [B,V,D]")
    if candidates.ndim != 4 or candidates.shape[0] != queries.shape[0]:
        raise ValueError("candidates must be [B,1+H,K,D]")
    if candidates.shape[1] < 2 or candidates.shape[-1] != queries.shape[-1]:
        raise ValueError("candidates must contain one positive and >=1 negative")
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")

    q = F.normalize(queries.float(), dim=-1)
    c = F.normalize(candidates.float(), dim=-1)
    token_scores = torch.einsum("bvd,bckd->bvck", q, c)
    scores = token_scores.amax(dim=-1)
    positive = scores[:, :, :1]
    negatives = scores[:, :, 1:]
    return F.softplus((negatives - positive + margin) / temperature)


@torch.no_grad()
def gradient_error_mode_rank(
    query: Tensor,
    candidates: Tensor,
    *,
    margin: float,
    temperature: float,
    eps: float = 1e-12,
) -> Tensor:
    """Participation-ratio rank of per-negative target-directed gradients."""
    if query.ndim != 2 or candidates.ndim != 4:
        raise ValueError("query/candidates must be [B,D] and [B,1+H,K,D]")
    q = F.normalize(query.float(), dim=-1)
    c = F.normalize(candidates.float(), dim=-1)
    token_scores = torch.einsum("bd,bckd->bck", q, c)
    winner = token_scores.argmax(dim=-1)
    gather = winner[:, :, None, None].expand(-1, -1, 1, c.shape[-1])
    chosen = c.gather(2, gather).squeeze(2)
    scores = token_scores.amax(dim=-1)
    x = (scores[:, 1:] - scores[:, :1] + margin) / temperature
    weight = torch.sigmoid(x) / temperature
    direction = chosen[:, :1] - chosen[:, 1:]
    tangent = direction - (
        direction * q[:, None]
    ).sum(dim=-1, keepdim=True) * q[:, None]
    modes = weight[:, :, None] * tangent
    gram = modes @ modes.transpose(1, 2)
    trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    trace_square = gram.square().sum(dim=(1, 2))
    rank = trace.square() / (trace_square + eps)
    return torch.where(trace > eps, rank, torch.zeros_like(rank))
