from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from diagnostics.geometry import feature_geometry


def coalition_oracle_values(utility: Tensor) -> Tensor:
    """Return v(S)=max(0,max_{k in S} u_k) for every one of 2**K subsets."""

    if utility.ndim != 2:
        raise ValueError("utility must be [N,K]")
    values = utility.detach().double().cpu()
    rows, candidates = values.shape
    coalition = values.new_zeros((rows, 1 << candidates))
    for subset in range(1, 1 << candidates):
        members = [slot for slot in range(candidates) if subset & (1 << slot)]
        coalition[:, subset] = values[:, members].max(dim=-1).values.clamp_min(0)
    return coalition


def exact_shapley_values(utility: Tensor) -> tuple[Tensor, Tensor]:
    """Exact per-row Shapley values for the STOP-anchored max-utility game."""

    coalition = coalition_oracle_values(utility)
    rows, candidates = utility.shape
    shapley = coalition.new_zeros((rows, candidates))
    factorial = [math.factorial(value) for value in range(candidates + 1)]
    denominator = factorial[candidates]
    for slot in range(candidates):
        slot_bit = 1 << slot
        for subset in range(1 << candidates):
            if subset & slot_bit:
                continue
            size = subset.bit_count()
            weight = factorial[size] * factorial[candidates - size - 1] / denominator
            shapley[:, slot] += weight * (
                coalition[:, subset | slot_bit] - coalition[:, subset]
            )
    return shapley, coalition


def _correlation_matrix(values: Tensor, epsilon: float = 1e-12) -> Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered
    scale = covariance.diag().clamp_min(0).sqrt()
    denominator = scale[:, None] * scale[None, :]
    nan = torch.full_like(covariance, float("nan"))
    return torch.where(denominator > epsilon, covariance / denominator, nan)


def unique_advantage(utility: Tensor) -> Tensor:
    """Utility over the best alternative candidate or STOP, per row and slot."""

    if utility.ndim != 2 or utility.shape[1] < 2:
        raise ValueError("unique advantage requires [N,K] with K >= 2")
    values = utility.detach().double().cpu()
    result = torch.empty_like(values)
    for slot in range(values.shape[1]):
        other = torch.cat([values[:, :slot], values[:, slot + 1 :]], dim=1)
        competitor = other.max(dim=-1).values.clamp_min(0)
        result[:, slot] = values[:, slot] - competitor
    return result


def functional_specialization_summary(
    utility: Tensor,
    *,
    dpp_temperature: float | None = None,
    useful_threshold: float | None = None,
    efficiency_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Functional slot quality, coalition oracle, and exact Shapley contribution."""

    if utility.ndim != 2 or utility.shape[0] == 0:
        raise ValueError("functional specialization requires a non-empty [N,K] matrix")
    values = utility.detach().double().cpu()
    rows, candidates = values.shape
    shapley, coalition = exact_shapley_values(values)
    full = coalition[:, -1]
    shapley_mean = shapley.mean(dim=0)
    all_value = float(full.mean())
    shapley_total = float(shapley_mean.sum())
    efficiency_error = abs(shapley_total - all_value)
    if efficiency_error > efficiency_tolerance:
        raise ArithmeticError(
            "Shapley efficiency failed: "
            f"sum={shapley_total}, coalition={all_value}, error={efficiency_error}"
        )

    best, best_slot = values.max(dim=-1)
    oracle_execute = best > 0
    tie_count = values.eq(best[:, None]).sum(dim=-1)
    unique_win = oracle_execute[:, None] & values.eq(best[:, None]) & tie_count[:, None].eq(1)
    oracle_execute_count = int(oracle_execute.sum())
    useful = None
    if dpp_temperature is not None and useful_threshold is not None:
        useful = torch.sigmoid(values / dpp_temperature) > useful_threshold

    per_slot = []
    single_values = []
    leave_one_out = []
    for slot in range(candidates):
        single = values[:, slot].clamp_min(0)
        single_mean = float(single.mean())
        single_values.append(single_mean)
        without = coalition[:, ((1 << candidates) - 1) ^ (1 << slot)]
        leave_one_out.append(float((full - without).mean()))
        occupied = oracle_execute & best_slot.eq(slot)
        per_slot.append(
            {
                "slot": slot,
                "mean_utility": float(values[:, slot].mean()),
                "median_utility": float(values[:, slot].median()),
                "positive_utility_fraction": float(values[:, slot].gt(0).double().mean()),
                "harmful_utility_fraction": float(values[:, slot].lt(0).double().mean()),
                "useful_dpp_quality_fraction": (
                    float(useful[:, slot].double().mean()) if useful is not None else None
                ),
                "oracle_occupancy_of_all_rows": float(occupied.double().mean()),
                "oracle_occupancy_given_oracle_execute": (
                    float(occupied.sum() / oracle_execute_count)
                    if oracle_execute_count
                    else None
                ),
                "unique_oracle_win_rate": float(unique_win[:, slot].double().mean()),
                "single_slot_oracle_value": single_mean,
                "leave_one_out_contribution": leave_one_out[-1],
                "shapley_contribution": float(shapley_mean[slot]),
            }
        )

    if shapley_total > efficiency_tolerance:
        shares = shapley_mean / shapley_mean.sum()
        positive_shares = shares.clamp_min(torch.finfo(shares.dtype).tiny)
        effective_k = float((-(shares * positive_shares.log()).sum()).exp())
        normalized_shares: list[float] | None = shares.tolist()
    else:
        effective_k = None
        normalized_shares = None

    c3_value = single_values[3] if candidates == 4 else None
    return {
        "sample_count": rows,
        "num_candidates": candidates,
        "per_slot": per_slot,
        "pairwise_slot_utility_correlation": [
            [float(value) if math.isfinite(float(value)) else None for value in row]
            for row in _correlation_matrix(values)
        ],
        "coalition_oracle": {
            "full_set_value": all_value,
            "single_slot_values": single_values,
            "leave_one_out_contributions": leave_one_out,
            "c3_only_value": c3_value,
            "all_minus_c3": all_value - c3_value if c3_value is not None else None,
        },
        "shapley": {
            "contribution_per_slot": shapley_mean.tolist(),
            "normalized_share_per_slot": normalized_shares,
            "efficiency_sum": shapley_total,
            "efficiency_target": all_value,
            "efficiency_absolute_error": efficiency_error,
        },
        "effective_functional_k": effective_k,
        "oracle_execute_count": oracle_execute_count,
        "oracle_execute_fraction": float(oracle_execute.double().mean()),
        "per_sample_shapley": shapley,
    }


def bootstrap_mean_interval(
    values: Tensor,
    *,
    samples: int,
    generator: torch.Generator,
    confidence: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Deterministic percentile bootstrap interval for a [N,K] mean."""

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("bootstrap values must be non-empty [N,K]")
    if samples <= 0:
        nan = torch.full((values.shape[1],), float("nan"), dtype=values.dtype)
        return nan, nan.clone()
    indices = torch.randint(
        values.shape[0],
        (samples, values.shape[0]),
        generator=generator,
    )
    estimates = values.index_select(0, indices.flatten()).reshape(
        samples, values.shape[0], values.shape[1]
    ).mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    return torch.quantile(estimates, tail, dim=0), torch.quantile(
        estimates, 1.0 - tail, dim=0
    )


def semantic_specialization_summary(
    utility: Tensor,
    shapley: Tensor,
    instruction_concepts: Sequence[Sequence[str]],
    *,
    min_support: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Multi-label conditional functional summaries without causal interpretation."""

    values = utility.detach().double().cpu()
    phi = shapley.detach().double().cpu()
    if values.shape != phi.shape or values.shape[0] != len(instruction_concepts):
        raise ValueError("utility, Shapley, and instruction concepts must align")
    concepts = sorted({item for row in instruction_concepts for item in row})
    advantages = unique_advantage(values)
    best, best_slot = values.max(dim=-1)
    oracle_execute = best > 0
    generator = torch.Generator().manual_seed(seed)
    reported = []
    insufficient = []
    for concept in concepts:
        mask = torch.tensor([concept in row for row in instruction_concepts])
        support = int(mask.sum())
        if support < min_support:
            insufficient.append({"concept": concept, "support_count": support})
            continue
        conditional_phi = phi[mask]
        low, high = bootstrap_mean_interval(
            conditional_phi,
            samples=bootstrap_samples,
            generator=generator,
        )
        conditional_advantage = advantages[mask]
        conditional_best = best_slot[mask]
        conditional_execute = oracle_execute[mask]
        execute_count = int(conditional_execute.sum())
        reported.append(
            {
                "concept": concept,
                "support_count": support,
                "conditional_shapley_mean": conditional_phi.mean(dim=0).tolist(),
                "conditional_shapley_ci_low": low.tolist(),
                "conditional_shapley_ci_high": high.tolist(),
                "conditional_unique_advantage_mean": conditional_advantage.mean(dim=0).tolist(),
                "positive_unique_advantage_rate": conditional_advantage.gt(0)
                .double()
                .mean(dim=0)
                .tolist(),
                "conditional_oracle_occupancy_of_all_rows": [
                    float((conditional_execute & conditional_best.eq(slot)).double().mean())
                    for slot in range(values.shape[1])
                ],
                "conditional_oracle_occupancy_given_execute": [
                    (
                        float((conditional_execute & conditional_best.eq(slot)).sum() / execute_count)
                        if execute_count
                        else None
                    )
                    for slot in range(values.shape[1])
                ],
            }
        )
    return {
        "minimum_support": min_support,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "reported_concepts": reported,
        "insufficient_support": insufficient,
    }


def _compact_geometry(values: Tensor, *, min_support: int) -> dict[str, Any]:
    count = int(values.shape[0])
    if count < min_support:
        return {"sample_count": count, "insufficient_support": True}
    geometry = feature_geometry(values)
    return {
        "sample_count": count,
        "insufficient_support": False,
        "participation_ratio": geometry["effective_rank_pr"],
        "effective_rank_entropy": geometry["effective_rank_entropy"],
        "mean_norm": geometry["mean_norm"],
        "zero_total_variance": geometry["zero_total_variance"],
        "degenerate_spectrum": geometry["degenerate_spectrum"],
    }


def conditional_geometry_summary(
    candidate_features: Tensor,
    utility: Tensor,
    shapley: Tensor,
    *,
    min_support: int,
) -> dict[str, Any]:
    """Per-slot geometry for all, useful, Shapley-positive, and oracle-win rows."""

    if candidate_features.ndim != 3:
        raise ValueError("candidate features must be [N,K,D]")
    if candidate_features.shape[:2] != utility.shape or utility.shape != shapley.shape:
        raise ValueError("features, utility, and Shapley must align on [N,K]")
    features = candidate_features.detach().float().cpu()
    values = utility.detach().double().cpu()
    phi = shapley.detach().double().cpu()
    best, best_slot = values.max(dim=-1)
    rows = []
    for slot in range(features.shape[1]):
        masks = {
            "all_valid_t0": torch.ones(features.shape[0], dtype=torch.bool),
            "positive_utility": values[:, slot] > 0,
            "positive_shapley": phi[:, slot] > 1e-12,
            "oracle_winner": (best > 0) & best_slot.eq(slot),
        }
        rows.append(
            {
                "slot": slot,
                "subsets": {
                    name: _compact_geometry(features[mask, slot], min_support=min_support)
                    for name, mask in masks.items()
                },
            }
        )
    return {"minimum_support": min_support, "per_slot": rows}


def concept_mil_responsibility(candidate_logits: Tensor, tau_mil: float) -> Tensor:
    """Exact candidate responsibility induced by ConceptSetAuxiliary log-sum-exp."""

    if candidate_logits.ndim != 3:
        raise ValueError("concept logits must be [B,K,M]")
    if tau_mil <= 0:
        raise ValueError("tau_mil must be positive")
    return torch.softmax(candidate_logits.float() / tau_mil, dim=1)


def summarize_slot_gradients(
    gradients: Tensor,
    *,
    zero_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Summarize [observations,K,D] candidate-specific gradient exposure."""

    if gradients.ndim != 3:
        raise ValueError("candidate gradients must be [N,K,D]")
    values = gradients.detach().float().cpu()
    norms = values.norm(dim=-1)
    energy = values.square().sum(dim=(0, 2))
    total_energy = float(energy.sum())
    return {
        "observation_count": values.shape[0],
        "per_slot": [
            {
                "slot": slot,
                "mean_gradient_norm": float(norms[:, slot].mean()),
                "median_gradient_norm": float(norms[:, slot].median()),
                "nonzero_gradient_fraction": float(
                    norms[:, slot].gt(zero_threshold).float().mean()
                ),
                "gradient_energy_fraction": (
                    float(energy[slot] / total_energy) if total_energy > 0 else 0.0
                ),
            }
            for slot in range(values.shape[1])
        ],
        "total_gradient_energy": total_energy,
    }


def gradient_interference_summary(
    gradients: dict[str, Tensor],
    *,
    task_name: str = "terminal",
    auxiliary_names: Sequence[str] = ("concept", "bind", "dpp"),
    zero_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Per-slot task/auxiliary cosine and norm ratio with zero-safe handling."""

    task = gradients[task_name].detach().float().cpu()
    if task.ndim != 3:
        raise ValueError("task candidate gradients must be [N,K,D]")
    result: dict[str, Any] = {}
    for name in auxiliary_names:
        auxiliary = gradients[name].detach().float().cpu()
        if auxiliary.shape != task.shape:
            raise ValueError("task and auxiliary gradients must share [N,K,D]")
        slots = []
        for slot in range(task.shape[1]):
            task_slot = task[:, slot]
            auxiliary_slot = auxiliary[:, slot]
            task_norm = task_slot.norm(dim=-1)
            auxiliary_norm = auxiliary_slot.norm(dim=-1)
            valid = (task_norm > zero_threshold) & (auxiliary_norm > zero_threshold)
            if valid.any():
                cosine = (task_slot[valid] * auxiliary_slot[valid]).sum(dim=-1) / (
                    task_norm[valid] * auxiliary_norm[valid]
                )
                cosine_mean = float(cosine.mean())
                cosine_median = float(cosine.median())
                negative_fraction = float(cosine.lt(0).float().mean())
            else:
                cosine_mean = None
                cosine_median = None
                negative_fraction = None
            ratio_valid = task_norm > zero_threshold
            norm_ratio = (
                float((auxiliary_norm[ratio_valid] / task_norm[ratio_valid]).mean())
                if ratio_valid.any()
                else None
            )
            slots.append(
                {
                    "slot": slot,
                    "mean_cosine": cosine_mean,
                    "median_cosine": cosine_median,
                    "negative_cosine_fraction": negative_fraction,
                    "valid_comparison_count": int(valid.sum()),
                    "mean_aux_to_terminal_norm_ratio": norm_ratio,
                }
            )
        result[f"{task_name}_vs_{name}"] = slots
    return result
