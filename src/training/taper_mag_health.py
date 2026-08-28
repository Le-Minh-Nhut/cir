from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


def _masked_values(values: Tensor, mask: Tensor) -> Tensor:
    return values.float()[mask]


@torch.no_grad()
def utility_health_metrics(
    predicted_gain: Tensor,
    teacher_gain: Tensor,
    *,
    active: Tensor | None = None,
    step_cost: float = 0.0,
    near_tie_band: float = 0.0,
    calibration_bins: int = 5,
) -> dict[str, Any]:
    """V4 action-quality diagnostics over on-policy states, with STOP fixed at zero."""
    if predicted_gain.shape != teacher_gain.shape or predicted_gain.ndim not in {2, 3}:
        raise ValueError("predicted_gain and teacher_gain must match [B,K] or [B,T,K]")
    if near_tie_band < 0 or calibration_bins <= 0:
        raise ValueError("near_tie_band must be non-negative and calibration_bins positive")
    if predicted_gain.ndim == 2:
        predicted_gain = predicted_gain[:, None]
        teacher_gain = teacher_gain[:, None]
    batch, steps, actions = predicted_gain.shape
    if active is None:
        active = torch.ones(batch, steps, dtype=torch.bool, device=predicted_gain.device)
    if active.shape != (batch, steps):
        raise ValueError("active must be [B,T]")
    stop_allowed = torch.arange(steps, device=active.device)[None, :] >= 1
    stop_allowed = stop_allowed.expand(batch, -1) & active
    predicted = torch.cat(
        [predicted_gain.float() - step_cost, torch.zeros(batch, steps, 1, device=active.device)],
        dim=-1,
    )
    teacher = torch.cat(
        [teacher_gain.float() - step_cost, torch.zeros(batch, steps, 1, device=active.device)],
        dim=-1,
    )
    minimum = torch.finfo(predicted.dtype).min
    predicted[:, 0, -1] = minimum
    teacher[:, 0, -1] = minimum
    predicted_action = predicted.argmax(dim=-1)
    oracle_action = teacher.argmax(dim=-1)
    oracle_value = teacher.max(dim=-1).values
    chosen_value = teacher.gather(-1, predicted_action.unsqueeze(-1)).squeeze(-1)
    regret = (oracle_value - chosen_value).clamp_min(0)
    valid_regret = _masked_values(regret, active)
    oracle_non_stop = active & oracle_action.ne(actions)
    valid_non_stop_regret = _masked_values(regret, oracle_non_stop)
    agreement = ((predicted_action == oracle_action) & active).sum() / active.sum().clamp_min(1)

    teacher_difference = teacher.unsqueeze(-1) - teacher.unsqueeze(-2)
    predicted_difference = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    upper = torch.triu(
        torch.ones(actions + 1, actions + 1, dtype=torch.bool, device=active.device),
        diagonal=1,
    )
    pair_mask = active[..., None, None] & upper
    available = torch.ones(
        batch, steps, actions + 1, dtype=torch.bool, device=active.device
    )
    available[:, 0, -1] = False
    pair_mask &= available.unsqueeze(-1) & available.unsqueeze(-2)
    pair_mask &= teacher_difference.abs() > near_tie_band
    pair_correct = (teacher_difference.sign() == predicted_difference.sign()) & pair_mask
    confident_mask = pair_mask & (predicted_difference.abs() > near_tie_band)

    real_predicted = predicted_gain.float()[active]
    real_teacher = teacher_gain.float()[active]
    calibration: list[dict[str, float | int | None]] = []
    if real_predicted.numel() == 0:
        raise ValueError("utility health requires at least one active state")
    minimum_prediction = float(real_predicted.min())
    maximum_prediction = float(real_predicted.max())
    if math.isclose(minimum_prediction, maximum_prediction):
        edges = torch.tensor(
            [minimum_prediction, maximum_prediction], device=real_predicted.device
        )
        bins = 1
    else:
        edges = torch.linspace(
            minimum_prediction,
            maximum_prediction,
            calibration_bins + 1,
            device=real_predicted.device,
        )
        bins = calibration_bins
    for index in range(bins):
        lower = edges[index]
        upper_edge = edges[index + 1]
        selected = (real_predicted >= lower) & (
            real_predicted <= upper_edge if index == bins - 1 else real_predicted < upper_edge
        )
        calibration.append(
            {
                "lower": float(lower),
                "upper": float(upper_edge),
                "count": int(selected.sum()),
                "mean_predicted_gain": float(real_predicted[selected].mean()) if selected.any() else None,
                "mean_teacher_gain": float(real_teacher[selected].mean()) if selected.any() else None,
            }
        )

    def quantile(values: Tensor, q: float) -> float:
        return float(torch.quantile(values, q)) if values.numel() else 0.0

    return {
        "near_tie_band": float(near_tie_band),
        "active_state_count": int(active.sum()),
        "top1_agreement": float(agreement),
        "mean_regret": float(valid_regret.mean()),
        "median_regret": quantile(valid_regret, 0.5),
        "p90_regret": quantile(valid_regret, 0.9),
        "p95_regret": quantile(valid_regret, 0.95),
        "mean_regret_oracle_non_stop": (
            float(valid_non_stop_regret.mean()) if valid_non_stop_regret.numel() else 0.0
        ),
        "false_stop_rate": float(
            ((predicted_action == actions) & (oracle_action != actions) & stop_allowed).sum()
            / stop_allowed.sum().clamp_min(1)
        ),
        "false_continue_rate": float(
            ((predicted_action != actions) & (oracle_action == actions) & stop_allowed).sum()
            / stop_allowed.sum().clamp_min(1)
        ),
        "positive_gain_rate": float(
            ((teacher_gain.max(dim=-1).values > 0) & active).sum()
            / active.sum().clamp_min(1)
        ),
        "pairwise_accuracy": float(pair_correct.sum() / pair_mask.sum().clamp_min(1)),
        "pairwise_pair_count": int(pair_mask.sum()),
        "confident_pair_accuracy": float(
            ((teacher_difference.sign() == predicted_difference.sign()) & confident_mask).sum()
            / confident_mask.sum().clamp_min(1)
        ),
        "confident_pair_count": int(confident_mask.sum()),
        "calibration_by_predicted_gain": calibration,
    }


@torch.no_grad()
def response_effective_rank(current_query: Tensor, candidate_queries: Tensor) -> dict[str, Any]:
    if (
        current_query.ndim != 2
        or candidate_queries.ndim != 3
        or candidate_queries.shape[0] != current_query.shape[0]
        or candidate_queries.shape[2] != current_query.shape[1]
    ):
        raise ValueError("expected current [B,D] and candidates [B,K,D]")
    deltas = candidate_queries.float() - current_query.float()[:, None]
    singular = torch.linalg.svdvals(deltas)
    effective = singular.sum(dim=-1).square() / singular.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "mean_effective_rank": float(effective.mean()),
        "median_effective_rank": float(effective.median()),
        "mean_singular_values": [float(value) for value in singular.mean(dim=0)],
    }


@torch.no_grad()
def dynamic_frozen_metrics(
    dynamic_actions: Tensor,
    frozen_actions: Tensor,
    dynamic_values: Tensor,
    frozen_values: Tensor,
    *,
    dynamic_retrieval: Tensor,
    frozen_retrieval: Tensor,
    dynamic_realized_gain: Tensor,
    frozen_realized_gain: Tensor,
    stop_index: int,
) -> dict[str, float]:
    if dynamic_actions.shape != frozen_actions.shape:
        raise ValueError("dynamic/frozen actions must align")
    later = torch.arange(dynamic_actions.shape[1], device=dynamic_actions.device)[None] > 0
    ordering_changed = (
        dynamic_values.argsort(dim=-1) != frozen_values.argsort(dim=-1)
    ).any(dim=-1)
    later_mask = later.expand_as(dynamic_actions)
    return {
        "ordering_change_rate": float(ordering_changed[later_mask].float().mean()) if dynamic_actions.shape[1] > 1 else 0.0,
        "top1_action_change_rate": float(
            (dynamic_actions[later_mask] != frozen_actions[later_mask]).float().mean()
        ) if dynamic_actions.shape[1] > 1 else 0.0,
        "retrieval_difference": float((dynamic_retrieval - frozen_retrieval).mean()),
        "realized_teacher_gain_difference": float(
            (dynamic_realized_gain - frozen_realized_gain).mean()
        ),
        "stop_difference": float(
            dynamic_actions.eq(stop_index).float().mean()
            - frozen_actions.eq(stop_index).float().mean()
        ),
    }


@torch.no_grad()
def repeat_staleness_metrics(
    actions: Tensor, teacher_gain: Tensor, active: Tensor, *, stop_index: int
) -> dict[str, float]:
    if actions.ndim != 2 or teacher_gain.shape[:2] != actions.shape:
        raise ValueError("repeat diagnostics expect actions [B,T] and gains [B,T,K]")
    if actions.shape[1] < 2:
        return {
            "repeat_frequency": 0.0,
            "teacher_gain_before_repeat": 0.0,
            "teacher_gain_when_reconsidered": 0.0,
            "repeated_gain_non_positive_fraction": 0.0,
        }
    previous = actions[:, :-1]
    current = actions[:, 1:]
    repeats = active[:, 1:] & previous.eq(current) & current.ne(stop_index)
    current_index = current.clamp_max(stop_index - 1).unsqueeze(-1)
    previous_index = previous.clamp_max(stop_index - 1).unsqueeze(-1)
    reconsidered = teacher_gain[:, 1:].gather(-1, current_index).squeeze(-1)
    before = teacher_gain[:, :-1].gather(-1, previous_index).squeeze(-1)
    count = repeats.sum().clamp_min(1)
    return {
        "repeat_frequency": float(repeats.sum() / (active[:, 1:].sum().clamp_min(1))),
        "teacher_gain_before_repeat": float((before * repeats).sum() / count),
        "teacher_gain_when_reconsidered": float((reconsidered * repeats).sum() / count),
        "repeated_gain_non_positive_fraction": float(
            ((reconsidered <= 0) & repeats).sum() / count
        ),
    }


@torch.no_grad()
def query_delta_clone_geometry(candidate_deltas: Tensor, best_indices: Tensor) -> dict[str, Any]:
    """Secondary response geometry only; never a causal operator intervention."""
    if candidate_deltas.ndim != 3 or best_indices.shape != candidate_deltas.shape[:1]:
        raise ValueError("clone controls expect deltas [B,K,D] and best indices [B]")
    batch, actions, width = candidate_deltas.shape
    best = candidate_deltas.gather(
        1, best_indices[:, None, None].expand(batch, 1, width)
    ).squeeze(1)
    mean = candidate_deltas.mean(dim=1)
    clone_best = best[:, None].expand(-1, actions, -1)
    clone_mean = mean[:, None].expand(-1, actions, -1)

    def rank(values: Tensor) -> float:
        singular = torch.linalg.svdvals(values.float())
        effective = singular.sum(dim=-1).square() / singular.square().sum(dim=-1).clamp_min(1e-12)
        return float(effective.mean())

    return {
        "clone_all_best_effective_rank": rank(clone_best),
        "clone_all_mean_effective_rank": rank(clone_mean),
        "repeat_best_delta_norm": float((2 * best).norm(dim=-1).mean()),
        "mean_repeat_delta_norm": float((2 * mean).norm(dim=-1).mean()),
        "operator_zero_delta_norm": 0.0,
        "operator_mean_delta_norm": float(mean.norm(dim=-1).mean()),
    }


@dataclass(slots=True)
class QueryGradientTracker:
    num_queries: int
    updates: int = 0
    norm_sum: Tensor = field(init=False)
    zero_count: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.norm_sum = torch.zeros(self.num_queries, dtype=torch.float64)
        self.zero_count = torch.zeros(self.num_queries, dtype=torch.long)

    def update(self, queries: nn.Parameter) -> None:
        if queries.grad is None:
            norms = torch.zeros(self.num_queries)
        else:
            norms = queries.grad.detach().float().norm(dim=-1).cpu()
        self.norm_sum += norms.double()
        self.zero_count += norms.eq(0)
        self.updates += 1

    def report(self) -> dict[str, Any]:
        denominator = max(self.updates, 1)
        means = self.norm_sum / denominator
        fractions = self.zero_count.double() / denominator
        return {
            "rolling_query_gradient_coverage": float((self.zero_count < denominator).float().mean()),
            "per_query_gradient_norm_mean": [float(value) for value in means],
            "per_query_zero_gradient_fraction": [float(value) for value in fractions],
            "optimizer_updates_observed": self.updates,
        }


def recursively_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(recursively_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(recursively_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True
