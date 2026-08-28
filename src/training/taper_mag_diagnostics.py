from __future__ import annotations

import torch
from torch import Tensor

from models.taper_mag.rollout import TaperOutput
from training.taper_mag_health import response_effective_rank, utility_health_metrics


def _entropy(probabilities: Tensor) -> Tensor:
    values = probabilities.float().clamp_min(1e-12)
    return -(values * values.log()).sum(dim=-1)


@torch.no_grad()
def summarize_training_diagnostics(
    output: TaperOutput,
    teacher_gain: Tensor,
    *,
    near_tie_band: float = 0.0,
    step_cost: float = 0.0,
    calibration_bins: int = 5,
) -> dict[str, float]:
    trace = output.trace
    num_queries = teacher_gain.shape[-1]
    active = trace.active
    actions = trace.actions
    executed = active & actions.ne(num_queries)
    action_count = executed.sum(dim=1).float()
    previous = actions[:, :-1]
    current = actions[:, 1:]
    repeat_valid = active[:, 1:] & previous.ne(num_queries) & current.ne(num_queries)
    repeat = (previous == current) & repeat_valid
    repeat_frequency = repeat.sum().float() / repeat_valid.sum().clamp_min(1)

    utility = utility_health_metrics(
        trace.predicted_gain,
        teacher_gain,
        active=active,
        near_tie_band=near_tie_band,
        step_cost=step_cost,
        calibration_bins=calibration_bins,
    )
    predicted_actions = trace.action_values.float().argmax(dim=-1)
    chosen_teacher = teacher_gain.gather(
        -1, predicted_actions.clamp_max(num_queries - 1).unsqueeze(-1)
    ).squeeze(-1)
    chosen_teacher = torch.where(
        predicted_actions.eq(num_queries), torch.zeros_like(chosen_teacher), chosen_teacher
    )
    sorted_gain = teacher_gain.sort(dim=-1, descending=True).values
    positive_rate = ((teacher_gain.max(dim=-1).values > 0) & active).sum().float() / active.sum().clamp_min(1)
    best_second = ((sorted_gain[..., 0] - sorted_gain[..., 1]) * active).sum() / active.sum().clamp_min(1)
    candidate_variance = trace.candidate_queries.float().var(dim=2, unbiased=False).mean()

    diagnostics: dict[str, float] = {
        "mean_action_count": float(action_count.mean()),
        "repeat_action_frequency": float(repeat_frequency),
        "candidate_query_variance": float(candidate_variance),
        "candidate_outcome_variance": float(candidate_variance),
        "support_mass": float(trace.support_mass.float().mean()),
        "support_saturation": float(trace.support_saturation.float().mean()),
        "delta_norm": float(trace.delta_norm.float().mean()),
        "state_norm": float(trace.state_norm.float().mean()),
        "oracle_best_gain": float(teacher_gain.max(dim=-1).values[active].mean()),
        "oracle_positive_gain_rate": float(positive_rate),
        "oracle_best_second_gap": float(best_second),
        "critic_top1_agreement": float(utility["top1_agreement"]),
        "critic_regret": float(utility["mean_regret"]),
        "critic_median_regret": float(utility["median_regret"]),
        "critic_p90_regret": float(utility["p90_regret"]),
        "critic_p95_regret": float(utility["p95_regret"]),
        "critic_regret_oracle_non_stop": float(utility["mean_regret_oracle_non_stop"]),
        "critic_false_stop_rate": float(utility["false_stop_rate"]),
        "critic_false_continue_rate": float(utility["false_continue_rate"]),
        "critic_pairwise_accuracy": float(utility["pairwise_accuracy"]),
        "critic_confident_pair_accuracy": float(utility["confident_pair_accuracy"]),
        "critic_near_tie_band": float(utility["near_tie_band"]),
        "realized_predicted_policy_gain": float(chosen_teacher[active].mean()),
        "text_attention_entropy": float(_entropy(output.diagnostics["text_attention"]).mean()),
        "visual_attention_entropy": float(_entropy(output.diagnostics["visual_attention"]).mean()),
        "query_query_cosine_offdiag": float(
            _off_diagonal_mean(output.diagnostics["query_cosine"])
        ),
        "operator_operator_cosine_offdiag": float(
            _off_diagonal_mean(output.diagnostics["operator_cosine"])
        ),
        "edit_gate_mean": float(output.diagnostics["edit_gate_mean"]),
        "edit_gate_std": float(output.diagnostics["edit_gate_std"]),
        "edit_gate_saturation": float(output.diagnostics["edit_gate_saturation"]),
    }
    rank = response_effective_rank(
        trace.current_queries.flatten(0, 1), trace.candidate_queries.flatten(0, 1)
    )
    diagnostics["response_effective_rank"] = float(rank["mean_effective_rank"])
    for step in range(actions.shape[1]):
        active_step = active[:, step]
        diagnostics[f"stop_fraction_t{step}"] = float(
            ((actions[:, step] == num_queries) & active_step).sum().float()
            / active_step.sum().clamp_min(1)
        )
    for action in range(num_queries):
        diagnostics[f"operator_usage_{action}"] = float(
            ((actions == action) & active).sum().float() / active.sum().clamp_min(1)
        )
    return diagnostics


def _off_diagonal_mean(matrix: Tensor) -> Tensor:
    size = matrix.shape[-1]
    mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix.float()[..., mask].mean()
