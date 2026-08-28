from __future__ import annotations

import torch
from torch import Tensor

from models.taper_mag.rollout import TaperOutput


def _entropy(probabilities: Tensor) -> Tensor:
    values = probabilities.float().clamp_min(1e-12)
    return -(values * values.log()).sum(dim=-1)


@torch.no_grad()
def summarize_training_diagnostics(
    output: TaperOutput, teacher_gain: Tensor
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

    teacher_values = torch.cat(
        [
            teacher_gain.float(),
            torch.zeros(*teacher_gain.shape[:-1], 1, device=teacher_gain.device),
        ],
        dim=-1,
    )
    predicted_values = trace.action_values.float()
    teacher_actions = teacher_values.argmax(dim=-1)
    predicted_actions = predicted_values.argmax(dim=-1)
    agreement = ((teacher_actions == predicted_actions) & active).sum().float() / active.sum().clamp_min(1)
    best_teacher = teacher_values.max(dim=-1).values
    chosen_teacher = teacher_values.gather(-1, predicted_actions.unsqueeze(-1)).squeeze(-1)
    regret = ((best_teacher - chosen_teacher) * active).sum() / active.sum().clamp_min(1)
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
        "delta_norm": float(trace.delta_norm.float().mean()),
        "oracle_best_gain": float(teacher_gain.max(dim=-1).values[active].mean()),
        "oracle_positive_gain_rate": float(positive_rate),
        "oracle_best_second_gap": float(best_second),
        "critic_top1_agreement": float(agreement),
        "critic_regret": float(regret),
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
