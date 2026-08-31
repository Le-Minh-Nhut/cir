from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.iag_srme.outputs import IAGSRMEOutput, RecurrentStepOutput


MATCHED_COMPUTE_CONTROLS = (
    "full",
    "zero_edit",
    "single_candidate",
    "repeat_candidate_1",
    "repeat_candidate_2",
    "repeat_candidate_3",
    "repeat_candidate_4",
    "repeat_best",
    "clone_candidate_1",
    "mean_candidate",
    "random_candidate",
    "frozen_t0_order",
    "frozen_t0_what",
)

FUNCTIONAL_ACTIVITY_EPSILON = 1e-8
FUNCTIONAL_RANK_EPSILON = 1e-8


def pairwise_cosine_matrix(values: Tensor) -> Tensor:
    """Return the full candidate-pair matrix while preserving every leading axis."""
    if values.ndim < 2:
        raise ValueError("pairwise cosine requires [...,K,F]")
    normalized = F.normalize(values.float(), dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def off_diagonal_values(matrix: Tensor) -> Tensor:
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("pairwise matrix must be [...,K,K]")
    candidates = matrix.shape[-1]
    mask = ~torch.eye(candidates, dtype=torch.bool, device=matrix.device)
    return matrix[..., mask]


def pairwise_cosine(values: Tensor) -> Tensor:
    matrix = pairwise_cosine_matrix(values)
    candidates = values.shape[-2]
    return off_diagonal_values(matrix).reshape(
        *values.shape[:-2], candidates, candidates - 1
    )


def functional_effect_activity(
    values: Tensor, epsilon: float = FUNCTIONAL_ACTIVITY_EPSILON
) -> tuple[Tensor, Tensor]:
    """Return effect norms and a numerical activity mask for [...,K,F]."""
    if values.ndim < 2:
        raise ValueError("candidate effects must be [...,K,F]")
    if epsilon < 0:
        raise ValueError("activity epsilon must be non-negative")
    norms = values.float().norm(dim=-1)
    return norms, norms > epsilon


def masked_pairwise_cosine(
    values: Tensor, epsilon: float = FUNCTIONAL_ACTIVITY_EPSILON
) -> tuple[Tensor, Tensor]:
    """Return cosine values and validity for pairs of active functional effects.

    Values at invalid locations are finite placeholders. Callers must use the returned
    mask and must not interpret those placeholders as cosine observations.
    """
    _, active = functional_effect_activity(values, epsilon)
    cosine = pairwise_cosine_matrix(values)
    valid = active.unsqueeze(-1) & active.unsqueeze(-2)
    return cosine, valid


def flatten_delta_z(delta_z: Tensor) -> Tensor:
    """Flatten spatial+channel axes only: [B,K,N,D] -> [B,K,N*D]."""
    if delta_z.ndim != 4:
        raise ValueError("delta_z must be [B,K,N,D]")
    return delta_z.flatten(start_dim=2)


def verify_same_parent_counterfactuals(step: RecurrentStepOutput) -> None:
    expected_states = step.current_state[:, None] + step.delta_z
    if not torch.equal(step.candidate_states, expected_states):
        raise AssertionError("candidate states do not branch from the same parent Z_t")
    expected_delta_q = step.candidate_queries - step.current_query[:, None]
    if not torch.equal(step.delta_q, expected_delta_q):
        raise AssertionError("delta_q is not candidate_query minus the same parent q_t")


def functional_effective_rank(
    delta_q: Tensor, epsilon: float = FUNCTIONAL_RANK_EPSILON
) -> Tensor:
    if delta_q.ndim < 2:
        raise ValueError("candidate effects must be [...,K,F]")
    singular_values = torch.linalg.svdvals(delta_q.float())
    singular_value_mass = singular_values.sum(dim=-1, keepdim=True)
    probabilities = singular_values / singular_value_mass.clamp_min(epsilon)
    active_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(epsilon).log()).sum(dim=-1)
    )
    return torch.where(
        singular_value_mass.squeeze(-1) > epsilon,
        active_rank,
        torch.zeros_like(active_rank),
    )


def summarize_trajectory(output: IAGSRMEOutput) -> dict[str, Tensor]:
    visual_supports = output.supports
    support_mass = visual_supports.float().sum(dim=-1)
    conditional_supports = (
        output.conditional_supports
        if output.conditional_supports is not None
        else visual_supports / support_mass[..., None].clamp_min(1e-8)
    )
    support_fraction = (conditional_supports > 0).float().mean(dim=-1)
    support_entropy = -(
        conditional_supports * conditional_supports.clamp_min(1e-8).log()
    ).sum(dim=-1)
    support_overlap = pairwise_cosine(conditional_supports)
    actions = torch.stack([step.selected_index for step in output.trace], dim=1)
    scores = torch.stack([step.scores for step in output.trace], dim=1)
    delta_q = torch.stack([step.delta_q for step in output.trace], dim=1)
    result = {
        "intent_pairwise_cosine": pairwise_cosine(output.intents),
        "grounding_support_fraction": support_fraction,
        "grounding_entropy": support_entropy,
        "grounding_overlap": support_overlap,
        "grounding_real_visual_mass": support_mass,
        "functional_delta_q_pairwise_cosine": pairwise_cosine(delta_q),
        "functional_effective_rank": functional_effective_rank(delta_q),
        "selected_candidate_distribution": torch.nn.functional.one_hot(
            actions, output.intents.shape[1] + 1
        )
        .float()
        .mean(dim=(0, 1)),
        "stop_frequency": actions.eq(output.intents.shape[1]).float().mean(),
        "scores_over_time": scores,
        "score_changes_over_time": scores[:, 1:] - scores[:, :-1],
    }
    if output.claims is not None:
        result["claim_mass"] = output.claims.sum(dim=-1)
    if output.visual_null_probabilities is not None:
        result["visual_null_probability"] = output.visual_null_probabilities
        result["visual_execution_confidence"] = output.visual_confidence
    if output.temporal_supports is not None:
        temporal = output.temporal_supports.float()
        result["temporal_grounding_support_mass"] = temporal.sum(dim=-1)
        result["temporal_grounding_support_fraction"] = (
            temporal > 0
        ).float().mean(dim=-1)
        result["temporal_grounding_entropy"] = -(
            temporal * temporal.clamp_min(1e-8).log()
        ).sum(dim=-1)
        if temporal.shape[1] > 1:
            result["temporal_grounding_same_candidate_cosine"] = (
                torch.nn.functional.cosine_similarity(
                    temporal[:, :-1], temporal[:, 1:], dim=-1
                )
            )
            result["temporal_grounding_l1_change"] = (
                temporal[:, 1:] - temporal[:, :-1]
            ).abs().sum(dim=-1)
    return result
