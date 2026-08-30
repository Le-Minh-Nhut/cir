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
)


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


def functional_effective_rank(delta_q: Tensor, epsilon: float = 1e-8) -> Tensor:
    if delta_q.ndim < 2:
        raise ValueError("candidate effects must be [...,K,F]")
    singular_values = torch.linalg.svdvals(delta_q.float())
    probabilities = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return torch.exp(-(probabilities * probabilities.clamp_min(epsilon).log()).sum(dim=-1))


def summarize_trajectory(output: IAGSRMEOutput) -> dict[str, Tensor]:
    supports = output.supports
    support_fraction = (supports > 0).float().mean(dim=-1)
    support_entropy = -(supports * supports.clamp_min(1e-8).log()).sum(dim=-1)
    support_overlap = pairwise_cosine(supports)
    actions = torch.stack([step.selected_index for step in output.trace], dim=1)
    scores = torch.stack([step.scores for step in output.trace], dim=1)
    delta_q = torch.stack([step.delta_q for step in output.trace], dim=1)
    result = {
        "intent_pairwise_cosine": pairwise_cosine(output.intents),
        "grounding_support_fraction": support_fraction,
        "grounding_entropy": support_entropy,
        "grounding_overlap": support_overlap,
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
    return result
