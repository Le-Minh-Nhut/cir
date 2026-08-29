from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.iag_srme.outputs import IAGSRMEOutput


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


def pairwise_cosine(values: Tensor) -> Tensor:
    normalized = F.normalize(values, dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    count = values.shape[-2]
    mask = ~torch.eye(count, dtype=torch.bool, device=values.device)
    return similarities[..., mask].reshape(*values.shape[:-2], count, count - 1)


def functional_effective_rank(delta_q: Tensor, epsilon: float = 1e-8) -> Tensor:
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
