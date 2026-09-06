from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

MATCHED_COMPUTE_CONTROLS: tuple[str, ...] = ()


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


def summarize_trajectory(output: dict[str, object]) -> dict[str, Tensor]:
    steps = output["steps"]
    if not steps:
        return {}
    supports = torch.cat([step["exec_mask"] for step in steps], dim=0)
    grounding = torch.cat([step["alpha_read"] for step in steps], dim=0)
    effects = torch.cat([step["delta_q"] for step in steps], dim=0)
    scores = torch.cat([step["scores"] for step in steps], dim=0)
    selected = torch.cat([step["selected_idx"] for step in steps], dim=0)
    candidates = scores.shape[-1]
    return {
        "read_entropy": -(grounding * grounding.clamp_min(1e-8).log()).sum(-1),
        "execution_area": supports.mean(-1),
        "grounding_overlap": pairwise_cosine(grounding),
        "functional_delta_q_pairwise_cosine": pairwise_cosine(effects),
        "functional_effective_rank": functional_effective_rank(effects),
        "score_mean": scores.mean(),
        "stop_frequency": selected.eq(candidates).float().mean(),
    }
