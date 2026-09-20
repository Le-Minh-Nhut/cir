from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

_EPS = 1e-8
_GENERIC_STAGES = ("proposals", "entities", "actions", "delta", "delta_q")


def _flatten_candidates(values: Tensor) -> Tensor:
    if values.ndim < 3:
        raise ValueError("candidate values must have shape [B,K,...]")
    return values.detach().flatten(start_dim=2)


def _pairwise_mean(values: Tensor, fn) -> Tensor:
    candidates = values.shape[1]
    if candidates < 2:
        return values.new_zeros(())
    pair_values = [fn(values[:, left], values[:, right]) for left in range(candidates) for right in range(left + 1, candidates)]
    return torch.stack(pair_values).mean()


def _pairwise_cosine(values: Tensor) -> Tensor:
    def cosine(left: Tensor, right: Tensor) -> Tensor:
        left = left.float()
        right = right.float()
        return (left * right).sum(dim=-1).div(
            left.norm(dim=-1) * right.norm(dim=-1) + _EPS
        ).mean()

    return _pairwise_mean(values, cosine)


def _effective_rank(values: Tensor) -> Tensor:
    # K is small; construct only the [B,K,K] Gram matrix, never a wide SVD workspace.
    batch, candidates, _ = values.shape
    gram = torch.empty(batch, candidates, candidates, device=values.device, dtype=torch.float32)
    for left in range(candidates):
        left_value = values[:, left].float()
        for right in range(left, candidates):
            inner = (left_value * values[:, right].float()).sum(dim=-1)
            gram[:, left, right] = inner
            gram[:, right, left] = inner
    singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
    return (
        singular_values.sum(dim=-1).square()
        / singular_values.square().sum(dim=-1).clamp_min(_EPS)
    ).mean()


def generic_diversity(values: Tensor) -> dict[str, float | list[float]]:
    """Detached within-sibling metrics for a [B,K,...] stage tensor."""

    flat = _flatten_candidates(values)
    candidates = flat.shape[1]
    mean = sum((flat[:, slot].float() for slot in range(candidates))) / candidates
    norms = torch.stack([flat[:, slot].float().norm(dim=-1) for slot in range(candidates)], dim=-1)
    spread = torch.stack(
        [(flat[:, slot].float() - mean).norm(dim=-1) for slot in range(candidates)], dim=-1
    ).mean(dim=-1)
    return {
        "pairwise_cosine": float(_pairwise_cosine(flat)),
        "effective_rank": float(_effective_rank(flat)),
        "spread": float(spread.mean()),
        "relative_spread": float((spread / (norms.mean(dim=-1) + _EPS)).mean()),
        "mean_norm": float(norms.mean()),
        **{f"mean_norm_c{slot}": float(norms[:, slot].mean()) for slot in range(candidates)},
    }


def alpha_read_diversity(values: Tensor) -> dict[str, float]:
    """Detached distribution metrics for normalized grounding [B,K,N]."""

    alpha = values.detach().float().clamp_min(0)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    def js(left: Tensor, right: Tensor) -> Tensor:
        midpoint = 0.5 * (left + right)
        return 0.5 * (
            (left * (left.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
            + (right * (right.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
        ).mean()

    return {
        "pairwise_cosine": float(_pairwise_cosine(alpha)),
        "pairwise_js_divergence": float(_pairwise_mean(alpha, js)),
        "entropy": float(-(alpha * alpha.clamp_min(_EPS).log()).sum(dim=-1).mean()),
        "argmax_agreement": float(
            _pairwise_mean(alpha, lambda left, right: left.argmax(dim=-1).eq(right.argmax(dim=-1)).float().mean())
        ),
    }


def exec_mask_diversity(values: Tensor) -> dict[str, float]:
    """Detached soft-mask metrics for executor support [B,K,N]."""

    masks = values.detach().float()
    flat = _flatten_candidates(masks)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = flat.norm(dim=-1)
    spread = centered.norm(dim=-1).mean(dim=-1)
    return {
        "pairwise_cosine": float(_pairwise_cosine(flat)),
        "soft_iou": float(
            _pairwise_mean(
                flat,
                lambda left, right: torch.minimum(left, right).sum(dim=-1).div(
                    torch.maximum(left, right).sum(dim=-1) + _EPS
                ).mean(),
            )
        ),
        "spread": float(spread.mean()),
        "relative_spread": float((spread / (norms.mean(dim=-1) + _EPS)).mean()),
        "mean_activation": float(masks.mean()),
    }


def _step_audit(step: Mapping[str, object]) -> dict[str, object]:
    tensors = {name: step[name] for name in (*_GENERIC_STAGES, "alpha_read", "exec_mask")}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("functional collapse audit requires tensor trajectory stages")
    generic = {name: generic_diversity(tensors[name]) for name in _GENERIC_STAGES}
    return {
        "live_samples": int(tensors["delta_q"].shape[0]),
        **generic,
        "alpha_read": alpha_read_diversity(tensors["alpha_read"]),
        "exec_mask": exec_mask_diversity(tensors["exec_mask"]),
    }


def _aggregate(steps: list[dict[str, object]]) -> dict[str, object]:
    if not steps:
        return {}
    total = sum(int(step["live_samples"]) for step in steps)

    def combine(values: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key in values[0]:
            if isinstance(values[0][key], list):
                result[key] = [
                    sum(float(value[key][slot]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
                    for slot in range(len(values[0][key]))
                ]
            else:
                result[key] = sum(float(value[key]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
        return result

    return {
        "live_samples": total,
        **{name: combine([step[name] for step in steps]) for name in _GENERIC_STAGES},
        "alpha_read": combine([step["alpha_read"] for step in steps]),
        "exec_mask": combine([step["exec_mask"] for step in steps]),
    }


@torch.no_grad()
def functional_collapse_audit(output: Mapping[str, object]) -> dict[str, object]:
    """Small detached, sample-weighted trajectory audit with per-step summaries."""

    trajectory = output.get("steps")
    if not isinstance(trajectory, list):
        raise TypeError("model output must contain a trajectory list")
    by_step = {f"t{step['timestep']}": _step_audit(step) for step in trajectory}
    return {"overall": _aggregate(list(by_step.values())), "by_step": by_step}


_PROPOSAL_STAGES = (
    "base_query",
    "expanded_query",
    "conditioned_residual",
    "query_pre_norm",
    "query_post_norm",
    "proposal_output",
)


def _proposal_step_audit(step: Mapping[str, object]) -> dict[str, object]:
    internal = step.get("proposal_internal")
    if not isinstance(internal, Mapping):
        raise TypeError("proposal internal audit requires proposal diagnostic stages")
    tensors = {name: internal[name] for name in _PROPOSAL_STAGES}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("proposal internal audit requires tensor stages")
    base = tensors["expanded_query"]
    conditioned = tensors["conditioned_residual"]
    pre = tensors["query_pre_norm"]
    base_norm = base.float().norm(dim=-1)
    displacement = (pre.float() - base.float()).norm(dim=-1) / (base_norm + _EPS)
    base_pre_cosine = (base.float() * pre.float()).sum(dim=-1) / (
        base_norm * pre.float().norm(dim=-1) + _EPS
    )
    post = generic_diversity(tensors["query_post_norm"])
    output = generic_diversity(tensors["proposal_output"])
    return {
        "live_samples": int(base.shape[0]),
        **{name: generic_diversity(tensors[name]) for name in _PROPOSAL_STAGES},
        "conditioner_to_base_norm_ratio": float(
            (conditioned.float().norm(dim=-1) / (base_norm + _EPS)).mean()
        ),
        "base_pre_cosine": float(base_pre_cosine.mean()),
        "mean_relative_displacement": float(displacement.mean()),
        **{
            f"mean_relative_displacement_c{slot}": float(displacement[:, slot].mean())
            for slot in range(base.shape[1])
        },
        "attention_diversity_retention": float(
            output["relative_spread"] / (post["relative_spread"] + _EPS)
        ),
        "attention_rank_retention": float(
            output["effective_rank"] / (post["effective_rank"] + _EPS)
        ),
    }


def _proposal_aggregate(steps: list[dict[str, object]]) -> dict[str, object]:
    if not steps:
        return {}
    total = sum(int(step["live_samples"]) for step in steps)

    def mean(name: str) -> object:
        value = steps[0][name]
        if isinstance(value, Mapping):
            return {
                key: mean_nested([step[name][key] for step in steps], total, steps)
                for key in value
            }
        return mean_nested([step[name] for step in steps], total, steps)

    return {"live_samples": total, **{name: mean(name) for name in steps[0] if name != "live_samples"}}


def mean_nested(values: list[object], total: int, steps: list[dict[str, object]]) -> object:
    if isinstance(values[0], list):
        return [
            sum(float(value[slot]) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total
            for slot in range(len(values[0]))
        ]
    return sum(float(value) * int(step["live_samples"]) for value, step in zip(values, steps, strict=True)) / total


@torch.no_grad()
def proposal_internal_audit(output: Mapping[str, object]) -> dict[str, object]:
    """Detached, sample-weighted ProposalNet stage audit."""
    trajectory = output.get("steps")
    if not isinstance(trajectory, list):
        raise TypeError("model output must contain a trajectory list")
    by_step = {f"t{step['timestep']}": _proposal_step_audit(step) for step in trajectory}
    return {"overall": _proposal_aggregate(list(by_step.values())), "by_step": by_step}
