from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

_EPS = 1e-8
_GENERIC_STAGES = ("proposals", "entities", "actions", "delta", "delta_q")


def _flatten_candidates(values: Tensor) -> Tensor:
    if values.ndim < 3:
        raise ValueError("candidate values must have shape [B,K,...]")
    return values.detach().float().flatten(start_dim=2)


def _pairwise_mean(values: Tensor, fn) -> Tensor:
    candidates = values.shape[1]
    if candidates < 2:
        return values.new_zeros(())
    pair_values = [fn(values[:, left], values[:, right]) for left in range(candidates) for right in range(left + 1, candidates)]
    return torch.stack(pair_values).mean()


def _pairwise_cosine(values: Tensor) -> Tensor:
    normalized = values / (values.norm(dim=-1, keepdim=True) + _EPS)
    return _pairwise_mean(normalized, lambda left, right: (left * right).sum(dim=-1).mean())


def _effective_rank(values: Tensor) -> Tensor:
    singular_values = torch.linalg.svdvals(values)
    return (singular_values.sum(dim=-1).square() / singular_values.square().sum(dim=-1).clamp_min(_EPS)).mean()


def generic_diversity(values: Tensor) -> dict[str, float | list[float]]:
    """Detached within-sibling metrics for a [B,K,...] stage tensor."""

    flat = _flatten_candidates(values)
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = flat.norm(dim=-1)
    spread = centered.norm(dim=-1).mean(dim=-1)
    return {
        "pairwise_cosine": float(_pairwise_cosine(flat)),
        "effective_rank": float(_effective_rank(flat)),
        "spread": float(spread.mean()),
        "relative_spread": float((spread / (norms.mean(dim=-1) + _EPS)).mean()),
        "mean_norm": float(norms.mean()),
        **{f"mean_norm_c{slot}": float(norms[:, slot].mean()) for slot in range(flat.shape[1])},
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
