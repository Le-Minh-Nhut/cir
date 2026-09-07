from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _flat(values: Tensor) -> Tensor:
    if values.ndim == 1:
        return values[:, None]
    if values.ndim == 2:
        return values
    return values.reshape(-1, values.shape[-1])


def _nan() -> float:
    return float("nan")


def feature_geometry(
    values: Tensor,
    *,
    variance_epsilon: float = 1e-12,
    near_zero_dim_threshold: float = 1e-4,
    zero_effect_threshold: float = 1e-8,
    max_cosine_samples: int = 2048,
) -> dict[str, Any]:
    """Cross-row geometry for [N,D], with explicit degenerate-spectrum flags."""

    x = _flat(values.detach()).float().cpu()
    count, dim = x.shape
    norms = x.norm(dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    total_variance = float(centered.square().mean(dim=0).sum()) if count else 0.0
    zero_variance = total_variance <= variance_epsilon
    zero_effect = bool((norms <= zero_effect_threshold).all()) if count else True
    result: dict[str, Any] = {
        "count": int(count),
        "dim": int(dim),
        "total_variance": total_variance,
        "zero_total_variance": zero_variance,
        "zero_effect": zero_effect,
        "degenerate_spectrum": count < 2 or zero_variance,
        "mean_norm": float(norms.mean()) if count else _nan(),
        "std_norm": float(norms.std(unbiased=False)) if count else _nan(),
        "min_norm": float(norms.min()) if count else _nan(),
        "max_norm": float(norms.max()) if count else _nan(),
    }

    if count < 2:
        result.update(
            {
                "effective_rank_pr": _nan(),
                "effective_rank_entropy": _nan(),
                "stable_rank": _nan(),
                "top1_explained_variance": _nan(),
                "top5_explained_variance": _nan(),
                "top10_explained_variance": _nan(),
                "mean_per_dim_std": _nan(),
                "median_per_dim_std": _nan(),
                "near_zero_dim_fraction": _nan(),
            }
        )
    else:
        std = x.std(dim=0, unbiased=False)
        result.update(
            {
                "mean_per_dim_std": float(std.mean()),
                "median_per_dim_std": float(std.median()),
                "near_zero_dim_fraction": float((std < near_zero_dim_threshold).float().mean()),
            }
        )
        if zero_variance:
            # Zero is a deliberate sentinel: this is not meaningful rank-one structure.
            result.update(
                {
                    "effective_rank_pr": 0.0,
                    "effective_rank_entropy": 0.0,
                    "stable_rank": 0.0,
                    "top1_explained_variance": 0.0,
                    "top5_explained_variance": 0.0,
                    "top10_explained_variance": 0.0,
                }
            )
        else:
            singular = torch.linalg.svdvals(centered)
            eigen = singular.square()
            total = eigen.sum()
            probability = eigen / total
            entropy = -(
                probability.clamp_min(torch.finfo(probability.dtype).tiny)
                * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
            ).sum()

            def explained(top_k: int) -> float:
                return float(eigen[: min(top_k, eigen.numel())].sum() / total)

            result.update(
                {
                    "effective_rank_pr": float(
                        total.square() / eigen.square().sum().clamp_min(variance_epsilon)
                    ),
                    "effective_rank_entropy": float(entropy.exp()),
                    "stable_rank": float(total / eigen.max()),
                    "top1_explained_variance": explained(1),
                    "top5_explained_variance": explained(5),
                    "top10_explained_variance": explained(10),
                }
            )

    if count < 2:
        cosine = {name: _nan() for name in ("mean", "median", "p90", "p99")}
    else:
        if count > max_cosine_samples:
            indices = torch.linspace(0, count - 1, max_cosine_samples).long()
            cosine_x = x.index_select(0, indices)
        else:
            cosine_x = x
        normalized = F.normalize(cosine_x, dim=-1)
        matrix = normalized @ normalized.T
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
        off_diagonal = matrix[mask]
        cosine = {
            "mean": float(off_diagonal.mean()),
            "median": float(off_diagonal.median()),
            "p90": float(torch.quantile(off_diagonal, 0.90)),
            "p99": float(torch.quantile(off_diagonal, 0.99)),
        }
    result.update(
        {
            "pairwise_cosine_mean": cosine["mean"],
            "pairwise_cosine_median": cosine["median"],
            "pairwise_cosine_p90": cosine["p90"],
            "pairwise_cosine_p99": cosine["p99"],
        }
    )
    return result


def variance_decomposition(
    values: Tensor, *, atol: float = 1e-6, rtol: float = 1e-5
) -> dict[str, Any]:
    """Population trace decomposition for X[B,K,D]."""

    if values.ndim != 3:
        raise ValueError(f"expected candidate tensor [B,K,D], got {tuple(values.shape)}")
    x = values.detach().float().cpu()
    grand_mean = x.mean(dim=(0, 1), keepdim=True)
    slot_means = x.mean(dim=0, keepdim=True)
    total = float((x - grand_mean).square().mean(dim=(0, 1)).sum())
    within = float((x - slot_means).square().mean(dim=(0, 1)).sum())
    between = float((slot_means - grand_mean).square().mean(dim=1).sum())
    error = abs(total - within - between)
    tolerance = atol + rtol * max(abs(total), 1.0)
    if error > tolerance:
        raise ArithmeticError(
            "candidate variance decomposition failed: "
            f"total={total}, within={within}, between={between}, error={error}"
        )
    denominator = max(total, torch.finfo(torch.float32).eps)
    return {
        "total_variance": total,
        "within_slot_variance": within,
        "between_slot_variance": between,
        "within_slot_variance_fraction": within / denominator if total > 0 else 0.0,
        "between_slot_variance_fraction": between / denominator if total > 0 else 0.0,
        "decomposition_abs_error": error,
        "decomposition_tolerance": tolerance,
        "decomposition_ok": True,
    }


def _candidate_geometry_one(values: Tensor) -> dict[str, Any]:
    pooled = feature_geometry(values.reshape(-1, values.shape[-1]))
    per_slot = []
    for slot in range(values.shape[1]):
        statistics = feature_geometry(values[:, slot])
        statistics["slot"] = slot
        per_slot.append(statistics)
    rank_pr = [item["effective_rank_pr"] for item in per_slot]
    rank_entropy = [item["effective_rank_entropy"] for item in per_slot]
    slot_centered = values - values.mean(dim=0, keepdim=True)
    return {
        "pooled": pooled,
        "per_slot": per_slot,
        "per_slot_summary": {
            "mean_per_slot_PR": float(torch.tensor(rank_pr).nanmean()),
            "min_per_slot_PR": min(rank_pr),
            "max_per_slot_PR": max(rank_pr),
            "mean_per_slot_entropy_rank": float(torch.tensor(rank_entropy).nanmean()),
        },
        "slot_centered_pooled": feature_geometry(
            slot_centered.reshape(-1, slot_centered.shape[-1])
        ),
        "variance_decomposition": variance_decomposition(values),
    }


def candidate_geometry(values: Tensor) -> dict[str, Any]:
    """Raw and direction-only decomposition for a candidate tensor [B,K,D]."""

    if values.ndim != 3:
        raise ValueError(f"expected candidate tensor [B,K,D], got {tuple(values.shape)}")
    raw = values.detach().float().cpu()
    normalized = F.normalize(raw, dim=-1)
    return {
        "raw": _candidate_geometry_one(raw),
        "l2_normalized": _candidate_geometry_one(normalized),
    }
