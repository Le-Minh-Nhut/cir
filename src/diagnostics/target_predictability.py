from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from diagnostics.feature_sufficiency import (
    COMPACT_FEATURE_FIELDS,
    LABEL_FIELD,
    validate_compact_features,
)
from diagnostics.feature_sufficiency import _LegacyEncoder
from diagnostics.feature_sufficiency import select_probe_scores
from models.iag_srme.utils.retrieval import marginal_teacher_utilities

PRIVILEGED_FIELDS = ("parent_target_cos", "candidate_target_cos", "positive_similarity_gain")
TARGET_CACHE_FIELDS = ("target_embeddings", "sample_ids", "target_ids")


def validate_target_cache_rows(rows: Mapping[str, Any]) -> dict[str, int]:
    if set(rows) != set(TARGET_CACHE_FIELDS):
        raise ValueError(f"invalid privileged target cache fields: {sorted(rows)}")
    targets = rows["target_embeddings"]
    if not isinstance(targets, Tensor) or targets.ndim != 2 or not torch.isfinite(targets).all():
        raise ValueError("target_embeddings must be finite [B,D]")
    if len(rows["sample_ids"]) != targets.shape[0] or len(rows["target_ids"]) != targets.shape[0]:
        raise ValueError("privileged target cache identifiers must match target rows")
    if any(target_id is None for target_id in rows["target_ids"]):
        raise ValueError("privileged target cache requires target IDs")
    return {"rows": targets.shape[0], "embedding_dim": targets.shape[1]}


def privileged_similarity_features(
    current_query: Tensor, candidate_queries: Tensor, target_embeddings: Tensor
) -> dict[str, Tensor]:
    """The only privileged inputs: true-target cosine geometry, never teacher labels."""

    if current_query.ndim != 2 or candidate_queries.ndim != 3:
        raise ValueError("queries must be [B,D] and [B,K,D]")
    if candidate_queries.shape[0] != current_query.shape[0] or target_embeddings.shape != current_query.shape:
        raise ValueError("privileged target/query batch shapes must match")
    parent = F.cosine_similarity(current_query.float(), target_embeddings.float(), dim=-1, eps=1e-8)
    candidate = F.cosine_similarity(
        candidate_queries.float(), target_embeddings[:, None].float(), dim=-1, eps=1e-8
    )
    return {
        "parent_target_cos": parent[:, None],
        "candidate_target_cos": candidate[:, :, None],
        "positive_similarity_gain": (candidate - parent[:, None])[:, :, None],
    }


def add_privileged_features(
    rows: Mapping[str, Any], target_rows: Mapping[str, Any]
) -> dict[str, Any]:
    """Join exact-order frozen target embeddings to otherwise immutable compact rows."""

    validate_compact_features(rows)
    validate_target_cache_rows(target_rows)
    if list(rows["sample_ids"]) != list(target_rows["sample_ids"]):
        raise ValueError("compact and privileged target cache sample ordering differs")
    features = privileged_similarity_features(
        rows["current_query"], rows["candidate_queries"], target_rows["target_embeddings"]
    )
    return {**rows, **features}


def validate_privileged_rows(rows: Mapping[str, Any]) -> dict[str, int]:
    base = {name: rows[name] for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD, "sample_ids") if name in rows}
    dimensions = validate_compact_features(base)
    if set(rows) != set(base) | set(PRIVILEGED_FIELDS):
        raise ValueError("target-aware probe receives only compact fields and allowed privileged similarities")
    batch, candidates = dimensions["rows"], dimensions["candidates"]
    expected = {
        "parent_target_cos": (batch, 1),
        "candidate_target_cos": (batch, candidates, 1),
        "positive_similarity_gain": (batch, candidates, 1),
    }
    for name, shape in expected.items():
        value = rows[name]
        if not isinstance(value, Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    return dimensions


class TargetAwareRetrievalIndependentProbe(nn.Module):
    """Matched retrieval-independent regressor plus three scalar true-target relations."""

    def __init__(
        self, state_dim: int, text_dim: int, action_dim: int, query_dim: int, width: int = 256
    ) -> None:
        super().__init__()
        self.encoder = _LegacyEncoder(state_dim, text_dim, action_dim, width)
        self.current_query = nn.Linear(query_dim, width)
        self.candidate_query = nn.Linear(query_dim, width)
        self.delta_query = nn.Linear(query_dim, width)
        self.privileged = nn.Linear(3, width)
        self.combine = nn.Sequential(nn.LayerNorm(5 * width), nn.Linear(5 * width, width), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, rows: Mapping[str, Tensor]) -> Tensor:
        validate_privileged_rows(rows)
        base = {name: rows[name] for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD, "sample_ids")}
        legacy = self.encoder(base)
        current = self.current_query(rows["current_query"])[:, None].expand_as(legacy)
        privileged = self.privileged(
            torch.cat(
                (
                    rows["parent_target_cos"][:, None].expand(-1, legacy.shape[1], -1),
                    rows["candidate_target_cos"],
                    rows["positive_similarity_gain"],
                ),
                dim=-1,
            )
        )
        retrieval = self.combine(
            torch.cat(
                (legacy, current, self.candidate_query(rows["candidate_queries"]), self.delta_query(rows["delta_q"]), privileged),
                dim=-1,
            )
        )
        return self.head(retrieval).squeeze(-1)


def target_aware_probe_from_rows(rows: Mapping[str, Any], *, width: int = 256) -> nn.Module:
    validate_privileged_rows(rows)
    return TargetAwareRetrievalIndependentProbe(
        int(rows["current_global"].shape[-1]),
        int(rows["text_global"].shape[-1]),
        int(rows["actions"].shape[-1]),
        int(rows["current_query"].shape[-1]),
        width,
    )


def deterministic_alternative_banks(
    target_ids: Sequence[str], *, anchor: int, bank_size: int, seeds: Sequence[int]
) -> list[list[int]]:
    """Retain anchor positive and deterministically prefer distinct negative target identities."""

    if not 1 <= bank_size <= len(target_ids):
        raise ValueError("bank_size must be within the target reservoir")
    anchor_id = target_ids[anchor]
    eligible = [index for index, target_id in enumerate(target_ids) if target_id != anchor_id]
    if len(eligible) < bank_size - 1:
        raise ValueError("target reservoir lacks enough negatives distinct from anchor target")
    groups: dict[str, list[int]] = {}
    for index in eligible:
        groups.setdefault(target_ids[index], []).append(index)
    result = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed) + anchor * 1_000_003)
        identities = list(groups)
        order = torch.randperm(len(identities), generator=generator).tolist()
        selected: list[int] = []
        for group_index in order:
            options = groups[identities[group_index]]
            selected.append(options[int(torch.randint(len(options), (), generator=generator))])
            if len(selected) == bank_size - 1:
                break
        if len(selected) < bank_size - 1:
            remaining = [index for index in eligible if index not in selected]
            extra = torch.randperm(len(remaining), generator=generator).tolist()
            selected.extend(remaining[index] for index in extra[: bank_size - 1 - len(selected)])
        result.append([anchor, *selected])
    return result


def recompute_utility_with_target_bank(
    current_query: Tensor,
    candidate_queries: Tensor,
    targets: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Use the production teacher exactly; bank index zero is the anchor positive."""

    if current_query.shape[0] != 1 or candidate_queries.shape[0] != 1:
        raise ValueError("alternative teacher utility is computed one anchor row at a time")
    positive = torch.zeros((1, targets.shape[0]), dtype=torch.bool, device=targets.device)
    positive[0, 0] = True
    negative = ~positive
    utility, valid = marginal_teacher_utilities(
        current_query, candidate_queries, targets, positive, negative, temperature
    )
    if not bool(valid.item()):
        raise RuntimeError("alternative teacher bank has no valid positive and negative")
    return utility[0]


def _correlation(left: Tensor, right: Tensor, *, rank: bool = False) -> float:
    left, right = left.float().flatten(), right.float().flatten()
    if rank:
        left, right = left.argsort().argsort().float(), right.argsort().argsort().float()
    left, right = left - left.mean(), right - right.mean()
    denominator = left.norm() * right.norm()
    return float(left.dot(right) / denominator) if float(denominator) > 1e-12 else float("nan")


def _oracle(values: Tensor, epsilon_stop: float) -> Tensor:
    return select_probe_scores(values, epsilon_stop=epsilon_stop)


def stability_metrics(canonical: Tensor, alternatives: Tensor, *, epsilon_stop: float) -> dict[str, Any]:
    """Pool-only label stability; alternatives are [pools, rows, candidate slots]."""

    if canonical.ndim != 2 or alternatives.ndim != 3 or alternatives.shape[1:] != canonical.shape:
        raise ValueError("canonical and alternative utilities must be [B,K] and [P,B,K]")
    pearson = [_correlation(canonical, pool) for pool in alternatives]
    spearman = [_correlation(canonical, pool, rank=True) for pool in alternatives]
    signs = [(pool > 0).eq(canonical > 0).float().mean().item() for pool in alternatives]
    canonical_orders = canonical[:, :, None] - canonical[:, None, :]
    agreement = []
    for pool in alternatives:
        pool_orders = pool[:, :, None] - pool[:, None, :]
        non_ties = canonical_orders.ne(0) & pool_orders.ne(0)
        agreement.append(float((canonical_orders.sign()[non_ties] == pool_orders.sign()[non_ties]).float().mean()))
    canonical_oracle = _oracle(canonical, epsilon_stop)
    alternative_oracle = torch.stack([_oracle(pool, epsilon_stop) for pool in alternatives])
    flips = (alternatives > 0).ne(canonical[None] > 0).any(dim=0)
    entry_std = alternatives.float().std(dim=0, unbiased=False)
    pairwise = [
        _correlation(alternatives[left], alternatives[right])
        for left in range(alternatives.shape[0])
        for right in range(left + 1, alternatives.shape[0])
    ]
    single = canonical_oracle.eq(canonical.shape[1])
    alt_stop = alternative_oracle.eq(canonical.shape[1])
    correlations = torch.tensor(pearson)
    finite = correlations[torch.isfinite(correlations)]
    spearman_values = torch.tensor(spearman)
    finite_spearman = spearman_values[torch.isfinite(spearman_values)]
    return {
        "canonical_vs_alternative": {
            "pearson_mean": float(finite.mean()) if finite.numel() else float("nan"),
            "pearson_std": float(finite.std(unbiased=False)) if finite.numel() else float("nan"),
            "pearson_min": min(pearson),
            "pearson_max": max(pearson),
            "spearman_mean": float(finite_spearman.mean()) if finite_spearman.numel() else float("nan"),
            "sign_agreement_at_zero_mean": sum(signs) / len(signs),
        },
        "utility_variance": {
            "mean_per_entry_std": float(entry_std.mean()),
            "median_per_entry_std": float(entry_std.median()),
            "p90_per_entry_std": float(torch.quantile(entry_std.flatten(), 0.9)),
            "canonical_utility_std": float(canonical.float().std(unbiased=False)),
            "pool_induced_to_canonical_std": float(entry_std.mean() / canonical.float().std(unbiased=False).clamp_min(1e-12)),
        },
        "candidate_ranking": {
            "pairwise_order_agreement_mean": sum(agreement) / len(agreement),
            "canonical_oracle_action_agreement_mean": float(alternative_oracle.eq(canonical_oracle).float().mean()),
            "rows_oracle_action_changes": float(alternative_oracle.ne(canonical_oracle).any(dim=0).float().mean()),
        },
        "stop": {
            "canonical_oracle_stop_rate": float(single.float().mean()),
            "alternative_oracle_stop_rate_mean": float(alt_stop.float().mean()),
            "alternative_oracle_stop_rate_std": float(alt_stop.float().mean(dim=1).std(unbiased=False)),
            "per_row_stop_execute_agreement": float(alt_stop.eq(single).float().mean()),
        },
        "sign": {"sign_flip_any_fraction": float(flips.float().mean()), "sign_flip_rate_by_slot": flips.float().mean(dim=0).tolist()},
        "per_slot": {"mean_utility_std": entry_std.mean(dim=0).tolist()},
        "pool_to_pool": {"pearson_mean": sum(pairwise) / len(pairwise), "pearson_min": min(pairwise)},
    }


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
