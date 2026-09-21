from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from diagnostics.selection import selector_timestep_summary
from losses.objective import absolute_gain_loss

COMPACT_FEATURE_FIELDS = (
    "current_global",
    "text_global",
    "actions",
    "local_mean",
    "candidate_global_delta",
    "current_query",
    "candidate_queries",
    "delta_q",
)
LABEL_FIELD = "teacher_utility"
FORBIDDEN_FEATURE_FIELDS = frozenset(
    {
        "target",
        "target_embedding",
        "target_pixels",
        "target_id",
        "positive_mask",
        "negative_mask",
        "teacher_retrieval_loss",
        "oracle_candidate",
        "oracle_stop",
        "parent_state",
        "candidate_states",
        "delta",
        "pixels",
    }
)


def compact_local_mean(delta: Tensor, exec_mask: Tensor) -> Tensor:
    """Match ScoreNet's target-free masked local-effect reduction exactly."""

    support = exec_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return delta.sum(dim=-2) / support


def compact_t0_features(step: Mapping[str, Tensor], teacher_utility: Tensor) -> dict[str, Tensor]:
    """Detach the compact, pre-selection t0 probe interface and its label."""

    required = {
        "current_global",
        "text_global",
        "actions",
        "delta",
        "exec_mask",
        "candidate_global",
        "current_query",
        "candidate_queries",
        "delta_q",
    }
    missing = required - set(step)
    if missing:
        raise ValueError(f"t0 step lacks compact probe fields: {sorted(missing)}")
    current_global = step["current_global"]
    return {
        "current_global": current_global.detach().float().cpu(),
        "text_global": step["text_global"].detach().float().cpu(),
        "actions": step["actions"].detach().float().cpu(),
        "local_mean": compact_local_mean(step["delta"], step["exec_mask"]).detach().float().cpu(),
        "candidate_global_delta": (
            step["candidate_global"] - current_global[:, None]
        ).detach().float().cpu(),
        "current_query": step["current_query"].detach().float().cpu(),
        "candidate_queries": step["candidate_queries"].detach().float().cpu(),
        "delta_q": step["delta_q"].detach().float().cpu(),
        LABEL_FIELD: teacher_utility.detach().float().cpu(),
    }


def validate_compact_features(rows: Mapping[str, Any]) -> dict[str, int]:
    """Reject target-derived or non-compact data before any probe forward pass."""

    expected = set(COMPACT_FEATURE_FIELDS) | {LABEL_FIELD, "sample_ids"}
    fields = set(rows)
    forbidden = fields & FORBIDDEN_FEATURE_FIELDS
    if forbidden:
        raise ValueError(f"target-derived or giant probe fields are forbidden: {sorted(forbidden)}")
    if fields != expected:
        raise ValueError(
            f"invalid compact probe fields: missing={sorted(expected - fields)}, "
            f"unexpected={sorted(fields - expected)}"
        )
    tensors = {name: rows[name] for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD)}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("compact probe tensors must be torch.Tensor values")
    utility = tensors[LABEL_FIELD]
    if utility.ndim != 2:
        raise ValueError("teacher_utility must be [B,K]")
    if not torch.isfinite(utility).all():
        raise ValueError("teacher_utility contains NaN or Inf")
    batch, candidates = utility.shape
    shapes = {
        "current_global": (batch, None),
        "text_global": (batch, None),
        "actions": (batch, candidates, None),
        "local_mean": (batch, candidates, None),
        "candidate_global_delta": (batch, candidates, None),
        "current_query": (batch, None),
        "candidate_queries": (batch, candidates, None),
        "delta_q": (batch, candidates, None),
    }
    for name, shape in shapes.items():
        value = tensors[name]
        if value.ndim != len(shape) or any(
            expected_size is not None and value.shape[index] != expected_size
            for index, expected_size in enumerate(shape)
        ):
            raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if len(rows["sample_ids"]) != batch:
        raise ValueError("sample_ids must have one entry per compact row")
    return {"rows": batch, "candidates": candidates}


def compact_cache_bytes(rows: Mapping[str, Any]) -> int:
    """Return the actual compact tensor payload size without writing it."""

    validate_compact_features(rows)
    return sum(rows[name].numel() * rows[name].element_size() for name in (*COMPACT_FEATURE_FIELDS, LABEL_FIELD))


def to_cache_precision(rows: Mapping[str, Any]) -> dict[str, Any]:
    validate_compact_features(rows)
    return {
        **{name: rows[name].to(dtype=torch.float16) for name in COMPACT_FEATURE_FIELDS},
        LABEL_FIELD: rows[LABEL_FIELD].to(dtype=torch.float32),
        "sample_ids": list(rows["sample_ids"]),
    }


def to_training_precision(rows: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    validate_compact_features(rows)
    return {
        **{name: rows[name].to(device=device, dtype=torch.float32) for name in COMPACT_FEATURE_FIELDS},
        LABEL_FIELD: rows[LABEL_FIELD].to(device=device, dtype=torch.float32),
        "sample_ids": list(rows["sample_ids"]),
    }


def cache_fingerprint(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _LegacyEncoder(nn.Module):
    def __init__(self, state_dim: int, text_dim: int, action_dim: int, width: int) -> None:
        super().__init__()
        self.context_global = nn.Linear(state_dim, width)
        self.context_text = nn.Linear(text_dim, width)
        self.action = nn.Linear(action_dim, width)
        self.local = nn.Linear(state_dim, width)
        self.global_effect = nn.Linear(state_dim, width)
        self.combine = nn.Sequential(nn.LayerNorm(5 * width), nn.Linear(5 * width, width), nn.GELU())

    def forward(self, rows: Mapping[str, Tensor]) -> Tensor:
        validate_compact_features(rows)
        context = self.context_global(rows["current_global"]) + self.context_text(rows["text_global"])
        action = self.action(rows["actions"])
        local = self.local(rows["local_mean"])
        global_effect = self.global_effect(rows["candidate_global_delta"])
        context = context[:, None].expand_as(action)
        return self.combine(
            torch.cat((context, action, local, global_effect, action * global_effect), dim=-1)
        )


class LegacyIndependentProbe(nn.Module):
    """Fresh per-candidate regressor over the raw information available to ScoreNet."""

    def __init__(self, state_dim: int, text_dim: int, action_dim: int, width: int = 256) -> None:
        super().__init__()
        self.encoder = _LegacyEncoder(state_dim, text_dim, action_dim, width)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, rows: Mapping[str, Tensor]) -> Tensor:
        return self.head(self.encoder(rows)).squeeze(-1)


class LegacySetRelativeProbe(nn.Module):
    """Legacy raw information plus permutation-equivariant sibling attention."""

    def __init__(self, state_dim: int, text_dim: int, action_dim: int, width: int = 256) -> None:
        super().__init__()
        self.encoder = _LegacyEncoder(state_dim, text_dim, action_dim, width)
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, rows: Mapping[str, Tensor]) -> Tensor:
        features = self.encoder(rows)
        attended, _ = self.attention(features, features, features, need_weights=False)
        return self.head(self.norm(features + attended)).squeeze(-1)


class RetrievalAugmentedIndependentProbe(nn.Module):
    """Independent legacy probe augmented only with target-free retrieval-space features."""

    def __init__(
        self, state_dim: int, text_dim: int, action_dim: int, query_dim: int, width: int = 256
    ) -> None:
        super().__init__()
        self.encoder = _LegacyEncoder(state_dim, text_dim, action_dim, width)
        self.current_query = nn.Linear(query_dim, width)
        self.candidate_query = nn.Linear(query_dim, width)
        self.delta_query = nn.Linear(query_dim, width)
        self.combine = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, width), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, rows: Mapping[str, Tensor]) -> Tensor:
        legacy = self.encoder(rows)
        current = self.current_query(rows["current_query"])[:, None].expand_as(legacy)
        retrieval = self.combine(
            torch.cat(
                (legacy, current, self.candidate_query(rows["candidate_queries"]), self.delta_query(rows["delta_q"])),
                dim=-1,
            )
        )
        return self.head(retrieval).squeeze(-1)


def probe_from_rows(name: str, rows: Mapping[str, Any], *, width: int = 256) -> nn.Module:
    dimensions = validate_compact_features(rows)
    del dimensions
    state_dim = int(rows["current_global"].shape[-1])
    text_dim = int(rows["text_global"].shape[-1])
    action_dim = int(rows["actions"].shape[-1])
    query_dim = int(rows["current_query"].shape[-1])
    if name == "legacy_independent":
        return LegacyIndependentProbe(state_dim, text_dim, action_dim, width)
    if name == "legacy_set_relative":
        return LegacySetRelativeProbe(state_dim, text_dim, action_dim, width)
    if name == "retrieval_augmented_independent":
        return RetrievalAugmentedIndependentProbe(state_dim, text_dim, action_dim, query_dim, width)
    raise ValueError(f"unsupported feature sufficiency probe: {name}")


def select_probe_scores(scores: Tensor, *, epsilon_stop: float) -> Tensor:
    """Production-equivalent raw-score STOP selection with routing disabled."""

    eligible = scores > epsilon_stop
    candidate = scores.masked_fill(~eligible, -torch.inf).argmax(dim=-1)
    return torch.where(eligible.any(dim=-1), candidate, torch.full_like(candidate, scores.shape[-1]))


def _rank(values: Tensor) -> Tensor:
    order = values.argsort().argsort().float()
    return order - order.mean()


def probe_metrics(scores: Tensor, utility: Tensor, *, epsilon_stop: float) -> dict[str, Any]:
    selected = select_probe_scores(scores, epsilon_stop=epsilon_stop)
    summary = selector_timestep_summary(utility, scores, selected, stop_threshold=epsilon_stop)
    calibration = summary["score_utility_calibration"]
    if calibration is None:
        return summary
    score_rank, utility_rank = _rank(scores.detach().flatten()), _rank(utility.detach().flatten())
    denominator = score_rank.norm() * utility_rank.norm()
    calibration["spearman"] = (
        float(score_rank.dot(utility_rank) / denominator)
        if float(denominator) > 1e-12
        else float("nan")
    )
    return summary


def probe_loss(scores: Tensor, utility: Tensor) -> Tensor:
    valid_rows = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    loss, _ = absolute_gain_loss(scores, utility, valid_rows, delta=1.0)
    return loss
