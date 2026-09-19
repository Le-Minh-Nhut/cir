from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


_STEP_SCALARS = (
    "total",
    "terminal",
    "candidate_credit_loss",
    "candidate_credit_weighted",
    "dpp_raw",
    "dpp_weighted",
    "dac_gradient_candidate_fraction",
    "dac_responsibility_concentration",
    "useful_candidate_count",
    "functional_pairwise_cosine",
    "functional_rank",
    "stop_rate",
)
_DAC_STATE = (
    "dac_stage",
    "dac_num_groups",
    "dac_group_size",
    "dac_split_interval_steps",
)


def _scalar(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected scalar tensor, got shape {tuple(value.shape)}")
        return float(value.detach().float().cpu())
    return float(value)


def _indexed_values(components: Mapping[str, Any], prefix: str) -> list[float]:
    indexed: list[tuple[int, float]] = []
    for name, value in components.items():
        if name.startswith(prefix):
            indexed.append((int(name.removeprefix(prefix)), _scalar(value)))
    return [value for _, value in sorted(indexed)]


def build_step_log_record(
    *,
    global_step: int,
    objective_global_step: int | None = None,
    epoch: int,
    components: Mapping[str, Tensor | float | int],
    gradient_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe record for a successful optimizer update."""

    record: dict[str, Any] = {
        "global_step": int(global_step),
        "objective_global_step": int(
            global_step if objective_global_step is None else objective_global_step
        ),
        "epoch": int(epoch),
    }
    for name in _STEP_SCALARS:
        if name in components:
            record[name] = _scalar(components[name])
    for name in _DAC_STATE:
        if name in components:
            record[name] = int(round(_scalar(components[name])))
    record["dac_responsibility_frequency"] = _indexed_values(
        components, "dac_responsibility_frequency_"
    )
    record["dac_group_winning_frequency"] = _indexed_values(
        components, "dac_group_winning_frequency_"
    )
    if gradient_metrics is not None:
        record.update({name: float(value) for name, value in gradient_metrics.items()})
    return record


def append_step_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")


def per_slot_gradient_l2(loss: Tensor, parameter: nn.Parameter) -> Tensor:
    """Measure a loss gradient per first-axis slot without populating ``.grad``."""

    if parameter.ndim < 1:
        raise ValueError("slot parameter must have at least one dimension")
    if not loss.requires_grad or not parameter.requires_grad:
        return torch.zeros(parameter.shape[0], device=parameter.device)
    gradient = torch.autograd.grad(
        loss,
        parameter,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros(parameter.shape[0], device=parameter.device)
    return gradient.detach().float().flatten(1).norm(dim=1)


def slot_gradient_metrics(
    norms: Tensor,
    *,
    prefix: str = "proposal_query_grad_l2",
) -> dict[str, float]:
    if norms.ndim != 1 or norms.numel() == 0:
        raise ValueError("slot gradient norms must be a non-empty vector")
    values = norms.detach().float().cpu()
    minimum = values.min()
    maximum = values.max()
    nonzero = values > 0
    epsilon = torch.finfo(values.dtype).eps
    metrics = {
        f"{prefix}_{index}": float(value)
        for index, value in enumerate(values)
    }
    metrics.update(
        {
            f"{prefix}_min": float(minimum),
            f"{prefix}_max": float(maximum),
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_max_min_ratio": float(maximum / minimum.clamp_min(epsilon)),
            f"{prefix}_nonzero_count": float(nonzero.sum()),
            f"{prefix}_nonzero_fraction": float(nonzero.float().mean()),
        }
    )
    return metrics


def proposal_query_gradient_diagnostics(
    model: nn.Module,
    losses: Mapping[str, Tensor],
) -> dict[str, float]:
    proposal = getattr(model, "proposal", None)
    queries = getattr(proposal, "queries", None)
    if not isinstance(queries, nn.Parameter):
        return {}
    sources = (
        ("total", "proposal_query_grad_l2"),
        ("terminal", "terminal_proposal_query_grad_l2"),
        ("candidate_credit_weighted", "candidate_credit_proposal_query_grad_l2"),
    )
    metrics: dict[str, float] = {}
    for loss_name, prefix in sources:
        loss = losses.get(loss_name)
        if isinstance(loss, Tensor):
            metrics.update(
                slot_gradient_metrics(
                    per_slot_gradient_l2(loss, queries),
                    prefix=prefix,
                )
            )
    return metrics


def validate_checkpoint_candidate_count(
    checkpoint: Mapping[str, Any],
    configured_num_candidates: int,
) -> None:
    metadata = checkpoint.get("metadata")
    model_config = metadata.get("model_config") if isinstance(metadata, Mapping) else None
    if not isinstance(model_config, Mapping) or "num_candidates" not in model_config:
        return
    stored = int(model_config["num_candidates"])
    configured = int(configured_num_candidates)
    if stored == configured:
        return
    remedy = (
        " Load this K=8 checkpoint with model=iag_srme_k8."
        if stored == 8
        else " Select a model config with the checkpoint's candidate count."
    )
    raise ValueError(
        "checkpoint candidate-count mismatch: "
        f"stored={stored}, configured={configured}.{remedy}"
    )
