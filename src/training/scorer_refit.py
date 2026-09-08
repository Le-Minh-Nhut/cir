from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from losses.objective import absolute_gain_loss


SCORER_TENSOR_FIELDS = (
    "current_global",
    "text_global",
    "actions",
    "delta",
    "exec_mask",
    "candidate_global",
    "teacher_utility",
    "timestep",
)
SCORER_BATCH_FIELDS = frozenset((*SCORER_TENSOR_FIELDS, "sample_ids"))


def validate_scorer_batch(batch: Mapping[str, Any], num_candidates: int) -> dict[str, int]:
    """Validate a detached raw-input cache batch before scorer-only training."""

    missing = SCORER_BATCH_FIELDS - set(batch)
    unexpected = set(batch) - SCORER_BATCH_FIELDS
    if missing or unexpected:
        raise ValueError(
            f"invalid scorer cache fields: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if any("target" in name for name in batch if name != "teacher_utility"):
        raise ValueError("target-derived values are forbidden in ScoreNet input fields")

    tensors = {name: batch[name] for name in SCORER_TENSOR_FIELDS}
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("all scorer tensor fields must be torch.Tensor values")
    utility = tensors["teacher_utility"]
    if utility.ndim != 2 or utility.shape[1] != num_candidates:
        raise ValueError("teacher_utility must be [B,K]")
    rows = utility.shape[0]
    expected_shapes = {
        "current_global": (rows, None),
        "text_global": (rows, None),
        "actions": (rows, num_candidates, None),
        "delta": (rows, num_candidates, None, None),
        "exec_mask": (rows, num_candidates, None),
        "candidate_global": (rows, num_candidates, None),
        "timestep": (rows,),
    }
    for name, pattern in expected_shapes.items():
        value = tensors[name]
        if value.ndim != len(pattern) or any(
            expected is not None and value.shape[index] != expected
            for index, expected in enumerate(pattern)
        ):
            raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")
    if tensors["delta"].shape[2] != tensors["exec_mask"].shape[2]:
        raise ValueError("delta and exec_mask patch dimensions differ")
    state_dims = {
        tensors["current_global"].shape[-1],
        tensors["actions"].shape[-1],
        tensors["delta"].shape[-1],
        tensors["candidate_global"].shape[-1],
    }
    if len(state_dims) != 1:
        raise ValueError("cached ScoreNet state/action/global dimensions differ")
    if tensors["timestep"].is_floating_point():
        raise ValueError("timestep metadata must use an integer dtype")
    sample_ids = batch["sample_ids"]
    if not isinstance(sample_ids, list) or len(sample_ids) != rows:
        raise ValueError("sample_ids must be a list with one entry per decision")
    if not all(torch.isfinite(value).all() for name, value in tensors.items() if name != "timestep"):
        raise ValueError("scorer cache contains NaN or Inf")
    if any(value.requires_grad or value.grad_fn is not None for value in tensors.values()):
        raise ValueError("scorer cache tensors must be detached")
    return {"rows": rows, "num_candidates": num_candidates}


def freeze_for_score_refit(model: nn.Module) -> list[str]:
    """Freeze the generator/backbone and enable every ScoreNet parameter."""

    score_net = getattr(model, "score_net")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in score_net.parameters():
        parameter.requires_grad_(True)
    return [f"score_net.{name}" for name, _ in score_net.named_parameters()]


def assert_scorer_optimizer_ownership(model: nn.Module, optimizer: Optimizer) -> None:
    assert_module_optimizer_ownership(model.score_net, optimizer)


def assert_module_optimizer_ownership(module: nn.Module, optimizer: Optimizer) -> None:
    """Require an optimizer to own every and only the given module parameters."""

    expected = {id(parameter) for parameter in module.parameters()}
    actual_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    actual = {id(parameter) for parameter in actual_parameters}
    if len(actual_parameters) != len(actual):
        raise RuntimeError("scorer optimizer contains duplicate parameter references")
    if actual != expected:
        raise RuntimeError(
            "optimizer must own every and only the requested module parameters: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )


def build_score_refit_optimizer(
    model: nn.Module, *, learning_rate: float, weight_decay: float
) -> AdamW:
    freeze_for_score_refit(model)
    optimizer = AdamW(
        list(model.score_net.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    assert_scorer_optimizer_ownership(model, optimizer)
    return optimizer


def build_standalone_score_optimizer(
    score_net: nn.Module, *, learning_rate: float, weight_decay: float
) -> AdamW:
    """Create the optimizer for a standalone ScoreNet copy used by stream_refit."""

    for parameter in score_net.parameters():
        parameter.requires_grad_(True)
    optimizer = AdamW(
        list(score_net.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    assert_module_optimizer_ownership(score_net, optimizer)
    return optimizer


def scorer_gain_loss(
    score_net: nn.Module,
    batch: Mapping[str, Any],
    *,
    huber_delta: float,
) -> tuple[Tensor, Tensor]:
    """Run all trainable ScoreNet projections and the gain-only Huber objective."""

    features = score_net.build_features(
        batch["current_global"],
        batch["text_global"],
        batch["actions"],
        batch["delta"],
        batch["exec_mask"],
        batch["candidate_global"],
    )
    predicted = score_net(features)
    valid_rows = torch.ones(
        predicted.shape[0], dtype=torch.bool, device=predicted.device
    )
    loss, _ = absolute_gain_loss(
        predicted,
        batch["teacher_utility"],
        valid_rows,
        huber_delta,
    )
    return loss, predicted


def streaming_scorer_step(
    score_net: nn.Module,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    *,
    huber_delta: float,
) -> tuple[Tensor, Tensor]:
    """One gain-only update with no cache or behavior-model mutation."""

    optimizer.zero_grad(set_to_none=True)
    loss, predicted = scorer_gain_loss(score_net, batch, huber_delta=huber_delta)
    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in score_net.parameters()
    ):
        raise FloatingPointError("non-finite standalone ScoreNet gradient during stream refit")
    optimizer.step()
    return loss.detach(), predicted.detach()


def scorer_minibatches(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[dict[str, Any]]:
    rows = len(batch["sample_ids"])
    order = torch.randperm(rows, generator=generator)
    for start in range(0, rows, batch_size):
        indices = order[start : start + batch_size]
        yield {
            **{
                name: batch[name].index_select(0, indices)
                for name in SCORER_TENSOR_FIELDS
            },
            "sample_ids": [batch["sample_ids"][index] for index in indices.tolist()],
        }


def parameter_fingerprint(model: nn.Module, *, score_net: bool) -> str:
    """Hash ScoreNet or non-ScoreNet parameters without retaining a model copy."""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("score_net.") != score_net:
            continue
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_parameter_fingerprint(module: nn.Module) -> str:
    """Hash every parameter in a standalone or behavior module."""

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def non_score_parameter_fingerprint(model: nn.Module) -> str:
    return parameter_fingerprint(model, score_net=False)
