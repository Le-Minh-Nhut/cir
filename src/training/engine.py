from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.images import ImageBatch
from losses.objective import IAGSRMEObjective
from models.iag_srme.model import IAGSRME


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    name: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool


def resolve_precision(name: str, device: torch.device) -> PrecisionPolicy:
    """Resolve the configured precision without conflating fp16 and bf16."""

    if name == "fp32":
        return PrecisionPolicy(name, False, None, False)
    if name == "fp16":
        return PrecisionPolicy(name, True, torch.float16, device.type == "cuda")
    if name == "bf16":
        return PrecisionPolicy(name, True, torch.bfloat16, False)
    raise ValueError(f"unsupported precision: {name}; expected fp32, fp16, or bf16")


def trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """Return the exact parameter objects that an optimizer must own."""

    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def assert_training_setup(
    model: nn.Module,
    objective: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> None:
    """Validate device ownership and optimizer identity before any update."""

    expected = trainable_parameters(model, objective)
    wrong_device = [parameter.device for parameter in expected if parameter.device != device]
    if wrong_device:
        raise RuntimeError(
            f"model/objective must be moved to {device} before optimizer construction; "
            f"found parameter on {wrong_device[0]}"
        )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_ids = {id(parameter) for parameter in expected}
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_parameters) != len(optimizer_ids):
        raise RuntimeError("optimizer contains duplicate parameter references")
    if optimizer_ids != expected_ids:
        missing = len(expected_ids - optimizer_ids)
        stale_or_extra = len(optimizer_ids - expected_ids)
        raise RuntimeError(
            "optimizer parameters do not match the live model/objective objects: "
            f"missing={missing}, stale_or_extra={stale_or_extra}"
        )


def set_epoch(loader: DataLoader[ImageBatch], epoch: int) -> None:
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)


def train_one_epoch(
    model: IAGSRME,
    objective: IAGSRMEObjective,
    loader: DataLoader[ImageBatch],
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    precision: PrecisionPolicy,
    epoch: int,
) -> dict[str, float]:
    model.train()
    objective.train()
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    progress = tqdm(loader, desc=f"train {epoch + 1}", dynamic_ncols=True)
    for cpu_batch in progress:
        batch = cpu_batch.to(device)
        if batch.target_pixels is None or any(value is None for value in batch.target_ids):
            raise ValueError("training batch requires raw target images and IDs")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=precision.autocast_enabled,
            dtype=precision.autocast_dtype,
        ):
            output = model(
                batch.reference_pixels,
                batch.input_ids,
                batch.attention_mask,
                batch.content_mask,
            )
            # Target encoding participates in terminal retrieval; the teacher path detaches it.
            target_embeddings = model.encode_global_images(batch.target_pixels)
            target_ids = [str(value) for value in batch.target_ids]
            components = objective(output, target_embeddings, target_ids)
            loss = components["total"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        for name, value in components.items():
            totals[name] += float(value.detach())
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if steps == 0:
        raise RuntimeError("empty training loader")
    return {name: value / steps for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: IAGSRME,
    objective: IAGSRMEObjective,
    optimizer: Optimizer,
    epoch: int,
    metric: float,
    precision: PrecisionPolicy,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "metadata": {
                "architecture": "iag-srme-v2-r0",
                "backbone_track": "B/FG-CLIP-v1",
                "backbone_checkpoint": model.backbone.checkpoint,
                "backbone_revision": model.backbone.revision,
                "recurrent_state": "penultimate_patch_tokens_without_cls",
                "global_query_initialization": "checkpoint_class_embedding",
                "patch_grid": model.backbone.patch_grid,
                "state_dim": model.backbone.state_dim,
                "dense_dim": model.backbone.dense_dim,
                "retrieval_dim": model.backbone.retrieval_dim,
                "retrieval_normalization": "l2_fp32",
                "model_config": asdict(model.config),
                "objective_config": asdict(objective.config),
                "teacher_policy": "identity_aware_false_negative_safe_in_batch",
                "stop_bootstrap_curriculum": "none",
                "precision": precision.name,
            },
        },
        path,
    )


def fit(
    model: IAGSRME,
    objective: IAGSRMEObjective,
    train_loader: DataLoader[ImageBatch],
    optimizer: Optimizer,
    evaluate: Callable[[IAGSRME], Mapping[str, float]],
    *,
    epochs: int,
    device: torch.device,
    output_dir: str | Path,
    precision: PrecisionPolicy,
    primary_metric: str = "mean_recall",
) -> None:
    assert_training_setup(model, objective, optimizer, device)
    destination = Path(output_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.scaler_enabled)
    best = float("-inf")
    for epoch in range(epochs):
        set_epoch(train_loader, epoch)
        training = train_one_epoch(
            model,
            objective,
            train_loader,
            optimizer,
            scaler,
            device,
            precision=precision,
            epoch=epoch,
        )
        validation = dict(evaluate(model))
        metric = float(validation[primary_metric])
        save_checkpoint(
            destination / "last.pt",
            model,
            objective,
            optimizer,
            epoch + 1,
            metric,
            precision,
        )
        if metric > best:
            best = metric
            save_checkpoint(
                destination / "best.pt",
                model,
                objective,
                optimizer,
                epoch + 1,
                metric,
                precision,
            )
        print(
            f"epoch={epoch + 1}/{epochs} total={training['total']:.4f} "
            f"{primary_metric}={metric:.3f} best={best:.3f}"
        )
