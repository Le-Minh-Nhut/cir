from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.images import ImageBatch
from losses.objective import IAGSRMEObjective
from losses.retrieval import positive_mask_from_ids
from models.iag_srme.model import IAGSRME


FASHIONIQ_EVALUATION_PROTOCOLS = ("fashioniq_original", "fashioniq_val")


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
            # Current target encoder participates normally in terminal retrieval. The marginal
            # evaluator detaches this bank inside MarginalActionLoss only.
            target_embeddings = model.encode_global_images(batch.target_pixels)
            target_ids = [str(value) for value in batch.target_ids]
            positives = positive_mask_from_ids(target_ids, device)
            components = objective(output, target_embeddings, positives)
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
    precision: PrecisionPolicy,
    validation_metrics: Mapping[str, Mapping[str, float]],
    selection_protocol: str | None,
    selection_metric: str = "mean_recall",
) -> None:
    serialized_metrics = {
        protocol: {name: float(value) for name, value in metrics.items()}
        for protocol, metrics in validation_metrics.items()
    }
    if set(serialized_metrics) != set(FASHIONIQ_EVALUATION_PROTOCOLS):
        raise ValueError("checkpoint requires both FashionIQ validation metric groups")
    if selection_protocol is not None and selection_protocol not in FASHIONIQ_EVALUATION_PROTOCOLS:
        raise ValueError("unsupported checkpoint selection protocol")
    selection_value = None
    if selection_protocol is not None:
        if selection_protocol not in serialized_metrics:
            raise ValueError("selection protocol is missing from validation metrics")
        if selection_metric not in serialized_metrics[selection_protocol]:
            raise ValueError("selection metric is missing from selected protocol metrics")
        selection_value = serialized_metrics[selection_protocol][selection_metric]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": selection_value,
            "metadata": {
                "backbone_type": model.backbone.backbone_type,
                "backbone_checkpoint": model.backbone.checkpoint,
                "backbone_revision": model.backbone.revision,
                "backbone_library": model.backbone.library,
                "backbone_library_version": model.backbone.library_version,
                "backbone_weights_repository": getattr(
                    model.backbone, "weights_repository", None
                ),
                "backbone_weights_revision": getattr(
                    model.backbone, "weights_revision", None
                ),
                # Retained for legacy readers. New dual-evaluation checkpoints use
                # selection_protocol as the authoritative provenance field.
                "evaluation_protocol": selection_protocol,
                "validation_metrics": serialized_metrics,
                "selection_protocol": selection_protocol,
                "selection_metric": selection_metric,
                "selection_metric_value": selection_value,
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
    evaluate: Callable[[IAGSRME], Mapping[str, Mapping[str, float]]],
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
    protocols = FASHIONIQ_EVALUATION_PROTOCOLS
    best = {protocol: float("-inf") for protocol in protocols}
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
        validation = {
            protocol: dict(metrics) for protocol, metrics in evaluate(model).items()
        }
        if set(validation) != set(protocols):
            raise ValueError("dual evaluation must return original and val protocol metrics")
        current = {
            protocol: float(validation[protocol][primary_metric]) for protocol in protocols
        }
        save_checkpoint(
            destination / "last.pt",
            model,
            objective,
            optimizer,
            epoch + 1,
            precision,
            validation,
            selection_protocol=None,
            selection_metric=primary_metric,
        )
        for protocol, filename in (
            ("fashioniq_original", "best_original.pt"),
            ("fashioniq_val", "best_val.pt"),
        ):
            if current[protocol] > best[protocol]:
                best[protocol] = current[protocol]
                save_checkpoint(
                    destination / filename,
                    model,
                    objective,
                    optimizer,
                    epoch + 1,
                    precision,
                    validation,
                    selection_protocol=protocol,
                    selection_metric=primary_metric,
                )
        print(
            f"epoch={epoch + 1}/{epochs} total={training['total']:.4f} "
            f"original_mean={current['fashioniq_original']:.3f} "
            f"original_best={best['fashioniq_original']:.3f} "
            f"val_mean={current['fashioniq_val']:.3f} "
            f"val_best={best['fashioniq_val']:.3f}"
        )
