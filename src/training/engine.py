from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
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
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    objective.train()
    amp_enabled = use_amp and device.type == "cuda"
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    progress = tqdm(loader, desc=f"train {epoch + 1}", dynamic_ncols=True)
    for cpu_batch in progress:
        batch = cpu_batch.to(device)
        if batch.target_pixels is None or any(value is None for value in batch.target_ids):
            raise ValueError("training batch requires raw target images and IDs")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(
                batch.reference_pixels,
                batch.input_ids,
                batch.attention_mask,
                batch.content_mask,
            )
            # Current target encoder participates normally in terminal retrieval. The marginal
            # evaluator detaches this bank inside MarginalActionLoss only.
            target_embeddings = model.encode_gallery(batch.target_pixels)
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
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric,
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
    use_amp: bool,
    primary_metric: str = "mean_recall",
) -> None:
    assert_training_setup(model, objective, optimizer, device)
    destination = Path(output_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
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
            use_amp=use_amp,
            epoch=epoch,
        )
        validation = dict(evaluate(model))
        metric = float(validation[primary_metric])
        save_checkpoint(destination / "last.pt", model, objective, optimizer, epoch + 1, metric)
        if metric > best:
            best = metric
            save_checkpoint(destination / "best.pt", model, objective, optimizer, epoch + 1, metric)
        print(
            f"epoch={epoch + 1}/{epochs} total={training['total']:.4f} "
            f"{primary_metric}={metric:.3f} best={best:.3f}"
        )
