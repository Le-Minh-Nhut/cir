from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from cache.features import (
    DenseImageFeatureCache,
    TextFeatureCache,
    get_dense_features_by_ids,
    get_features_by_ids,
    get_text_features_with_global_by_sample_ids,
)
from datasets.common import CIRBatch


def prepare_entity_action_batch(
    batch: CIRBatch,
    device: torch.device,
    global_image_features: torch.Tensor,
    image_name_to_idx: dict[str, int],
    dense_image_cache: DenseImageFeatureCache,
    text_cache: TextFeatureCache,
) -> dict[str, object]:
    target_ids = list(batch.target_ids)
    if any(target_id is None for target_id in target_ids):
        raise ValueError("Training sample is missing target_id")
    target_ids_str = [str(target_id) for target_id in target_ids]

    def globals_for(image_ids: list[str]) -> torch.Tensor:
        values = get_features_by_ids(image_ids, global_image_features, image_name_to_idx)
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError("FG-CLIP2 global image cache must be [N,1,D]")
        return values[:, 0].to(device=device, dtype=torch.float32).detach()

    reference_dense, reference_dense_mask = get_dense_features_by_ids(
        batch.reference_ids, dense_image_cache
    )
    target_dense, target_dense_mask = get_dense_features_by_ids(
        target_ids_str, dense_image_cache
    )
    text_states, attention_mask, content_mask, text_global = (
        get_text_features_with_global_by_sample_ids(
            batch.sample_ids, batch.modification_texts, text_cache
        )
    )
    if (content_mask.to(torch.bool) & ~attention_mask.to(torch.bool)).any():
        raise ValueError("Text content cache includes padding")
    prepared: dict[str, object] = {
        "reference_global": globals_for(batch.reference_ids),
        "reference_dense": reference_dense.to(device=device).detach(),
        "reference_dense_mask": reference_dense_mask.to(device=device).detach(),
        "target_global": globals_for(target_ids_str),
        "target_dense": target_dense.to(device=device).detach(),
        "target_dense_mask": target_dense_mask.to(device=device).detach(),
        "text_global": text_global.to(device=device, dtype=torch.float32).detach(),
        "text_states": text_states.to(device=device, dtype=torch.float32).detach(),
        "text_content_mask": content_mask.to(device=device, dtype=torch.bool).detach(),
        "target_ids": target_ids_str,
    }
    return prepared


def _total_loss(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    if not weights:
        raise ValueError("loss weights must not be empty")
    missing = set(weights) - set(losses)
    if missing:
        raise KeyError(f"Model omitted configured loss terms: {sorted(missing)}")
    return sum((float(weight) * losses[name] for name, weight in weights.items()), start=torch.zeros((), device=next(iter(losses.values())).device))


def _checkpoint(model: nn.Module, *, epoch: int, metric: float) -> dict[str, object]:
    provenance = model.experiment_provenance()  # type: ignore[attr-defined]
    return {
        "model_state_dict": model.state_dict(),
        "experiment_provenance": provenance,
        "epoch": epoch,
        "mean_recall": metric,
    }


def load_entity_action_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("A8.0 checkpoint must contain model_state_dict and provenance")
    expected = model.experiment_provenance()  # type: ignore[attr-defined]
    if checkpoint.get("experiment_provenance") != expected:
        raise RuntimeError(
            "A8.0 checkpoint provenance mismatch: "
            f"expected={expected}, found={checkpoint.get('experiment_provenance')}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def fit_entity_action_binding(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    evaluate_fn: Callable[[nn.Module], dict[str, float]],
    prepare_batch_fn: Callable[[CIRBatch, torch.device], dict[str, object]],
    *,
    num_epochs: int,
    device: torch.device,
    loss_weights: dict[str, float],
    output_dir: str | Path,
    use_amp: bool,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best = float("-inf")
    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    for epoch in range(num_epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        progress = tqdm(train_loader, desc=f"A8.0 train [{epoch + 1}]", dynamic_ncols=True)
        for raw_batch in progress:
            optimizer.zero_grad(set_to_none=True)
            batch = prepare_batch_fn(raw_batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                components = model.compute_loss(batch)  # type: ignore[attr-defined]
                total = _total_loss(components, loss_weights)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["total_loss"] += float(total.detach())
            for name, value in components.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(loss=f"{float(total.detach()):.4f}")
        if steps == 0:
            raise RuntimeError("Training loader produced no batches")

        model.eval()
        with torch.no_grad():
            validation = evaluate_fn(model)
        mean_recall = float(validation["mean_recall"])
        torch.save(_checkpoint(model, epoch=epoch + 1, metric=mean_recall), output_dir / "last.pt")
        if mean_recall > best:
            best = mean_recall
            torch.save(_checkpoint(model, epoch=epoch + 1, metric=mean_recall), output_dir / "best.pt")
        averaged = {name: value / steps for name, value in totals.items()}
        persisted_metrics: dict[str, float | int] = {"epoch": epoch + 1}
        persisted_metrics.update(
            {f"train/{name}": value for name, value in averaged.items()}
        )
        persisted_metrics.update(
            {f"val/{name}": float(value) for name, value in validation.items()}
        )
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(persisted_metrics, allow_nan=False) + "\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} | loss={averaged['total_loss']:.4f} | "
            f"R@10={validation['recall_at_10']:.2f} | R@50={validation['recall_at_50']:.2f} | "
            f"mean={mean_recall:.2f} | best={best:.2f} | "
            f"relation_cos={averaged.get('diagnostic/relation_offdiag_cosine_mean', float('nan')):.3f}"
        )
