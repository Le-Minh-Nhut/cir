from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

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


class JSONLLogger:
    """Tiny append-only run logger; one self-contained JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), allow_nan=True) + "\n")


def _small_trainable_named(module: nn.Module | None) -> tuple[str, nn.Parameter] | None:
    if module is None:
        return None
    candidates = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    return min(candidates, key=lambda item: item[1].numel(), default=None)


def build_optimizer_probes(model: IAGSRME) -> dict[str, tuple[str, nn.Parameter]]:
    """One representative tensor per scientifically relevant trainable group."""

    modules = {
        "text_encoder": getattr(model.backbone.model, "text_model", None),
        "proposal": model.proposal,
        "grounder": model.grounder,
        "action_fusion": model.action_fusion,
        "executor": model.executor,
        "score_net": model.score_net,
    }
    probes = {}
    for group, module in modules.items():
        selected = _small_trainable_named(module)
        if selected is not None:
            probes[group] = selected
    return probes


def _gradient_diagnostics(parameters: list[nn.Parameter]) -> tuple[int, int]:
    nonfinite = 0
    elements = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        elements += gradient.numel()
        nonfinite += int((~torch.isfinite(gradient)).sum())
    return nonfinite, elements


def _probe_diagnostics(
    probes: Mapping[str, tuple[str, nn.Parameter]],
    before: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float | str | None]]:
    report = {}
    for group, (name, parameter) in probes.items():
        value = parameter.detach().float()
        gradient = parameter.grad
        update = value - before[group]
        parameter_norm = float(value.norm())
        update_norm = float(update.norm())
        report[group] = {
            "parameter": name,
            "parameter_norm": parameter_norm,
            "gradient_norm": (
                float(gradient.detach().float().norm()) if gradient is not None else None
            ),
            "parameter_delta_norm": update_norm,
            "update_to_weight_ratio": update_norm / max(parameter_norm, 1e-12),
        }
    return report


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


def parameter_count_diagnostics(model: IAGSRME, objective: IAGSRMEObjective) -> dict[str, int]:
    """Compact parameter ownership summary for fine-tuning ablations."""

    def count(modules: nn.Module | tuple[nn.Module, ...], trainable: bool = True) -> int:
        selected = (modules,) if isinstance(modules, nn.Module) else modules
        return sum(
            parameter.numel()
            for module in selected
            for parameter in module.parameters()
            if not trainable or parameter.requires_grad
        )

    task_modules = (
        model.proposal,
        model.grounder,
        model.action_fusion,
        model.executor,
        model.score_net,
    )
    return {
        "total_parameters": count((model, objective), trainable=False),
        "total_trainable_parameters": count((model, objective)),
        "trainable_vision_parameters": count(model.backbone.model.vision_model),
        "trainable_visual_projection_parameters": count(model.backbone.model.visual_projection),
        "trainable_text_parameters": count(model.backbone.model.text_model),
        "trainable_text_adapter_parameters": count(model.backbone.text_adapter),
        "trainable_q_g_parameters": (
            model.backbone.q_G.numel() if model.backbone.q_G.requires_grad else 0
        ),
        "trainable_iag_srme_parameters": count(task_modules),
        "trainable_objective_parameters": count(objective),
    }


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
    logger: JSONLLogger | None = None,
    optimizer_step_start: int = 0,
    batch_step_start: int = 0,
    log_interval: int = 1,
) -> dict[str, float]:
    model.train()
    objective.train()
    totals: defaultdict[str, float] = defaultdict(float)
    steps = 0
    true_optimizer_steps = optimizer_step_start
    batch_steps = batch_step_start
    parameters = trainable_parameters(model, objective)
    probes = build_optimizer_probes(model)
    progress = tqdm(loader, desc=f"train {epoch + 1}", dynamic_ncols=True)
    for cpu_batch in progress:
        batch_steps += 1
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
            components = objective(output, target_embeddings, target_ids, batch.modification_texts)
            loss = components["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite_gradients, gradient_elements = _gradient_diagnostics(parameters)
        before = {
            group: parameter.detach().float().clone() for group, (_, parameter) in probes.items()
        }
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        scale_decreased = scale_after < scale_before
        skipped = bool(precision.scaler_enabled and scale_decreased)
        if not skipped:
            true_optimizer_steps += 1
        steps += 1
        for name, value in components.items():
            totals[name] += float(value.detach())
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if logger is not None and (steps % max(log_interval, 1) == 0):
            component_values = {name: float(value.detach()) for name, value in components.items()}
            logger.write(
                {
                    "record_type": "train_update",
                    "epoch": epoch + 1,
                    "optimizer_step": true_optimizer_steps,
                    "batch_step": batch_steps,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "amp_scale": scale_after,
                    "amp_scale_decreased": scale_decreased,
                    "amp_skipped_update_inferred": skipped,
                    "nonfinite_gradient_count": nonfinite_gradients,
                    "gradient_element_count": gradient_elements,
                    **component_values,
                    "parameter_updates": _probe_diagnostics(probes, before),
                }
            )
    if steps == 0:
        raise RuntimeError("empty training loader")
    result = {name: value / steps for name, value in totals.items()}
    result["optimizer_step_count"] = float(true_optimizer_steps)
    result["batch_step_count"] = float(batch_steps)
    return result


def save_checkpoint(
    path: Path,
    model: IAGSRME,
    objective: IAGSRMEObjective,
    optimizer: Optimizer,
    epoch: int,
    metric: float,
    precision: PrecisionPolicy,
    *,
    scaler: torch.amp.GradScaler | None = None,
    optimizer_step: int = 0,
    batch_step: int = 0,
    run_metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "optimizer_step": optimizer_step,
            "batch_step": batch_step,
            "epoch": epoch,
            "metric": metric,
            "metadata": {
                "architecture": "iag-srme-v2-r0",
                "backbone_track": "B/FG-CLIP-v1",
                "backbone_checkpoint": model.backbone.checkpoint,
                "backbone_revision": model.backbone.revision,
                "recurrent_state": "penultimate_patch_tokens_without_cls",
                "global_readout_mode": model.backbone.global_readout_mode,
                "readout_experiment": model.backbone.readout_experiment,
                "finetune_policy": model.backbone.finetune_policy,
                "experiment_identity": model.backbone.experiment_identity,
                "train_vision": model.backbone.train_vision,
                "train_text": model.backbone.train_text,
                "train_text_projection": model.backbone.train_text_projection,
                "global_query_initialization": (
                    "checkpoint_class_embedding"
                    if model.backbone.global_readout_mode == "learned_qg"
                    else "image_specific_penultimate_cls"
                ),
                "patch_grid": model.backbone.patch_grid,
                "state_dim": model.backbone.state_dim,
                "dense_dim": model.backbone.dense_dim,
                "retrieval_dim": model.backbone.retrieval_dim,
                "retrieval_normalization": "l2_fp32",
                "model_config": asdict(model.config),
                "objective_config": asdict(objective.config),
                "teacher_policy": "identity_aware_false_negative_safe_in_batch",
                "concept_parser_version": objective.config.concept_parser_version,
                "concept_vocabulary_size": (
                    len(objective.concept.vocabulary.concepts)
                    if objective.concept is not None
                    else 0
                ),
                "concept_vocabulary_fingerprint": (
                    objective.concept.vocabulary.fingerprint
                    if objective.concept is not None
                    else None
                ),
                "concept_vocabulary": (
                    objective.concept.vocabulary.concepts if objective.concept is not None else ()
                ),
                "correspondence": "disabled",
                "stop_bootstrap_curriculum": "none",
                "precision": precision.name,
                "number_of_optimizer_updates": optimizer_step,
                "run": dict(run_metadata or {}),
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
    logging_interval: int = 1,
    run_metadata: Mapping[str, Any] | None = None,
) -> None:
    assert_training_setup(model, objective, optimizer, device)
    destination = Path(output_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=precision.scaler_enabled)
    logger = JSONLLogger(destination / "metrics.jsonl")
    logger.write({"record_type": "run_start", **dict(run_metadata or {})})
    best = float("-inf")
    optimizer_step = 0
    batch_step = 0
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
            logger=logger,
            optimizer_step_start=optimizer_step,
            batch_step_start=batch_step,
            log_interval=logging_interval,
        )
        optimizer_step = int(training.pop("optimizer_step_count"))
        batch_step = int(training.pop("batch_step_count"))
        validation = dict(evaluate(model))
        logger.write(
            {
                "record_type": "validation",
                "epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "batch_step": batch_step,
                **validation,
            }
        )
        metric = float(validation[primary_metric])
        save_checkpoint(
            destination / "last.pt",
            model,
            objective,
            optimizer,
            epoch + 1,
            metric,
            precision,
            scaler=scaler,
            optimizer_step=optimizer_step,
            batch_step=batch_step,
            run_metadata=run_metadata,
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
                scaler=scaler,
                optimizer_step=optimizer_step,
                batch_step=batch_step,
                run_metadata=run_metadata,
            )
        print(
            f"epoch={epoch + 1}/{epochs} total={training['total']:.4f} "
            f"{primary_metric}={metric:.3f} best={best:.3f}"
        )
