from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from backbones.fgclip2_base import (
    FGCLIP2_BASE_MODEL_ID,
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    TokenizedTextBatch,
    VisionTuningConfig,
)
from cache.taper_mag import GLOBAL_DIRECTORY, GlobalImageCache
from datasets.common import CIRBatch, collate_cir_samples
from datasets.fashioniq import resolve_fashioniq_correction_dicts
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    macro_average_fashioniq,
)
from models.one_shot_control import FGCLIP2OneShotControl, OneShotControlConfig
from models.taper_controls import (
    ReferenceOnlyControl,
    SimpleSumControl,
    TextControlConfig,
    TextOnlyControl,
)
from training.checkpointing import load_checkpoint, save_checkpoint
from training.fashioniq_runtime import CATEGORIES, load_config, make_dataset, seed_everything
from training.taper_control_registry import control_entry
from training.taper_mag_losses import terminal_bidirectional_infonce


ControlID = Literal["M0", "M1", "M2", "M3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-same-backbone M0-M3 CIR controls")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    return parser.parse_args()


def _control_id(config: dict[str, Any]) -> ControlID:
    experiment_id = str(config.get("control", {}).get("id", "")).upper()
    entry = control_entry(experiment_id)
    if experiment_id not in {"M0", "M1", "M2", "M3"} or not entry.available_in_this_pass:
        raise ValueError("Control runner supports exactly M0, M1, M2, and M3")
    return cast(ControlID, experiment_id)


def _validate_global_cache(
    cache: GlobalImageCache,
    config: dict[str, Any],
    split: str,
) -> None:
    expected = {
        "model_id": FGCLIP2_BASE_MODEL_ID,
        "revision": str(config["backbone"]["revision"]),
        "feature_dim": 768,
        "normalization": "L2",
        "cache_kind": "global",
        "image_scope": "complete_split",
        "split": split,
        "complete_split": True,
    }
    differences = {
        key: {"expected": value, "actual": getattr(cache.manifest, key)}
        for key, value in expected.items()
        if getattr(cache.manifest, key) != value
    }
    if differences:
        raise RuntimeError(f"Control/global cache mismatch: {differences}")


def build_control_model(
    control_id: ControlID,
    *,
    text_dim: int = 768,
    retrieval_dim: int = 768,
    hidden_dim: int = 768,
    variant: str | None = None,
) -> nn.Module:
    if control_id == "M0":
        return ReferenceOnlyControl()
    text_config = TextControlConfig(text_dim=text_dim, retrieval_dim=retrieval_dim)
    if control_id == "M1":
        return TextOnlyControl(text_config)
    if control_id == "M2":
        return SimpleSumControl(text_config)
    if variant not in {None, "gated_mlp_combiner"}:
        raise ValueError(f"Unsupported controlled M3 variant: {variant}")
    return FGCLIP2OneShotControl(
        OneShotControlConfig(
            text_dim=text_dim,
            retrieval_dim=retrieval_dim,
            hidden_dim=hidden_dim,
        )
    )


def _tokenize_online(
    batch: CIRBatch,
    backbone: FGCLIP2BaseBackbone,
    device: torch.device,
) -> tuple[TokenizedTextBatch, Tensor]:
    tokenized = backbone.tokenize_texts(tuple(batch.modification_texts))
    moved = TokenizedTextBatch(
        tokenized.input_ids.to(device),
        tokenized.attention_mask.to(device),
        tokenized.content_mask.to(device),
    )
    return moved, backbone.encode_text_tokens(moved)


def control_query(
    control_id: ControlID,
    model: nn.Module,
    batch: CIRBatch,
    cache: GlobalImageCache,
    backbone: FGCLIP2BaseBackbone | None,
    device: torch.device,
) -> Tensor:
    if control_id == "M0":
        return cast(ReferenceOnlyControl, model)(
            cache.by_ids(tuple(batch.reference_ids)).to(device).float()
        )
    if backbone is None:
        raise RuntimeError(f"{control_id} requires the online text backbone")
    tokenized, text_tokens = _tokenize_online(batch, backbone, device)
    if control_id == "M1":
        return cast(TextOnlyControl, model)(text_tokens, tokenized.content_mask)
    reference = cache.by_ids(tuple(batch.reference_ids)).to(device).float()
    if control_id == "M2":
        return cast(SimpleSumControl, model)(reference, text_tokens, tokenized.content_mask)
    return cast(FGCLIP2OneShotControl, model)(
        reference, text_tokens, tokenized.content_mask
    )


def _optimizer(
    model: nn.Module,
    backbone: FGCLIP2BaseBackbone,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    categorized: dict[tuple[str, bool], list[nn.Parameter]] = {}
    seen: set[int] = set()

    def add(category: str, name: str, parameter: nn.Parameter) -> None:
        if not parameter.requires_grad:
            return
        if id(parameter) in seen:
            raise RuntimeError(f"Duplicate control optimizer parameter: {name}")
        seen.add(id(parameter))
        lowered = name.lower()
        decay = parameter.ndim >= 2 and not lowered.endswith("bias") and "norm" not in lowered
        categorized.setdefault((category, decay), []).append(parameter)

    for name, parameter in model.named_parameters():
        add("composer", name, parameter)
    for name, parameter in backbone.model.named_parameters():
        add("text", f"backbone.{name}", parameter)
    learning_rates = {
        "composer": float(config["optimizer"]["actor_lr"]),
        "text": float(config["optimizer"]["text_lr"]),
    }
    groups = [
        {
            "params": parameters,
            "lr": learning_rates[category],
            "weight_decay": float(config["optimizer"]["weight_decay"]) if decay else 0.0,
            "group_name": f"{category}_{'decay' if decay else 'no_decay'}",
        }
        for (category, decay), parameters in categorized.items()
    ]
    return torch.optim.AdamW(
        groups,
        betas=tuple(config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
    )


def parameter_report(
    control_id: ControlID,
    model: nn.Module,
    backbone: FGCLIP2BaseBackbone | None,
) -> dict[str, int | str]:
    composer = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    text = (
        sum(parameter.numel() for parameter in backbone.model.text_model.parameters() if parameter.requires_grad)
        if backbone is not None
        else 0
    )
    return {
        "control": control_id,
        "composer_trainable_params": composer,
        "text_trainable_params": text,
        "total_trainable_params": composer + text,
    }


@torch.no_grad()
def validate_control(
    control_id: ControlID,
    model: nn.Module,
    backbone: FGCLIP2BaseBackbone | None,
    cache: GlobalImageCache,
    config: dict[str, Any],
    corrections: dict[str, dict[str, str]] | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if backbone is not None:
        backbone.eval()
    results: dict[str, dict[str, float]] = {}
    dataset_root = Path(config["data"]["dataset_root"])
    for category in CATEGORIES:
        dataset = make_dataset(config, "val", (category,), corrections)
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_cir_samples,
        )
        gallery_ids = build_fashioniq_gallery(
            protocol=config["data"]["validation_protocol"],
            split_root=dataset_root / "image_splits",
            category=category,
            annotations=dataset.annotations,
            split="val",
        )
        gallery = cache.by_ids(gallery_ids).to(device).float()
        scores: list[Tensor] = []
        target_ids: list[str] = []
        for batch in loader:
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                query = control_query(control_id, model, batch, cache, backbone, device)
            scores.append((query.float() @ gallery.T).cpu())
            target_ids.extend(str(target_id) for target_id in batch.target_ids)
        results[category] = evaluate_fashioniq_category(
            torch.cat(scores), target_ids, gallery_ids
        )
    metrics = macro_average_fashioniq(results)
    for category, values in results.items():
        metrics.update({f"{category}_{key}": value for key, value in values.items()})
    return metrics


def _build_backbone(config: dict[str, Any], device: torch.device) -> FGCLIP2BaseBackbone:
    vision = VisionTuningConfig(**config["backbone"]["vision_tuning"])
    if vision.mode != "frozen" or vision.num_unfrozen_blocks != 0:
        raise ValueError("Exact controls require frozen FG-CLIP2 vision")
    dtype = (
        torch.bfloat16
        if config["runtime"]["precision"] == "bf16" and device.type == "cuda"
        else torch.float32
    )
    return FGCLIP2BaseBackbone(
        revision=config["backbone"]["revision"],
        dtype=dtype,
        text_tuning=TextTuningConfig(**config["backbone"]["text_tuning"]),
        vision_tuning=vision,
    ).to(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    control_id = _control_id(config)
    seed_everything(int(config["seed"]))
    corrections = resolve_fashioniq_correction_dicts(
        Path(config["data"]["dataset_root"]) / "captions",
        str(config["data"]["correction_policy"]),
    )
    device = torch.device(
        config["runtime"]["device"]
        if config["runtime"]["device"] != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    cache_base = (
        Path(config["data"]["cache_root"])
        / "fashioniq"
        / "fgclip2-base"
        / str(config["backbone"]["revision"])
    )
    val_cache = GlobalImageCache.load(cache_base / "val" / GLOBAL_DIRECTORY)
    _validate_global_cache(val_cache, config, "val")
    backbone = None if control_id == "M0" else _build_backbone(config, device)
    model = build_control_model(
        control_id,
        text_dim=backbone.contract.text_dim if backbone is not None else 768,
        retrieval_dim=backbone.contract.retrieval_dim if backbone is not None else 768,
        hidden_dim=int(config.get("model", {}).get("hidden_dim", 768)),
        variant=config.get("model", {}).get("variant"),
    ).to(device)
    print(json.dumps(parameter_report(control_id, model, backbone), sort_keys=True))
    if control_id == "M0":
        if args.resume is not None:
            raise ValueError("M0 has no trainable checkpoint to resume")
        metrics = validate_control(control_id, model, None, val_cache, config, corrections, device)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    assert backbone is not None
    train_cache = GlobalImageCache.load(cache_base / "train" / GLOBAL_DIRECTORY)
    _validate_global_cache(train_cache, config, "train")
    manifest_hashes = {
        "train_global": train_cache.manifest.sha256,
        "val_global": val_cache.manifest.sha256,
    }
    if args.validate_only:
        if args.resume is None:
            raise ValueError("--validate-only requires --resume")
        payload = load_checkpoint(
            args.resume,
            model=model,
            backbone=backbone,
            expected_manifest_hashes=manifest_hashes,
        )
        if payload["stage"] != control_id:
            raise RuntimeError("Control checkpoint ID differs from current config")
        if payload["config"]["training"].get("max_optimizer_updates") != config[
            "training"
        ].get("max_optimizer_updates"):
            raise RuntimeError("Control checkpoint optimizer-update budget differs from config")
        metrics = validate_control(
            control_id, model, backbone, val_cache, config, corrections, device
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    dataset = make_dataset(config, "train", CATEGORIES, corrections)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_train_samples, len(dataset)))
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_cir_samples,
        pin_memory=device.type == "cuda",
    )
    optimizer = _optimizer(model, backbone, config)
    accumulation = int(config["training"]["gradient_accumulation"])
    epochs = int(config["training"]["epochs"])
    epoch_updates = math.ceil(len(loader) / accumulation)
    configured_max = config["training"].get("max_optimizer_updates")
    max_updates = int(configured_max) if configured_max is not None else epoch_updates * epochs
    if max_updates <= 0:
        raise ValueError("max_optimizer_updates must be positive")
    warmup = max(1, round(0.05 * max_updates))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(max_updates - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_metrics = {"mean_recall": -float("inf")}
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=model,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_manifest_hashes=manifest_hashes,
        )
        if payload["stage"] != control_id:
            raise RuntimeError("Control checkpoint ID differs from current config")
        saved_budget = payload["config"]["training"].get("max_optimizer_updates")
        if saved_budget != config["training"].get("max_optimizer_updates"):
            raise RuntimeError("Control checkpoint optimizer-update budget differs from config")
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_metrics = dict(payload["best_metrics"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        backbone.train()
        for micro_step, batch in enumerate(loader):
            if global_step >= max_updates:
                break
            if any(target is None for target in batch.target_ids):
                raise ValueError("Control training sample is missing target ID")
            target_ids = tuple(str(target) for target in batch.target_ids)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                query = control_query(control_id, model, batch, train_cache, backbone, device)
                targets = train_cache.by_ids(target_ids).to(device).float()
                loss = terminal_bidirectional_infonce(
                    query,
                    targets,
                    target_ids,
                    tuple((target_id,) for target_id in target_ids),
                    temperature=float(config["loss"]["retrieval_temperature"]),
                )
            (loss / accumulation).backward()
            should_update = (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    float(config["optimizer"]["gradient_clip"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        metrics = validate_control(
            control_id, model, backbone, val_cache, config, corrections, device
        )
        print(json.dumps({"epoch": epoch + 1, "global_step": global_step, **metrics}, sort_keys=True))
        improved = metrics["mean_recall"] > best_metrics["mean_recall"]
        if improved:
            best_metrics = dict(metrics)
        checkpoint_names = {"last.ckpt"}
        if improved:
            checkpoint_names.add("best_retrieval_valid.ckpt")
        for checkpoint_name in checkpoint_names:
            save_checkpoint(
                output_dir / checkpoint_name,
                model=model,
                backbone=backbone,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                stage=control_id,
                curriculum_state={"max_optimizer_updates": max_updates},
                resolved_config=config,
                manifest_hashes=manifest_hashes,
                best_metrics=best_metrics,
            )
        if global_step >= max_updates:
            break


if __name__ == "__main__":
    main()
