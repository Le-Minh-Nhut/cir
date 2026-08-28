from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from backbones.fgclip2_base import FGCLIP2BaseBackbone, TextTuningConfig, TokenizedTextBatch, VisionTuningConfig
from cache.taper_mag import GLOBAL_DIRECTORY, GlobalImageCache
from datasets.common import CIRBatch, collate_cir_samples
from datasets.fashioniq import resolve_fashioniq_correction_dicts
from evaluation.fashioniq import build_fashioniq_gallery, evaluate_fashioniq_category, macro_average_fashioniq
from models.one_shot_control import FGCLIP2OneShotControl, OneShotControlConfig
from training.taper_mag_losses import terminal_bidirectional_infonce
from training.checkpointing import load_checkpoint, save_checkpoint
from train_taper_mag import CATEGORIES, load_config, make_dataset, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-backbone FG-CLIP2 one-shot CIR control")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _online_inputs(
    batch: CIRBatch,
    cache: GlobalImageCache,
    backbone: FGCLIP2BaseBackbone,
    device: torch.device,
) -> tuple[torch.Tensor, TokenizedTextBatch]:
    reference = cache.by_ids(tuple(batch.reference_ids)).to(device).float()
    tokenized = backbone.tokenize_texts(tuple(batch.modification_texts))
    return reference, TokenizedTextBatch(
        tokenized.input_ids.to(device),
        tokenized.attention_mask.to(device),
        tokenized.content_mask.to(device),
    )


def _optimizer(
    model: FGCLIP2OneShotControl,
    backbone: FGCLIP2BaseBackbone,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    categorized: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}

    def add(category: str, name: str, parameter: torch.nn.Parameter) -> None:
        if not parameter.requires_grad:
            return
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


@torch.no_grad()
def validate(
    model: FGCLIP2OneShotControl,
    backbone: FGCLIP2BaseBackbone,
    cache: GlobalImageCache,
    config: dict[str, Any],
    corrections: dict[str, dict[str, str]] | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
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
        scores: list[torch.Tensor] = []
        targets: list[str] = []
        for batch in loader:
            reference, tokenized = _online_inputs(batch, cache, backbone, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                text = backbone.encode_text_tokens(tokenized)
                query = model(reference, text, tokenized.content_mask)
            scores.append((query @ gallery.T).cpu())
            targets.extend(str(target) for target in batch.target_ids)
        results[category] = evaluate_fashioniq_category(torch.cat(scores), targets, gallery_ids)
    metrics = macro_average_fashioniq(results)
    for category, values in results.items():
        metrics.update({f"{category}_{key}": value for key, value in values.items()})
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("experiment") != "fgclip2_one_shot_control":
        raise ValueError("One-shot entry point requires experiment=fgclip2_one_shot_control")
    seed_everything(int(config["seed"]))
    corrections = resolve_fashioniq_correction_dicts(
        Path(config["data"]["dataset_root"]) / "captions",
        str(config["data"]["correction_policy"]),
    )
    vision_cfg = VisionTuningConfig(**config["backbone"]["vision_tuning"])
    if vision_cfg.mode != "frozen" or vision_cfg.num_unfrozen_blocks != 0:
        raise ValueError("One-shot control requires frozen FG-CLIP2 vision")
    device = torch.device(
        config["runtime"]["device"]
        if config["runtime"]["device"] != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if config["runtime"]["precision"] == "bf16" and device.type == "cuda" else torch.float32
    backbone = FGCLIP2BaseBackbone(
        revision=config["backbone"]["revision"],
        dtype=dtype,
        text_tuning=TextTuningConfig(**config["backbone"]["text_tuning"]),
        vision_tuning=vision_cfg,
    ).to(device)
    model = FGCLIP2OneShotControl(
        OneShotControlConfig(
            text_dim=backbone.contract.text_dim,
            retrieval_dim=backbone.contract.retrieval_dim,
            hidden_dim=int(config["model"]["hidden_dim"]),
        )
    ).to(device)
    print(json.dumps({"one_shot_trainable_params": model.trainable_parameter_count}))
    cache_base = Path(config["data"]["cache_root"]) / "fashioniq" / "fgclip2-base" / backbone.revision
    train_cache = GlobalImageCache.load(cache_base / "train" / GLOBAL_DIRECTORY)
    val_cache = GlobalImageCache.load(cache_base / "val" / GLOBAL_DIRECTORY)
    manifest_hashes = {
        "train_global": train_cache.manifest.sha256,
        "val_global": val_cache.manifest.sha256,
    }
    if args.validate_only:
        if args.resume is None:
            raise ValueError("--validate-only requires --resume")
        load_checkpoint(
            args.resume,
            model=model,
            backbone=backbone,
            expected_manifest_hashes=manifest_hashes,
        )
        print(json.dumps(validate(model, backbone, val_cache, config, corrections, device), indent=2))
        return
    dataset = make_dataset(config, "train", CATEGORIES, corrections)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_cir_samples,
    )
    optimizer = _optimizer(model, backbone, config)
    accumulation = int(config["training"]["gradient_accumulation"])
    epochs = int(config["training"]["epochs"])
    updates = max(1, math.ceil(len(loader) / accumulation) * epochs)
    warmup = max(1, round(0.05 * updates))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(updates - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
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
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_metrics = dict(payload["best_metrics"])
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        backbone.train()
        for micro_step, batch in enumerate(loader):
            if any(target is None for target in batch.target_ids):
                raise ValueError("Training sample is missing target ID")
            targets = tuple(str(target) for target in batch.target_ids)
            reference, tokenized = _online_inputs(batch, train_cache, backbone, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                text = backbone.encode_text_tokens(tokenized)
                query = model(reference, text, tokenized.content_mask)
                target = train_cache.by_ids(targets).to(device).float()
                loss = terminal_bidirectional_infonce(
                    query, target, targets, tuple((target_id,) for target_id in targets),
                    temperature=float(config["loss"]["retrieval_temperature"]),
                )
            (loss / accumulation).backward()
            if (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    float(config["optimizer"]["gradient_clip"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        metrics = validate(model, backbone, val_cache, config, corrections, device)
        print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True))
        best_metrics = dict(metrics) if metrics["mean_recall"] > best_metrics["mean_recall"] else best_metrics
        save_checkpoint(
            output / "last.ckpt",
            model=model,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            stage="one_shot_control",
            curriculum_state={},
            resolved_config=config,
            manifest_hashes=manifest_hashes,
            best_metrics=best_metrics,
        )


if __name__ == "__main__":
    main()
