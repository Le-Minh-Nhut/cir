from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from backbones.fgclip2_base import (
    FGCLIP2BaseBackbone,
    TextTuningConfig,
    VisionTuningConfig,
)
from cache.taper_mag import FeatureSourcePolicy, FrozenVisionCache, stable_json_hash
from datasets.common import CIRBatch, collate_cir_samples
from datasets.fashioniq import (
    FashionIQDataset,
    resolve_fashioniq_correction_dicts,
)
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    macro_average_fashioniq,
)
from models.taper_mag.contracts import PolicyBatch, SupervisionBatch
from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.rollout import RolloutConfig
from training.checkpointing import load_checkpoint, save_checkpoint
from training.marginal_gain_teacher import MarginalGainTeacher
from training.negative_bank import NegativeBank
from training.taper_mag_engine import (
    CurriculumStage,
    EngineConfig,
    TaperMAGTrainingEngine,
    encode_policy_batch,
)
from training.taper_mag_diagnostics import summarize_training_diagnostics
from training.taper_mag_optimizer import (
    OptimizerConfig,
    build_optimizer,
    format_parameter_report,
)
from training.text_drift import TextDriftMonitor, text_block_gradient_norms


CATEGORIES = ("dress", "shirt", "toptee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAPER-MAG V4 on FashionIQ")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported TAPER-MAG config schema")
    return config


def make_dataset(
    config: dict[str, Any],
    split: str,
    categories: tuple[str, ...],
    correction_dicts: dict[str, dict[str, str]] | None,
) -> FashionIQDataset:
    annotation_root = Path(config["data"]["dataset_root"]) / "captions"
    policy_key = "train_caption_policy" if split == "train" else "validation_caption_policy"
    return FashionIQDataset(
        annotation_root,
        split,
        categories,
        caption_policy=config["data"][policy_key],
        correction_dicts=correction_dicts,
        seed=int(config["seed"]),
    )


def build_policy_batch(
    batch: CIRBatch,
    cache: FrozenVisionCache,
    backbone: FGCLIP2BaseBackbone,
    device: torch.device,
) -> PolicyBatch:
    reference_ids = tuple(batch.reference_ids)
    texts = tuple(batch.modification_texts)
    local, local_mask, shapes = cache.dense_by_ids(reference_ids)
    reference_global = cache.global_by_ids(reference_ids)
    tokenized = backbone.tokenize_texts(texts)
    return PolicyBatch(
        reference_ids=reference_ids,
        modification_texts=texts,
        reference_local=local.to(device).float(),
        reference_local_mask=local_mask.to(device),
        reference_global=reference_global.to(device).float(),
        text_input_ids=tokenized.input_ids.to(device),
        text_attention_mask=tokenized.attention_mask.to(device),
        text_content_mask=tokenized.content_mask.to(device),
        spatial_shapes=shapes.to(device),
    )


def build_supervision_batch(
    batch: CIRBatch,
    cache: FrozenVisionCache,
    device: torch.device,
) -> SupervisionBatch:
    if any(target_id is None for target_id in batch.target_ids):
        raise ValueError("Training/validation FashionIQ sample is missing target ID")
    target_ids = tuple(str(target_id) for target_id in batch.target_ids)
    target_global = cache.global_by_ids(target_ids)
    return SupervisionBatch(
        target_embedding=target_global.to(device).float(),
        target_ids=target_ids,
        positive_ids=tuple((target_id,) for target_id in target_ids),
    )


def validate_cache(cache: FrozenVisionCache, backbone: FGCLIP2BaseBackbone, split: str) -> None:
    manifest = backbone.manifest()
    processor_hash = stable_json_hash(manifest.image_processor_config)
    for cache_manifest, expected_dim, expected_normalization in (
        (cache.global_manifest, backbone.contract.retrieval_dim, "L2"),
        (cache.dense_manifest, backbone.contract.vision_dim, "none"),
    ):
        expected = {
            "model_id": backbone.model_id,
            "revision": backbone.revision,
            "processor_config_hash": processor_hash,
            "feature_dim": expected_dim,
            "patch_policy": manifest.vision_patch_policy,
            "split": split,
            "normalization": expected_normalization,
            "complete_split": True,
            "cache_kind": "global" if expected_normalization == "L2" else "dense_reference",
            "image_scope": "complete_split" if expected_normalization == "L2" else "reference_only",
        }
        actual = cache_manifest.__dict__ if hasattr(cache_manifest, "__dict__") else {
            field: getattr(cache_manifest, field) for field in expected
        }
        differences = {key: (value, actual[key]) for key, value in expected.items() if actual[key] != value}
        if differences:
            raise RuntimeError(f"Cache/backbone contract mismatch: {differences}")


@torch.no_grad()
def validate_fashioniq(
    model: TaperMAG,
    backbone: FGCLIP2BaseBackbone,
    cache: FrozenVisionCache,
    config: dict[str, Any],
    device: torch.device,
    stage: CurriculumStage,
    correction_dicts: dict[str, dict[str, str]] | None,
) -> dict[str, float]:
    model.eval()
    backbone.eval()
    category_results: dict[str, dict[str, float]] = {}
    dataset_root = Path(config["data"]["dataset_root"])
    for category in CATEGORIES:
        dataset = make_dataset(config, "val", (category,), correction_dicts)
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
        gallery = cache.global_by_ids(gallery_ids).to(device).float()
        scores: list[torch.Tensor] = []
        target_ids: list[str] = []
        for raw_batch in loader:
            policy = build_policy_batch(raw_batch, cache, backbone, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                encoded = encode_policy_batch(backbone, policy)
                if stage in {CurriculumStage.ACTOR_WARMUP, CurriculumStage.UTILITY_SHADOW}:
                    rollout = RolloutConfig(max_steps=1, selection_mode="uniform")
                else:
                    rollout = RolloutConfig(
                        max_steps=int(config["training"]["horizon"]), selection_mode="learned"
                    )
                query = model(encoded, rollout).final_query.float()
            scores.append((query @ gallery.T).cpu())
            target_ids.extend(str(target_id) for target_id in raw_batch.target_ids)
        category_results[category] = evaluate_fashioniq_category(
            torch.cat(scores), target_ids, gallery_ids
        )
    average = macro_average_fashioniq(category_results)
    metrics = dict(average)
    for category, values in category_results.items():
        metrics.update({f"{category}_{key}": value for key, value in values.items()})
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    correction_dicts = resolve_fashioniq_correction_dicts(
        Path(config["data"]["dataset_root"]) / "captions",
        str(config["data"]["correction_policy"]),
    )
    stage = CurriculumStage(config["training"]["stage"])
    engine_config = EngineConfig(
        stage=stage,
        horizon=int(config["training"]["horizon"]),
        step_cost=float(config["policy"]["step_cost"]),
        retrieval_temperature=float(config["loss"]["retrieval_temperature"]),
        utility_weight=float(config["loss"]["utility_weight"]),
        oracle_mix=float(config["training"].get("oracle_mix", 0.0)),
        straight_through=config["training"].get("straight_through"),
    )
    engine_config.validate()
    device = torch.device(
        config["runtime"]["device"]
        if config["runtime"]["device"] != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    text_cfg = TextTuningConfig(**config["backbone"]["text_tuning"])
    vision_cfg = VisionTuningConfig(**config["backbone"]["vision_tuning"])
    if vision_cfg.mode != "frozen" or vision_cfg.num_unfrozen_blocks != 0:
        raise ValueError("This branch requires vision_tuning.mode=frozen and num_unfrozen_blocks=0")
    FeatureSourcePolicy(
        text_encoder_trainable=text_cfg.mode != "frozen",
        vision_encoder_trainable=False,
        gallery_projection_trainable=False,
        use_cached_text_states=False,
        use_cached_reference_dense=True,
        use_cached_reference_global=True,
        use_cached_gallery_global=True,
    ).validate()
    dtype = torch.bfloat16 if config["runtime"]["precision"] == "bf16" else torch.float32
    backbone = FGCLIP2BaseBackbone(
        revision=config["backbone"]["revision"],
        dtype=dtype if device.type == "cuda" else torch.float32,
        text_tuning=text_cfg,
        vision_tuning=vision_cfg,
    ).to(device)
    taper = TaperMAG(
        TaperMAGConfig(
            text_dim=backbone.contract.text_dim,
            vision_dim=backbone.contract.vision_dim,
            retrieval_dim=backbone.contract.retrieval_dim,
            d_model=int(config["model"]["d_model"]),
            num_queries=int(config["model"]["num_queries"]),
            max_steps=int(config["training"]["horizon"]),
        )
    ).to(device)
    print(format_parameter_report(backbone, taper))

    cache_base = (
        Path(config["data"]["cache_root"])
        / "fashioniq"
        / "fgclip2-base"
        / backbone.revision
    )
    train_cache = FrozenVisionCache.load(cache_base / "train")
    val_cache = FrozenVisionCache.load(cache_base / "val")
    validate_cache(train_cache, backbone, "train")
    validate_cache(val_cache, backbone, "val")
    manifest_hashes = {
        "train_global": train_cache.global_manifest.sha256,
        "train_dense": train_cache.dense_manifest.sha256,
        "val_global": val_cache.global_manifest.sha256,
        "val_dense": val_cache.dense_manifest.sha256,
    }

    if args.validate_only:
        if args.resume is None:
            raise ValueError("--validate-only requires --resume with a trained checkpoint")
        load_checkpoint(
            args.resume,
            model=taper,
            backbone=backbone,
            expected_manifest_hashes=manifest_hashes,
        )
        metrics = validate_fashioniq(
            taper, backbone, val_cache, config, device, stage, correction_dicts
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    dataset: Dataset = make_dataset(config, "train", CATEGORIES, correction_dicts)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_cir_samples,
        pin_memory=device.type == "cuda",
    )
    optimizer = build_optimizer(taper, backbone, OptimizerConfig(**config["optimizer"]))
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
    bank = NegativeBank(
        train_cache.global_embeddings,
        tuple(image_id for image_id, _ in sorted(train_cache.name_to_idx.items(), key=lambda item: item[1])),
        hard_negatives=int(config["teacher"]["hard_negatives"]),
    )
    engine = TaperMAGTrainingEngine(
        backbone,
        taper,
        bank,
        MarginalGainTeacher(float(config["teacher"]["retrieval_temperature"])),
    )
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "backbone_manifest.json").write_text(
        json.dumps(asdict(backbone.manifest()), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    drift_batch = backbone.tokenize_texts(
        ["make it red", "add long sleeves", "remove the pattern", "make it more formal"]
    )
    drift_snapshot = TextDriftMonitor.capture(backbone, drift_batch)
    start_epoch = 0
    global_step = 0
    best_metrics = {"mean_recall": -float("inf")}
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=taper,
            backbone=backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_manifest_hashes=manifest_hashes,
        )
        if payload["stage"] != stage.value:
            raise RuntimeError("Resume checkpoint stage differs from resolved config")
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_metrics = dict(payload["best_metrics"])

    optimizer.zero_grad(set_to_none=True)
    last_gradient_norms: dict[str, float] = {}
    last_diagnostics: dict[str, float] = {}
    for epoch in range(start_epoch, epochs):
        taper.train()
        backbone.train()
        epoch_loss = 0.0
        for micro_step, raw_batch in enumerate(loader):
            policy = build_policy_batch(raw_batch, train_cache, backbone, device)
            supervision = build_supervision_batch(raw_batch, train_cache, device)
            use_bf16 = device.type == "cuda" and config["runtime"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                result = engine.step(policy, supervision, engine_config)
                scaled_loss = result.loss / accumulation
            scaled_loss.backward()
            epoch_loss += float(result.loss.detach())
            last_diagnostics = summarize_training_diagnostics(
                result.model_output, result.teacher_gain
            )
            if (micro_step + 1) % accumulation == 0 or micro_step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    max_norm=float(config["optimizer"]["gradient_clip"]),
                )
                last_gradient_norms = text_block_gradient_norms(backbone)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        metrics = validate_fashioniq(
            taper, backbone, val_cache, config, device, stage, correction_dicts
        )
        drift = TextDriftMonitor.measure(backbone, drift_snapshot)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "stage": stage.value,
            "train_loss": epoch_loss / max(len(loader), 1),
            **metrics,
            **drift,
            **last_gradient_norms,
            **last_diagnostics,
        }
        with (output_dir / "metrics_val.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        if metrics["mean_recall"] > best_metrics["mean_recall"]:
            best_metrics = dict(metrics)
            checkpoint_name = "best_retrieval_valid.ckpt"
        else:
            checkpoint_name = "last.ckpt"
        for name in {checkpoint_name, "last.ckpt"}:
            save_checkpoint(
                output_dir / name,
                model=taper,
                backbone=backbone,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                stage=stage.value,
                curriculum_state={
                    "horizon": engine_config.horizon,
                    "oracle_mix": engine_config.oracle_mix,
                    "step_cost": engine_config.step_cost,
                },
                resolved_config=config,
                manifest_hashes=manifest_hashes,
                best_metrics=best_metrics,
            )


if __name__ == "__main__":
    main()
