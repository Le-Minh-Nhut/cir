from __future__ import annotations

from pathlib import Path
import json

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from evaluation.fashioniq import build_validation_datasets, evaluate_fashioniq
from runtime import resolve_device
from train import CATEGORIES, build_model


def validate_checkpoint_backbone_metadata(
    metadata: object,
    expected_checkpoint: str,
    expected_revision: str,
    expected_readout_mode: str | None = None,
    expected_readout_experiment: str | None = None,
    expected_finetune_policy: str | None = None,
    expected_train_vision: bool | None = None,
    expected_train_text: bool | None = None,
    expected_train_text_projection: bool | None = None,
    expected_model_config: dict[str, object] | None = None,
    allow_counterfactual: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no reproducible backbone metadata")
    actual = (metadata.get("backbone_checkpoint"), metadata.get("backbone_revision"))
    expected = (expected_checkpoint, expected_revision)
    if actual != expected:
        raise ValueError(f"checkpoint backbone mismatch: stored={actual}, configured={expected}")
    mismatches: list[dict[str, object]] = []
    if expected_readout_mode is not None:
        actual_mode = metadata.get("global_readout_mode")
        if actual_mode != expected_readout_mode:
            mismatches.append(
                {
                    "field": "global_readout_mode",
                    "stored": actual_mode,
                    "configured": expected_readout_mode,
                }
            )
    expected_policy = {
        "readout_experiment": expected_readout_experiment,
        "finetune_policy": expected_finetune_policy,
        "train_vision": expected_train_vision,
        "train_text": expected_train_text,
        "train_text_projection": expected_train_text_projection,
    }
    for field, configured in expected_policy.items():
        if configured is not None and metadata.get(field) != configured:
            mismatches.append(
                {
                    "field": field,
                    "stored": metadata.get(field),
                    "configured": configured,
                }
            )
    stored_model = metadata.get("model_config")
    if expected_model_config is not None:
        if not isinstance(stored_model, dict):
            mismatches.append(
                {
                    "field": "model_config",
                    "stored": None,
                    "configured": expected_model_config,
                }
            )
        else:
            for field, configured in expected_model_config.items():
                stored = stored_model.get(field)
                if stored != configured:
                    mismatches.append({"field": field, "stored": stored, "configured": configured})
    if mismatches and not allow_counterfactual:
        fields = ", ".join(str(item["field"]) for item in mismatches)
        policy_fields = {
            "readout_experiment",
            "finetune_policy",
            "train_vision",
            "train_text",
            "train_text_projection",
        }
        if fields == "global_readout_mode":
            kind = "readout"
        elif any(item["field"] in policy_fields for item in mismatches):
            kind = "fine-tuning"
        else:
            kind = "behavior"
        raise ValueError(
            f"checkpoint {kind} mismatch ({fields}); pass "
            "+allow_counterfactual_eval=true to label an intentional override"
        )
    return mismatches


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("pass checkpoint=/absolute/or/repository/relative/best.pt")
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    model, tokenizer, processor = build_model(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    counterfactual = bool(cfg.get("allow_counterfactual_eval", False))
    mismatches = validate_checkpoint_backbone_metadata(
        checkpoint.get("metadata"),
        str(cfg.backbone.checkpoint),
        str(cfg.backbone.revision),
        str(cfg.backbone.global_readout_mode),
        expected_readout_experiment=str(cfg.backbone.readout_experiment),
        expected_finetune_policy=str(cfg.backbone.finetune_policy),
        expected_train_vision=bool(cfg.backbone.train_vision),
        expected_train_text=bool(cfg.backbone.train_text),
        expected_train_text_projection=bool(cfg.backbone.train_text_projection),
        expected_model_config={
            "num_candidates": int(cfg.model.num_candidates),
            "max_steps": int(cfg.model.max_steps),
            "stop_enabled": bool(cfg.model.stop_enabled),
            "epsilon_stop": float(cfg.model.epsilon_stop),
        },
        allow_counterfactual=counterfactual,
    )
    if mismatches:
        print("[evaluate] COUNTERFACTUAL checkpoint overrides:", mismatches)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / str(cfg.dataset.annotation_dir)
    image_store = DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir))
    collator = FashionIQImageCollator(
        image_store, tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=False
    )
    loaders = {}
    annotations = {}
    validation_datasets = build_validation_datasets(
        annotation_root,
        CATEGORIES,
        str(cfg.experiment.val_caption_policy),
        seed=int(cfg.seed),
    )
    for category, dataset in validation_datasets.items():
        annotations[category] = dataset.annotations
        loaders[category] = DataLoader(
            dataset,
            batch_size=int(cfg.experiment.eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.experiment.num_workers),
            collate_fn=collator,
        )
    metrics = evaluate_fashioniq(
        model,
        loaders,
        annotations,
        protocol=str(cfg.protocol.name),
        split_root=dataset_root / str(cfg.dataset.split_dir),
        split=str(cfg.protocol.split),
        image_store=image_store,
        image_processor=processor,
        device=device,
        gallery_batch_size=int(cfg.experiment.gallery_batch_size),
        num_workers=int(cfg.experiment.num_workers),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "counterfactual": bool(mismatches),
        "configuration_mismatches": mismatches,
        "configured_behavior": {
            "num_candidates": int(cfg.model.num_candidates),
            "max_steps": int(cfg.model.max_steps),
            "stop_enabled": bool(cfg.model.stop_enabled),
            "epsilon_stop": float(cfg.model.epsilon_stop),
            "global_readout_mode": str(cfg.backbone.global_readout_mode),
            "finetune_policy": str(cfg.backbone.finetune_policy),
            "backbone_checkpoint": str(cfg.backbone.checkpoint),
            "backbone_revision": str(cfg.backbone.revision),
        },
        "metrics": metrics,
    }
    output_path = Path(HydraConfig.get().runtime.output_dir) / "evaluation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
