from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cache.features import load_features, load_text_features
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset
from evaluation.fashioniq_encoder import (
    FASHIONIQ_ENCODER_CATEGORIES,
    evaluate_fashioniq_encoder,
)
from runtime import configure_torch_runtime, resolve_device, seed_everything
from train_fashioniq_encoder import (
    CAPTION_POLICY,
    build_fashioniq_encoder_model,
    load_encoder_correction_dicts,
)


def _load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError("TAPER checkpoint must contain a state-dict mapping")
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith("teacher.")
    ]
    if invalid_missing:
        raise RuntimeError(f"TAPER checkpoint is missing keys: {invalid_missing}")
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"TAPER checkpoint has unexpected keys: {incompatible.unexpected_keys}"
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.dataset.get("name", "")) != "fashioniq":
        raise ValueError("src/evaluate_fashioniq_encoder.py requires dataset=fashioniq")
    if str(cfg.protocol.get("name", "")) != "fashioniq_encoder":
        raise ValueError(
            "src/evaluate_fashioniq_encoder.py requires protocol=fashioniq_encoder"
        )
    if str(cfg.experiment.get("name", "")) != "taper_e2e":
        raise ValueError(
            "src/evaluate_fashioniq_encoder.py requires experiment=taper_e2e"
        )

    print("Dataset: FashionIQ")
    print("Protocol mode: ENCODER")

    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("Provide +checkpoint=/path/to/best.pt")
    checkpoint_path = Path(checkpoint_value).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TAPER checkpoint not found: {checkpoint_path}")

    seed_everything(seed=cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(
        deterministic=cfg.runtime.deterministic,
        benchmark=cfg.runtime.benchmark,
    )
    device = resolve_device(
        device_name=cfg.runtime.device,
        accelerator_index=cfg.runtime.accelerator_index,
    )
    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    correction_dicts = load_encoder_correction_dicts(annotation_root)
    cache_root = Path(cfg.paths.cache_root) / "fashioniq" / "csmcir" / "val"
    retrieval_features, retrieval_name_to_idx = load_features(
        cache_root / "retrieval"
    )
    native_features, native_name_to_idx = load_features(cache_root / "native")
    text_cache = load_text_features(cache_root / "text")

    val_loaders = {}
    val_annotations = {}
    for category in FASHIONIQ_ENCODER_CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[category],
            caption_policy=CAPTION_POLICY,
            correction_dicts=correction_dicts,
        )
        val_loaders[category] = DataLoader(
            dataset,
            batch_size=cfg.experiment.eval_batch_size,
            shuffle=False,
            num_workers=cfg.experiment.num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=True,
        )
        val_annotations[category] = dataset.annotations

    model = build_fashioniq_encoder_model(cfg, device)
    _load_checkpoint(model, checkpoint_path, device)
    metrics = evaluate_fashioniq_encoder(
        model,
        val_loaders,
        val_annotations,
        retrieval_features=retrieval_features,
        native_features=native_features,
        retrieval_name_to_idx=retrieval_name_to_idx,
        native_name_to_idx=native_name_to_idx,
        text_cache=text_cache,
        device=device,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    output_value = cfg.get("metrics_output")
    output_path = (
        Path(output_value)
        if output_value is not None
        else Path(cfg.paths.output_root) / "fashioniq" / "encoder" / "metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "FashionIQ",
                "protocol_mode": "encoder",
                "split": "val",
                "caption_policy": CAPTION_POLICY,
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
