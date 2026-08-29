from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from evaluation.fashioniq import build_validation_datasets, evaluate_fashioniq
from runtime import resolve_device
from train import CATEGORIES, build_model


def validate_checkpoint_backbone_metadata(
    metadata: object, expected_checkpoint: str, expected_revision: str
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint has no reproducible backbone metadata")
    actual = (metadata.get("backbone_checkpoint"), metadata.get("backbone_revision"))
    expected = (expected_checkpoint, expected_revision)
    if actual != expected:
        raise ValueError(f"checkpoint backbone mismatch: stored={actual}, configured={expected}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("pass checkpoint=/absolute/or/repository/relative/best.pt")
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    model, tokenizer, processor = build_model(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    validate_checkpoint_backbone_metadata(
        checkpoint.get("metadata"),
        str(cfg.backbone.checkpoint),
        str(cfg.backbone.revision),
    )
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
    print(metrics)


if __name__ == "__main__":
    main()
