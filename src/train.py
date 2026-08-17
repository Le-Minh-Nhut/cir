from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cache.features import load_features
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset
from evaluation.fashioniq import evaluate_fashioniq
from runtime import configure_torch_runtime, resolve_device, seed_everything
from training.engine import fit, prepare_batch


CATEGORIES = ("dress", "shirt", "toptee")


def build_train_loader(annotation_root: str | Path, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    dataset = FashionIQDataset(annotation_root=annotation_root, split="train", categories=CATEGORIES, caption_policy="randomized_four_way", seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)


def build_val_loaders(annotation_root: str | Path, *, batch_size: int, num_workers: int) -> tuple[dict[str, DataLoader], dict[str, list]]:
    val_loaders = {}
    val_annotations = {}

    for category in CATEGORIES:
        dataset = FashionIQDataset(annotation_root=annotation_root, split="val", categories=[category], caption_policy="ordered_and",)
        val_loaders[category] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)
        val_annotations[category] = dataset.annotations

    return val_loaders, val_annotations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(seed=cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)
    device = resolve_device(device_name=cfg.runtime.device, accelerator_index=cfg.runtime.accelerator_index)

    print(f"Device: {device}")
    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"
    cache_root = Path(cfg.paths.cache_root)
    train_feature_dir = cache_root / "fashioniq" / "fgclip2_large" / "train"
    val_feature_dir = cache_root / "fashioniq" / "fgclip2_large" / "val"


    train_loader = build_train_loader(annotation_root=annotation_root, batch_size=cfg.experiment.batch_size, num_workers=cfg.experiment.num_workers, seed=cfg.seed)
    val_loaders, val_annotations = build_val_loaders(annotation_root=annotation_root, batch_size=cfg.experiment.eval_batch_size, num_workers=cfg.experiment.num_workers)

    print(f"Train queries: {len(train_loader.dataset)}")
    for category, loader in val_loaders.items():
        print(f"Val {category}: {len(loader.dataset)} queries")

    train_features, train_name_to_idx = load_features(train_feature_dir)
    val_features, val_name_to_idx = load_features(val_feature_dir)
    print("Train image features:", tuple(train_features.shape))
    print("Val image features:", tuple(val_features.shape))

    raise RuntimeError("Training wiring reached the teacher/text-encoder boundary. Teacher and text encoder must be selected before TAPER can be instantiated.")


if __name__ == "__main__":
    main()