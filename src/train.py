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
from models.taper import TAPER
from runtime import configure_torch_runtime, resolve_device, seed_everything
from teachers.csmcir import CSMCIRStage1Teacher
from training.engine import fit, prepare_batch


CATEGORIES = ("dress", "shirt", "toptee")


def build_train_loader(annotation_root: str | Path, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    dataset = FashionIQDataset(annotation_root=annotation_root, split="train", categories=CATEGORIES, caption_policy="randomized_four_way", seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)


def build_val_loaders(annotation_root: str | Path, *, batch_size: int, num_workers: int):
    val_loaders = {}
    val_annotations = {}

    for category in CATEGORIES:
        dataset = FashionIQDataset(annotation_root=annotation_root, split="val", categories=[category], caption_policy="ordered_and")
        val_loaders[category] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_cir_samples, pin_memory=True)
        val_annotations[category] = dataset.annotations

    return val_loaders, val_annotations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(seed=cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)
    device = resolve_device(device_name=cfg.runtime.device, accelerator_index=cfg.runtime.accelerator_index)

    print("Device:", device)

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"
    cache_root = Path(cfg.paths.cache_root)

    train_retrieval, train_retrieval_idx = load_features(cache_root / "fashioniq" / "csmcir" / "train" / "retrieval")
    train_native, train_native_idx = load_features(cache_root / "fashioniq" / "csmcir" / "train" / "native")
    val_retrieval, val_retrieval_idx = load_features(cache_root / "fashioniq" / "csmcir" / "val" / "retrieval")
    val_native, val_native_idx = load_features(cache_root / "fashioniq" / "csmcir" / "val" / "native")

    if train_retrieval_idx != train_native_idx:
        raise ValueError("Train retrieval/native caches are not aligned")

    if val_retrieval_idx != val_native_idx:
        raise ValueError("Val retrieval/native caches are not aligned")

    print("Train retrieval:", tuple(train_retrieval.shape))
    print("Train native:", tuple(train_native.shape))
    print("Val retrieval:", tuple(val_retrieval.shape))
    print("Val native:", tuple(val_native.shape))

    train_loader = build_train_loader(
        annotation_root=annotation_root,
        batch_size=cfg.experiment.batch_size,
        num_workers=cfg.experiment.num_workers,
        seed=cfg.seed,
    )

    val_loaders, val_annotations = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=cfg.experiment.eval_batch_size,
        num_workers=cfg.experiment.num_workers,
    )

    teacher = CSMCIRStage1Teacher(
        csmcir_root="teacher/repos/CSMCIR",
        checkpoint_path="teacher/checkpoints/csmcir/fashioniq_tuned_clip_best.pt",
        device=str(device),
    ).to(device).eval()

    model = TAPER(
        teacher,
        text_dim=768,
        reference_dim=1408,
        teacher_text_dim=768,
        teacher_query_dim=256,
        query_dim=256,
        slot_dim=512,
        state_dim=512,
        num_slots=4,
        num_primitives=8,
    ).to(device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.experiment.lr,
        weight_decay=cfg.experiment.weight_decay,
    )

    prepare_batch_fn = lambda batch, device: prepare_batch(batch, device, train_retrieval, train_native, train_retrieval_idx, teacher)

    def evaluate_fn(model):
        return evaluate_fashioniq(
            model,
            val_loaders,
            val_annotations,
            protocol=cfg.protocol.name,
            split_root=split_root,
            split="val",
            retrieval_features=val_retrieval,
            native_features=val_native,
            name_to_idx=val_retrieval_idx,
            teacher=teacher,
            device=device,
        )

    fit(
        model,
        train_loader,
        optimizer,
        evaluate_fn,
        num_epochs=cfg.experiment.num_epochs,
        device=device,
        loss_weights={"retrieval_loss": 1.0},
        primary_metric="mean_recall",
        output_dir=cfg.paths.output_root,
        use_amp=True,
        prepare_batch_fn=prepare_batch_fn,
    )


if __name__ == "__main__":
    main()