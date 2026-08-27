from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from backbones.fgclip2 import (
    FGCLIP2_LARGE_DIM,
    FGCLIP2_LARGE_MODEL_ID,
    FGCLIP2_LARGE_REVISION,
    validate_fgclip2_revision,
)
from cache.features import (
    load_dense_image_features,
    load_feature_manifest,
    load_features,
    load_text_features,
    validate_feature_manifest,
    validate_text_cache_subdir,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import (
    FashionIQDataset,
    load_correction_dict,
    validate_correction_policy,
)
from evaluation.entity_action_binding import evaluate_entity_action_binding
from models.entity_action_binding import EntityActionBindingCIR
from runtime import configure_torch_runtime, resolve_device, seed_everything
from training.entity_action_binding import (
    fit_entity_action_binding,
    prepare_entity_action_batch,
)

CATEGORIES = ("dress", "shirt", "toptee")


def load_correction_dicts(annotation_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionIQ correction dictionary: {path}")
        result[category] = load_correction_dict(path)
    return result


def build_loaders(
    annotation_root: Path,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    train_caption_policy: str,
    val_caption_policy: str,
    correction_dicts: dict[str, dict[str, str]] | None,
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, list]]:
    train_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy=train_caption_policy,
        correction_dicts=correction_dicts,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=torch.cuda.is_available(),
    )
    val_loaders: dict[str, DataLoader] = {}
    val_annotations: dict[str, list] = {}
    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root,
            "val",
            [category],
            caption_policy=val_caption_policy,
            correction_dicts=correction_dicts,
            seed=seed,
        )
        val_loaders[category] = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=torch.cuda.is_available(),
        )
        val_annotations[category] = dataset.annotations
    return train_loader, val_loaders, val_annotations


def build_model(cfg: DictConfig) -> EntityActionBindingCIR:
    model_cfg = cfg.experiment.model
    if int(model_cfg.dim) != FGCLIP2_LARGE_DIM:
        raise ValueError(f"A8.0 requires dim={FGCLIP2_LARGE_DIM}")
    return EntityActionBindingCIR(
        dim=int(model_cfg.dim),
        num_relations=int(model_cfg.num_relations),
        fusion_hidden_dim=int(model_cfg.fusion_hidden_dim),
        retrieval_temperature=float(model_cfg.retrieval_temperature),
        entity_action_temperature=float(model_cfg.entity_action_temperature),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.experiment.get("name", "")) != "encoder_binding_e2e":
        raise ValueError(
            "This entry point requires experiment=encoder_binding_e2e"
        )
    seed_everything(cfg.seed, deterministic=cfg.runtime.deterministic)
    configure_torch_runtime(
        deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark
    )
    device = resolve_device(cfg.runtime.device, cfg.runtime.accelerator_index)
    backbone_cfg = cfg.experiment.backbone
    if str(backbone_cfg.model_id) != FGCLIP2_LARGE_MODEL_ID:
        raise ValueError("A8.0 requires qihoo360/fg-clip2-large")
    revision = validate_fgclip2_revision(str(backbone_cfg.revision))
    if revision != FGCLIP2_LARGE_REVISION:
        raise ValueError(f"A8.0 requires revision={FGCLIP2_LARGE_REVISION}")

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"
    feature_root = Path(cfg.paths.cache_root) / "fashioniq" / "fgclip2-large"
    correction_policy = validate_correction_policy(str(cfg.experiment.correction_policy))
    text_subdir = validate_text_cache_subdir(
        str(cfg.experiment.text_cache_subdir), correction_policy
    )
    correction_dicts = (
        load_correction_dicts(annotation_root) if correction_policy == "fashioniq" else None
    )

    caches: dict[str, tuple[torch.Tensor, dict[str, int], object, object]] = {}
    for split in ("train", "val"):
        global_dir = feature_root / split / str(cfg.experiment.global_image_cache_subdir)
        dense_dir = feature_root / split / str(cfg.experiment.dense_image_cache_subdir)
        text_dir = feature_root / split / text_subdir
        for cache_dir, cache_name in ((global_dir, "images"), (dense_dir, "dense_images")):
            manifest = load_feature_manifest(cache_dir)
            validate_feature_manifest(
                manifest,
                model_id=FGCLIP2_LARGE_MODEL_ID,
                revision=revision,
                cache_name=f"{split}/{cache_name}",
            )
        text_manifest = load_feature_manifest(text_dir)
        validate_feature_manifest(
            text_manifest,
            model_id=FGCLIP2_LARGE_MODEL_ID,
            revision=revision,
            cache_name=f"{split}/{text_subdir}",
            correction_policy=correction_policy,
        )
        globals_, global_index = load_features(global_dir)
        dense = load_dense_image_features(dense_dir)
        text = load_text_features(text_dir)
        if dense.manifest.get("feature_dim") != FGCLIP2_LARGE_DIM:
            raise ValueError("Dense cache has wrong FG-CLIP2 feature dimension")
        if text.global_features is None:
            raise FileNotFoundError(
                f"A8.0 requires {text_dir / 'global.npy'}; rerun text precompute with --save-global"
            )
        if text.global_features.shape[1] != FGCLIP2_LARGE_DIM:
            raise ValueError("Text global cache has wrong FG-CLIP2 feature dimension")
        caches[split] = (globals_, global_index, dense, text)

    train_loader, val_loaders, val_annotations = build_loaders(
        annotation_root,
        train_batch_size=int(cfg.experiment.batch_size),
        eval_batch_size=int(cfg.experiment.eval_batch_size),
        num_workers=int(cfg.experiment.num_workers),
        seed=int(cfg.seed),
        train_caption_policy=str(cfg.experiment.train_caption_policy),
        val_caption_policy=str(cfg.experiment.val_caption_policy),
        correction_dicts=correction_dicts,
    )
    model = build_model(cfg).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(cfg.experiment.lr), weight_decay=float(cfg.experiment.weight_decay)
    )
    train_global, train_index, train_dense, train_text = caches["train"]
    val_global, val_index, val_dense, val_text = caches["val"]

    print("Experiment: FG-CLIP2 Entity–Action Binding (no TAPER/QASA/CSMCIR)")
    print("Device:", device)
    print("Model/revision:", FGCLIP2_LARGE_MODEL_ID, revision)
    print("Relations:", model.num_relations, "(one shared query bank)")
    print("Train dense tokens:", train_dense.manifest["total_token_count"])
    print("Text globals: required and present")

    def prepare(raw_batch, batch_device):
        return prepare_entity_action_batch(
            raw_batch, batch_device, train_global, train_index, train_dense, train_text
        )

    def evaluate(current_model):
        return evaluate_entity_action_binding(
            current_model,
            val_loaders,
            val_annotations,
            protocol=str(cfg.protocol.name),
            split_root=split_root,
            split="val",
            global_features=val_global,
            global_name_to_idx=val_index,
            dense_cache=val_dense,
            text_cache=val_text,
            device=device,
            gallery_batch_size=int(cfg.experiment.gallery_batch_size),
        )

    fit_entity_action_binding(
        model,
        train_loader,
        optimizer,
        evaluate,
        prepare,
        num_epochs=int(cfg.experiment.num_epochs),
        device=device,
        loss_weights=dict(cfg.experiment.loss_weights),
        output_dir=cfg.paths.output_root,
        use_amp=str(cfg.runtime.precision) == "fp16",
    )


if __name__ == "__main__":
    main()
