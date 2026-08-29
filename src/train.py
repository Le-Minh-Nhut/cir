from __future__ import annotations

from functools import partial
from pathlib import Path
import os
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
import hydra
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from evaluation.fashioniq import evaluate_fashioniq
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig, IAGSRMECore
from models.iag_srme.backbone import assert_cache_legal
from runtime import configure_torch_runtime, resolve_device, seed_everything
from training.engine import fit, resolve_precision, trainable_parameters


CATEGORIES = ("dress", "shirt", "toptee")


def build_model(cfg: DictConfig) -> tuple[IAGSRME, object, object]:
    regime = FGCLIPRegime(
        checkpoint=str(cfg.backbone.checkpoint),
        revision=str(cfg.backbone.revision),
        train_vision=bool(cfg.backbone.train_vision),
        train_text=bool(cfg.backbone.train_text),
        train_text_projection=bool(cfg.backbone.train_text_projection),
        trust_remote_code=bool(cfg.backbone.trust_remote_code),
    )
    assert_cache_legal(regime.train_vision, cfg.backbone.get("image_cache_path"))
    backbone = FGCLIPBackbone.from_pretrained(regime, int(cfg.model.width))
    tokenizer, processor = FGCLIPBackbone.load_processor(
        regime.checkpoint, regime.revision, regime.trust_remote_code
    )
    model_config = IAGSRMEConfig(
        width=int(cfg.model.width),
        num_candidates=int(cfg.model.num_candidates),
        max_steps=int(cfg.model.max_steps),
        num_heads=int(cfg.model.num_heads),
        retrieval_dim=backbone.retrieval_dim,
        lambda_z=float(cfg.model.lambda_z),
        query_cap=float(cfg.model.query_cap),
        selector_temperature=float(cfg.model.selector_temperature),
        selector_gumbel_noise=bool(cfg.model.selector_gumbel_noise),
        enable_claim_head=bool(cfg.model.enable_claim_head),
        enable_factor_head=bool(cfg.model.enable_factor_head),
        factor_dim=(None if cfg.model.factor_dim is None else int(cfg.model.factor_dim)),
    )
    return IAGSRME(backbone, IAGSRMECore(model_config)), tokenizer, processor


def build_objective(cfg: DictConfig) -> IAGSRMEObjective:
    objective_config = ObjectiveConfig(
        **{key: value for key, value in cfg.objective.items() if key != "name"}
    )
    return IAGSRMEObjective(objective_config, width=int(cfg.model.width))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark)
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    model, tokenizer, processor = build_model(cfg)
    objective = build_objective(cfg)
    model.to(device)
    objective.to(device)

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / str(cfg.dataset.annotation_dir)
    split_root = dataset_root / str(cfg.dataset.split_dir)
    image_store = DirectoryImageStore(dataset_root / str(cfg.dataset.image_dir))
    train_dataset = FashionIQDataset(
        annotation_root,
        "train",
        CATEGORIES,
        caption_policy=str(cfg.experiment.train_caption_policy),
        seed=int(cfg.seed),
    )
    train_collator = FashionIQImageCollator(
        image_store, tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.experiment.batch_size),
        shuffle=True,
        num_workers=int(cfg.experiment.num_workers),
        pin_memory=True,
        collate_fn=train_collator,
    )
    val_loaders = {}
    val_annotations = {}
    val_collator = FashionIQImageCollator(
        image_store, tokenizer, processor, int(cfg.backbone.max_text_length), include_targets=False
    )
    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root,
            "val",
            [category],
            caption_policy=str(cfg.experiment.val_caption_policy),
            seed=int(cfg.seed),
        )
        val_annotations[category] = dataset.annotations
        val_loaders[category] = DataLoader(
            dataset,
            batch_size=int(cfg.experiment.eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.experiment.num_workers),
            pin_memory=True,
            collate_fn=val_collator,
        )
    optimizer = AdamW(
        trainable_parameters(model, objective),
        lr=float(cfg.experiment.learning_rate),
        weight_decay=float(cfg.experiment.weight_decay),
    )
    evaluate = partial(
        evaluate_fashioniq,
        val_loaders=val_loaders,
        val_annotations=val_annotations,
        protocol=str(cfg.protocol.name),
        split_root=split_root,
        split=str(cfg.protocol.split),
        image_store=image_store,
        image_processor=processor,
        device=device,
        gallery_batch_size=int(cfg.experiment.gallery_batch_size),
        num_workers=int(cfg.experiment.num_workers),
    )
    fit(
        model,
        objective,
        train_loader,
        optimizer,
        evaluate,
        epochs=int(cfg.experiment.epochs),
        device=device,
        output_dir=str(cfg.paths.output_root),
        precision=precision,
    )


if __name__ == "__main__":
    main()
