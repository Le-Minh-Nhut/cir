from __future__ import annotations

from functools import partial
from pathlib import Path
import json
import os
import subprocess

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset, compose_fashioniq_caption
from evaluation.fashioniq import evaluate_fashioniq
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig
from models.iag_srme.utils.backbone import assert_cache_legal
from models.iag_srme.utils.semantic import ConceptVocabulary
from runtime import configure_torch_runtime, resolve_device, seed_everything
from training.engine import (
    fit,
    parameter_count_diagnostics,
    resolve_precision,
    trainable_parameters,
)


CATEGORIES = ("dress", "shirt", "toptee")


def git_identity() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "git_sha": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
    }


def persist_run_configuration(
    cfg: DictConfig,
    output_dir: Path,
    model: IAGSRME,
    objective: IAGSRMEObjective,
    processor: object,
    precision_name: str,
) -> dict[str, object]:
    """Persist the actual resolved run, independent of later YAML changes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    (output_dir / "resolved_config.yaml").write_text(resolved_yaml, encoding="utf-8")
    metadata: dict[str, object] = {
        **git_identity(),
        "seed": int(cfg.seed),
        "dataset": str(cfg.dataset.name),
        "dataset_root": str(cfg.dataset.root),
        "train_split": "train",
        "validation_split": str(cfg.protocol.split),
        "train_caption_policy": str(cfg.experiment.train_caption_policy),
        "validation_caption_policy": str(cfg.experiment.val_caption_policy),
        "image_processor": type(processor).__name__,
        "backbone_checkpoint": str(cfg.backbone.checkpoint),
        "backbone_revision": str(cfg.backbone.revision),
        "global_readout_mode": str(cfg.backbone.global_readout_mode),
        "finetune_policy": str(cfg.backbone.finetune_policy),
        "num_candidates": int(cfg.model.num_candidates),
        "max_steps": int(cfg.model.max_steps),
        "stop_enabled": bool(cfg.model.stop_enabled),
        "epsilon_stop": float(cfg.model.epsilon_stop),
        "batch_size": int(cfg.experiment.batch_size),
        "gradient_accumulation": 1,
        "precision": precision_name,
        "optimizer": "AdamW",
        "learning_rate": float(cfg.experiment.learning_rate),
        "weight_decay": float(cfg.experiment.weight_decay),
        "checkpoint_selection_metric": "mean_recall",
        "parameter_counts": parameter_count_diagnostics(model, objective),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_model(cfg: DictConfig) -> tuple[IAGSRME, object, object]:
    regime = FGCLIPRegime(
        checkpoint=str(cfg.backbone.checkpoint),
        revision=str(cfg.backbone.revision),
        train_vision=bool(cfg.backbone.train_vision),
        train_text=bool(cfg.backbone.train_text),
        train_text_projection=bool(cfg.backbone.train_text_projection),
        trust_remote_code=bool(cfg.backbone.trust_remote_code),
        global_readout_mode=str(cfg.backbone.global_readout_mode),
        readout_experiment=str(cfg.backbone.readout_experiment),
        finetune_policy=str(cfg.backbone.finetune_policy),
        experiment_identity=str(cfg.backbone.experiment_identity),
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
        exec_dim=int(cfg.model.exec_dim),
        epsilon_stop=float(cfg.model.epsilon_stop),
        stop_enabled=bool(cfg.model.stop_enabled),
        read_scale_init=float(cfg.model.read_scale_init),
        exec_scale_init=float(cfg.model.exec_scale_init),
        exec_bias_init=float(cfg.model.exec_bias_init),
        score_dropout=float(cfg.model.score_dropout),
    )
    return IAGSRME(backbone, model_config), tokenizer, processor


def build_concept_vocabulary(
    dataset: FashionIQDataset, config: ObjectiveConfig
) -> ConceptVocabulary:
    # Vocabulary construction is deterministic and sees training instructions only.
    instructions = [
        compose_fashioniq_caption(annotation.captions, "ordered_and")
        for annotation in dataset.annotations
    ]
    return ConceptVocabulary.build(
        instructions,
        min_frequency=config.concept_min_frequency,
        max_size=config.concept_max_size,
    )


@torch.no_grad()
def encode_concept_prototypes(
    model: IAGSRME,
    tokenizer: object,
    vocabulary: ConceptVocabulary,
    max_text_length: int,
    batch_size: int = 128,
) -> torch.Tensor:
    """Freeze the current FG-CLIP text-global representation of each concept."""

    was_training = model.backbone.training
    model.backbone.eval()
    device = next(model.backbone.parameters()).device
    prototypes = []
    for start in range(0, len(vocabulary.concepts), batch_size):
        concepts = list(vocabulary.concepts[start : start + batch_size])
        tokenized = tokenizer(
            concepts,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        content_mask = attention_mask.clone()
        content_mask[:, 0] = False
        final_positions = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        content_mask.scatter_(1, final_positions[:, None], False)
        _, text_global = model.backbone.encode_text(input_ids, attention_mask, content_mask)
        prototypes.append(text_global.float().cpu())
    model.backbone.train(was_training)
    return torch.cat(prototypes, dim=0)


def build_objective(
    cfg: DictConfig,
    model: IAGSRME,
    tokenizer: object,
    train_dataset: FashionIQDataset,
) -> IAGSRMEObjective:
    objective_config = ObjectiveConfig(
        **{key: value for key, value in cfg.objective.items() if key != "name"}
    )
    vocabulary = None
    prototypes = None
    if objective_config.concept_enabled:
        vocabulary = build_concept_vocabulary(train_dataset, objective_config)
        prototypes = encode_concept_prototypes(
            model,
            tokenizer,
            vocabulary,
            int(cfg.backbone.max_text_length),
        )
    return IAGSRMEObjective(
        objective_config,
        width=int(cfg.model.width),
        state_dim=model.backbone.state_dim,
        concept_vocabulary=vocabulary,
        concept_prototypes=prototypes,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed), bool(cfg.runtime.deterministic))
    configure_torch_runtime(
        deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark)
    )
    device = resolve_device(str(cfg.runtime.device), int(cfg.runtime.accelerator_index))
    precision = resolve_precision(str(cfg.runtime.precision), device)
    model, tokenizer, processor = build_model(cfg)

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
    objective = build_objective(cfg, model, tokenizer, train_dataset)
    model.to(device)
    objective.to(device)
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
    output_dir = Path(str(cfg.paths.output_root))
    run_metadata = persist_run_configuration(
        cfg, output_dir, model, objective, processor, precision.name
    )
    fit(
        model,
        objective,
        train_loader,
        optimizer,
        evaluate,
        epochs=int(cfg.experiment.epochs),
        device=device,
        output_dir=output_dir,
        precision=precision,
        logging_interval=int(cfg.experiment.get("logging_interval", 1)),
        run_metadata=run_metadata,
    )


if __name__ == "__main__":
    main()
