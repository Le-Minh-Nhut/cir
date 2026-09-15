from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cache.features import load_features, load_text_features
from datasets.cirr import CIRRDataset, load_cirr_image_mapping
from datasets.common import collate_cir_samples
from evaluation.cirr_encoder import evaluate_cirr_encoder
from models.taper import TAPER
from runtime import configure_torch_runtime, resolve_device, seed_everything
from teachers.csmcir_compose import CSMCIRComposeTeacher


def build_cirr_encoder_model(cfg: DictConfig, device: torch.device) -> TAPER:
    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.experiment.teacher.csmcir_root,
        checkpoint_path=cfg.experiment.teacher.checkpoint_path,
    ).to(device).eval()
    model_cfg = cfg.experiment.model
    return TAPER(
        teacher,
        text_dim=model_cfg.text_dim,
        reference_dim=model_cfg.reference_dim,
        teacher_text_dim=model_cfg.teacher_text_dim,
        teacher_query_dim=model_cfg.teacher_query_dim,
        query_dim=model_cfg.query_dim,
        slot_dim=model_cfg.slot_dim,
        state_dim=model_cfg.state_dim,
        num_slots=model_cfg.num_slots,
        num_primitives=model_cfg.num_primitives,
        mask_temperature=model_cfg.mask_temperature,
        router_temperature=model_cfg.router_temperature,
        retrieval_temperature=model_cfg.retrieval_temperature,
        neutral_mode=model_cfg.neutral_mode,
        slot_gate_threshold=model_cfg.slot_gate_threshold,
        hard_slot_gating_during_training=model_cfg.hard_slot_gating_during_training,
        gate_mode=model_cfg.gate_mode,
        st_gate_recovery=model_cfg.st_gate_recovery,
        alpha_max=model_cfg.alpha_max,
        counterfactual_chunk_size=model_cfg.counterfactual_chunk_size,
        num_refine_iters=model_cfg.num_refine_iters,
        residual_bias_strength=model_cfg.residual_bias_strength,
        residual_depletion_power=model_cfg.residual_depletion_power,
        residual_eps=model_cfg.residual_eps,
        randomize_slot_order_during_training=(
            model_cfg.randomize_slot_order_during_training
        ),
    ).to(device)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if str(cfg.dataset.get("name", "")) != "cirr":
        raise ValueError("src/train_cirr_encoder.py requires dataset=cirr")
    if str(cfg.protocol.get("name", "")) != "cirr_encoder":
        raise ValueError("src/train_cirr_encoder.py requires protocol=cirr_encoder")
    if str(cfg.experiment.get("name", "")) != "taper_e2e":
        raise ValueError("src/train_cirr_encoder.py requires experiment=taper_e2e")

    print("Dataset: CIRR")
    print("Protocol mode: ENCODER")

    from training.engine import fit, prepare_batch

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
    version = str(cfg.dataset.version)
    cache_root = Path(cfg.paths.cache_root) / "cirr" / "csmcir"

    train_retrieval, train_retrieval_idx = load_features(
        cache_root / "train" / "retrieval"
    )
    train_native, train_native_idx = load_features(cache_root / "train" / "native")
    val_retrieval, val_retrieval_idx = load_features(
        cache_root / "val" / "retrieval"
    )
    val_native, val_native_idx = load_features(cache_root / "val" / "native")
    train_text = load_text_features(cache_root / "train" / "text")
    val_text = load_text_features(cache_root / "val" / "text")

    train_dataset = CIRRDataset(dataset_root / "captions", "train", version=version)
    val_dataset = CIRRDataset(dataset_root / "captions", "val", version=version)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=True,
        num_workers=cfg.experiment.num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.experiment.eval_batch_size,
        shuffle=False,
        num_workers=cfg.experiment.num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=True,
    )
    gallery_ids = list(
        load_cirr_image_mapping(
            dataset_root / "image_splits", "val", version=version
        )
    )

    model = build_cirr_encoder_model(cfg, device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.experiment.lr,
        weight_decay=cfg.experiment.weight_decay,
    )
    output_dir = Path(cfg.paths.output_root) / "cirr" / "encoder"
    best_validation_score = float("-inf")
    validation_epoch = 0

    def prepare_batch_fn(batch, batch_device):
        return prepare_batch(
            batch,
            batch_device,
            train_retrieval,
            train_native,
            train_retrieval_idx,
            train_native_idx,
            train_text,
        )

    def evaluate_fn(current_model):
        nonlocal best_validation_score, validation_epoch
        validation_epoch += 1
        metrics = evaluate_cirr_encoder(
            current_model,
            val_loader,
            gallery_ids=gallery_ids,
            retrieval_features=val_retrieval,
            native_features=val_native,
            retrieval_name_to_idx=val_retrieval_idx,
            native_name_to_idx=val_native_idx,
            text_cache=val_text,
            device=device,
        )
        print(
            "CIRR ENCODER validation | "
            + " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )
        payload = {
            "dataset": "CIRR",
            "protocol_mode": "encoder",
            "split": "val",
            "version": version,
            "epoch": validation_epoch,
            "metrics": metrics,
        }
        with (output_dir / "metrics_last.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        if metrics["selection_score"] > best_validation_score:
            best_validation_score = metrics["selection_score"]
            with (output_dir / "metrics_best.json").open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
        return metrics

    fit(
        model,
        train_loader,
        optimizer,
        evaluate_fn,
        num_epochs=cfg.experiment.num_epochs,
        device=device,
        loss_weights=dict(cfg.experiment.loss_weights),
        primary_metric="selection_score",
        output_dir=output_dir,
        use_amp=cfg.runtime.precision == "fp16",
        prepare_batch_fn=prepare_batch_fn,
    )


if __name__ == "__main__":
    main()
