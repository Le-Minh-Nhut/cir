from __future__ import annotations

import os
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)
import json
import math
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from cache.features import get_features_by_ids, load_features
from datasets.common import CIRBatch, collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_fashioniq_split_ids
from evaluation.edit_slot_stage1 import build_tcfr_cache, evaluate_stage1_edit_slots
from models.taper import TAPER
from runtime import collect_environment_metadata, configure_torch_runtime, resolve_device, seed_everything

CATEGORIES = ("dress", "shirt", "toptee")

STAGE1_STATE_PREFIXES = (
    "slot_query_projection.",
    "text_key_projection.",
    "slot_mlp.",
    "slot_gate.",
)

STAGE1_STATE_EXACT_KEYS = {
    "slot_queries",
    "neutral_embedding",
}

def build_train_loader(
    annotation_root: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    caption_policy: str,
) -> DataLoader:
    dataset = FashionIQDataset(
        annotation_root=annotation_root,
        split="train",
        categories=CATEGORIES,
        caption_policy=caption_policy,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cir_samples,
        pin_memory=torch.cuda.is_available(),
    )

def build_val_loaders(
    annotation_root: str | Path,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[category],
            caption_policy="ordered_and",
        )
        loaders[category] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders

def encode_teacher_text(
    teacher: nn.Module,
    texts: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """
    Runtime teacher contract:

        teacher.encode_text_tokens(texts)

    must return either:

        text_states, text_attention_mask

    or:

        text_states, text_attention_mask, text_content_mask

    text_content_mask is optional in the evaluator, but if
    slot_coverage_loss has non-zero weight during Stage 1,
    train_stage1.py will require it.
    """
    if not hasattr(teacher, "encode_text_tokens"):
        raise AttributeError("Stage-1 teacher must implement encode_text_tokens(texts).")
    with torch.no_grad():
        output = teacher.encode_text_tokens(texts)
    if not isinstance(output, (tuple, list)):
        raise TypeError("teacher.encode_text_tokens() must return a tuple/list.")
    if len(output) == 2:
        text_states, text_attention_mask = output
        text_content_mask = None
    elif len(output) == 3:
        (
            text_states,
            text_attention_mask,
            text_content_mask,
        ) = output
    else:
        raise ValueError(
            "teacher.encode_text_tokens() must return either "
            "(text_states, text_attention_mask) or "
            "(text_states, text_attention_mask, text_content_mask)."
        )
    if not isinstance(text_states, Tensor):
        raise TypeError("text_states must be a Tensor")
    if not isinstance(text_attention_mask, Tensor):
        raise TypeError("text_attention_mask must be a Tensor")
    if text_content_mask is not None and not isinstance(text_content_mask, Tensor):
        raise TypeError("text_content_mask must be a Tensor")
    text_states = text_states.to(
        device=device,
        non_blocking=True,
    )
    text_attention_mask = text_attention_mask.to(
        device=device,
        non_blocking=True,
    )
    if text_content_mask is not None:
        text_content_mask = text_content_mask.to(
            device=device,
            non_blocking=True,
        )
    if text_states.ndim != 3:
        raise ValueError(f"text_states must have shape [B, N, D], got {tuple(text_states.shape)}")
    if text_attention_mask.shape != text_states.shape[:2]:
        raise ValueError("text_attention_mask shape must match text_states[:2]")
    if text_content_mask is not None and text_content_mask.shape != text_attention_mask.shape:
        raise ValueError("text_content_mask must match text_attention_mask")
    if not torch.isfinite(text_states).all():
        raise ValueError("text_states contains NaN or Inf")
    return (
        text_states,
        text_attention_mask,
        text_content_mask,
    )

def prepare_stage1_batch(
    batch: CIRBatch,
    *,
    model: TAPER,
    reference_features: Tensor,
    reference_name_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    """
    Stage 1 only needs:

        teacher-native reference features
        teacher-compatible token states

    It does NOT need:
        target image features
        TAPER executor state
        primitive bank
        final TAPER query

    The tensor is called `reference_features` because that is the
    current build_edit_slots() API, but these features must belong
    to the frozen teacher's compose() input space.
    """
    teacher_reference_features = get_features_by_ids(
        image_ids=batch.reference_ids,
        features=reference_features,
        name_to_idx=reference_name_to_idx,
    ).to(
        device=device,
        non_blocking=True,
    )
    (
        text_states,
        text_attention_mask,
        text_content_mask,
    ) = encode_teacher_text(
        teacher=model.teacher,
        texts=batch.modification_texts,
        device=device,
    )
    prepared: dict[str, object] = {
        "sample_ids": batch.sample_ids,
        "reference_ids": batch.reference_ids,
        "target_ids": batch.target_ids,
        "categories": batch.categories,
        "reference_features": teacher_reference_features,
        "text_states": text_states,
        "text_attention_mask": text_attention_mask,
    }
    if text_content_mask is not None:
        prepared["text_content_mask"] = text_content_mask
    return prepared

def compute_stage1_losses(model: TAPER, batch: dict[str, object]) -> dict[str, Tensor]:
    reference_features = batch["reference_features"]
    text_states = batch["text_states"]
    text_attention_mask = batch["text_attention_mask"]
    if not isinstance(reference_features, Tensor):
        raise TypeError("reference_features must be Tensor")
    if not isinstance(text_states, Tensor):
        raise TypeError("text_states must be Tensor")
    if not isinstance(text_attention_mask, Tensor):
        raise TypeError("text_attention_mask must be Tensor")
    
    content_mask = batch.get("text_content_mask")
    if content_mask is not None and not isinstance(content_mask, Tensor):
        raise TypeError("text_content_mask must be Tensor")
    slot_output = model.build_edit_slots(
        reference_features=reference_features,
        text_states=text_states,
        text_attention_mask=text_attention_mask,
        text_content_mask=content_mask,
    )
    return model._slot_regularizers(
        slot_masks=slot_output["slot_masks"],
        slot_effects=slot_output["slot_effects"],
        slot_gates=slot_output["slot_gates"],
        text_attention_mask=text_attention_mask,
        text_content_mask=content_mask,
    )

def compute_total_loss(
    loss_dict: dict[str, Tensor],
    loss_weights: dict[str, float],
) -> Tensor:
    if not loss_weights:
        raise ValueError("Stage-1 loss_weights must not be empty")
    total_loss: Tensor | None = None
    for name, weight in loss_weights.items():
        if name not in loss_dict:
            raise KeyError(f"Stage-1 loss '{name}' does not exist. Available: {sorted(loss_dict)}")
        weighted = float(weight) * loss_dict[name]
        if total_loss is None:
            total_loss = weighted
        else:
            total_loss = total_loss + weighted
    assert total_loss is not None
    return total_loss

def configure_stage1_trainable_parameters(model: TAPER) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters: list[nn.Parameter] = []
    model.slot_queries.requires_grad_(True)
    parameters.append(model.slot_queries)
    modules = (
        model.slot_query_projection,
        model.text_key_projection,
        model.slot_mlp,
        model.slot_gate,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if model.neutral_mode == "learned":
        if not isinstance(model.neutral_embedding, nn.Parameter):
            raise TypeError("neutral_embedding should be Parameter when neutral_mode='learned'")
        model.neutral_embedding.requires_grad_(True)
        parameters.append(model.neutral_embedding)
    if not parameters:
        raise RuntimeError("No Stage-1 trainable parameters found")
    return parameters

def get_amp_settings(
    device: torch.device,
    precision: str,
) -> tuple[bool, torch.dtype | None, bool]:
    precision = precision.lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be fp32, fp16, or bf16")
    if device.type != "cuda" or precision == "fp32":
        return False, None, False
    if precision == "fp16":
        return True, torch.float16, True
    return True, torch.bfloat16, False

def train_stage1_one_epoch(
    *,
    model: TAPER,
    train_loader: DataLoader,
    prepare_batch_fn,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    loss_weights: dict[str, float],
    device: torch.device,
    epoch: int,
    precision: str,
    log_every_n_steps: int,
) -> dict[str, float]:
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)
    model.train()
    amp_enabled, amp_dtype, _ = get_amp_settings(
        device=device,
        precision=precision,
    )
    running_total = 0.0
    running_components: defaultdict[str, float] = defaultdict(float)
    num_steps = 0
    progress = tqdm(
        train_loader,
        desc=f"Stage1 [{epoch + 1}]",
        dynamic_ncols=True,
    )
    coverage_weight = float(loss_weights.get("slot_coverage_loss", 0.0))
    for step, raw_batch in enumerate(progress):
        batch = prepare_batch_fn(raw_batch)
        if coverage_weight != 0.0 and "text_content_mask" not in batch:
            raise RuntimeError(
                "slot_coverage_loss has non-zero weight, but "
                "teacher.encode_text_tokens() did not provide "
                "text_content_mask. Do not silently train with "
                "coverage=0."
            )
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            assert amp_dtype is not None
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            )
        else:
            autocast_context = nullcontext()
        with autocast_context:
            loss_dict = compute_stage1_losses(
                model=model,
                batch=batch,
            )
            total_loss = compute_total_loss(
                loss_dict=loss_dict,
                loss_weights=loss_weights,
            )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite Stage-1 loss at epoch={epoch + 1}, step={step}")
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_value = float(total_loss.detach().item())
        running_total += total_value
        for name, value in loss_dict.items():
            running_components[name] += float(value.detach().item())
        num_steps += 1
        if log_every_n_steps > 0 and (step + 1) % log_every_n_steps == 0:
            progress.set_postfix(loss=f"{total_value:.4f}")
    if num_steps == 0:
        raise RuntimeError("Stage-1 train loader produced no batches")
    metrics: dict[str, float] = {
        "total_loss": running_total / num_steps,
    }
    for name, total in running_components.items():
        metrics[name] = total / num_steps
    return metrics

def get_stage1_state_dict(
    model: TAPER,
) -> dict[str, Tensor]:
    stage1_state: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        keep = name in STAGE1_STATE_EXACT_KEYS or name.startswith(STAGE1_STATE_PREFIXES)
        if keep:
            stage1_state[name] = tensor.detach().cpu()
    if "slot_queries" not in stage1_state:
        raise RuntimeError("Stage-1 checkpoint is missing slot_queries")
    return stage1_state

def save_stage1_checkpoint(
    path: str | Path,
    *,
    model: TAPER,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    cfg: DictConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": 1,
        "epoch": epoch,
        "stage1_state": get_stage1_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": OmegaConf.to_container(
            cfg,
            resolve=True,
        ),
    }
    torch.save(checkpoint, path)

def append_metrics(
    path: str | Path,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "epoch": epoch,
    }
    for name, value in train_metrics.items():
        record[f"train/{name}"] = float(value)
    for name, value in val_metrics.items():
        if isinstance(value, bool):
            record[name] = value
        elif isinstance(value, (int, float)):
            record[name] = float(value)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

def build_stage1_model(cfg: DictConfig, device: torch.device) -> TAPER:
    teacher = instantiate(cfg.stage1.teacher)
    if not isinstance(teacher, nn.Module):
        raise TypeError("cfg.stage1.teacher must instantiate an nn.Module")
    teacher = teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if not hasattr(teacher, "compose"):
        raise AttributeError("Stage-1 teacher must implement compose(...)")
    if not hasattr(teacher, "encode_text_tokens"):
        raise AttributeError("Stage-1 teacher must implement encode_text_tokens(texts)")
    m = cfg.stage1.model
    model = TAPER(
        teacher=teacher,
        text_dim=int(m.text_dim),
        reference_dim=int(m.reference_dim),
        teacher_query_dim=int(m.teacher_query_dim),
        query_dim=int(m.query_dim),
        slot_dim=int(m.slot_dim),
        state_dim=int(m.state_dim),
        num_slots=int(m.num_slots),
        num_primitives=int(m.num_primitives),
        mask_temperature=float(m.mask_temperature),
        router_temperature=float(m.router_temperature),
        retrieval_temperature=float(m.retrieval_temperature),
        neutral_mode=str(m.neutral_mode),
        slot_gate_threshold=float(m.slot_gate_threshold),
        hard_slot_gating_during_training=bool(m.hard_slot_gating_during_training),
        overlap_margin=float(m.overlap_margin),
        effect_diversity_margin=float(m.effect_diversity_margin),
        alpha_max=float(m.alpha_max),
    )
    model = model.to(device)
    configure_stage1_trainable_parameters(model)
    return model

def assert_finite_chunked(features: Tensor, *, name: str, chunk_rows: int = 32) -> None:
    for start in range(0, features.shape[0], chunk_rows,):
        end = min(start + chunk_rows, features.shape[0],)

        if not torch.isfinite(features[start:end]).all().item():
            raise ValueError(f"{name} contains NaN/Inf in rows {start}:{end}")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if "stage1" not in cfg:
        raise KeyError(
            "Missing cfg.stage1. Create the Stage-1 config before running src/train_stage1.py."
        )
    seed_everything(seed=int(cfg.seed), deterministic=bool(cfg.runtime.deterministic))
    configure_torch_runtime(deterministic=bool(cfg.runtime.deterministic), benchmark=bool(cfg.runtime.benchmark),)
    device = resolve_device(device_name=str(cfg.runtime.device), accelerator_index=int(cfg.runtime.accelerator_index),)
    print(f"Device: {device}")
    print(f"Precision: {cfg.runtime.precision}")
    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"
    output_dir = Path(cfg.paths.output_root) / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.jsonl"
    environment_path = output_dir / "environment.json"
    with environment_path.open("w", encoding="utf-8") as file:
        json.dump(
            collect_environment_metadata(),
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    train_loader = build_train_loader(
        annotation_root=annotation_root,
        batch_size=int(cfg.stage1.batch_size),
        num_workers=int(cfg.stage1.num_workers),
        seed=int(cfg.seed),
        caption_policy=str(cfg.stage1.train_caption_policy),
    )
    val_loaders = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=int(cfg.stage1.eval_batch_size),
        num_workers=int(cfg.stage1.num_workers),
    )
    print(f"Stage-1 train queries: {len(train_loader.dataset)}")
    for category, loader in val_loaders.items():
        print(f"Stage-1 val {category}: {len(loader.dataset)}")
    (
        train_reference_features,
        train_reference_name_to_idx,
    ) = load_features(Path(cfg.stage1.features.train_reference_dir))
    (
        val_reference_features,
        val_reference_name_to_idx,
    ) = load_features(Path(cfg.stage1.features.val_reference_dir))
    teacher_galleries = {}
    for category in CATEGORIES:
        features, name_to_idx = load_features(Path(cfg.stage1.features.val_gallery_dir) / category)
        teacher_galleries[category] = (features, name_to_idx)
    if train_reference_features.ndim < 2:
        raise ValueError("train_reference_features must have shape [N,...,D]")

    if val_reference_features.ndim < 2:
        raise ValueError("val_reference_features must have shape [N,...,D]")

    assert_finite_chunked(train_reference_features, name="train teacher reference cache")
    assert_finite_chunked(val_reference_features, name="val teacher reference cache")
    for category, (gallery_features, gallery_name_to_idx) in teacher_galleries.items():
        if gallery_features.ndim < 2:
            raise ValueError(f"{category} teacher gallery must have shape [G,...,D]")

        if gallery_features.shape[0] != len(gallery_name_to_idx):
            raise ValueError(f"{category} teacher gallery feature/index count mismatch")

        assert_finite_chunked(gallery_features, name=f"{category} teacher gallery")
    gallery_ids_by_category: dict[str, list[str]] = {}
    for category in CATEGORIES:
        gallery_ids_by_category[category] = load_fashioniq_split_ids(
            split_root=split_root,
            split="val",
            category=category,
        )
    model = build_stage1_model(
        cfg=cfg,
        device=device,
    )
    stage1_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in stage1_parameters)
    print(f"Stage-1 trainable parameters: {trainable_count:,}")
    loss_weights = {str(name): float(weight) for name, weight in cfg.stage1.loss_weights.items()}
    if not loss_weights:
        raise ValueError("cfg.stage1.loss_weights must not be empty")
    if all(weight == 0.0 for weight in loss_weights.values()):
        raise ValueError("At least one Stage-1 loss weight must be non-zero")
    if (
        float(
            loss_weights.get(
                "slot_coverage_loss",
                0.0,
            )
        )
        == 0.0
    ):
        print(
            "WARNING: slot_coverage_loss weight is zero. Stage 1 has no explicit coverage pressure."
        )
    def prepare_train_batch(
        batch: CIRBatch,
    ) -> dict[str, object]:
        return prepare_stage1_batch(
            batch,
            model=model,
            reference_features=train_reference_features,
            reference_name_to_idx=(train_reference_name_to_idx),
            device=device,
        )
    def prepare_val_batch(
        batch: CIRBatch,
    ) -> dict[str, object]:
        return prepare_stage1_batch(
            batch,
            model=model,
            reference_features=val_reference_features,
            reference_name_to_idx=(val_reference_name_to_idx),
            device=device,
        )
    optimizer = AdamW(
        stage1_parameters,
        lr=float(cfg.stage1.lr),
        weight_decay=float(cfg.stage1.weight_decay),
    )
    (
        _,
        _,
        use_grad_scaler,
    ) = get_amp_settings(
        device=device,
        precision=str(cfg.runtime.precision),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )
    print("Building fixed TCFR hard-negative cache...")
    model.eval()
    evaluation_config = OmegaConf.to_container(
        cfg.stage1.evaluation,
        resolve=True,
    )
    if not isinstance(
        evaluation_config,
        dict,
    ):
        raise TypeError("cfg.stage1.evaluation must resolve to a dictionary")
    with torch.no_grad():
        tcfr_cache = build_tcfr_cache(
            model=model,
            val_loaders=val_loaders,
            prepare_batch_fn=prepare_val_batch,
            teacher_galleries=teacher_galleries,
            gallery_ids_by_category=gallery_ids_by_category,
            config=evaluation_config,
            device=device,
        )
    best_tcfr = float("-inf")
    best_epoch = -1
    previous_anchor = None
    num_epochs = int(cfg.stage1.num_epochs)
    primary_metric = "stage1/tcfr_margin_drop"
    for epoch in range(num_epochs):
        train_metrics = train_stage1_one_epoch(
            model=model,
            train_loader=train_loader,
            prepare_batch_fn=prepare_train_batch,
            optimizer=optimizer,
            scaler=scaler,
            loss_weights=loss_weights,
            device=device,
            epoch=epoch,
            precision=str(cfg.runtime.precision),
            log_every_n_steps=int(cfg.logging.log_every_n_steps),
        )
        model.eval()
        with torch.no_grad():
            val_metrics, current_anchor = (
                evaluate_stage1_edit_slots(
                    model=model,
                    val_loaders=val_loaders,
                    prepare_batch_fn=prepare_val_batch,
                    teacher_galleries=teacher_galleries,
                    gallery_ids_by_category=gallery_ids_by_category,
                    tcfr_cache=tcfr_cache,
                    previous_anchor=previous_anchor,
                    config=evaluation_config,
                    device=device,
                )
            )
        previous_anchor = current_anchor
        if primary_metric not in val_metrics:
            raise KeyError(f"Stage-1 evaluator did not return '{primary_metric}'")
        current_tcfr = float(val_metrics[primary_metric])
        if not math.isfinite(current_tcfr):
            raise FloatingPointError("Stage-1 TCFR is not finite")
        health_ok = bool(
            val_metrics.get(
                "stage1/health_ok",
                True,
            )
        )
        save_stage1_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            cfg=cfg,
        )
        if current_tcfr > best_tcfr and health_ok:
            best_tcfr = current_tcfr
            best_epoch = epoch + 1
            save_stage1_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                cfg=cfg,
            )
            print(f"Saved best.pt | TCFR={best_tcfr:.6f}")
        append_metrics(
            metrics_path,
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        zero_active = val_metrics.get("stage1/zero_active_rate")
        all_active = val_metrics.get("stage1/all_active_rate")
        mask_stability = val_metrics.get("stage1/matched_mask_stability")
        effect_stability = val_metrics.get("stage1/matched_effect_stability")
        gate_drift = val_metrics.get("stage1/matched_gate_drift")
        message = (
            f"Epoch {epoch + 1}/{num_epochs}"
            f" | train={train_metrics['total_loss']:.4f}"
            f" | TCFR={current_tcfr:.6f}"
            f" | best={best_tcfr:.6f}"
        )
        if zero_active is not None:
            message += f" | zero={float(zero_active):.3f}"
        if all_active is not None:
            message += f" | all={float(all_active):.3f}"
        if mask_stability is not None:
            message += f" | S_mask={float(mask_stability):.3f}"
        if effect_stability is not None:
            message += f" | S_effect={float(effect_stability):.3f}"
        if gate_drift is not None:
            message += f" | D_gate={float(gate_drift):.3f}"
        print(message)
    print()
    print("Stage-1 training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best TCFR: {best_tcfr:.6f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")

if __name__ == "__main__":
    main()
