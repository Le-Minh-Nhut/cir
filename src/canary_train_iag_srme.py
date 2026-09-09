from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.images import FashionIQImageCollator
from datasets.common import DirectoryImageStore
from datasets.fashioniq import FashionIQDataset
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig
from runtime import configure_torch_runtime, seed_everything
from training.engine import (
    assert_training_setup,
    parameter_count_diagnostics,
    resolve_precision,
    trainable_parameters,
)


CHECKPOINT = "qihoo360/fg-clip-base"
REVISION = "454d76372c2cf5eb48fa0d871fd0534481484d97"
CATEGORIES = ("dress", "shirt", "toptee")


def _gradients_are_finite(parameters: Iterable[nn.Parameter]) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
        if parameter.grad is not None
    )


def _complete_amp_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    tracked: dict[str, nn.Parameter],
    *,
    scale_before: float,
    overflow: bool,
) -> tuple[float, dict[str, float]]:
    before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    changes = {
        name: float((parameter.detach().float() - before[name].float()).abs().max())
        for name, parameter in tracked.items()
    }
    if overflow:
        if not scaler.is_enabled() or scale_after >= scale_before:
            raise FloatingPointError("AMP overflow was not skipped with scale backoff")
        if any(change != 0.0 for change in changes.values()):
            raise AssertionError("parameters changed during a skipped AMP step")
    return scale_after, changes


def build_model(
    device: torch.device,
    global_readout_mode: str = "learned_qg",
    finetune_policy: str = "full",
) -> tuple[IAGSRME, object, object]:
    train_vision = finetune_policy == "full"
    readout_experiment = "R0-QG" if global_readout_mode == "learned_qg" else "R0-NCLS"
    suffix = "FULL" if finetune_policy == "full" else "TEXT"
    regime = FGCLIPRegime(
        checkpoint=CHECKPOINT,
        revision=REVISION,
        train_vision=train_vision,
        train_text=True,
        train_text_projection=False,
        global_readout_mode=global_readout_mode,
        readout_experiment=readout_experiment,
        finetune_policy=finetune_policy,
        experiment_identity=f"{readout_experiment}-{suffix}",
    )
    backbone = FGCLIPBackbone.from_pretrained(regime, text_width=256)
    tokenizer, processor = FGCLIPBackbone.load_processor(CHECKPOINT, REVISION)
    config = IAGSRMEConfig(width=256, num_context_edits=4, num_heads=8)
    return IAGSRME(backbone, config).to(device), tokenizer, processor


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential-context FashionIQ canary")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--global-readout-mode",
        choices=("learned_qg", "native_cls"),
        default="learned_qg",
    )
    parser.add_argument(
        "--finetune-policy",
        choices=("full", "text_only"),
        default="full",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("FASHIONIQ_ROOT", "data/fashionIQ_dataset"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)
    configure_torch_runtime(deterministic=True, benchmark=False)
    precision = resolve_precision(args.precision, device)
    model, tokenizer, processor = build_model(
        device, args.global_readout_mode, args.finetune_policy
    )

    root = Path(args.dataset_root)
    dataset = FashionIQDataset(
        root / "captions",
        "train",
        CATEGORIES,
        caption_policy="ordered_and",
        seed=args.seed,
    )
    objective = IAGSRMEObjective(ObjectiveConfig()).to(device)
    parameter_counts = parameter_count_diagnostics(model, objective)
    collator = FashionIQImageCollator(
        DirectoryImageStore(root / "images"),
        tokenizer,
        processor,
        include_targets=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-5, weight_decay=0.01)
    assert_training_setup(model, objective, optimizer, device)
    scaler = torch.amp.GradScaler(device.type, enabled=precision.scaler_enabled)

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    records = []
    for step_index in range(args.steps):
        batch = next(iterator).to(device)
        assert batch.target_pixels is not None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=precision.autocast_enabled,
            dtype=precision.autocast_dtype,
        ):
            output = model(
                batch.reference_pixels,
                batch.input_ids,
                batch.attention_mask,
                batch.content_mask,
            )
            targets = model.encode_global_images(batch.target_pixels)
            target_ids = [str(value) for value in batch.target_ids]
            losses = objective(output, targets, target_ids)
        scaler.scale(losses["total"]).backward()
        gradient_norms = {
            name: sum(
                float(parameter.grad.detach().float().norm())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            for name, module in {
                "proposal": model.proposal,
                "grounder": model.grounder,
                "action_fusion": model.action_fusion,
                "executor": model.executor,
                "text_model": model.backbone.model.text_model,
                "text_adapter": model.backbone.text_adapter,
            }.items()
        }
        if len(output["steps"]) != model.config.num_context_edits:
            raise AssertionError("every configured context edit must execute exactly once")
        if not _gradients_are_finite(trainable_parameters(model, objective)):
            raise FloatingPointError("non-finite gradient in sequential canary")
        if any(value <= 0.0 for value in gradient_norms.values()):
            raise AssertionError(f"missing sequential gradient: {gradient_norms}")
        scaler.step(optimizer)
        scaler.update()
        records.append(
            {
                "step": step_index + 1,
                "loss": float(losses["total"].detach()),
                "terminal": float(losses["terminal"].detach()),
                "slot_diagnostics": {
                    name: float(value)
                    for name, value in losses.items()
                    if name.startswith("slot_")
                },
                "gradient_norms": gradient_norms,
                "sequential_steps": len(output["steps"]),
            }
        )
    memory = (
        {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        if device.type == "cuda"
        else None
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "global_readout_mode": args.global_readout_mode,
                "finetune_policy": args.finetune_policy,
                "experiment_identity": model.backbone.experiment_identity,
                "parameter_counts": parameter_counts,
                "peak_cuda_memory": memory,
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
