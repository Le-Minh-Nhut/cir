from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from backbones.fgclip2_base import FGCLIP2BaseBackbone


def git_metadata() -> dict[str, str]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": str(bool(run("status", "--porcelain"))).lower(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    backbone: FGCLIP2BaseBackbone,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    stage: str,
    curriculum_state: dict[str, Any],
    resolved_config: dict[str, Any],
    manifest_hashes: dict[str, str],
    best_metrics: dict[str, float],
    ema_state: dict[str, Any] | None = None,
    checkpoint_reason: str = "unspecified",
    validation_metrics: dict[str, Any] | None = None,
    policy_metrics: dict[str, Any] | None = None,
    functional_health_metrics: dict[str, Any] | None = None,
    selection_state: dict[str, Any] | None = None,
    dataset_epoch: int | None = None,
    micro_step: int | None = None,
) -> None:
    trainable_backbone = {
        name: parameter.detach().cpu()
        for name, parameter in backbone.model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "trainable_backbone": trainable_backbone,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema_state,
        "epoch": epoch,
        "global_step": global_step,
        "stage": stage,
        "curriculum_state": curriculum_state,
        "rng": rng_state(),
        "config": resolved_config,
        "manifest_hashes": manifest_hashes,
        "best_metrics": best_metrics,
        "checkpoint_selection_reason": checkpoint_reason,
        "validation_metrics": validation_metrics or {},
        "policy_metrics": policy_metrics or {},
        "functional_health_metrics": functional_health_metrics or {},
        "checkpoint_selection_state": selection_state,
        "resume_contract": "deterministic_epoch_boundary_only",
        "dataset_epoch": dataset_epoch,
        "micro_step": micro_step,
        "git": git_metadata(),
        "torch_version": torch.__version__,
    }
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    backbone: FGCLIP2BaseBackbone,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    expected_manifest_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("micro_step") is not None:
        raise RuntimeError(
            "Mid-epoch resume is unsupported; canonical checkpoints resume at epoch boundaries"
        )
    if expected_manifest_hashes is not None and payload["manifest_hashes"] != expected_manifest_hashes:
        raise RuntimeError("Checkpoint feature manifest hashes do not match current caches")
    model.load_state_dict(payload["model"], strict=True)
    named_parameters = dict(backbone.model.named_parameters())
    expected_names = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
    if set(payload["trainable_backbone"]) != expected_names:
        raise RuntimeError("Checkpoint trainable text parameter set disagrees with tuning config")
    with torch.no_grad():
        for name, value in payload["trainable_backbone"].items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return payload
