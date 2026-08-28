from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.checkpointing import git_metadata
from training.negative_bank import MINING_IMPLEMENTATION_VERSION, MMAP_MINING_CHUNK_SIZE


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matching_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    if not root.exists():
        return {"status": "unavailable"}
    files = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    if not files:
        return {"status": "unavailable"}
    return {str(path.relative_to(root)): _hash_file(path) for path in files}


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    config: dict[str, Any],
    backbone_manifest: Any,
    cache_manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    backbone = asdict(backbone_manifest) if hasattr(backbone_manifest, "__dataclass_fields__") else dict(backbone_manifest)
    dataset_root = Path(config["data"]["dataset_root"])
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    physical_batch = int(config["training"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation"])
    return {
        "schema_version": 1,
        "git": git_metadata(),
        "backbone": {
            "model_id": backbone["model_id"],
            "revision": backbone["revision"],
            "transformers_version": backbone["transformers_version"],
            "manifest_sha256": getattr(backbone_manifest, "sha256", _json_hash(backbone)),
            "tokenizer_config_sha256": _json_hash(backbone["tokenizer_config"]),
            "image_processor_config_sha256": _json_hash(backbone["image_processor_config"]),
            "text_tuning": config["backbone"]["text_tuning"],
            "vision_tuning": config["backbone"]["vision_tuning"],
        },
        "dataset": {
            "name": config["data"]["dataset"],
            "annotation_hashes": _matching_hashes(annotation_root, ("cap.*.train.json", "cap.*.val.json")),
            "split_hashes": _matching_hashes(split_root, ("split.*.train.json", "split.*.val.json")),
            "train_caption_policy": config["data"]["train_caption_policy"],
            "validation_caption_policy": config["data"]["validation_caption_policy"],
            "correction_policy": config["data"]["correction_policy"],
            "correction_hashes": _matching_hashes(annotation_root, ("correction_dict_*.json",)),
            "validation_protocol": config["data"]["validation_protocol"],
            "external_generated_data": False,
        },
        "cache_manifest_hashes": dict(sorted(cache_manifest_hashes.items())),
        "seed": int(config["seed"]),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "precision": config["runtime"]["precision"],
            "configured_device": config["runtime"]["device"],
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda or "unavailable",
            "cuda_driver": "unavailable_not_exposed_portably_by_torch",
            "gpu_name": gpu_name,
            "amp_scaler": "not_used_bf16" if config["runtime"]["precision"] == "bf16" else "not_configured",
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "batching": {
            "physical_batch": physical_batch,
            "gradient_accumulation": accumulation,
            "effective_batch": physical_batch * accumulation,
        },
        "optimizer": config["optimizer"],
        "scheduler": {
            "name": "linear_5pct_warmup_cosine_to_0.1",
            "warmup_fraction": 0.05,
            "max_optimizer_updates": config["training"].get("max_optimizer_updates"),
        },
        "teacher": {
            "hard_negatives": int(config["teacher"]["hard_negatives"]),
            "temperature": float(config["teacher"]["retrieval_temperature"]),
            "negative_bank_manifest_hashes": dict(sorted(cache_manifest_hashes.items())),
            "mining_implementation_version": MINING_IMPLEMENTATION_VERSION,
            "mmap_chunk_size": MMAP_MINING_CHUNK_SIZE,
            "normalization": "L2 FP32 before top-k",
        },
        "policy": {
            "step_cost": float(config["policy"]["step_cost"]),
            "accepted_health_gates": sorted(config["training"].get("approved_health_gates", [])),
        },
        "resume_contract": {
            "fresh_run": "seeded; deterministic audit subset and caption sampling",
            "checkpoint": "epoch-boundary only",
            "mid_epoch": "unsupported and rejected",
            "dataset_epoch_recorded": True,
        },
    }


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
