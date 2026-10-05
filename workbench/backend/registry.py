from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "models.yaml"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ValueError("unsupported model registry")
    for model in registry["models"]:
        if model["native_protocol"] is not None and model["native_protocol"] not in registry["protocols"]:
            raise ValueError(f"unknown protocol for {model['model_id']}")
        for checkpoint in model["checkpoint_variants"]:
            if checkpoint["evaluation_noise_pct"] != 0:
                raise ValueError(f"non-clean evaluation policy for {model['model_id']}")
    return registry


def model_by_id(model_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    for model in registry["models"]:
        if model["model_id"] == model_id:
            return model
    raise KeyError(model_id)


def checkpoint_by_id(model: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    for checkpoint in model["checkpoint_variants"]:
        if checkpoint["checkpoint_id"] == checkpoint_id:
            return checkpoint
    raise KeyError(checkpoint_id)


def checkpoint_path(model_id: str, checkpoint: dict[str, Any], root: Path = CHECKPOINT_ROOT) -> Path:
    return root / model_id / checkpoint["filename"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_availability(model: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    path = checkpoint_path(model["model_id"], checkpoint)
    downloaded = path.is_file()
    actual_sha = sha256_file(path) if downloaded else None
    expected_sha = checkpoint.get("expected_sha256")
    valid = downloaded and (expected_sha is None or actual_sha == expected_sha)
    reasons: list[str] = []
    if checkpoint["checkpoint_mapping_status"] == "UNVERIFIED":
        reasons.append("checkpoint_mapping_unverified")
    if checkpoint.get("download_url") is None:
        reasons.append("checkpoint_download_url_unavailable")
    if not downloaded:
        reasons.append("checkpoint_not_downloaded")
    if downloaded and not valid:
        reasons.append("checkpoint_hash_mismatch")
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "filename": checkpoint["filename"],
        "checkpoint_metadata_known": checkpoint["status"] not in {"BLOCKED"},
        "checkpoint_downloaded": downloaded,
        "checkpoint_path": str(path),
        "checkpoint_hash_valid": valid if downloaded else None,
        "runnable": not reasons,
        "blocking_reasons": reasons,
    }
