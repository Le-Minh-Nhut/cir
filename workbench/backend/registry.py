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
        if bool(model["source_available"]) != bool(model.get("source_dir")):
            raise ValueError(f"source_dir mismatch for {model['model_id']}")
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
    path = checkpoint_path(model["model_id"], checkpoint, CHECKPOINT_ROOT)
    downloaded = path.is_file()
    local_sha = sha256_file(path) if downloaded else None
    official_sha = checkpoint.get("expected_sha256")
    official_sha_match = local_sha == official_sha if downloaded and official_sha else None
    mapping_verified = checkpoint["checkpoint_mapping_status"] not in {"UNVERIFIED"}
    source_synced = bool(model.get("source_dir")) and (ROOT / "third_party" / model["source_dir"]).is_dir()
    adapter_command_ready = mapping_verified and model["model_id"] in {"csmcir", "encoder"}
    runtime_verified = model.get("reproduction_status") == "VERIFIED"
    runnable = downloaded and source_synced and mapping_verified and (official_sha is None or official_sha_match is True) and adapter_command_ready
    blocking_reasons = []
    if not mapping_verified:
        blocking_reasons.append("checkpoint_mapping_unverified")
    if not downloaded:
        blocking_reasons.append("checkpoint_not_installed")
    if official_sha and official_sha_match is False:
        blocking_reasons.append("official_hash_mismatch")
    if not adapter_command_ready:
        blocking_reasons.append("adapter_command_unverified")
    return {"checkpoint_id": checkpoint["checkpoint_id"], "filename": checkpoint["filename"], "source_metadata": checkpoint["status"] != "BLOCKED", "source_synced_locally": source_synced, "automatic_download_available": checkpoint.get("download_url") is not None, "checkpoint_downloaded": downloaded, "local_sha256": local_sha, "official_sha256_known": official_sha is not None, "official_sha256_match": official_sha_match, "mapping_verified": mapping_verified, "adapter_command_ready": adapter_command_ready, "runtime_verified": runtime_verified, "runnable": runnable, "checkpoint_path": str(path), "blocking_reasons": blocking_reasons}
