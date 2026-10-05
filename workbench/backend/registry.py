from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "models.yaml"
AUXILIARY_ASSET_REGISTRY_PATH = ROOT / "registry" / "auxiliary_assets.yaml"
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


def load_auxiliary_assets(path: Path = AUXILIARY_ASSET_REGISTRY_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("assets"), list):
        raise ValueError("unsupported auxiliary asset registry")
    required = {
        "model_id", "family", "required_for", "source_type", "source_repo", "source_revision", "source_path",
        "destination_scope", "destination_path", "automatic_download_available", "expected_sha256", "provenance_status",
    }
    for asset in registry["assets"]:
        if required - asset.keys():
            raise ValueError("incomplete auxiliary asset record")
        available = asset["automatic_download_available"]
        destination = PurePosixPath(asset["destination_path"])
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError(f"unsafe auxiliary destination for {asset['model_id']}: {asset['destination_path']}")
        if asset["destination_scope"] not in {"fashioniq_root", "model_source"}:
            raise ValueError(f"unknown auxiliary destination scope: {asset['destination_scope']}")
        if available != (asset["source_type"] == "author_huggingface" and bool(asset["source_repo"]) and bool(asset["source_revision"]) and bool(asset["source_path"])):
            raise ValueError(f"auxiliary source mismatch for {asset['model_id']}: {asset['destination_path']}")
    return registry["assets"]


def auxiliary_assets_for_model(model_id: str) -> list[dict[str, Any]]:
    return [asset for asset in load_auxiliary_assets() if asset["model_id"] == model_id]


def auxiliary_destination(asset: dict[str, Any], third_party_root: Path, fashioniq_root: Path) -> Path:
    if asset["destination_scope"] == "fashioniq_root":
        return fashioniq_root / asset["destination_path"]
    if asset["destination_scope"] == "model_source":
        return third_party_root / model_by_id(asset["model_id"])["source_dir"] / asset["destination_path"]
    raise ValueError(f"unknown auxiliary destination scope: {asset['destination_scope']}")


def auxiliary_source_url(asset: dict[str, Any]) -> str | None:
    if not asset["automatic_download_available"]:
        return None
    return f"{asset['source_repo']}/resolve/{asset['source_revision']}/{asset['source_path']}"


def auxiliary_model_ids() -> set[str]:
    return {asset["model_id"] for asset in load_auxiliary_assets()}


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


def checkpoint_path(model_id: str, checkpoint: dict[str, Any], root: Path | None = None) -> Path:
    from workbench.backend.operator_config import resolve_config

    return (root or resolve_config().WORKBENCH_CHECKPOINT_ROOT) / model_id / checkpoint["filename"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_availability(model: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    downloaded = path.is_file()
    local_sha = sha256_file(path) if downloaded else None
    official_sha = checkpoint.get("expected_sha256")
    official_sha_match = local_sha == official_sha if downloaded and official_sha else None
    mapping_verified = checkpoint["checkpoint_mapping_status"] not in {"UNVERIFIED"}
    source_synced = bool(model.get("source_dir")) and (config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]).is_dir()
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
    return {"checkpoint_id": checkpoint["checkpoint_id"], "filename": checkpoint["filename"], "source_metadata": checkpoint["status"] != "BLOCKED", "source_root": str(config.WORKBENCH_THIRD_PARTY_ROOT), "checkpoint_root": str(config.WORKBENCH_CHECKPOINT_ROOT), "source_synced_locally": source_synced, "automatic_download_available": checkpoint.get("download_url") is not None, "checkpoint_downloaded": downloaded, "local_sha256": local_sha, "official_sha256_known": official_sha is not None, "official_sha256_match": official_sha_match, "mapping_verified": mapping_verified, "adapter_command_ready": adapter_command_ready, "runtime_verified": runtime_verified, "runnable": runnable, "checkpoint_path": str(path), "blocking_reasons": blocking_reasons}
