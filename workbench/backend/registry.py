from __future__ import annotations

import hashlib
from workbench.backend.fashioniq_layout import paths_for_layout
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "models.yaml"
AUXILIARY_ASSET_REGISTRY_PATH = ROOT / "registry" / "auxiliary_assets.yaml"
PREPARATION_CONTRACT_REGISTRY_PATH = ROOT / "registry" / "preparation_contracts.yaml"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ValueError("unsupported model registry")
    layouts = {"fashioniq_standard", "fashioniq_ilearn_resized", "fashioniq_ilearn_resized_training", "fashioniq_clvc_resized", "fashioniq_dcnet"}
    mapping_statuses = {"VERIFIED_METADATA", "UNVERIFIED", "URL_UNRESOLVED", "ARTIFACT_ASSOCIATED_UNVALIDATED"}
    preparation_contracts = load_preparation_contracts()
    for model in registry["models"]:
        native_protocol = model["native_protocol"]
        supported_protocols = model["supported_protocols"]
        if native_protocol is not None and native_protocol not in registry["protocols"]:
            raise ValueError(f"unknown protocol for {model['model_id']}")
        if any(protocol not in registry["protocols"] for protocol in supported_protocols):
            raise ValueError(f"unknown supported protocol for {model['model_id']}")
        if native_protocol is not None and native_protocol not in supported_protocols:
            raise ValueError(f"native protocol unsupported for {model['model_id']}")
        if model.get("literature_split_label") not in {"original", "val", None}:
            raise ValueError(f"unknown literature split label for {model['model_id']}")
        if native_protocol is not None and model.get("literature_split_label") != registry["protocols"][native_protocol]["literature_split_label"]:
            raise ValueError(f"literature split label mismatch for {model['model_id']}")
        if bool(model["source_available"]) != bool(model.get("source_dir")):
            raise ValueError(f"source_dir mismatch for {model['model_id']}")
        if model["source_available"] and model.get("fashioniq_layout") not in layouts:
            raise ValueError(f"unknown FashionIQ layout for {model['model_id']}")
        if "research_generation" in model and model["research_generation"] not in {"legacy", "recent"}:
            raise ValueError(f"unknown research generation for {model['model_id']}")
        if "publication_year" in model and model["publication_year"] is not None and (not isinstance(model["publication_year"], int) or model["publication_year"] < 2020):
            raise ValueError(f"invalid publication year for {model['model_id']}")
        if "preparation_contract" in model and model["preparation_contract"] not in {"manual_standard", "manual_ilearn_resized", "manual_clvc_resized", "manual_dcnet", "unavailable"}:
            raise ValueError(f"unknown preparation contract for {model['model_id']}")
        if model.get("preparation_id") not in {None, *preparation_contracts}:
            raise ValueError(f"unknown preparation contract for {model['model_id']}")
        if model.get("native_protocol_condition") and native_protocol is not None:
            raise ValueError(f"conditional native protocol must remain unavailable for {model['model_id']}")
        variants = {checkpoint["checkpoint_id"] for checkpoint in model["checkpoint_variants"]}
        for bundle in model.get("checkpoint_bundles", []):
            if {"bundle_id", "required_checkpoint_ids", "structure", "mapping_status", "notes"} - bundle.keys() or not bundle["required_checkpoint_ids"] or not set(bundle["required_checkpoint_ids"]).issubset(variants):
                raise ValueError(f"invalid checkpoint bundle for {model['model_id']}")
        for checkpoint in model["checkpoint_variants"]:
            if checkpoint["evaluation_noise_pct"] != 0:
                raise ValueError(f"non-clean evaluation policy for {model['model_id']}")
            if checkpoint["checkpoint_mapping_status"] not in mapping_statuses:
                raise ValueError(f"unknown checkpoint mapping state for {model['model_id']}")
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


def load_preparation_contracts(path: Path = PREPARATION_CONTRACT_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("schema_version") != 1 or not isinstance(registry.get("contracts"), list):
        raise ValueError("unsupported preparation contract registry")
    contracts: dict[str, dict[str, Any]] = {}
    required = {"preparation_id", "source_family", "raw_inputs", "generated_artifacts", "external_assets", "deterministic_status", "automation_policy", "notes"}
    asset_required = {"name", "kind", "state", "required_for_evaluation", "required_for_preparation", "expected_local_location", "source", "optional", "notes"}
    for contract in registry["contracts"]:
        if required - contract.keys() or contract["preparation_id"] in contracts:
            raise ValueError("invalid preparation contract")
        for group in ("raw_inputs", "generated_artifacts"):
            for artifact in contract[group]:
                if {"name", "kind", "scope", "paths"} - artifact.keys() or artifact["scope"] != "fashioniq_root":
                    raise ValueError(f"invalid preparation artifact for {contract['preparation_id']}")
                for path_value in artifact["paths"]:
                    path = PurePosixPath(path_value)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"unsafe preparation path for {contract['preparation_id']}")
        for asset in contract["external_assets"]:
            if asset_required - asset.keys() or not isinstance(asset["required_for_evaluation"], bool) or not isinstance(asset["required_for_preparation"], bool) or not isinstance(asset["optional"], bool):
                raise ValueError(f"invalid external asset for {contract['preparation_id']}")
            location = asset["expected_local_location"]
            if location is not None:
                path = PurePosixPath(location)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe external asset path for {contract['preparation_id']}")
        contracts[contract["preparation_id"]] = contract
    return contracts


def preparation_contract_for_model(model: dict[str, Any]) -> dict[str, Any] | None:
    preparation_id = model.get("preparation_id")
    if preparation_id is None:
        return None
    try:
        return load_preparation_contracts()[preparation_id]
    except KeyError as error:
        raise ValueError(f"unknown preparation contract for {model['model_id']}") from error


def preparation_paths(contract: dict[str, Any], root: Path, group: str) -> tuple[Path, ...]:
    return tuple(root / path for artifact in contract[group] for path in artifact["paths"])


def required_runtime_assets(model: dict[str, Any]) -> list[dict[str, Any]]:
    contract = preparation_contract_for_model(model)
    if contract is None:
        return []
    return [asset for asset in contract["external_assets"] if asset["required_for_evaluation"] and not asset["optional"]]


def external_asset_blockers(model: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for asset in required_runtime_assets(model):
        location = asset["expected_local_location"]
        if location is None:
            blockers.append(f"external runtime asset provenance unresolved: {asset['name']}")
        else:
            blockers.append(f"external runtime asset missing: {asset['name']} ({location})")
    return blockers


def checkpoint_missing_paths(path: Path, checkpoint: dict[str, Any]) -> list[Path]:
    if checkpoint.get("artifact_type") == "run_directory":
        return ([] if path.is_dir() else [path]) + [path / name for name in checkpoint.get("required_files", []) if not (path / name).is_file()]
    return [] if path.is_file() else [path]


def checkpoint_is_present(path: Path, checkpoint: dict[str, Any]) -> bool:
    return not checkpoint_missing_paths(path, checkpoint)


def checkpoint_bundle_missing_paths(model: dict[str, Any], root: Path) -> dict[str, list[Path]]:
    variants = {checkpoint["checkpoint_id"]: checkpoint for checkpoint in model["checkpoint_variants"]}
    missing: dict[str, list[Path]] = {}
    for bundle in model.get("checkpoint_bundles", []):
        paths = [path for checkpoint_id in bundle["required_checkpoint_ids"] for path in checkpoint_missing_paths(checkpoint_path(model["model_id"], variants[checkpoint_id], root), variants[checkpoint_id])]
        if paths:
            missing[bundle["bundle_id"]] = paths
    return missing


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


def fashioniq_required_paths(model: dict[str, Any], root: Path) -> tuple[Path, ...]:
    layout = model.get("fashioniq_layout")
    if layout is None:
        return ()
    return paths_for_layout(root, layout)


def checkpoint_availability(model: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    downloaded = checkpoint_is_present(path, checkpoint)
    local_sha = sha256_file(path) if downloaded and path.is_file() else None
    official_sha = checkpoint.get("expected_sha256")
    official_sha_match = local_sha == official_sha if downloaded and official_sha else None
    mapping_verified = checkpoint["checkpoint_mapping_status"] == "VERIFIED_METADATA"
    source_synced = bool(model.get("source_dir")) and (config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]).is_dir()
    adapter_command_ready = bool(model.get("command_status") == "COMMAND_AUDITED" or (model.get("command_status") is None and model["model_id"] in {"csmcir", "encoder"}))
    runtime_verified = model.get("reproduction_status") == "VERIFIED"
    external_blockers = external_asset_blockers(model)
    bundle_missing = checkpoint_bundle_missing_paths(model, config.WORKBENCH_CHECKPOINT_ROOT)
    runnable = downloaded and source_synced and mapping_verified and (official_sha is None or official_sha_match is True) and adapter_command_ready and not external_blockers and not bundle_missing
    blocking_reasons = []
    if not mapping_verified:
        blocking_reasons.append("checkpoint_mapping_unverified")
    if not downloaded:
        blocking_reasons.append("checkpoint_not_installed")
    if official_sha and official_sha_match is False:
        blocking_reasons.append("official_hash_mismatch")
    if not adapter_command_ready:
        blocking_reasons.append("adapter_command_unverified")
    blocking_reasons.extend(external_blockers)
    blocking_reasons.extend(f"checkpoint_bundle_incomplete:{bundle_id}" for bundle_id in bundle_missing)
    return {"checkpoint_id": checkpoint["checkpoint_id"], "filename": checkpoint["filename"], "source_metadata": checkpoint["status"] != "BLOCKED", "source_root": str(config.WORKBENCH_THIRD_PARTY_ROOT), "checkpoint_root": str(config.WORKBENCH_CHECKPOINT_ROOT), "source_synced_locally": source_synced, "automatic_download_available": checkpoint.get("download_url") is not None, "checkpoint_downloaded": downloaded, "local_sha256": local_sha, "official_sha256_known": official_sha is not None, "official_sha256_match": official_sha_match, "mapping_verified": mapping_verified, "adapter_command_ready": adapter_command_ready, "runtime_verified": runtime_verified, "runnable": runnable, "checkpoint_path": str(path), "blocking_reasons": blocking_reasons}
