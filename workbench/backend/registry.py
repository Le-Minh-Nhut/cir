from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from workbench.backend.fashioniq_layout import paths_for_layout

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "models.yaml"
AUXILIARY_ASSET_REGISTRY_PATH = ROOT / "registry" / "auxiliary_assets.yaml"
PREPARATION_CONTRACT_REGISTRY_PATH = ROOT / "registry" / "preparation_contracts.yaml"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        custom = os.environ.get("WORKBENCH_REGISTRY_PATH")
        path = Path(custom).resolve() if custom else REGISTRY_PATH
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
            if "cache_path_suffix" in asset:
                path = PurePosixPath(asset["cache_path_suffix"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe external asset cache path for {contract['preparation_id']}")
            expected_sha = asset.get("expected_sha256")
            expected_prefix = asset.get("expected_sha256_prefix")
            if expected_sha and expected_prefix:
                raise ValueError(f"external asset must use one checksum form for {contract['preparation_id']}")
            if expected_sha and (len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha.lower())):
                raise ValueError(f"invalid external SHA-256 for {contract['preparation_id']}")
            if expected_prefix and (len(expected_prefix) < 8 or len(expected_prefix) > 64 or any(char not in "0123456789abcdef" for char in expected_prefix.lower())):
                raise ValueError(f"invalid external SHA-256 prefix for {contract['preparation_id']}")
            if asset.get("location_env_var") and not asset["location_env_var"].isidentifier():
                raise ValueError(f"invalid external asset location variable for {contract['preparation_id']}")
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


def external_asset_statuses(model: dict[str, Any]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for asset in required_runtime_assets(model):
        configured = os.environ.get(asset.get("location_env_var", ""))
        location = configured.strip() if configured and configured.strip() else asset.get("expected_local_location")
        if location is None and asset.get("cache_path_suffix"):
            cache_root = os.environ.get(asset.get("cache_root_env_var", ""))
            root = Path(cache_root).expanduser() if cache_root else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "torch"
            location = str(root / asset["cache_path_suffix"])
        expected_sha = asset.get("expected_sha256")
        expected_prefix = asset.get("expected_sha256_prefix")
        if location is None:
            state = asset.get("state", "UNKNOWN")
            status = "UNRESOLVED_PROVENANCE" if state == "BLOCKED_EXTERNAL_ASSET_PROVENANCE" else "UNCONFIGURED_LOCATION"
            statuses.append({"name": asset["name"], "status": status, "path": None,
                             "expected_sha256": expected_sha, "expected_sha256_prefix": expected_prefix,
                             "actual_sha256": None})
            continue
        path = Path(location).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        path = path.resolve()
        status = "CONFIGURED_MISSING"
        actual_sha = None
        if path.exists():
            if not path.is_file():
                status = "INVALID_FILE_TYPE"
            elif path.stat().st_size == 0:
                status = "EMPTY_PLACEHOLDER"
            else:
                actual_sha = sha256_file(path)
                if expected_sha and actual_sha != expected_sha:
                    status = "CHECKSUM_MISMATCH"
                elif expected_prefix and not actual_sha.startswith(expected_prefix.lower()):
                    status = "CHECKSUM_MISMATCH"
                elif expected_sha or expected_prefix:
                    status = "VERIFIED_LOCAL_ASSET"
                else:
                    status = "EXISTS_UNVERIFIED_CHECKSUM"
        statuses.append({"name": asset["name"], "status": status, "path": str(path),
                         "expected_sha256": expected_sha, "expected_sha256_prefix": expected_prefix,
                         "actual_sha256": actual_sha})
    return statuses


def external_asset_blockers(model: dict[str, Any]) -> list[str]:
    messages = {
        "UNRESOLVED_PROVENANCE": "provenance unresolved",
        "UNCONFIGURED_LOCATION": "local location unconfigured",
        "CONFIGURED_MISSING": "file missing",
        "INVALID_FILE_TYPE": "expected regular file, found non-file",
        "EMPTY_PLACEHOLDER": "empty placeholder file",
        "CHECKSUM_MISMATCH": "checksum mismatch",
        "EXISTS_UNVERIFIED_CHECKSUM": "file exists but no verified checksum is available",
    }
    return [
        f"external runtime asset {messages[item['status']]}: {item['name']}" + (f" ({item['path']})" if item["path"] else "")
        for item in external_asset_statuses(model) if item["status"] != "VERIFIED_LOCAL_ASSET"
    ]


def replay_readiness(model: dict[str, Any]) -> dict[str, Any]:
    """Metadata-only replay infrastructure status. Not runtime readiness."""
    return {
        "model_id": model["model_id"],
        "replay_code": model.get("replay_status", "NOT_IMPLEMENTED"),
        "command_status": model.get("command_status"),
        "checkpoint_provenance": model.get("reproduction_status", "NOT_RUN"),
        "category_model_policy": model.get("category_model_policy", "shared"),
        "environment_documented": model.get("environment") is not None,
        "protocol": model.get("native_protocol"),
    }


def checkpoint_missing_paths(path: Path, checkpoint: dict[str, Any]) -> list[Path]:
    artifact_type = checkpoint.get("artifact_type", "file")
    if artifact_type in {"run_directory", "bundle_directory", "bundle_placeholder"}:
        members = checkpoint.get("bundle_members")
        required_files = [member["filename"] for member in members] if members else checkpoint.get("required_files", [])
        if not _regular_directory(path):
            return [path]
        missing = []
        for name in required_files:
            member_path = path / name
            try:
                _safe_relative_file(path, name)
            except ValueError:
                missing.append(member_path)
                continue
            if member_path.stat().st_size == 0:
                missing.append(member_path)
            else:
                member = next((item for item in members or [] if item["filename"] == name), None)
                expected_sha = member.get("expected_sha256") if member else None
                if expected_sha and sha256_file(member_path) != expected_sha:
                    missing.append(member_path)
        return missing
    if not _regular_file(path) or path.stat().st_size == 0:
        return [path]
    expected_sha = checkpoint.get("expected_sha256")
    return [path] if expected_sha and sha256_file(path) != expected_sha else []


def _regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _regular_directory(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir()


def _safe_relative_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"unsafe artifact member path: {name}")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"symlink artifact member is not allowed: {candidate}")
    if not _regular_file(candidate):
        raise ValueError(f"artifact member is not a regular file: {candidate}")
    return candidate

def checkpoint_is_present(path: Path, checkpoint: dict[str, Any]) -> bool:
    return not checkpoint_missing_paths(path, checkpoint)


def checkpoint_bundle_missing_paths(model: dict[str, Any], root: Path) -> dict[str, list[Path]]:
    variants = {checkpoint["checkpoint_id"]: checkpoint for checkpoint in model["checkpoint_variants"]}
    missing: dict[str, list[Path]] = {}
    for bundle in model.get("checkpoint_bundles", []):
        paths = [path for checkpoint_id in bundle["required_checkpoint_ids"]
                 for path in checkpoint_missing_paths(checkpoint_path(model["model_id"], variants[checkpoint_id], root), variants[checkpoint_id])]
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


def source_runtime_artifacts(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Paths *inside a model's source checkout* the workbench legitimately creates.

    Everything here is source-backed: each entry traces to a declared registry
    contract (an audited auxiliary asset destination, or the CSMCIR fixed canonical
    dataset link). Anything not declared here is an unauthorized source change.
    """
    model_id = model["model_id"]
    if not model.get("source_dir"):
        return []
    artifacts: list[dict[str, Any]] = []
    for asset in auxiliary_assets_for_model(model_id):
        if asset["destination_scope"] != "model_source":
            continue
        destination = PurePosixPath(asset["destination_path"])
        if destination.is_absolute() or ".." in destination.parts:
            continue
        artifacts.append({
            "kind": "declared_auxiliary_asset",
            "path": asset["destination_path"],
            "family": asset["family"],
            "expected_sha256": asset.get("expected_sha256"),
            "provenance_status": asset.get("provenance_status"),
            "source": asset.get("source_path"),
        })
    if model_id == "csmcir":
        artifacts.append({
            "kind": "canonical_dataset_link",
            "path": "fashionIQ_dataset",
            "expects": "FASHIONIQ_ROOT",
        })
    return artifacts


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


def sha256_file(path: Path, required_files: list[str] | None = None) -> str:
    """Hash file bytes for single files, or sorted relative-path/content pairs for directories.

    For directory artifacts (run_directory, bundle_directory): pass required_files to hash
    only the declared meaningful members, excluding unrelated logs or metadata. For a generic
    directory with no required_files restriction, all regular non-symlink children are hashed.
    Hashes are distinct by artifact type: file and directory with identical content yield
    different digests.
    """
    digest = hashlib.sha256()
    if _regular_file(path):
        _update_file_digest(digest, path)
        return digest.hexdigest()
    if _regular_directory(path):
        # Directory artifact: prefix + sorted (relative-path, file-bytes) pairs.
        digest.update(b"directory\0")
        files = ([_safe_relative_file(path, name) for name in required_files]
                 if required_files is not None else _regular_tree_files(path))
        members = sorted(files, key=lambda item: item.relative_to(path).as_posix())
        # The member list itself is hashed first, length-prefixed, so a directory
        # artifact cannot be re-identified by adding or removing a member and the
        # per-member framing cannot be confused with another member's content.
        names = [member.relative_to(path).as_posix().encode("utf-8") for member in members]
        digest.update(len(names).to_bytes(8, "big"))
        for name in names:
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
        for member, name in zip(members, names):
            digest.update(b"member\0")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            _update_file_digest(digest, member)
        return digest.hexdigest()
    raise ValueError(f"artifact is not a regular file or non-symlink directory: {path}")


def _regular_tree_files(root: Path) -> list[Path]:
    files = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"symlink artifact member is not allowed: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"non-regular artifact member: {candidate}")
        files.append(candidate)
    return files


def _update_file_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def fashioniq_required_paths(model: dict[str, Any], root: Path) -> tuple[Path, ...]:
    layout = model.get("fashioniq_layout")
    if layout is None:
        return ()
    return paths_for_layout(root, layout)


def checkpoint_availability(model: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.runtime import runtime_blockers

    config = resolve_config()
    path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    downloaded = checkpoint_is_present(path, checkpoint)
    required_files = checkpoint.get("required_files")
    if checkpoint.get("bundle_members"):
        required_files = [member["filename"] for member in checkpoint["bundle_members"]]
    local_sha = sha256_file(path, required_files) if downloaded else None
    official_sha = checkpoint.get("expected_sha256")
    official_sha_match = local_sha == official_sha if downloaded and official_sha else None
    mapping_verified = checkpoint["checkpoint_mapping_status"] == "VERIFIED_METADATA"
    source_synced = bool(model.get("source_dir")) and (config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]).is_dir()
    command_status = model.get("command_status")
    adapter_command_ready = command_status in {"COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"}
    if command_status is None:
        from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
        adapter = ADAPTERS.get(model["model_id"])
        adapter_command_ready = adapter is not None and getattr(adapter, "command", None) is not OfficialScriptAdapter.command
    reasons = runtime_blockers(model, checkpoint, config, model.get("native_protocol"))
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "filename": checkpoint["filename"],
        "source_metadata": checkpoint["status"] != "BLOCKED",
        "source_root": str(config.WORKBENCH_THIRD_PARTY_ROOT),
        "checkpoint_root": str(config.WORKBENCH_CHECKPOINT_ROOT),
        "source_synced_locally": source_synced,
        "automatic_download_available": checkpoint.get("download_url") is not None,
        "checkpoint_downloaded": downloaded,
        "local_sha256": local_sha,
        "official_sha256_known": official_sha is not None,
        "official_sha256_match": official_sha_match,
        "mapping_verified": mapping_verified,
        "adapter_command_ready": adapter_command_ready,
        "runtime_verified": model.get("reproduction_status") == "VERIFIED",
        "runnable": not reasons,
        "checkpoint_path": str(path),
        "blocking_reasons": reasons,
    }
