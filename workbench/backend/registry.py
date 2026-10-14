from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
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
    mapping_statuses = {"VERIFIED_METADATA", "UNVERIFIED", "ARTIFACT_ASSOCIATED_UNVALIDATED"}
    #: A variant's ``status`` records where its bytes come from; ``checkpoint_mapping_status``
    #: records whether the upstream repository establishes WHICH benchmark variant it is. They are
    #: separate facts about the same asset, so they have separate vocabularies: a mapping can be
    #: verified while the file has no source at all, and an unresolved URL says nothing about the
    #: mapping. Letting one value be legal in both places previously hid an entry that had the two
    #: fields transposed, which made a runnable model unresolvable.
    source_statuses = {"URL_VERIFIED", "NO_CLEAN_CHECKPOINT", "URL_UNRESOLVED", "NO_SOURCE",
                       "CHECKPOINT_MAPPING_UNVERIFIED"}
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
        link = model.get("canonical_dataset_link")
        if link is not None:
            link_path = PurePosixPath(link)
            if link_path.is_absolute() or ".." in link_path.parts or len(link_path.parts) != 1:
                raise ValueError(
                    f"canonical_dataset_link must be a single in-checkout name for {model['model_id']}")
        extensions = model.get("fashioniq_image_extensions")
        if extensions is not None:
            if not isinstance(extensions, list) or not extensions:
                raise ValueError(f"fashioniq_image_extensions must be a non-empty list for {model['model_id']}")
            for extension in extensions:
                if not isinstance(extension, str) or not extension.startswith(".") or len(extension) < 2:
                    raise ValueError(f"invalid image extension {extension!r} for {model['model_id']}")
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
            # The two fields must agree on whether the mapping is established. Only one source
            # state means "we do not know which benchmark variant this is"; every other state
            # records a fact about obtainability, which is independent of the mapping. This is what
            # catches the transposition: a mapping field claiming verification while the source
            # field says the mapping is unresolved. A variant that records no source state is not
            # making a claim about the mapping, so there is nothing to contradict.
            source_state = checkpoint.get("status")
            if source_state is not None:
                if source_state not in source_statuses:
                    raise ValueError(f"unknown checkpoint source state {source_state!r} for "
                                     f"{model['model_id']}")
                mapping_unverified = source_state == "CHECKPOINT_MAPPING_UNVERIFIED"
                if mapping_unverified != (checkpoint["checkpoint_mapping_status"] == "UNVERIFIED"):
                    raise ValueError(
                        f"checkpoint source state {source_state!r} and mapping state "
                        f"{checkpoint['checkpoint_mapping_status']!r} disagree for {model['model_id']}")
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
        if asset["destination_scope"] not in {"fashioniq_root", "model_source", "user_cache"}:
            raise ValueError(f"unknown auxiliary destination scope: {asset['destination_scope']}")
        if asset["destination_scope"] == "user_cache" and asset.get("cache_root_env_var") not in {"XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME"}:
            raise ValueError(f"unknown auxiliary cache root for {asset['model_id']}: {asset.get('cache_root_env_var')}")
        if available:
            if asset["source_type"] == "author_huggingface":
                valid_src = bool(asset["source_repo"]) and bool(asset["source_revision"]) and bool(asset["source_path"])
            elif asset["source_type"] == "direct_url":
                valid_src = bool(asset["source_repo"])
            else:
                valid_src = False
            if not valid_src:
                raise ValueError(f"auxiliary source mismatch for {asset['model_id']}: {asset['destination_path']}")
        else:
            if asset["source_type"] != "none":
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
    if asset["destination_scope"] == "user_cache":
        # Some consumers hard-code their cache directory instead of honouring an environment
        # variable, and the registry has to match the CONSUMER, not a convention. OpenAI's CLIP does
        # exactly this: `clip.load` uses `download_root or os.path.expanduser("~/.cache/clip")` and
        # never reads XDG_CACHE_HOME, so staging RN50x4 under an operator's XDG_CACHE_HOME put the
        # weights where the evaluator would not look and left it downloading them itself.
        fixed = asset.get("cache_root_fixed")
        if fixed:
            return Path(fixed).expanduser() / asset["destination_path"]
        env_var = asset["cache_root_env_var"]
        root = Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        if env_var in {"TORCH_HOME", "HF_HOME"} and not os.environ.get(env_var):
            root /= "torch" if env_var == "TORCH_HOME" else "huggingface"
        return root / asset["destination_path"]
    if asset["destination_scope"] == "fashioniq_root":
        return fashioniq_root / asset["destination_path"]
    if asset["destination_scope"] == "model_source":
        return third_party_root / model_by_id(asset["model_id"])["source_dir"] / asset["destination_path"]
    raise ValueError(f"unknown auxiliary destination scope: {asset['destination_scope']}")


def auxiliary_source_url(asset: dict[str, Any]) -> str | None:
    if not asset["automatic_download_available"]:
        return None
    if asset.get("source_type") == "direct_url":
        return asset["source_repo"]
    if asset.get("source_type") == "author_huggingface":
        return f"{asset['source_repo']}/resolve/{asset['source_revision']}/{asset['source_path']}"
    return None


@dataclass(frozen=True)
class AuxiliaryValidationResult:
    ok: bool
    status: str
    issues: list[str]

    def __iter__(self):
        return iter((self.ok, self.issues))


def validate_auxiliary_asset(
    asset: dict[str, Any] | Any,
    path: Path | None = None,
    *,
    fashioniq_root: Path | None = None,
    split_ids: set[str] | None = None,
) -> AuxiliaryValidationResult:
    if isinstance(asset, dict):
        rec = asset
    else:
        rec = {
            k: getattr(asset, k)
            for k in (
                "model_id", "family", "destination", "required", "automatic_download_available",
                "source_url", "source_revision", "expected_sha256", "provenance_status",
                "expected_sha256_prefix", "expected_size_bytes", "content_contract",
            )
            if hasattr(asset, k)
        }
    if path is None:
        target = rec.get("destination") or rec.get("path")
        path = Path(target) if target else None
    if path is None:
        return AuxiliaryValidationResult(False, "MISSING", ["no path specified for asset validation"])
    path = Path(path)

    if not path.is_file():
        return AuxiliaryValidationResult(False, "MISSING", [f"MISSING {path}"])

    try:
        size = path.stat().st_size
    except OSError as err:
        return AuxiliaryValidationResult(False, "MISSING", [f"MISSING {path}: {err}"])

    if size == 0:
        return AuxiliaryValidationResult(False, "INVALID_EMPTY_FILE", [f"INVALID EMPTY FILE {path}"])

    issues: list[str] = []
    status = "VALID"

    expected_size = rec.get("expected_size_bytes")
    if expected_size is not None and size != expected_size:
        issues.append(f"{path}: {size} bytes (expected {expected_size} bytes)")

    expected_sha = rec.get("expected_sha256")
    expected_prefix = rec.get("expected_sha256_prefix")
    digest = None
    if expected_sha is not None or expected_prefix is not None:
        try:
            digest = sha256_file(path)
        except Exception as err:
            issues.append(f"checksum computation error for {path}: {err}")
        if digest is not None:
            if expected_sha is not None and digest != expected_sha:
                issues.append(f"{path}: {size} bytes sha256={digest} OFFICIAL SHA256 MISMATCH")
                status = "CHECKSUM_MISMATCH"
            elif expected_prefix is not None and not digest.startswith(expected_prefix):
                issues.append(f"{path}: {size} bytes sha256={digest} OFFICIAL SHA256 PREFIX MISMATCH")
                status = "CHECKSUM_MISMATCH"

    contract = rec.get("content_contract")
    is_json = path.suffix == ".json" or contract is not None
    if is_json:
        data = None
        try:
            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except Exception as err:
            issues.append(f"INVALID JSON {path}: {err}")
            if status == "VALID":
                status = "INVALID_JSON"

        if data is not None and contract:
            if not isinstance(data, dict):
                issues.append(f"SCHEMA MISMATCH {path}: expected JSON object mapping, got {type(data).__name__}")
                if status == "VALID":
                    status = "SCHEMA_MISMATCH"
            else:
                exp_count = contract.get("expected_record_count")
                if exp_count is not None and len(data) != exp_count:
                    issues.append(f"RECORD COUNT MISMATCH {path}: expected {exp_count}, got {len(data)}")
                    if status == "VALID":
                        status = "RECORD_COUNT_MISMATCH"

                req_fields = contract.get("record_key_fields")
                if req_fields:
                    for k, v in data.items():
                        if not isinstance(v, dict):
                            issues.append(f"SCHEMA MISMATCH {path}: record for {k} is not a dict")
                            status = "SCHEMA_MISMATCH"
                            break
                        missing_fields = set(req_fields) - set(v.keys())
                        if missing_fields:
                            issues.append(f"SCHEMA MISMATCH {path}: record {k} missing required fields: {sorted(missing_fields)}")
                            if status == "VALID":
                                status = "SCHEMA_MISMATCH"
                            break

                id_source = contract.get("id_source")
                check_ids = split_ids
                if check_ids is None and id_source:
                    split_file = fashioniq_root / id_source if fashioniq_root else None
                    try:
                        if split_file is None:
                            raise OSError("FashionIQ root was not provided")
                        split_data = json.loads(split_file.read_text(encoding="utf-8"))
                        if not isinstance(split_data, list):
                            raise ValueError("expected JSON list of split ids")
                        check_ids = set(split_data)
                    except (OSError, json.JSONDecodeError, ValueError) as err:
                        issues.append(f"SPLIT ID SOURCE INVALID {path}: cannot read declared split {split_file or id_source}: {err}")
                if check_ids is not None:
                    keys = set(data.keys())
                    unknown_ids = keys - check_ids
                    missing_ids = check_ids - keys
                    if unknown_ids or missing_ids:
                        issues.append(f"SPLIT ID MISMATCH {path}: {len(unknown_ids)} keys not in declared split, {len(missing_ids)} split ids missing")
                        if status == "VALID":
                            status = "SPLIT_ID_MISMATCH"

    provenance = rec.get("provenance_status")
    if provenance != "VERIFIED":
        status = "OPERATOR_SUPPLIED_UNVERIFIED"
        issues.append(f"OPERATOR_SUPPLIED_UNVERIFIED {path}: provenance status is {provenance}")
        return AuxiliaryValidationResult(False, status, issues)

    if issues:
        return AuxiliaryValidationResult(False, status if status != "VALID" else "INVALID", issues)

    return AuxiliaryValidationResult(True, "VERIFIED", [])

def auxiliary_model_ids() -> set[str]:
    return {asset["model_id"] for asset in load_auxiliary_assets()}

def caption_alias_evidence(asset: dict[str, Any], path: Path, config: Any) -> dict[str, Any] | None:
    """Describe a staged caption alias, or None when the file there is not one.

    The download manifest is the evidence of what the workbench itself wrote; a file at an
    unverified asset's destination counts as a deliberate alias only when the manifest records it as
    one AND the bytes still match what was staged. Anything else is someone else's file and stays a
    blocker, so an operator's own COT captions are never silently reclassified as a variant.
    """
    alias = asset.get("caption_alias_from")
    if not alias:
        return None
    manifest = Path(getattr(config, "WORKBENCH_ARTIFACTS_ROOT", "")) / "auxiliary" / "download_manifest.json"
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("destination") != str(path):
            continue
        if entry.get("staged_as") != "SOURCE_COMPATIBLE_CAPTION_ALIAS":
            return None
        recorded = entry.get("downloaded_sha256")
        try:
            actual = sha256_file(path)
        except OSError:
            return None
        if not recorded or actual != recorded:
            return None
        # The manifest only proves the workbench wrote these bytes; it cannot prove they are the
        # published stand-in, because a manifest is a local file and a run could have written one.
        # The registry declares the published digest, so requiring the staged bytes to equal it ties
        # the file on disk to the published source rather than to a claim about it. Without this a
        # hand-written manifest could relabel arbitrary bytes as a variant.
        expected = alias.get("expected_sha256")
        if not expected or actual != expected:
            return None
        return {
            "staged_as": entry["staged_as"],
            "alias_source_path": entry.get("alias_source_path") or alias.get("source_path"),
            "source_revision": entry.get("source_revision") or alias.get("source_revision"),
            "staged_sha256": actual,
            "published_sha256": expected,
            "equivalence": "SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN",
        }
    return None


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
    # Any model whose pinned evaluator hard-codes a dataset root declares it here; the same
    # declaration is what authorizes the link as a runtime artifact rather than a source edit.
    for model_with_link in load_registry()["models"]:
        if model_with_link["model_id"] != model_id:
            continue
        link = model_with_link.get("canonical_dataset_link")
        if link:
            artifacts.append({
                "kind": "canonical_dataset_link",
                "path": link,
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
