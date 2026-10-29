from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable

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
            if link_path.is_absolute() or ".." in link_path.parts:
                raise ValueError(
                    f"canonical_dataset_link must be a relative in-checkout path for "
                    f"{model['model_id']}")
            if len(link_path.parts) > 2:
                raise ValueError(
                    f"canonical_dataset_link must name a path at most one directory deep for "
                    f"{model['model_id']}: {link!r}")
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
            # One direction only: a source state that says the mapping is unresolved must not
            # coexist with a mapping field claiming verification (the transposition). The
            # converse is deliberately NOT enforced: an obtainability fact such as
            # URL_UNRESOLVED says nothing about which benchmark variant the asset is, so an
            # unresolved source with an UNVERIFIED mapping is honest, not contradictory.
            source_state = checkpoint.get("status")
            if source_state is not None:
                if source_state not in source_statuses:
                    raise ValueError(f"unknown checkpoint source state {source_state!r} for "
                                     f"{model['model_id']}")
                # One direction only: a source state that says the mapping is unresolved must not
                # coexist with a mapping field claiming verification (the transposition). The
                # converse is deliberately NOT enforced: an obtainability fact such as
                # URL_UNRESOLVED says nothing about which benchmark variant the asset is, so an
                # unresolved source with an UNVERIFIED mapping is honest, not contradictory.
                if source_state == "CHECKPOINT_MAPPING_UNVERIFIED" and \
                        checkpoint["checkpoint_mapping_status"] != "UNVERIFIED":
                    raise ValueError(
                        f"checkpoint source state {source_state!r} and mapping state "
                        f"{checkpoint['checkpoint_mapping_status']!r} disagree for {model['model_id']}")
    return registry


def expected_asset_size(asset: dict[str, Any]) -> int | None:
    """The expected byte size of an auxiliary asset, honoring documented test fixture overrides."""
    hub_files = asset.get("hub_files") or []
    dest = hub_files[0] if asset.get("hub_repo_id") and hub_files else asset.get("destination_path") or asset.get("cache_path_suffix") or ""
    if dest:
        name_key = Path(dest).name.upper().replace(".", "_")
        env_key = f"WORKBENCH_TEST_ASSET_SIZE_{name_key}"
        try:
            from workbench.backend.model_settings import setting as _setting
            val = _setting(env_key)
        except Exception:
            val = os.environ.get(env_key)
        if val is not None and val.strip().isdigit():
            return int(val.strip())
    return asset.get("expected_size_bytes")


def load_auxiliary_assets(path: Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        custom = os.environ.get("WORKBENCH_AUXILIARY_REGISTRY_PATH")
        path = Path(custom).resolve() if custom else AUXILIARY_ASSET_REGISTRY_PATH
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
        elif asset["required_for"] == "official_evaluation" and asset["source_type"] != "unverified_external":
            if asset["source_type"] != "author_huggingface" or not bool(asset["caption_alias_from"]):
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
                if {"name", "kind", "scope", "paths"} - artifact.keys() or artifact["scope"] not in {"fashioniq_root", "evaluator_cwd"}:
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
            # A directory-shaped asset (a hub repo cache) has no file to size or hash, and a
            # `location_type` typo would silently fall through to the file branch and then report a
            # cache directory as `INVALID_FILE_TYPE`. Validated here so the declaration is either
            # correct or refused.
            if asset.get("location_type") not in {None, "file", "directory"}:
                raise ValueError(f"invalid external asset location_type for {contract['preparation_id']}")
            if asset.get("location_type") == "directory" and (asset.get("expected_sha256") or asset.get("expected_sha256_prefix")):
                raise ValueError(
                    f"a directory external asset cannot carry a file digest for {contract['preparation_id']}")
            cache_root_var = asset.get("cache_root_env_var")
            if cache_root_var is not None and not cache_root_var.isidentifier():
                raise ValueError(f"invalid external asset cache root variable for {contract['preparation_id']}")
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


def preparation_paths(contract: dict[str, Any], root: Path, group: str,
                      evaluator_root: Path | None = None) -> tuple[Path, ...]:
    """Resolve an artifact's paths against the root its DECLARED scope names.

    `fashioniq_root` artifacts live in the dataset root, which is what `root` already means. An
    `evaluator_cwd` artifact -- a generated cache the native code opens relative to the evaluator's
    own working directory -- must resolve against the model checkout instead. Resolving both against
    the dataset root made readiness inspect a path the run never opens, so a missing native input
    passed and the failure surfaced only inside a long evaluation.
    """
    resolved: list[Path] = []
    for artifact in contract[group]:
        base = evaluator_root if artifact["scope"] == "evaluator_cwd" else root
        if artifact["scope"] == "evaluator_cwd" and base is None:
            # Without the checkout there is nothing to resolve against; the caller that cannot supply
            # it must not receive a path that looks checked.
            continue
        resolved.extend(base / path for path in artifact["paths"])
    return tuple(resolved)


def required_runtime_assets(model: dict[str, Any]) -> list[dict[str, Any]]:
    contract = preparation_contract_for_model(model)
    if contract is None:
        return []
    return [asset for asset in contract["external_assets"] if asset["required_for_evaluation"] and not asset["optional"]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def evaluator_root(model: dict[str, Any]) -> Path | None:
    """The working directory the pinned evaluator runs in, or None when it has no checkout.

    This is the root an external asset declared `evaluator_cwd` must resolve against, and it is the
    SAME directory the runner chdirs into (`evaluate_models.source_cwd`, and the replay wrappers'
    own `os.chdir`). Deriving it from the configured third-party root rather than from the repository
    root is what keeps a checker, a preparation step and an evaluator pointed at one set of bytes
    when the operator relocates that root (`WORKBENCH_THIRD_PARTY_ROOT`) or mounts it differently in
    a container.
    """
    from workbench.backend.operator_config import resolve_config

    source_dir = model.get("source_dir")
    if not source_dir:
        return None
    return resolve_config().WORKBENCH_THIRD_PARTY_ROOT / source_dir

def asset_location_key(asset: dict[str, Any]) -> tuple[str | None, str] | None:
    """The (environment variable, relative path) an asset's local location is keyed by.

    One identity for "where this asset lives", so an admission decision, a preparation step and the
    evaluator's own loader cannot disagree about which file they mean. `None` when the asset declares
    neither a location variable nor a path -- i.e. when its location is derived some other way (the
    tokenizer cache entry).
    """
    env_var = asset.get("location_env_var")
    location = asset.get("expected_local_location")
    if not env_var and not location:
        return None
    return (env_var, str(location)) if location else (env_var, "")

def evaluator_relative_asset_paths(model: dict[str, Any]) -> dict[str, str]:
    """`{asset name: path the evaluator loader opens}` for assets its OWN code opens by a literal.

    An asset whose only declaration is a repository-relative path cannot be relocated by an
    environment variable without editing pinned source, so the asset's real identity is the checkout
    it lives in: `<evaluator root>/<relative path>`. The declaration stays the
    single registry fact; this is how it is READ. Used both to resolve the asset and to tell the
    source authority which files inside the checkout legitimate preparation may create.
    """
    contract = preparation_contract_for_model(model)
    resolved: dict[str, str] = {}
    for asset in (contract or {}).get("external_assets", []):
        location = asset.get("expected_local_location")
        if not location or asset.get("location_env_var"):
            continue
        resolved[asset["name"]] = str(location)
    return resolved

def external_asset_statuses(model: dict[str, Any]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    checkout_relative = evaluator_relative_asset_paths(model)
    root = evaluator_root(model)
    for asset in required_runtime_assets(model):
        # `location_env_var` is optional in the contract schema and is legitimately absent for an
        # asset that has no override variable (the tokenizer cache entry). Reading it with a default
        # only covered a MISSING key; an explicit `null` reached `os.environ.get(None)`, which raises
        # `TypeError: str expected, not NoneType` instead of reporting the asset's state.
        env_var = asset.get("location_env_var")
        from workbench.backend.model_settings import setting as _setting

        configured = _setting(env_var) if env_var else None
        # A cache-shaped asset resolves BELOW the cache root, whichever way the root was named.
        # `location_env_var` for the tokenizer is `HF_HOME` itself, and the loader turns the repo id
        # into `hub/models--<repo>/` under that root, so reading the configured value as the asset
        # directory pointed the checker at the cache root and left `refs`/`snapshots` unexamined.
        cache_shaped = bool(asset.get("cache_path_suffix"))
        location = None if cache_shaped else (configured if configured else asset.get("expected_local_location"))
        if location is None and cache_shaped:
            # `cache_root_env_var` names the variable that RELOCATES this cache, and its default is the
            # one the consumer itself uses. The tokenizer is read by Hugging Face's loader, whose root
            # is `HF_HOME` (default `~/.cache/huggingface`), NOT `~/.cache/torch` -- resolving it under
            # the torch root made a correctly populated cache report as missing.
            #
            # A HUB asset resolves one level below any cache root: the loader turns the repo id into
            # `hub/models--<repo>/` itself, so a configured `HF_HOME` (or `location_env_var: HF_HOME`)
            # is the CACHE ROOT, not the asset directory. Returning the bare root for a
            # `cache_path_suffix` ending in a repo directory made the checker inspect a directory the
            suffix = str(asset["cache_path_suffix"])
            root_var = asset.get("cache_root_env_var") or "TORCH_HOME"
            # Read through the same settings reader as `location_env_var` above: an operator who
            # advertised `HF_HOME` in the local workbench env file must get the same root here as
            # everywhere else, or the checker and the loader disagree about where the cache is.
            configured_root = _setting(root_var)
            if configured_root and configured_root.strip():
                root = Path(configured_root.strip()).expanduser()
            else:
                root = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
                root = root / ("huggingface" if root_var == "HF_HOME" else "torch")
            hub_prefix, _, repo_segment = suffix.partition("models--")
            if hub_prefix == "hub/" and repo_segment:
                location = str(hf_hub_repo_dir(repo_segment.replace("--", "/"), root))
            else:
                location = str(root / suffix)
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
            relative = checkout_relative.get(asset["name"])
            # A path the model's own pinned code opens from its working directory belongs to the
            # CHECKOUT. Joining it to the repository root put TME's `weight/eva_vit_g.pth` one
            # directory above the file the loader reads, so a correctly staged asset was reported
            # missing and a wrongly staged one would have been accepted by the check the evaluator
            # never performs.
            path = (root / relative) if (relative and root is not None) else (_repo_root() / path)
        path = path.resolve()
        if asset.get("location_type") == "directory":
            # A directory-shaped asset (a hub repo cache) has no single file to size or hash. What
            # makes it usable is the resolution its CONSUMER performs: a revision through the cache's
            # refs, and the payload below that revision. Reporting "the directory exists and holds
            # something" was wrong in both directions -- an orphan snapshot no ref names read as
            # present, and a repo holding only a vocabulary read as complete while the loader that
            # opens the model weights would still fail.
            missing: list[str] = []
            snapshot = None
            if is_hub_cache_asset(asset):
                identity = hub_cache_identity(asset, path)
                snapshot = identity["snapshot"]
                missing = hub_cache_missing_files(asset, path)
            elif path.is_dir():
                snapshot = hub_cache_snapshot(path)
            if not path.exists():
                status = "CONFIGURED_MISSING"
            elif not path.is_dir():
                status = "INVALID_FILE_TYPE"
            elif not any(path.iterdir()):
                status = "EMPTY_PLACEHOLDER"
            elif snapshot is None:
                # Populated, but nothing in it resolves: the R22-03 shape, where a snapshot with a
                # vocabulary and no active ref passed a directory check and then failed a repo-id
                # lookup. Named separately from "missing" because the remedy is different.
                status = "UNRESOLVED_REF"
            elif missing:
                status = "INCOMPLETE_CACHE"
            else:
                status = "EXISTS_UNVERIFIED_CHECKSUM"
            statuses.append({"name": asset["name"], "status": status, "path": str(path),
                             "expected_sha256": None, "expected_sha256_prefix": None,
                             "actual_sha256": None,
                             "cache_snapshot": str(snapshot) if snapshot is not None else None,
                             "cache_missing_files": missing})
            continue
        status = "CONFIGURED_MISSING"
        actual_sha = None
        exp_size = expected_asset_size(asset)
        if path.exists():
            if not path.is_file():
                status = "INVALID_FILE_TYPE"
            elif path.stat().st_size == 0:
                status = "EMPTY_PLACEHOLDER"
            elif exp_size is not None and path.stat().st_size != exp_size:
                # A declared size the local file does not match is a fact about the FILE, and saying
                # "file missing" for a file that is present and wrong sends an operator looking for
                # something to download when what they have is something to replace.
                status = "SIZE_MISMATCH"
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
                         "actual_sha256": actual_sha, "expected_size_bytes": exp_size,
                         "actual_size_bytes": (path.stat().st_size if path.is_file() else None)})
    return statuses


#: One wording per unadmitted state, shared by every consumer so readiness, doctor, the API and a
#: replay wrapper cannot describe the same file differently.
_ASSET_BLOCKER_MESSAGES = {
    "UNRESOLVED_PROVENANCE": "provenance unresolved",
    "UNCONFIGURED_LOCATION": "local location unconfigured",
    "CONFIGURED_MISSING": "file missing",
    "INVALID_FILE_TYPE": "expected regular file, found non-file",
    "EMPTY_PLACEHOLDER": "empty placeholder file",
    "CHECKSUM_MISMATCH": "checksum mismatch",
    "SIZE_MISMATCH": "size mismatch against the declared byte count",
    "SUSPECT_PAYLOAD": "content is not a plausible payload (uniform bytes across the whole file)",
    "EXISTS_UNVERIFIED_CHECKSUM": "file exists but no verified checksum is available",
}

#: Statuses no operator action can fix, and which therefore block every consumer.
UNADMITTED_ASSET_STATUSES = (
    "UNRESOLVED_PROVENANCE",
    "UNCONFIGURED_LOCATION",
    "CONFIGURED_MISSING",
    "INVALID_FILE_TYPE",
    "EMPTY_PLACEHOLDER",
    "CHECKSUM_MISMATCH",
    "SIZE_MISMATCH",
)

def _has_substantive_payload(path: Path, asset: dict[str, Any]) -> bool:
    """Whether a file declared without an author digest actually contains a plausible payload.

    The size check catches a truncated download, but NOT a file that is exactly the declared length
    yet contains nothing: a sparse or zero-filled allocation has the right size and no weights, and
    admitting it would let a model be launched against a file that is structurally a container of
    zeros. When no author digest exists, this is the only content evidence the workbench can obtain
    without loading the asset, so a file whose entire length is a single repeated byte is refused.
    The check reads at most a bounded prefix and a bounded suffix, so it costs nothing on a 2 GB
    weight file.
    """
    try:
        size = path.stat().st_size
        if size == 0:
            return False
        block = 1 << 20
        with path.open("rb") as handle:
            head = handle.read(min(block, size))
            if not head:
                return False
            if any(byte != head[0] for byte in head):
                return True
            if size > block:
                handle.seek(max(0, size - block))
                tail = handle.read(block)
                if any(byte != head[0] for byte in tail):
                    return True
        return False
    except OSError:
        return False

def asset_admission_status(asset: dict[str, Any], status: dict[str, Any]) -> str:
    """The one admission decision every consumer of an external asset makes.

    Three states, because there are three genuinely different situations:

    * `VERIFIED_LOCAL_ASSET` -- an author-published digest was declared and the local bytes match it.
    * `LOCALLY_RECORDED_ASSET` -- the asset is author-linked and complete (present, non-empty, right
      size) but the author publishes no digest to compare against. The workbench records the local
      digest and says plainly it is not an author verification. Refusing this state made the shared
      preflight and the TME wrapper contradict each other on the same bytes: the wrapper admitted
      them with a warning and its own evidence, while `external_asset_blockers` and
      `preflight.model_requirements` rejected them, so a correctly staged asset blocked readiness.
      The gap this closes is a policy disagreement, not a checksum gap: no published value exists to
      verify against, and inventing one would be worse than admitting the gap.
    * anything else -- reject. Missing, malformed, mismatched, placeholder or
      unresolved-provenance assets are never admitted, by any caller.

    A `cache`-derived asset (a tokenizer cache whose readiness the loader itself must establish, not
    its parent directory's contents) is NOT admitted here; it stays an explicit rejection so its own
    consumer decides, with a real load.
    """
    state = status["status"]
    if state == "VERIFIED_LOCAL_ASSET":
        return state
    if (state == "EXISTS_UNVERIFIED_CHECKSUM" and not is_cache_asset(asset)
            and not is_hub_cache_asset(asset) and asset.get("location_type") != "directory"):
        path = Path(status.get("path") or "")
        try:
            size = path.stat().st_size
        except OSError:
            return "CONFIGURED_MISSING"
        if not path.is_file():
            return "INVALID_FILE_TYPE"
        if size == 0:
            return "EMPTY_PLACEHOLDER"
        expected_size = expected_asset_size(asset)
        if expected_size is not None and size != expected_size:
            return "SIZE_MISMATCH"
        if not _has_substantive_payload(path, asset):
            return "SUSPECT_PAYLOAD"
        return "LOCALLY_RECORDED_ASSET"
    return state

def admitted_asset_statuses(status: str) -> bool:
    return status in {"VERIFIED_LOCAL_ASSET", "LOCALLY_RECORDED_ASSET"}

def is_cache_asset(asset: dict[str, Any]) -> bool:
    """True for an asset whose local location is a CACHE its consumer must load.

    Shape, not location: `cache_path_suffix` is also used to compose a default FILENAME for a
    plain file asset (DCNet's ResNet-50 weights), and treating those as caches made the blocker text
    claim a `.pth` file had "no snapshots/ revision directory". The declaration that marks a
    directory-shaped cache is `location_type: directory`.
    """
    return asset.get("location_type") == "directory" and bool(asset.get("cache_path_suffix"))

def asset_admission_detail(asset: dict[str, Any], status: dict[str, Any]) -> str:
    """How the admission decision was reached, in the words the operator sees."""
    if is_cache_asset(asset):
        reason = cache_asset_blocker(asset, status)
        if reason:
            return f"cache unusable: {reason}"
        return "cache layout verified; loaded by its consumer, not author-checksum-verified"
    admission = asset_admission_status(asset, status)
    if admission == "VERIFIED_LOCAL_ASSET":
        return "verified against the published digest"
    if admission == "LOCALLY_RECORDED_ASSET":
        return ("author digest unavailable; local digest recorded, not verified against a "
                "published value")
    return _ASSET_BLOCKER_MESSAGES.get(admission, f"state {admission}")

def _asset_admitted(asset: dict[str, Any], status: dict[str, Any]) -> bool:
    if is_cache_asset(asset):
        # A cache-shaped location is judged by whether its consumer can actually load it, not by
        # whether the directory has something in it.
        return cache_asset_blocker(asset, status) is None
    return admitted_asset_statuses(asset_admission_status(asset, status))

def external_asset_blockers(model: dict[str, Any], contract_asset: Callable[[str], dict[str, Any] | None] | None = None) -> list[str]:
    """Every required evaluation asset this workbench will not admit.

    `contract_asset` lets a caller that already holds the declaration map (the TME wrapper does) use
    its own view of the same facts; the default reads the contract itself.
    """
    lookup = contract_asset or (lambda name: _asset_declaration(model, name))
    return [
        f"external runtime asset {asset_admission_detail(lookup(item['name']) or {}, item)}: "
        f"{item['name']}" + (f" ({item['path']})" if item["path"] else "")
        for item in external_asset_statuses(model)
        if not _asset_admitted(lookup(item["name"]) or {}, item)
    ]

def _asset_declaration(model: dict[str, Any], name: str) -> dict[str, Any] | None:
    contract = preparation_contract_for_model(model)
    for asset in (contract or {}).get("external_assets", []):
        if asset["name"] == name:
            return asset
    return None

def cache_asset_blocker(asset: dict[str, Any], status: dict[str, Any]) -> str | None:
    """Why a cache-shaped asset is unusable, or None when it can actually be loaded.

    A populated directory is not a usable Hugging Face cache. The pinned LAVIS loader calls
    `BertTokenizer.from_pretrained("bert-base-uncased")` and then builds the Qformer with
    `BertLMHeadModel.from_pretrained("bert-base-uncased", config=...)`, and BOTH resolve the repo id
    through the cache. So the shared admission asks the loader's own question in two steps: the
    cache must resolve a REPOSITORY the way the loader resolves it -- a `refs/<revision>` entry
    naming an existing snapshot that holds the declared payload -- and, when `transformers` is
    importable in the checking interpreter, the tokenizer must actually load OFFLINE from that cache
    root.

    Two corrections to the earlier check, both reproduced against the real loader:

    * a repository cache is looked up by REPO ID, so a snapshot no ref names cannot be reached
      however complete it looks (R22-03), and
    * a directory holding only a snapshot cannot be reached at all: the loader reads
      `<root>/hub/models--<owner>--<name>/`, so a check that accepts a bare `snapshots/` directory
      certifies a layout the loader never produces.

    Neither step downloads anything: a readiness report never reaches the network.
    """
    if not status.get("path") or not (asset.get("cache_path_suffix") or is_hub_cache_asset(asset)):
        return None
    path = Path(status["path"])

    if is_hub_cache_asset(asset):
        repo = path
        refs = repo / "refs"
        if not repo.is_dir():
            return f"no hub cache directory at {repo}"
        if not refs.is_dir() or not any(refs.iterdir()):
            return (f"{repo} holds no active revision under refs/ (a snapshot no ref names cannot "
                    f"be resolved by repo id)")
        ref_main = refs / "main"
        if not ref_main.is_file():
            return f"ref main is missing under {refs} (unqualified native loader lookup demands refs/main)"
        snapshot = hub_cache_snapshot(repo)
        if snapshot is None:
            return f"ref main names a snapshot that is not present under {repo / 'snapshots'}"
        expected_rev = hub_cache_declared_revision(hub_cache_repo_id(asset))
        if expected_rev and snapshot.name != expected_rev:
            return f"ref main points to snapshot {snapshot.name}, expected pinned revision {expected_rev}"
        # The repo directory THIS CALLER resolved, so the payload question is about the cache being
        # checked rather than about whatever `HF_HOME` happens to hold.
        missing = hub_cache_missing_files(asset, repo)
        if missing:
            return (f"{snapshot} is missing the declared payload {', '.join(missing)} "
                    f"(a tokenizer without the model weights fails in the Qformer loader)")
        integrity_problem = hub_cache_integrity_problem(
            asset, repo, snapshot, hub_cache_declared_files(hub_cache_repo_id(asset)))
        if integrity_problem:
            return integrity_problem
        status["cache_snapshot"] = str(snapshot)
    else:
        snapshots = path / "snapshots"
        if not snapshots.is_dir():
            return "cache has no `snapshots/` revision directory"
        candidates = [
            child for child in sorted(snapshots.iterdir())
            if child.is_dir() and any((child / name).is_file()
                                      for name in ("vocab.txt", "tokenizer.json", "spiece.model"))
        ]
        if not candidates:
            return "cache holds no revision snapshot with tokenizer files (vocab.txt/tokenizer.json)"
        snapshot = candidates[-1]

    try:
        import transformers  # noqa: F401
    except ImportError:
        # The model environment owns loading this cache; this interpreter can only certify layout.
        status["admission_note"] = ("cache layout verified; load-test belongs to the model "
                                    "environment (not author-checksum-verified)")
        return None

    # The load-test resolves the REPO ID through this cache root, offline, which is exactly what the
    # pinned constructor does. Loading a snapshot DIRECTORY instead would test a different question:
    # it passes on a cache the native repo-id lookup cannot find, which is how a tokenizer-only
    # cache came to be certified for a run that then failed in `BertLMHeadModel.from_pretrained`.
    # The load-test also runs against the cache this caller RESOLVED: `path` is the repo directory
    # for a hub asset, so its cache root is that directory's grandparent, and pointing the library
    # at the host's `HF_HOME` instead tested a directory nobody asked about.
    repo_id = hub_cache_repo_id(asset) if is_hub_cache_asset(asset) else ""
    root = (path.parent.parent if is_hub_cache_asset(asset) else path.parent.parent)
    target = repo_id if repo_id else str(snapshot)
    environment_backup = {key: os.environ.get(key)
                          for key in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        # Pass explicit cache_dir=<root>/hub and local_files_only=True so the read-only check
        # does not touch the network or default cache even when transformers was imported earlier.
        if is_hub_cache_asset(asset):
            AutoTokenizer.from_pretrained(target, cache_dir=str(root / "hub"), local_files_only=True)
        else:
            AutoTokenizer.from_pretrained(target, local_files_only=True)
    except Exception as error:  # a cache the loader cannot read is a blocker, not a warning
        return f"tokenizer does not load offline from {target} under {root}: {type(error).__name__}: {error}"
    finally:
        for key, value in environment_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return None


#: Digest cache for the availability view only. `checkpoint_availability` runs on every
#: `/api/models` request and hashes whole checkpoints (LIMN's three are 1.79 GB each), which
#: made a single request take ~85 s. The key carries size and mtime_ns, so a replaced or
#: rewritten file rehashes; the authoritative hashing behind verification, receipts and the
#: ledger is deliberately NOT cached and still reads every byte.
_AVAILABILITY_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}

def _cached_availability_digest(path: Path, required_files: list[str] | None) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _AVAILABILITY_DIGEST_CACHE.get(key)
    if cached is None:
        cached = sha256_file(path, required_files)
        _AVAILABILITY_DIGEST_CACHE[key] = cached
    return cached

#: Recorded digests from the checkpoint download manifest, keyed by resolved destination.
#: The manifest is written when the workbench itself downloads and verifies a file, so the
#: availability view can report a digest without re-reading gigabytes on every request.
#: A manifest claim is only reused while the file's size still matches the recorded size;
#: anything else falls back to hashing the bytes.
_CHECKPOINT_MANIFEST_CACHE: tuple[float, dict[str, dict[str, Any]]] | None = None

def _recorded_checkpoint_digests(config: Any = None) -> dict[str, dict[str, Any]]:
    global _CHECKPOINT_MANIFEST_CACHE
    if config is None:
        from workbench.backend.operator_config import resolve_config
        config = resolve_config()
    manifest_path = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "download_manifest.json"
    try:
        mtime = manifest_path.stat().st_mtime
    except OSError:
        return {}
    if _CHECKPOINT_MANIFEST_CACHE is not None and _CHECKPOINT_MANIFEST_CACHE[0] == mtime:
        return _CHECKPOINT_MANIFEST_CACHE[1]
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    recorded: dict[str, dict[str, Any]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        key = _checkpoint_manifest_key(entry.get("model_id"), entry.get("filename"))
        if key is not None:
            recorded[key] = entry
    _CHECKPOINT_MANIFEST_CACHE = (mtime, recorded)
    return recorded

def _checkpoint_manifest_key(model_id: Any, filename: Any) -> str | None:
    """The manifest identifies a downloaded checkpoint by model id and file name; it predates
    any `destination` field, so the key is built from the pair."""
    if not isinstance(model_id, str) or not isinstance(filename, str):
        return None
    return f"{model_id}\0{filename}"

def _availability_digest(config: Any = None, model_id: str = "", checkpoint: dict[str, Any] | None = None, path: Path | None = None) -> str | None:
    """Digest for the availability view: the manifest's recorded hash when the file on disk
    still has the size the manifest recorded, otherwise a real read of the bytes.

    A bundle declaration hashes a set of member files with directory framing, which cannot be
    reconstructed from per-file hashes, so those fall back to hashing (once per process).
    """
    if config is None:
        from workbench.backend.operator_config import resolve_config
        config = resolve_config()
    if path is None or checkpoint is None:
        return None
    if checkpoint.get("bundle_members"):
        return _cached_availability_digest(path, checkpoint.get("required_files"))
    entry = _recorded_checkpoint_digests(config).get(
        _checkpoint_manifest_key(model_id, checkpoint.get("filename") or path.name) or "")
    if entry is not None:
        recorded_sha = entry.get("downloaded_sha256")
        recorded_size = entry.get("size_bytes")
        recorded_ctime = entry.get("ctime_ns")
        try:
            stat_result = path.stat()
        except OSError:
            return None
        # Size AND ctime: size alone let a same-length in-place rewrite (mtime restored) produce a
        # forged `official_sha256_match: True` while the verifier refused the same file -- the
        # presence half of this defect, here in the digest half. ctime cannot be set backwards by
        # any userspace call; manifests written before this record none and fall through to hashing.
        if (recorded_sha and recorded_size == stat_result.st_size
                and recorded_ctime is not None and recorded_ctime == stat_result.st_ctime_ns):
            return recorded_sha
    return _cached_availability_digest(path, checkpoint.get("required_files"))

def checkpoint_missing_paths(path: Path, checkpoint: dict[str, Any],
                             model_id: str | None = None,
                             config: Any = None) -> list[Path]:
    if model_id is None:
        model_id = checkpoint.get("model_id") or (path.parent.name if path.parent else None)
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
            try:
                st = member_path.stat()
            except OSError:
                missing.append(member_path)
                continue
            if st.st_size == 0:
                missing.append(member_path)
            else:
                member = next((item for item in members or [] if item["filename"] == name), None)
                expected_sha = member.get("expected_sha256") if member else None
                if expected_sha:
                    digest = None
                    if model_id:
                        entry = _recorded_checkpoint_digests(config).get(
                            _checkpoint_manifest_key(model_id, name) or "")
                        if entry is not None:
                            recorded_sha = entry.get("downloaded_sha256")
                            recorded_size = entry.get("size_bytes")
                            # Size ALONE is not a witness: a same-length in-place rewrite (with
                            # mtime restored) read as present for all three members without a
                            # single byte being hashed, so a tampered bundle was reported present
                            # while the verifier refused it. ctime cannot be set backwards by any
                            # userspace call, so requiring it closes the gap; manifests written
                            # before this record no ctime and simply fall through to hashing.
                            recorded_ctime = entry.get("ctime_ns")
                            if recorded_sha and recorded_size == st.st_size and \
                                    recorded_ctime is not None and recorded_ctime == st.st_ctime_ns:
                                digest = recorded_sha
                    if digest is None:
                        digest = sha256_file(member_path)
                    if digest != expected_sha:
                        missing.append(member_path)
        return missing
    if not _regular_file(path):
        return [path]
    try:
        st = path.stat()
    except OSError:
        return [path]
    if st.st_size == 0:
        return [path]
    expected_sha = checkpoint.get("expected_sha256")
    # A container is not a weight. When the registry declares that this artifact arrives inside an
    # archive, its published bytes are the EXTRACTED member, so container-shaped bytes at the
    # destination are the wrong object even though the file exists, is regular, and is non-empty.
    # Without this, a previously mis-installed gzip/tar file satisfied every presence check and the
    # run proceeded to a loader that could not read it.
    if checkpoint.get("archive_member") and _is_compressed_container(path):
        return [path]
    # A declared size is part of the artifact's identity for a single-file artifact, not only for a
    # download in progress. A truncated extraction publishes the right NAME with the wrong BYTES, and
    # without this check it read as present. (Bundles already compare member sizes through their
    # manifest; this is the same rule applied to the single-file shape.)
    declared_size = checkpoint.get("expected_size_bytes")
    if type(declared_size) is int and declared_size > 0 and st.st_size != declared_size:
        return [path]
    if not expected_sha:
        return []
    actual_sha = _availability_digest(config, model_id or "", checkpoint, path)
    return [path] if actual_sha != expected_sha else []


def _is_compressed_container(path: Path) -> bool:
    """True for the container formats that can hide a weight, regardless of the file's name.

    ZIP is deliberately NOT on this list. `torch.save` has written its own archives with the zip
    format since PyTorch 1.6, so `PK\\x03\\x04` is the NORMAL first four bytes of a modern `.pth` or
    `.pt` state dictionary, and treating it as a container rejected a legitimate checkpoint. Verified
    against the real artifact: TME's extracted `fashioniq_0_best_model.pth` begins `504b0304` and is
    2728382104 bytes, which is the declared member size - it is a valid torch zip, not an archive.
    A zip wrapper around a weight is not a shape the author endpoints produce, and refusing one is the
    downloader's business, not this presence check's.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    if head[:2] == b"\x1f\x8b" or head[:2] == b"BZ" or head[:6] == b"\xfd7zXZ\x00":
        return True
    # A bare tar keeps "ustar" at offset 257, which is outside an 8-byte read.
    try:
        with path.open("rb") as handle:
            return handle.read(262)[257:262] == b"ustar"
    except OSError:
        return False

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

def checkpoint_is_present(path: Path, checkpoint: dict[str, Any],
                         model_id: str | None = None,
                         config: Any = None) -> bool:
    return not checkpoint_missing_paths(path, checkpoint, model_id=model_id, config=config)


def checkpoint_bundle_missing_paths(model: dict[str, Any], root: Path,
                                    config: Any = None) -> dict[str, list[Path]]:
    variants = {checkpoint["checkpoint_id"]: checkpoint for checkpoint in model["checkpoint_variants"]}
    missing: dict[str, list[Path]] = {}
    for bundle in model.get("checkpoint_bundles", []):
        paths = [path for checkpoint_id in bundle["required_checkpoint_ids"]
                 for path in checkpoint_missing_paths(
                     checkpoint_path(model["model_id"], variants[checkpoint_id], root),
                     variants[checkpoint_id],
                     model_id=model["model_id"],
                     config=config,
                 )]
        if paths:
            missing[bundle["bundle_id"]] = paths
    return missing


def auxiliary_assets_for_model(model_id: str) -> list[dict[str, Any]]:
    return [asset for asset in load_auxiliary_assets() if asset["model_id"] == model_id]


#: The canonical Hugging Face hub directory a repo id resolves to. `transformers` builds it from
#: the repo id (`models--<owner>--<name>`) and then reads `refs/<rev>` to find the active
#: `snapshots/<rev>/` directory, so a cache holding files directly at `hub/models--<repo>/` is NOT
#: a cache the loader can read. One function answers "where does a repo id live" for the checker,
#: the auxiliary downloader and the offline bootstrap step, so those three cannot disagree about
#: the directory a real loader consults.
def hf_hub_cache_root() -> Path:
    """The Hugging Face cache ROOT: `HF_HOME`, else the XDG default.

    Read through the shared settings reader, so the operator's local `workbench.env` behaves like the
    process environment -- which is what that file documents for every registry-declared name. The
    configured value is the ROOT, never the asset directory: the loader resolves a repo id to
    `<root>/hub/models--<owner>--<name>/`, so a consumer that stops at the configured value inspects
    a directory the loader never opens (R22-02).
    """
    try:
        from workbench.backend.model_settings import resolve_path_setting

        resolved = resolve_path_setting("HF_HOME")
        if resolved is not None:
            return resolved
    except Exception:
        pass
    configured = os.environ.get("HF_HOME")
    if configured and configured.strip():
        from workbench.backend.operator_config import _DEFAULT_REPO_ROOT

        p = Path(configured.strip()).expanduser()
        return p if p.is_absolute() else (_DEFAULT_REPO_ROOT / p).resolve()
    return (Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface").resolve()


def hf_hub_repo_dir(repo_id: str, root: Path | None = None) -> Path:
    return (root or hf_hub_cache_root()) / "hub" / ("models--" + repo_id.replace("/", "--"))


def hub_cache_snapshot(repo: Path, revision: str = "main") -> Path | None:
    """The snapshot a repo directory's refs point at, or None when no ref resolves to one.

    The loader finds a revision through `refs/<rev>`; a snapshot directory no ref names cannot be
    reached by a repo-id lookup however complete it looks, so it is not a candidate.
    Unqualified calls demand refs/main.
    """
    refs = repo / "refs"
    if not refs.is_dir():
        return None
    ref = refs / revision
    if not ref.is_file():
        return None
    target_sha = ref.read_text(encoding="utf-8").strip()
    snapshot = repo / "snapshots" / target_sha
    if snapshot.is_dir():
        return snapshot
    return None


def hub_cache_layout_problem(repo_id: str, root: Path | None = None, revision: str = "main",
                             *, check_transfer: bool = True) -> str | None:
    """Why a Hub cache cannot resolve, or None.

    The repair publisher may skip transfer-record comparison while replacing that record;
    revision, declared-size, and substantive-payload checks still apply. All readers keep the
    default transfer-record check.
    """
    repo = hf_hub_repo_dir(repo_id, root)
    if not repo.is_dir():
        return f"no hub cache directory at {repo}"
    refs = repo / "refs"
    if not refs.is_dir() or not any(refs.iterdir()):
        return (f"{repo} holds no active revision under refs/ "
                "(a snapshot no ref points at cannot be resolved by repo id)")
    ref = refs / revision
    if not ref.is_file():
        return f"ref {revision} is missing under {refs} (unqualified native loader lookup demands refs/{revision})"
    sha = ref.read_text(encoding="utf-8").strip()
    snapshot = repo / "snapshots" / sha
    if not snapshot.is_dir():
        return f"ref {ref.name} names a snapshot that is not present: {snapshot}"
    expected_revision = hub_cache_declared_revision(repo_id)
    if expected_revision and sha != expected_revision:
        return f"ref {ref.name} points to snapshot {sha}, expected pinned revision {expected_revision}"
    asset = {"hub_repo_id": repo_id}
    missing = hub_cache_missing_files(asset, repo)
    if missing:
        return f"{snapshot} is missing the declared payload {', '.join(missing)}"
    integrity_problem = hub_cache_integrity_problem(
        asset, repo, snapshot, hub_cache_declared_files(repo_id), check_transfer=check_transfer)
    if integrity_problem:
        return integrity_problem
    if not hub_cache_declared_files(repo_id) and not any(
            (snapshot / name).is_file()
            for name in ("pytorch_model.bin", "model.safetensors", "tf_model.h5")):
        return f"{snapshot} holds no model payload (pytorch_model.bin/model.safetensors)"
    return None



def is_hub_cache_asset(asset: dict[str, Any]) -> bool:
    """True for an asset whose destination is a Hugging Face HUB repo cache directory.

    SHAPE, not declaration source, and the shape is read from two registries that describe the same
    cache from different sides:

    * an AUXILIARY asset names the repo (`hub_repo_id`) because acquisition needs it, and
    * a PREPARATION-CONTRACT asset describes the same cache as the loader sees it, through
      `location_type: directory` plus `cache_path_suffix: hub/models--<owner>--<name>`.

    A predicate that accepted only `hub_repo_id` therefore answered `False` for every consumer that
    passes a contract asset -- the readiness check, `cache_asset_blocker`, the TME input blocker and
    the TME preflight all do -- so those paths silently fell back to the pre-fix behaviour: an orphan
    snapshot read as present, and the declared payload was never required.
    """
    if asset.get("hub_repo_id"):
        return True
    suffix = str(asset.get("cache_path_suffix") or "")
    prefix, _, repo_segment = suffix.partition("models--")
    return bool(repo_segment) and prefix == "hub/" and (
        asset.get("location_type") == "directory" or bool(asset.get("cache_root_env_var")))


def hub_cache_repo_id(asset: dict[str, Any]) -> str:
    """The repo id a hub asset resolves, from either registry's declaration of it.

    The auxiliary registry names it directly; a preparation contract describes the same cache by its
    loader-visible path (`hub/models--<owner>--<name>`), and the two must yield one id or the checker
    and the downloader end up inspecting different directories.
    """
    declared = asset.get("hub_repo_id")
    if declared:
        return str(declared)
    _, _, repo_segment = str(asset.get("cache_path_suffix") or "").partition("models--")
    return repo_segment.replace("--", "/")


def hub_cache_root(asset: dict[str, Any]) -> Path:
    """The cache ROOT a hub asset resolves below.

    `cache_root_env_var` names the variable that relocates this cache (Hugging Face's loader uses
    `HF_HOME`), and a configured value is the ROOT, not the repo directory: an operator who sets
    `HF_HOME=<dir>` gets `<dir>/hub/models--<repo>/`. Reading it as the asset directory was the
    R22-02 defect, so there is ONE resolver and the checker, the readiness report, the budget, the
    acquisition plan and the downloader all call it.
    """
    variable = asset.get("cache_root_env_var") or "HF_HOME"
    if variable != "HF_HOME":
        try:
            from workbench.backend.model_settings import resolve_path_setting

            resolved = resolve_path_setting(variable)
            if resolved is not None:
                return resolved
        except Exception:
            pass
        configured = os.environ.get(variable)
        if configured and configured.strip():
            from workbench.backend.operator_config import _DEFAULT_REPO_ROOT

            p = Path(configured.strip()).expanduser()
            return p if p.is_absolute() else (_DEFAULT_REPO_ROOT / p).resolve()
    return hf_hub_cache_root()


def hub_cache_repo_dir(asset: dict[str, Any]) -> Path:
    """The repo cache directory a hub asset resolves to, under its own cache root."""
    return hf_hub_repo_dir(hub_cache_repo_id(asset), hub_cache_root(asset))


def hub_cache_declared_files(repo_id: str) -> list[str]:
    """The payload files a hub repo is declared to need, from ONE declaration.

    Two registries describe the same repository: the auxiliary one names what acquisition must fetch
    (`hub_files`), and a preparation contract names the same cache by the path its loader opens.
    Reading the file list only from the asset in hand therefore produced an EMPTY set for every
    contract asset -- so an incomplete cache passed every payload check, and a cache holding nothing
    but a vocabulary was certified for a run whose Qformer loader then failed on the absent weights.
    Looking the repo up in the auxiliary registry keeps the two in step without a second copy of the
    list.
    """
    for candidate in load_auxiliary_assets():
        if candidate.get("hub_repo_id") == repo_id and candidate.get("hub_files"):
            return [str(name) for name in candidate["hub_files"]]
    return []

def hub_cache_declared_revision(repo_id: str) -> str | None:
    for candidate in load_auxiliary_assets():
        if candidate.get("hub_repo_id") == repo_id:
            revision = candidate.get("source_revision")
            return str(revision) if revision else None
    return None

def hub_cache_declared_size(repo_id: str) -> int | None:
    for candidate in load_auxiliary_assets():
        if candidate.get("hub_repo_id") == repo_id:
            size = expected_asset_size(candidate)
            return size if type(size) is int and size > 0 else None
    return None

def hub_cache_size_problem(asset: dict[str, Any], repo: Path | None = None) -> str | None:
    identity = hub_cache_identity(asset, repo)
    expected = hub_cache_declared_size(identity["repo_id"])
    if expected is None or hub_cache_missing_files(asset, repo):
        return None
    actual = identity["size_bytes"]
    if actual != expected:
        return (f"PAYLOAD_SIZE_MISMATCH {identity['snapshot']}: declared payload has {actual} bytes, "
                f"registry declares {expected} bytes")
    return None


def hub_cache_payload_problem(asset: dict[str, Any], repo: Path | None = None) -> str | None:
    identity = hub_cache_identity(asset, repo)
    if identity["snapshot"] is None or hub_cache_missing_files(asset, repo):
        return None
    for name in identity["declared_files"]:
        path = identity["snapshot"] / name
        if not _has_substantive_payload(path, asset):
            return f"{identity['snapshot']} declared payload {name} is empty or has no substantive content"
    return None


def hub_cache_transfer_problem(repo_id: str, repo: Path, snapshot: Path,
                               declared_files: list[str]) -> str | None:
    """Check per-file transfer evidence when this cache was acquired by the workbench."""
    try:
        from workbench.backend.operator_config import resolve_config
        from workbench.scripts.download_auxiliary_assets import load_manifest, manifest_path

        entries = load_manifest(manifest_path(resolve_config().WORKBENCH_ARTIFACTS_ROOT))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return f"hub cache transfer manifest is unreadable: {type(error).__name__}: {error}"
    destination = str(repo.resolve())
    record = next((item for item in entries if item.get("destination") == destination
                   and item.get("hub_repo_id") == repo_id), None)
    if record is None:
        return None
    if record.get("source_revision") != snapshot.name:
        return (f"HUB_TRANSFER_REVISION_MISMATCH {repo}: manifest records "
                f"{record.get('source_revision')}, active snapshot is {snapshot.name}")
    file_records = record.get("hub_file_records")
    if not isinstance(file_records, list):
        return f"HUB_TRANSFER_RECORD_INVALID {repo}: per-file transfer evidence is missing"
    expected_names = sorted(declared_files)
    recorded_names = sorted(item.get("name") for item in file_records if isinstance(item, dict))
    if recorded_names != expected_names:
        return f"HUB_TRANSFER_RECORD_INVALID {repo}: manifest file list differs from declared payload"
    normalized = []
    seen: set[Path] = set()
    actual_total = 0
    for item in sorted(file_records, key=lambda value: value["name"]):
        path = snapshot / item["name"]
        try:
            actual_size = path.stat().st_size
            actual_digest = hub_payload_sha256(path)
            real_path = path.resolve()
            if real_path not in seen:
                seen.add(real_path)
                actual_total += actual_size
        except OSError:
            return f"HUB_TRANSFER_FILE_MISSING {path}"
        except ValueError as error:
            return f"HUB_TRANSFER_FILE_INVALID {path}: {error}"
        if item.get("size_bytes") != actual_size or item.get("sha256") != actual_digest:
            return f"HUB_TRANSFER_DIGEST_MISMATCH {path}: cache bytes differ from acquired transfer"
        normalized.append({"name": item["name"], "size_bytes": actual_size,
                           "sha256": actual_digest})
    payload = json.dumps({"hub_repo_id": repo_id, "source_revision": snapshot.name,
                          "hub_file_records": normalized}, sort_keys=True, separators=(",", ":"))
    transfer_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if record.get("downloaded_sha256") != transfer_digest or record.get("size_bytes") != actual_total:
        return f"HUB_TRANSFER_DIGEST_MISMATCH {repo}: manifest transfer summary differs from cache"
    return None

def hub_cache_integrity_problem(asset: dict[str, Any], repo: Path, snapshot: Path,
                                declared_files: list[str], *, check_transfer: bool = True) -> str | None:
    problem = hub_cache_size_problem(asset, repo)
    if problem:
        return problem
    problem = hub_cache_payload_problem(asset, repo)
    if problem:
        return problem
    if check_transfer:
        return hub_cache_transfer_problem(hub_cache_repo_id(asset), repo, snapshot, declared_files)
    return None


def hub_cache_identity(asset: dict[str, Any], repo: Path | None = None) -> dict[str, Any]:
    """The resolved identity of a hub cache asset: where it lives and what is in it.

    `repo` overrides where the cache is LOOKED FOR. A caller that already resolved a destination (the
    auxiliary validator, whose `path` argument exists precisely so a caller can point at one cache)
    must be able to ask about THAT directory; re-deriving it from `HF_HOME` instead made an isolated
    fixture describe the host's cache, which is both wrong and untestable.

    One record for four consumers that must not disagree -- the readiness report, the budget, the
    acquisition plan and the downloader's verification. `files` counts the payload the LOADER reads
    (the snapshot's real filenames, followed through the symlinks the hub lays down), while
    `size_bytes` sums only the blobs actually stored, so a correct cache is never reported as
    exceeding twice its own declaration.
    """
    repo_id = hub_cache_repo_id(asset)
    repo = repo if repo is not None else hf_hub_repo_dir(repo_id, hub_cache_root(asset))
    snapshot = hub_cache_snapshot(repo)
    declared = [str(name) for name in (asset.get("hub_files") or [])] or hub_cache_declared_files(repo_id)
    files: list[str] = []
    size = 0
    seen: set[Path] = set()
    if snapshot is not None:
        for name in sorted(declared):
            entry = snapshot / name
            if entry.is_file():
                files.append(name)
                try:
                    real_path = entry.resolve()
                    if real_path not in seen and real_path.is_file():
                        seen.add(real_path)
                        size += real_path.stat().st_size
                except OSError:
                    pass
    return {"repo_id": repo_id, "path": repo, "snapshot": snapshot, "present": repo.is_dir(),
            "files": files, "size_bytes": size,
            "declared_files": declared}


def hub_cache_missing_files(asset: dict[str, Any], repo: Path | None = None) -> list[str]:
    """The declared payload files of a hub repo that its active snapshot is missing."""
    identity = hub_cache_identity(asset, repo)
    if identity["snapshot"] is None:
        return list(identity["declared_files"])
    return [name for name in identity["declared_files"] if name not in set(identity["files"])]


def _guard_auxiliary_destination(root: Path, path: str) -> Path:
    resolved_root = root.resolve()
    destination = (root / path).resolve()
    if not destination.is_relative_to(resolved_root):
        raise ValueError(f"auxiliary destination escapes root: {destination}")
    return destination


def auxiliary_destination(asset: dict[str, Any], third_party_root: Path, fashioniq_root: Path) -> Path:
    if asset["destination_scope"] == "user_cache":
        # Some consumers hard-code their cache directory instead of honouring an environment
        # variable, and the registry has to match the CONSUMER, not a convention. OpenAI's CLIP does
        # exactly this: `clip.load` uses `download_root or os.path.expanduser("~/.cache/clip")` and
        # never reads XDG_CACHE_HOME, so staging RN50x4 under an operator's XDG_CACHE_HOME put the
        # weights where the evaluator would not look and left it downloading them itself.
        fixed = asset.get("cache_root_fixed")
        if fixed:
            root = Path(fixed).expanduser()
        elif asset["destination_path"].startswith("hub/models--") and asset.get("cache_root_env_var") in {"HF_HOME", None}:
            # A Hugging Face HUB asset, not a file under a cache root. The loader resolves the repo
            # id to `hub/models--<owner>--<name>/` and then reads `refs/` and `snapshots/` inside it;
            # a file written to any other level of the cache is one the loader never opens. Routing
            # through the shared resolver keeps this destination equal to the directory the checker
            # inspects and the bootstrap writes.
            parts = PurePosixPath(asset["destination_path"]).parts
            repo = parts[1][len("models--"):].replace("--", "/") if len(parts) > 1 else ""
            return _guard_auxiliary_destination(
                hf_hub_cache_root(), str(hf_hub_repo_dir(repo).relative_to(hf_hub_cache_root())))
        else:
            env_var = asset["cache_root_env_var"]
            root = Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            if env_var in {"TORCH_HOME", "HF_HOME"} and not os.environ.get(env_var):
                root /= "torch" if env_var == "TORCH_HOME" else "huggingface"
        return _guard_auxiliary_destination(root, asset["destination_path"])
    if asset["destination_scope"] == "fashioniq_root":
        return _guard_auxiliary_destination(fashioniq_root, asset["destination_path"])
    if asset["destination_scope"] == "model_source":
        root = third_party_root / model_by_id(asset["model_id"])["source_dir"]
        return _guard_auxiliary_destination(root, asset["destination_path"])
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
                "source_url", "source_type", "source_repo", "source_revision", "expected_sha256", "provenance_status",
                "expected_sha256_prefix", "expected_size_bytes", "content_contract",
                "hub_repo_id", "hub_files", "cache_root_env_var",
            )
            if hasattr(asset, k)
        }
    if path is None:
        target = rec.get("destination") or rec.get("path")
        path = Path(target) if target else None
    if path is None:
        return AuxiliaryValidationResult(False, "MISSING", ["no path specified for asset validation"])
    path = Path(path)

    if is_hub_cache_asset(rec):
        # A hub cache is a DIRECTORY the loader resolves, not a file to size or hash. Without this
        # branch the directory fell into the "is it a file?" test below and every correctly staged
        # cache was reported MISSING -- which is how a complete cache, and the budget derived from
        # the same asset, both went wrong. What makes the cache usable is the resolution its consumer
        # performs: `refs/<rev>` must name a snapshot and that snapshot must hold the declared
        # payload. Layout only: loading the files belongs to the model environment.
        if not path.exists():
            return AuxiliaryValidationResult(False, "MISSING", [f"MISSING {path}"])
        if not path.is_dir():
            return AuxiliaryValidationResult(False, "INVALID_FILE_TYPE", [f"NOT A DIRECTORY {path}"])
        refs = path / "refs"
        if not refs.is_dir():
            return AuxiliaryValidationResult(False, "INVALID_FILE_TYPE", [f"NOT A DIRECTORY {refs}"])
        ref_main = refs / "main"
        if not ref_main.is_file():
            return AuxiliaryValidationResult(
                False, "UNRESOLVED_REF",
                [f"ref main is missing under {refs} (unqualified native loader lookup demands refs/main)"])
        snapshot = hub_cache_snapshot(path)
        if snapshot is None:
            return AuxiliaryValidationResult(
                False, "UNRESOLVED_REF",
                [f"ref main under {refs} names a snapshot that is not present in snapshots/"])
        expected_rev = hub_cache_declared_revision(hub_cache_repo_id(rec))
        if expected_rev and snapshot.name != expected_rev:
            return AuxiliaryValidationResult(
                False, "PINNED_REVISION_MISMATCH",
                [f"ref main points to snapshot {snapshot.name}, expected pinned revision {expected_rev}"])
        missing = hub_cache_missing_files(rec, path)
        if missing:
            return AuxiliaryValidationResult(
                False, "INCOMPLETE_CACHE",
                [f"INCOMPLETE_CACHE {snapshot}: missing declared payload {', '.join(missing)}"])
        integrity_problem = hub_cache_integrity_problem(
            rec, path, snapshot, hub_cache_declared_files(hub_cache_repo_id(rec)))
        if integrity_problem:
            status = ("PAYLOAD_SIZE_MISMATCH" if integrity_problem.startswith("PAYLOAD_SIZE_MISMATCH")
                      else "INVALID_PAYLOAD" if "no substantive content" in integrity_problem
                      else "INVALID_TRANSFER" if integrity_problem.startswith("HUB_TRANSFER_")
                      else "INVALID_CACHE")
            return AuxiliaryValidationResult(False, status, [integrity_problem])
        return AuxiliaryValidationResult(True, "VERIFIED", [])

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

    expected_size = expected_asset_size(rec)
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
        author_linked_without_digest = (
            rec.get("source_type") in {"direct_url", "author_huggingface"}
            and bool(rec.get("source_repo"))
            and rec.get("expected_sha256") is None
            and rec.get("expected_sha256_prefix") is None
        )
        # A digest-less asset has no published value to verify against, so the ONLY content evidence
        # available is the file itself. Without this the size check alone admits a sparse or
        # zero-filled file of exactly the declared length -- structurally a container of zeros with
        # no weights in it -- which is precisely the state `asset_admission_status` already refuses
        # as SUSPECT_PAYLOAD. Both layers now ask the same question of the same bytes, so external
        # admission and auxiliary admission cannot contradict each other again in either direction.
        has_payload = _has_substantive_payload(path, rec)
        if author_linked_without_digest and not issues and status == "VALID" and has_payload:
            return AuxiliaryValidationResult(True, "LOCALLY_RECORDED_ASSET", [])
        if author_linked_without_digest and not issues and status == "VALID":
            return AuxiliaryValidationResult(
                False, "SUSPECT_PAYLOAD",
                [f"SUSPECT_PAYLOAD {path}: no author digest to verify against, and the file holds no "
                 f"substantive payload (every byte sampled is identical); a placeholder of the "
                 f"declared size is not an asset"])
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
    # A model whose pinned source cannot run as published may declare one approved patch. The
    # declaration carries the exact base and derived digests, so the checkout is accepted only while
    # it holds that one approved delta, and any other modification still blocks.
    for patch in model.get("approved_source_patches") or []:
        artifacts.append({
            "kind": "approved_source_patch",
            "path": patch["path"],
            "base_sha256": patch.get("base_sha256"),
            "derived_sha256": patch.get("derived_sha256"),
            "patch_file": patch.get("patch_file"),
            "patch_sha256": patch.get("patch_sha256"),
            "reason": patch.get("reason"),
        })
    return artifacts


def model_by_id(model_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    for model in registry["models"]:
        if model["model_id"] == model_id:
            return model
    raise KeyError(model_id)


def checkpoint_by_id(model: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    """Resolve a checkpoint selection to the record an evaluation runs against.

    A declared bundle is a first-class selection (LIMN needs all three per-category
    checkpoints for one complete macro result). The bundle is anchored on its first required
    member so path/hash handling stays uniform, while `bundle_id` and the member list record
    what was actually selected. Without this, every caller that goes through the adapter or
    `run_eval.py` raised `KeyError` on a legitimate bundle id.

    The derived record carries THREE consistent views of its members, because different
    consumers need different things and one shape cannot serve all of them:

    * ``bundle_members`` -- the member *records*, each carrying its ``filename``. Every consumer
      that has to name or hash the files a bundle is made of reads this
      (``checkpoint_missing_paths``, ``checkpoint_availability``, ``evaluate_models``' plan
      building and ``completion_validator``).
    * ``required_files`` -- the member filenames. This is the key those consumers' hashing and
      availability helpers already thread through (`checkpoint_missing_paths`,
      `_availability_digest`), so a bundle must populate it too or its digest path silently took
      the single-file branch.
    * ``required_checkpoint_ids`` -- the member *checkpoint ids*, the registry's own vocabulary,
      for any reader that resolves members through the registry.

    Before this, ``bundle_members`` was the raw id list while five consumers indexed it as
    records: ``checkpoint_availability`` and ``completion_validator`` raised ``TypeError`` on a
    bundle, and ``completion_validator`` swallowed it in a bare ``except`` and fell back to
    ``Path(".")`` -- so a bundle's completion proof was validated against the process's working
    directory instead of its weights.
    """
    for checkpoint in model["checkpoint_variants"]:
        if checkpoint["checkpoint_id"] == checkpoint_id:
            return checkpoint
    for bundle in model.get("checkpoint_bundles") or []:
        if bundle.get("bundle_id") != checkpoint_id:
            continue
        required = list(bundle.get("required_checkpoint_ids") or [])
        if not required:
            break
        variants = []
        for member_id in required:
            member = next((item for item in model["checkpoint_variants"]
                           if item["checkpoint_id"] == member_id), None)
            if member is None:
                break
            variants.append(member)
        else:
            # The anchor is a FULL variant record (the derived bundle inherits its protocol,
            # mapping state, environment and source facts); `bundle_members` is the *normalized*
            # member view. The two are deliberately different shapes: spreading a normalized
            # member as the anchor silently dropped every other field a caller reads.
            anchor = variants[0]
            # The canonical member shape a proof, a manifest and a reader all speak: `filename`
            # (the file that gets hashed), `expected_sha256` (the declared digest, when the
            # registry states one), and the member's own checkpoint id. `checkpoint_missing_paths`
            # reads the first two for a bundle/placeholder artifact, so a member record without
            # them made a complete bundle verify as missing.
            members = [{"filename": str(member["filename"]),
                        "expected_sha256": member.get("expected_sha256"),
                        "checkpoint_id": str(member["checkpoint_id"])}
                       for member in variants]
            return {**anchor, "checkpoint_id": bundle["bundle_id"],
                    "bundle_id": bundle["bundle_id"],
                    # The anchor's own `artifact_type` (usually absent, i.e. `file`) describes ONE
                    # member, not the bundle -- and `checkpoint_missing_paths` keys its member
                    # sweep on this field, so leaving it as `file` made a bundle whose members are
                    # all present be reported as a missing directory. `bundle_placeholder` is the
                    # registry's own vocabulary for "a directory of declared members"
                    # (`checkpoint_missing_paths` treats it identically to `bundle_directory`), so
                    # reusing it here keeps the derived record inside the registry's schema.
                    "artifact_type": "bundle_placeholder",
                    "bundle_members": members,
                    "required_files": [str(member["filename"]) for member in variants],
                    "required_checkpoint_ids": required,
                    # The bundle's OWN mapping status is a distinct fact from its anchor member's:
                    # `checkpoint_selection` filters bundles on this key, and `model_limn.Adapter.status`
                    # reads it to decide readiness. Deriving the record without it made both see None,
                    # so a complete, VERIFIED_METADATA bundle was refused as "mapping unresolved".
                    "mapping_status": bundle.get("mapping_status")}
        break
    raise KeyError(checkpoint_id)


def checkpoint_path(model_id: str, checkpoint: dict[str, Any], root: Path | None = None) -> Path:
    """Where the artifact this selection names lives.

    Three shapes, all of which must resolve to the directory or file the *members* actually occupy:

    * a **bundle** is not one file. Its members are the per-category weights, which live together
      in the model's checkpoint directory, so the artifact's path is that directory. Returning the
      anchor member's own filename would name one member and leave the other two unresolvable --
      which is how a bundle's manifest came out with a single member and its completion proof was
      rejected as a digest mismatch.
    * a **directory** artifact declares the directory name in `filename` and its members in
      `required_files`; the path is that directory.
    * a **file** artifact's path is the file.

    The last two are the same expression, so there is one branch and no third convention to drift.
    """
    from workbench.backend.operator_config import resolve_config

    base = (root or resolve_config().WORKBENCH_CHECKPOINT_ROOT) / model_id
    if checkpoint.get("bundle_id"):
        return base
    return base / checkpoint["filename"]


def checkpoint_member_root(checkpoint: Path, members: list[dict[str, Any]] | None = None,
                           artifact_type: str | None = None,
                           is_multi_member: bool | None = None) -> Path:
    """The directory the checkpoint's MEMBERS occupy -- one rule for every consumer.

    ``checkpoint_path`` resolves a multi-member artifact (LIMN's three per-category weights) to the
    directory holding them, and a single-artifact selection to that file. Both shapes need the same
    answer here, and the answer differs per shape:

    * a multi-member bundle keeps its members as SIBLINGS inside the resolved directory, so the
      member root IS that directory -- ``checkpoint.parent`` would name the whole checkpoint root and
      every member lookup would come up one level too high;
    * a single checkpoint is itself the one member, so its members live in ``checkpoint.parent``.

    This was duplicated as slightly different expressions in the adapter and in the observer, and the
    observer's copy was wrong for bundles. One function, so it cannot drift again.
    """
    if is_multi_member is None:
        is_multi_member = artifact_type in ("category_specific_whole_models", "run_directory",
                                            "bundle_directory", "bundle_placeholder")
    if artifact_type is not None or is_multi_member:
        # The declaration answers, so the result does not change when the weights arrive.
        return checkpoint if is_multi_member else checkpoint.parent
    # No declaration available (a bare path): fall back to the filesystem, which is the best
    # available answer and is what a caller passing only a path can mean.
    return checkpoint if checkpoint.is_dir() else checkpoint.parent


def hub_payload_sha256(path: Path) -> str:
    """Hash Hub payload bytes after resolving Hugging Face's snapshot symlinks."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FileNotFoundError(f"Hub payload member is missing: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"Hub payload member is not a regular file: {path}")
    return sha256_file(resolved)


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
    downloaded = checkpoint_is_present(path, checkpoint, model_id=model["model_id"], config=config)
    required_files = checkpoint.get("required_files")
    if checkpoint.get("bundle_members"):
        required_files = [member["filename"] for member in checkpoint["bundle_members"]]
    local_sha = _availability_digest(config, model["model_id"], checkpoint, path) if downloaded else None
    official_sha = checkpoint.get("expected_sha256")
    # A BUNDLE's `local_sha256` is a directory digest over its declared members, while the
    # registry's `expected_sha256` on the derived record is the *anchor member's* file digest
    # (inherited from `checkpoint_by_id`, which only redirects `filename`/`checkpoint_id`). Comparing
    # the two reported `official_sha256_match: False` on a pristine, correctly downloaded LIMN
    # bundle -- a false FAIL that `pc.sh` refuses on -- so the comparison is made only for the
    # artifacts whose digest framing can agree. For a bundle, the per-member declarations are the
    # comparable facts, and `checkpoint_missing_paths` already checks each member against them.
    official_sha_comparable = official_sha is not None and not checkpoint.get("bundle_members")
    official_sha_match = (local_sha == official_sha
                          if downloaded and official_sha_comparable else None)
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
        # Stated as what it is: for a bundle the registry declares a member digest, not a bundle
        # digest, so there is no bundle-level expectation to compare against.
        "official_sha256_known": official_sha is not None and not checkpoint.get("bundle_members"),
        "official_sha256_match": official_sha_match,
        "mapping_verified": mapping_verified,
        "adapter_command_ready": adapter_command_ready,
        "runtime_verified": model.get("reproduction_status") == "VERIFIED",
        "runnable": not reasons,
        "checkpoint_path": str(path),
        "blocking_reasons": reasons,
    }
