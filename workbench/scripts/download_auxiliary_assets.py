#!/usr/bin/env python3
"""Download verified CSMCIR auxiliary assets; never invent unresolved sources."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import (
    auxiliary_assets_for_model, auxiliary_destination, auxiliary_model_ids,
    auxiliary_source_url, hf_hub_repo_dir, sha256_file, validate_auxiliary_asset,
)

@dataclass(frozen=True)
class Asset:
    model_id: str
    family: str
    filename: str
    destination: Path
    required: bool
    automatic_download_available: bool
    source_url: str | None
    source_revision: str | None
    expected_sha256: str | None
    provenance_status: str
    expected_sha256_prefix: str | None = None
    expected_size_bytes: int | None = None
    content_contract: dict[str, Any] | None = None
    #: A published asset that can stand in for an unattainable one — only for an asset whose
    #: `provenance_status` is not VERIFIED and which declares it in the registry (CSMCIR's
    #: COT_ours2 captions point at the author's Qwen captions). Absent means the asset has no
    #: compatible stand-in and stays strictly unattainable.
    caption_alias_from: dict[str, Any] | None = None
    source_type: str | None = None
    source_repo: str | None = None
    #: Set for an asset whose destination is a Hugging Face HUB cache directory: the repo id the
    #: loader resolves. Acquisition then happens through the same library the loader uses, and the
    #: per-file download path below must not touch it (its destination is a directory, not a file).
    hub_repo_id: str | None = None
    #: The only files of that repo the run needs. A hub repo publishes several frameworks' weights
    #: and a card; fetching the whole repository is both slower and larger than the declaration the
    #: budget was computed from.
    hub_files: tuple[str, ...] | None = None

def assets(model_ids: set[str], third_party_root: Path, fashioniq_root: Path) -> tuple[Asset, ...]:
    return tuple(
        Asset(
            asset["model_id"],
            asset["family"],
            Path(asset["destination_path"]).name,
            auxiliary_destination(asset, third_party_root, fashioniq_root),
            asset["required_for"] == "official_evaluation",
            asset["automatic_download_available"],
            auxiliary_source_url(asset),
            asset["source_revision"],
            asset["expected_sha256"],
            asset["provenance_status"],
            asset.get("expected_sha256_prefix"),
            asset.get("expected_size_bytes"),
            asset.get("content_contract"),
            asset.get("caption_alias_from"),
            asset.get("source_type"),
            asset.get("source_repo"),
            asset.get("hub_repo_id"),
            tuple(asset["hub_files"]) if asset.get("hub_files") else None,
        )
        for model_id in model_ids
        for asset in auxiliary_assets_for_model(model_id)
    )


def manifest_path(artifacts_root: Path) -> Path:
    return artifacts_root / "auxiliary" / "download_manifest.json"


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def report(items: tuple[Asset, ...]) -> None:
    for asset in items:
        if asset.hub_repo_id:
            problem = hub_cache_layout_problem(asset.hub_repo_id)
            state = "present" if problem is None else "missing"
            source_unverified = asset.provenance_status != "VERIFIED"
        else:
            state = "present" if asset.destination.is_file() else "missing"
            prefix = getattr(asset, "expected_sha256_prefix", None)
            source_unverified = asset.provenance_status != "VERIFIED" or (asset.expected_sha256 is None and prefix is None)
        source = asset.source_url or "unverified author download source"
        print(
            f"{asset.family}: destination={asset.destination} state={state} "
            f"required={str(asset.required).lower()} "
            f"automatic_download_available={str(asset.automatic_download_available).lower()} "
            f"source_unverified={str(source_unverified).lower()} "
            f"source={source}"
        )

def verify(asset: Asset, *, fashioniq_root: Path | None = None) -> bool:
    if asset.hub_repo_id:
        problem = hub_cache_layout_problem(asset.hub_repo_id)
        if problem:
            print(f"{asset.family}: {problem}")
            return False
        print(f"{asset.family}: cache layout verified")
        return True
    res = validate_auxiliary_asset(asset, asset.destination, fashioniq_root=fashioniq_root)
    if res.status == "MISSING":
        print(f"MISSING {asset.destination}")
        return False
    if res.status == "INVALID_EMPTY_FILE":
        print(f"INVALID EMPTY FILE {asset.destination}")
        return False
    size = asset.destination.stat().st_size
    digest = sha256_file(asset.destination)
    prefix = getattr(asset, "expected_sha256_prefix", None)
    if asset.expected_sha256 is None and not prefix:
        print(f"{asset.destination}: {size} bytes sha256={digest} — checksum unavailable; provenance not checksum-verified")
    elif (asset.expected_sha256 and digest != asset.expected_sha256) or (prefix and not digest.startswith(prefix)):
        print(f"{asset.destination}: {size} bytes sha256={digest} OFFICIAL SHA256 MISMATCH")
    else:
        print(f"{asset.destination}: {size} bytes sha256={digest} OFFICIAL SHA256 MATCH")
    if asset.provenance_status != "VERIFIED":
        print(f"UNRESOLVED REQUIRED PROVENANCE {asset.destination}: {asset.provenance_status}")
    for issue in res.issues:
        if (
            "OFFICIAL SHA256 MISMATCH" not in issue
            and "OPERATOR_SUPPLIED_UNVERIFIED" not in issue
            and "MISSING" not in issue
            and "INVALID EMPTY FILE" not in issue
        ):
            print(issue)
    return res.ok

def hub_cache_layout_problem(repo_id: str) -> str | None:
    """The registry's own hub-layout diagnosis, imported rather than duplicated.

    One definition answers this for the checker, the downloader and the bootstrap; a second copy
    here could disagree with the one the readiness report prints.
    """
    from workbench.backend.registry import hub_cache_layout_problem as _problem

    return _problem(repo_id)


def hub_cache_missing_files(asset: "Asset") -> list[str]:
    """The declared payload files of this asset's hub repo that its active snapshot is missing.

    Layout resolution and payload completeness are two questions: a cache whose `refs/` resolve and
    whose model weights are present can still be missing `config.json`, and a plan that asks only the
    first question schedules a download for a cache the download would then consider complete.
    """
    from workbench.backend.registry import hub_cache_missing_files as _missing

    return _missing({"hub_repo_id": asset.hub_repo_id, "hub_files": list(asset.hub_files or []),
                     "cache_root_env_var": "HF_HOME"})


def bootstrap_hub_asset(asset: "Asset") -> tuple[bool, str]:
    """Acquire a Hugging Face hub asset into the cache the loader reads, bounded to its declaration.

    Uses the pinned `transformers`/`huggingface_hub` in THIS interpreter — the model environment
    owns the loader, and the same library writes the layout it will later read. Only the files the
    registry declares are fetched, so the transfer is bounded by a stated byte count rather than by
    whatever the repository happens to contain.
    """
    repo_id = asset.hub_repo_id
    assert repo_id is not None
    from huggingface_hub import snapshot_download

    allow = list(asset.hub_files or [])
    limit = asset.expected_size_bytes if type(asset.expected_size_bytes) is int and asset.expected_size_bytes > 0 else None
    # The destination is resolved from the WORKBENCH root below, and the library must write into
    # exactly that root. `snapshot_download` reads `HF_HOME` from its own environment, which
    # `hf_hub_cache_root()` already answers through the shared settings reader, so the two agree by
    # construction. Passing `cache_dir` instead was tried and REVERTED after it was reproduced
    # splitting a cache in two (`<dir>/hub/…` for snapshot symlinks, `<dir>/refs` for ref
    # bookkeeping), which staged 421 MB whose own verification then refused it.
    try:
        path = snapshot_download(repo_id=repo_id,
                                 revision=asset.source_revision or None,
                                 allow_patterns=allow or None)
    except Exception as error:
        return False, f"hub bootstrap failed for {repo_id}: {type(error).__name__}: {error}"
    repo = hf_hub_repo_dir(repo_id)
    # `blobs/` holds the bytes; `snapshots/` holds symlinks into it. Counting `rglob` would follow
    # those links and report every file twice, which is how a correct acquisition appears to exceed
    # its own bound.
    used = sum(path_file.stat().st_size for path_file in repo.rglob("*")
               if path_file.is_file() and not path_file.is_symlink())
    # PUBLISH the reference. `snapshot_download(revision=<sha>)` writes `snapshots/<sha>/` and NO
    # `refs/<rev>` entry, and a snapshot no ref names cannot be reached by the repo-id lookup the
    # pinned loader performs -- so without this step the acquisition would produce a cache that its
    # own verification refuses, which is exactly what happened the first time this ran. The ref is
    # written only after the payload is present, so an interrupted download leaves the cache
    # unreferenced and therefore visibly incomplete rather than silently usable.
    if asset.source_revision:
        refs = repo / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        ref = refs / "main"
        if not ref.exists() or ref.read_text(encoding="utf-8").strip() != asset.source_revision:
            ref.write_text(asset.source_revision, encoding="utf-8")
    if limit is not None and used > limit:
        # Bounded acquisition, checked AFTER the fact because the library does not report the
        # transfer size up front. Reported rather than silently accepted: a hub repo that grew past
        # its declaration is a fact the operator needs before the run budgets on it.
        return False, (f"hub bootstrap for {repo_id} wrote {used} bytes, above the declared bound "
                       f"of {limit} bytes")
    problem = hub_cache_layout_problem(repo_id)
    if problem:
        return False, problem
    print(f"STAGED HUB CACHE {repo} ({used} bytes, files: {', '.join(sorted(allow)) or 'all'})")
    return True, str(path)


#: Alias staging is opt-in. The registry names an alias source only for an asset with no verified
#: provenance (CSMCIR's COT_ours2 captions), and even then the run must ask for it: see
#: `workbench/docs/COT_OURS2_AUDIT.md` for why the published stand-in is source-compatible but not
#: proven identical to the file the authors evaluated with.
CAPTION_ALIAS_FLAG = "--stage-caption-aliases"

def alias_source_url(alias: dict[str, Any], source_repo: str) -> str | None:
    source_path = alias.get("source_path")
    if not source_path:
        return None
    return f"{source_repo}/resolve/{alias['source_revision']}/{source_path}"

def alias_content_problems(asset: Asset, staged: Path, *, fashioniq_root: Path | None) -> list[str]:
    """Reject a published stand-in whose content cannot serve as the declared asset.

    The digest proves the bytes are the published file; it does not prove the published file is
    usable here. These are the properties the pinned evaluator actually depends on: a JSON object
    keyed by FashionIQ image id, exactly covering the declared split, with a non-empty Final_Caption
    for every id — the field every executing consumer reads.
    """
    from workbench.backend.registry import validate_auxiliary_asset

    alias = asset.caption_alias_from or {}
    res = validate_auxiliary_asset(
        {
            "destination": str(staged),
            "expected_sha256": alias.get("expected_sha256"),
            "expected_size_bytes": alias.get("expected_size_bytes"),
            "content_contract": asset.content_contract,
            "provenance_status": "VERIFIED",
        },
        staged,
        fashioniq_root=fashioniq_root,
    )
    problems = list(res.issues)
    if not res.ok:
        return problems or [f"{staged}: content did not satisfy the declared contract"]
    data = json.loads(staged.read_text(encoding="utf-8"))
    blank = [key for key, value in data.items() if not str(value.get("Final_Caption", "")).strip()]
    if blank:
        problems.append(f"{staged}: {len(blank)} records have an empty Final_Caption (e.g. {blank[0]})")
    return problems

def stage_caption_alias(asset: Asset, *, fashioniq_root: Path | None = None,
                        managed: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Copy the published stand-in to the unattainable asset's destination.

    A copy, never a symlink: the runtime treats a symlink at a declared auxiliary asset as an
    unauthorized source change, and the file must live inside the model's checkout so the pinned
    evaluator's `base_path` resolves it. The copy is verified against the alias's published digest
    before it replaces anything, and the destination's own registry hash is deliberately NOT
    satisfied — the staged bytes are a variant, and the caller keeps reporting them as one.

    Ownership comes from the download manifest, never from the filename or the registry: a file the
    manifest does not attribute to this destination is the operator's, and staging refuses to touch
    it. Asking for `--stage-caption-aliases` authorizes creating a variant the workbench manages; it
    does not authorize destroying data someone else put there.
    """
    alias = asset.caption_alias_from
    if not alias:
        return False, f"{asset.destination}: no caption alias is declared for this asset"
    if asset.provenance_status == "VERIFIED":
        return False, f"{asset.destination}: refusing to alias a checksum-verified asset"
    source_path = alias.get("source_path")
    if not source_path or fashioniq_root is None:
        return False, f"{asset.destination}: alias has no resolvable published source"
    # The published stand-in is the Qwen file the downloader already stages, so the source is the
    # staged Qwen caption, not a second download of the same bytes.
    staged = fashioniq_root / "qwen_captions" / Path(source_path).name
    if not staged.is_file():
        return False, f"{asset.destination}: the published caption is not staged yet: {staged}"
    digest = sha256_file(staged)
    if alias.get("expected_sha256") and digest != alias["expected_sha256"]:
        return False, f"{asset.destination}: staged caption {staged} is not the declared published file (sha256={digest})"
    content_problems = alias_content_problems(asset, staged, fashioniq_root=fashioniq_root)
    if content_problems:
        return False, f"{asset.destination}: published caption is unusable here: {content_problems[0]}"
    if asset.destination.is_symlink():
        return False, f"{asset.destination}: a symlink is never written as a declared auxiliary asset"
    if asset.destination.exists() and str(asset.destination) not in (managed or {}):
        return False, (f"{asset.destination}: refusing to overwrite an operator-owned file (not recorded in the "
                       f"download manifest); move it aside to stage the alias")
    asset.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = asset.destination.with_suffix(asset.destination.suffix + ".alias.part")
    try:
        shutil.copyfile(staged, temporary)
        if sha256_file(temporary) != digest:
            raise RuntimeError("alias copy did not reproduce the published bytes")
        os.replace(temporary, asset.destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True, f"ALIASED {asset.destination} <- {staged} sha256={digest}"

def record_alias(entries: list[dict], asset: Asset) -> list[dict]:
    """Record an alias as workbench-managed, so a corrupt variant is repaired and resume sees it."""
    alias = asset.caption_alias_from or {}
    digest = sha256_file(asset.destination)
    remaining = [entry for entry in entries if entry["destination"] != str(asset.destination)]
    remaining.append(
        {
            "model_id": asset.model_id,
            "asset_family": asset.family,
            "filename": asset.filename,
            "destination": str(asset.destination),
            "source_url": alias_source_url(alias, "https://huggingface.co/peng12138/CSMCIR"),
            "source_revision": alias.get("source_revision"),
            "downloaded_sha256": digest,
            "size_bytes": asset.destination.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "official_hash_available": False,
            "official_sha256": None,
            # The distinction that keeps this honest: these bytes came from a source-compatible
            # stand-in, not from an artifact the registry can vouch for.
            "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
            "alias_source_path": alias.get("source_path"),
            "alias_source_sha256": alias.get("expected_sha256"),
        }
    )
    return remaining


#: Upper bound on any single download. When the registry declares an expected size it is the bound;
#: otherwise this ceiling still stops a runaway or hostile response from filling the disk. A declared
#: size is the authoritative check; this is the backstop for assets whose size the author never
#: published.
_DOWNLOAD_CEILING = 8 * 1024 * 1024 * 1024


def download(url: str, partial: Path, limit: int = _DOWNLOAD_CEILING) -> None:
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            status = response.getcode()
            if status != 200:
                raise RuntimeError(f"unexpected HTTP status {status} for auxiliary asset download")
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise RuntimeError(
                        f"response exceeded the size limit of {limit} bytes; "
                        f"refusing to continue writing {partial}"
                    )
                output.write(chunk)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} downloading {url}: {error.reason}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"network error downloading {url}: {error.reason}") from error


def record_download(entries: list[dict], asset: Asset) -> list[dict]:
    digest = sha256_file(asset.destination)
    remaining = [entry for entry in entries if entry["destination"] != str(asset.destination)]
    remaining.append(
        {
            "model_id": asset.model_id,
            "asset_family": asset.family,
            "filename": asset.filename,
            "destination": str(asset.destination),
            "source_url": asset.source_url,
            "source_revision": asset.source_revision,
            "downloaded_sha256": digest,
            "size_bytes": asset.destination.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "official_hash_available": asset.expected_sha256 is not None,
            "official_sha256": asset.expected_sha256,
        }
    )
    return remaining


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list registered auxiliary assets")
    selector.add_argument("--model", choices=tuple(sorted(auxiliary_model_ids())), help="select one asset-enabled model")
    selector.add_argument("--all", action="store_true", help="select all registered asset-enabled models")
    parser.add_argument("--dry-run", action="store_true", help="show exact URLs and destinations without changes")
    parser.add_argument("--verify-only", action="store_true", help="verify all required local auxiliary assets without network")
    parser.add_argument("--force-redownload", action="store_true", help="replace verified-download destinations atomically")
    parser.add_argument(
        CAPTION_ALIAS_FLAG,
        action="store_true",
        help="stage a published, source-compatible caption at a registered asset's destination when that "
             "asset has no verified source (CSMCIR COT_ours2). Opt-in: the result is a labelled variant, "
             "not official parity. See workbench/docs/COT_OURS2_AUDIT.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    config = resolve_config()
    model_ids = {args.model} if args.model else auxiliary_model_ids()
    items = assets(model_ids, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    # A Hugging Face HUB asset is a DIRECTORY-shaped cache, not a file to fetch at one URL. Its
    # consumer (`transformers`) resolves the repo id to `hub/models--<repo>/` and reads `refs/<rev>`
    # to find the active `snapshots/<rev>/` directory; a single file written at any other level is a
    # file the loader never opens. It is therefore acquired by the same library the loader uses,
    # offline-verifiable afterwards, and split out of the per-file path below.
    hub_items = tuple(asset for asset in items if asset.hub_repo_id)
    file_items = tuple(asset for asset in items if not asset.hub_repo_id)
    report(items)
    downloadable = tuple(asset for asset in file_items if asset.automatic_download_available)
    unavailable = tuple(asset for asset in file_items if asset.required and not asset.automatic_download_available)
    hub_downloadable = tuple(asset for asset in hub_items if asset.automatic_download_available)
    if args.list:
        return 0
    if args.verify_only:
        failed = False
        for asset in items:
            if asset.required:
                failed |= not verify(asset, fashioniq_root=config.FASHIONIQ_ROOT)
        return int(failed)
    aliasable = tuple(asset for asset in items if asset.caption_alias_from and asset.required)
    if args.dry_run and args.stage_caption_aliases:
        for asset in aliasable:
            staged = config.FASHIONIQ_ROOT / "qwen_captions" / Path(asset.caption_alias_from["source_path"]).name
            print(f"DRY RUN: caption alias {asset.destination} <- {staged} "
                  f"(published sha256={asset.caption_alias_from.get('expected_sha256')}, "
                  f"currently {'staged' if staged.is_file() else 'not staged'})")
        print("DRY RUN: no filesystem mutation.")
    # Deciding whether a file may be replaced needs evidence of WHO put it there, and the download
    # manifest is that evidence: it only ever records a file the workbench itself fetched. Inferring
    # ownership from the registry's `provenance_status` was wrong in both directions — it replaced an
    # operator's own file whenever the registry happened to call the asset VERIFIED, and it kept a
    # corrupt managed file forever whenever the upstream provenance was not VERIFIED. Operator-owned
    # files are never touched.
    manifest = load_manifest(manifest_path(config.WORKBENCH_ARTIFACTS_ROOT))
    recorded_digest = {entry["destination"]: entry.get("downloaded_sha256") for entry in manifest}

    def _needs_replacement(asset) -> bool:
        if args.force_redownload:
            return True
        if not asset.destination.is_file():
            return True
        if asset.destination.stat().st_size == 0:
            return True
        # Not in the manifest means the workbench did not download it: it is the operator's file.
        if str(asset.destination) not in recorded_digest:
            return False
        try:
            digest = sha256_file(asset.destination)
        except OSError:
            return False
        # A managed file is replaced when it no longer matches what the workbench recorded, or the
        # published hash the registry declares. Either mismatch means the bytes on disk are not the
        # bytes that were verified at acquisition.
        if digest != recorded_digest[str(asset.destination)]:
            return True
        if asset.expected_sha256:
            return digest != asset.expected_sha256
        prefix = getattr(asset, "expected_sha256_prefix", None)
        return bool(prefix) and not digest.startswith(prefix)

    selected = tuple(asset for asset in downloadable if _needs_replacement(asset))
    # A hub cache is incomplete when ANY declared payload file is absent, not only when its model
    # weights are: `hub_cache_layout_problem` answers the narrower question "can the loader resolve
    # MODEL weights here", so gating acquisition on it alone left a cache missing `config.json` or
    # `tokenizer.json` unrepaired -- the plan would schedule the download form and the download would
    # then no-op, exiting 0 with the cache still incomplete.
    hub_selected = tuple(asset for asset in hub_items
                         if asset.automatic_download_available
                         and (hub_cache_layout_problem(asset.hub_repo_id) is not None
                              or hub_cache_missing_files(asset)))
    repaired = tuple(asset for asset in selected if asset.destination.is_file())
    if args.dry_run:
        for asset in selected:
            action = "REPLACE corrupt" if asset in repaired else "download"
            print(f"DRY RUN: {action} {asset.source_url} -> {asset.destination}")
        for asset in hub_selected:
            # A hub asset's destination is a directory the loader resolves, not one URL, so the
            # planned action names the repo and the two things resolution needs: the revision its
            # ref must point at, and the payload files this run actually requires.
            print(f"DRY RUN: hub bootstrap {asset.hub_repo_id}@{asset.source_revision or 'default'} "
                  f"files={list(asset.hub_files or ['all'])} -> {hf_hub_repo_dir(asset.hub_repo_id)}")
        for asset in unavailable:
            if not asset.destination.is_file():
                print(f"{asset.family}: BLOCKED — required asset has no verified automatic acquisition source")
        print("DRY RUN: no network request and no filesystem mutation.")
        return 0
    for asset in hub_selected:
        ok, message = bootstrap_hub_asset(asset)
        if not ok:
            print(f"BLOCKED: {asset.family}: {message}", file=sys.stderr)
            return 1
    entries = manifest
    for asset in selected:
        assert asset.source_url is not None
        partial = asset.destination.with_suffix(asset.destination.suffix + ".part")
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        try:
            limit = asset.expected_size_bytes if type(asset.expected_size_bytes) is int and asset.expected_size_bytes > 0 else _DOWNLOAD_CEILING
            download(asset.source_url, partial, limit)
            if partial.stat().st_size == 0:
                raise RuntimeError("downloaded auxiliary asset is empty")
            digest = sha256_file(partial)
            if asset.expected_sha256 and digest != asset.expected_sha256:
                raise RuntimeError("official SHA256 mismatch")
            prefix = getattr(asset, "expected_sha256_prefix", None)
            if prefix and not digest.startswith(prefix):
                raise RuntimeError("official SHA256 prefix mismatch")
            os.replace(partial, asset.destination)
        except Exception as error:
            partial.unlink(missing_ok=True)
            print(f"BLOCKED: {asset.destination}: {error}", file=sys.stderr)
            return 1
        entries = record_download(entries, asset)
        save_manifest(manifest_path(config.WORKBENCH_ARTIFACTS_ROOT), entries)
        prefix = getattr(asset, "expected_sha256_prefix", None)
        hash_status = "official SHA256 verified" if asset.expected_sha256 else "official SHA256 prefix verified" if prefix else "local hash; official SHA256 unavailable"
        print(f"DOWNLOADED {asset.destination}: sha256={sha256_file(asset.destination)} ({hash_status})")
    if not selected:
        print("verified auxiliary downloads: complete")
    if args.stage_caption_aliases:
        alias_failed = False
        for asset in aliasable:
            try:
                ok, message = stage_caption_alias(
                    asset, fashioniq_root=config.FASHIONIQ_ROOT, managed=recorded_digest
                )
            except Exception as error:
                print(f"BLOCKED: {asset.destination}: {error}", file=sys.stderr)
                alias_failed = True
                continue
            print(message)
            if ok:
                entries = record_alias(entries, asset)
                save_manifest(manifest_path(config.WORKBENCH_ARTIFACTS_ROOT), entries)
            else:
                # A refusal is a failed staging operation, not a warning: the caller asked for a
                # variant and did not get one, so the run must not report success.
                print(f"BLOCKED: {message}", file=sys.stderr)
                alias_failed = True
        if alias_failed:
            return 1
    for asset in unavailable:
        if not asset.destination.is_file():
            print(f"BLOCKED: required {asset.family} has no verified automatic acquisition source: {asset.destination}")
    # A required asset that is still absent after staging is an unresolved requirement, and the
    # caller must see it as failure rather than as a completed run.
    if any(asset.required and not asset.destination.is_file() for asset in unavailable):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
