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
    auxiliary_source_url, sha256_file, validate_auxiliary_asset,
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
        state = "present" if asset.destination.is_file() else "missing"
        source = asset.source_url or "unverified author download source"
        prefix = getattr(asset, "expected_sha256_prefix", None)
        source_unverified = asset.provenance_status != "VERIFIED" or (asset.expected_sha256 is None and prefix is None)
        print(
            f"{asset.family}: destination={asset.destination} state={state} "
            f"required={str(asset.required).lower()} "
            f"automatic_download_available={str(asset.automatic_download_available).lower()} "
            f"source_unverified={str(source_unverified).lower()} "
            f"source={source}"
        )

def verify(asset: Asset, *, fashioniq_root: Path | None = None) -> bool:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    config = resolve_config()
    model_ids = {args.model} if args.model else auxiliary_model_ids()
    items = assets(model_ids, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    report(items)
    downloadable = tuple(asset for asset in items if asset.automatic_download_available)
    unavailable = tuple(asset for asset in items if asset.required and not asset.automatic_download_available)
    if args.list:
        return 0
    if args.verify_only:
        failed = False
        for asset in items:
            if asset.required:
                failed |= not verify(asset, fashioniq_root=config.FASHIONIQ_ROOT)
        return int(failed)
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
    repaired = tuple(asset for asset in selected if asset.destination.is_file())
    if args.dry_run:
        for asset in selected:
            action = "REPLACE corrupt" if asset in repaired else "download"
            print(f"DRY RUN: {action} {asset.source_url} -> {asset.destination}")
        for asset in unavailable:
            if not asset.destination.is_file():
                print(f"{asset.family}: BLOCKED — required asset has no verified automatic acquisition source")
        print("DRY RUN: no network request and no filesystem mutation.")
        return 0
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
    for asset in unavailable:
        if not asset.destination.is_file():
            print(f"BLOCKED: required {asset.family} has no verified automatic acquisition source: {asset.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
