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
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination, auxiliary_model_ids, auxiliary_source_url, sha256_file

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
        )
        for model_id in model_ids
        for asset in auxiliary_assets_for_model(model_id)
    )


def manifest_path(repo_root: Path) -> Path:
    return repo_root / "workbench" / "artifacts" / "auxiliary" / "download_manifest.json"


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def report(items: tuple[Asset, ...]) -> None:
    for asset in items:
        state = "present" if asset.destination.is_file() else "missing"
        source = asset.source_url or "unverified author download source"
        print(
            f"{asset.family}: destination={asset.destination} state={state} "
            f"required={str(asset.required).lower()} "
            f"automatic_download_available={str(asset.automatic_download_available).lower()} "
            f"source_unverified={str(asset.provenance_status != 'VERIFIED').lower()} "
            f"source={source}"
        )


def download(url: str, partial: Path) -> None:
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            status = response.getcode()
            if status != 200:
                raise RuntimeError(f"unexpected HTTP status {status} for auxiliary asset download")
            shutil.copyfileobj(response, output, length=1024 * 1024)
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
            "downloaded_at": datetime.now(UTC).isoformat(),
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
        return 0 if all(asset.destination.is_file() for asset in items if asset.required) else 1
    selected = tuple(asset for asset in downloadable if args.force_redownload or not asset.destination.is_file())
    if args.dry_run:
        for asset in selected:
            print(f"DRY RUN: download {asset.source_url} -> {asset.destination}")
        for asset in unavailable:
            if not asset.destination.is_file():
                print(f"{asset.family}: BLOCKED — required asset has no verified automatic acquisition source")
        print("DRY RUN: no network request and no filesystem mutation.")
        return 0
    entries = load_manifest(manifest_path(config.CIR_REPO_ROOT))
    for asset in selected:
        assert asset.source_url is not None
        partial = asset.destination.with_suffix(asset.destination.suffix + ".part")
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        try:
            download(asset.source_url, partial)
            os.replace(partial, asset.destination)
        except Exception as error:
            partial.unlink(missing_ok=True)
            print(f"BLOCKED: {asset.destination}: {error}", file=sys.stderr)
            return 1
        entries = record_download(entries, asset)
        save_manifest(manifest_path(config.CIR_REPO_ROOT), entries)
        print(f"DOWNLOADED {asset.destination}: sha256={sha256_file(asset.destination)} (local hash; official hash unavailable)")
    if not selected:
        print("verified auxiliary downloads: complete")
    for asset in unavailable:
        if not asset.destination.is_file():
            print(f"BLOCKED: required {asset.family} has no verified automatic acquisition source: {asset.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
