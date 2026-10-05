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
from workbench.backend.registry import sha256_file

MODEL = "csmcir"
AUTHOR_REPOSITORY = "https://huggingface.co/peng12138/CSMCIR"
CATEGORIES = ("dress", "shirt", "toptee")


@dataclass(frozen=True)
class Asset:
    family: str
    filename: str
    destination: Path
    required: bool
    automatic_download_available: bool
    source_url: str | None
    source_unverified: bool = False


def assets(third_party_root: Path, fashioniq_root: Path) -> tuple[Asset, ...]:
    qwen = tuple(
        Asset(
            "Qwen captions",
            f"{category}_cot_val.json",
            fashioniq_root / "qwen_captions" / f"{category}_cot_val.json",
            True,
            True,
            f"{AUTHOR_REPOSITORY}/resolve/main/fashioniq_qwen_captions/qwen_captions/{category}_cot_val.json",
        )
        for category in CATEGORIES
    )
    cot = tuple(
        Asset(
            "COT_ours2 captions",
            f"{category}_cot_val.json",
            third_party_root / "CSMCIR" / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json",
            True,
            False,
            None,
            True,
        )
        for category in CATEGORIES
    )
    return (*qwen, *cot)


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
            f"source_unverified={str(asset.source_unverified).lower()} "
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
            "model_id": MODEL,
            "asset_family": asset.family,
            "filename": asset.filename,
            "destination": str(asset.destination),
            "source_url": asset.source_url,
            "downloaded_sha256": digest,
            "size_bytes": asset.destination.stat().st_size,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "official_hash_available": False,
        }
    )
    return remaining


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list required auxiliary assets")
    selector.add_argument("--model", choices=(MODEL,), help="select CSMCIR")
    parser.add_argument("--dry-run", action="store_true", help="show exact URLs and destinations without changes")
    parser.add_argument("--verify-only", action="store_true", help="verify all required local auxiliary assets without network")
    parser.add_argument("--force-redownload", action="store_true", help="replace verified-download destinations atomically")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    config = resolve_config()
    items = assets(config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    report(items)
    qwen = tuple(asset for asset in items if asset.automatic_download_available)
    cot = tuple(asset for asset in items if not asset.automatic_download_available)
    if args.list:
        return 0
    if args.verify_only:
        return 0 if all(asset.destination.is_file() for asset in items if asset.required) else 1
    selected = tuple(asset for asset in qwen if args.force_redownload or not asset.destination.is_file())
    if args.dry_run:
        for asset in selected:
            print(f"DRY RUN: download {asset.source_url} -> {asset.destination}")
        print("COT_ours2 captions: BLOCKED — official download source not yet verified")
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
        print("Qwen captions: READY")
    elif all(asset.destination.is_file() for asset in qwen):
        print("Qwen captions: READY")
    if any(not asset.destination.is_file() for asset in cot):
        print("COT_ours2 captions: BLOCKED — official download source not yet verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
