#!/usr/bin/env python3
"""Download only registry-declared model checkpoints. Run --dry-run first."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.registry import CHECKPOINT_ROOT, checkpoint_path, load_registry, sha256_file


def selected(args: argparse.Namespace, models: list[dict]) -> list[tuple[dict, dict]]:
    if args.model:
        models = [item for item in models if item["model_id"] == args.model]
        if not models:
            raise ValueError(f"unknown model: {args.model}")
    records: list[tuple[dict, dict]] = []
    for model in models:
        variants = model["checkpoint_variants"]
        if args.checkpoint:
            variants = [item for item in variants if item["checkpoint_id"] == args.checkpoint]
            if not variants and args.model:
                raise ValueError(f"unknown checkpoint for {args.model}: {args.checkpoint}")
        if len(variants) > 1 and model["model_id"] == "pair" and not args.all:
            print("PAIR has unresolved variants; select --checkpoint pair_b1 or pair_b2.", file=sys.stderr)
            continue
        records.extend((model, checkpoint) for checkpoint in variants)
    return records


def describe(model: dict, checkpoint: dict, destination: Path) -> None:
    noise = checkpoint["checkpoint_training_noise_pct"]
    print(f"Model: {model['method_name']} ({model['model_id']})")
    print(f"Checkpoint: {checkpoint['checkpoint_id']} / {checkpoint['filename']}")
    print(f"Source URL: {checkpoint.get('download_url') or checkpoint.get('source_url') or 'unavailable'}")
    print(f"Checkpoint training noise: {f'{noise}%' if noise is not None else 'unknown'}")
    print("Evaluation policy: clean / 0%")
    print(f"FashionIQ protocol: {model.get('native_protocol') or 'unavailable'}")
    print(f"Destination: {destination}")


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def save_manifest(path: Path, manifest: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def verify(model: dict, checkpoint: dict, destination: Path) -> bool:
    if not destination.is_file():
        print(f"MISSING {destination}")
        return False
    digest = sha256_file(destination)
    expected = checkpoint.get("expected_sha256")
    valid = expected is None or digest == expected
    print(f"{destination}: {destination.stat().st_size} bytes sha256={digest} {'OK' if valid else 'MISMATCH'}")
    return valid


def _write_response(response, partial: Path, mode: str) -> None:
    with partial.open(mode) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _valid_content_range(value: str | None, offset: int) -> bool:
    match = re.fullmatch(r"bytes (\d+)-\d+/\d+|bytes (\d+)-\d+/\*", value or "")
    return match is not None and int(match.group(1) or match.group(2)) == offset


def download(url: str, partial: Path) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        status = response.getcode()
        if offset and status == 206 and _valid_content_range(response.headers.get("Content-Range"), offset):
            _write_response(response, partial, "ab")
            return
        if status == 200:
            _write_response(response, partial, "wb")
            return
        if offset and status == 206:
            raise RuntimeError(f"server returned incompatible Content-Range for resume: {response.headers.get('Content-Range')!r}")
        raise RuntimeError(f"unexpected HTTP status {status} for checkpoint download")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list registry checkpoint records")
    group.add_argument("--model", help="select one model")
    group.add_argument("--all", action="store_true", help="select all downloadable model records")
    parser.add_argument("--checkpoint", help="select one checkpoint variant; required for unresolved PAIR variants")
    parser.add_argument("--dry-run", action="store_true", help="print planned operations; never access network")
    parser.add_argument("--verify-only", action="store_true", help="verify local files only; never access network")
    parser.add_argument("--force-redownload", action="store_true", help="replace existing local file after explicit request")
    parser.add_argument("--output-root", type=Path, default=CHECKPOINT_ROOT)
    args = parser.parse_args()
    try:
        items = selected(args, load_registry()["models"])
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for model, checkpoint in items:
            describe(model, checkpoint, checkpoint_path(model["model_id"], checkpoint, args.output_root))
            print(f"Status: {checkpoint['status']}\n")
        return
    manifest_path = args.output_root / "download_manifest.json"
    manifest = load_manifest(manifest_path)
    for model, checkpoint in items:
        destination = checkpoint_path(model["model_id"], checkpoint, args.output_root)
        describe(model, checkpoint, destination)
        if args.verify_only:
            verify(model, checkpoint, destination)
            continue
        if checkpoint.get("download_url") is None:
            print("BLOCKED: direct official download URL unavailable; registry remains unmodified.", file=sys.stderr)
            continue
        if destination.exists() and not args.force_redownload:
            if verify(model, checkpoint, destination):
                continue
            raise RuntimeError(f"refusing overwrite of {destination}; use --force-redownload")
        if args.dry_run:
            print("DRY RUN: no network request and no filesystem mutation.\n")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if args.force_redownload:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
        download(checkpoint["download_url"], partial)
        os.replace(partial, destination)
        digest = sha256_file(destination)
        expected = checkpoint.get("expected_sha256")
        if expected and digest != expected:
            raise RuntimeError(f"sha256 mismatch for {destination}")
        manifest = [entry for entry in manifest if not (entry["model_id"] == model["model_id"] and entry["checkpoint_id"] == checkpoint["checkpoint_id"])]
        manifest.append({"model_id": model["model_id"], "checkpoint_id": checkpoint["checkpoint_id"], "filename": checkpoint["filename"], "source_url": checkpoint["download_url"], "downloaded_sha256": digest, "size_bytes": destination.stat().st_size, "downloaded_at": datetime.now(UTC).isoformat(), "official_hash_available": expected is not None, "official_hash_match": digest == expected if expected else None})
        save_manifest(manifest_path, manifest)
        print(f"DOWNLOADED {destination}: {destination.stat().st_size} bytes sha256={digest}\n")


if __name__ == "__main__":
    main()
