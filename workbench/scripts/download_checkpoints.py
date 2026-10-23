#!/usr/bin/env python3
"""Download only registry-declared model checkpoints. Run --dry-run first."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_path, load_registry, sha256_file


def selected(args: argparse.Namespace, models: list[dict]) -> list[tuple[dict, dict]]:
    if args.model:
        models = [item for item in models if item["model_id"] == args.model]
        if not models:
            raise ValueError(f"unknown model: {args.model}")
    requested = [args.checkpoint] if isinstance(args.checkpoint, str) else list(args.checkpoint or [])
    records: list[tuple[dict, dict]] = []
    for model in models:
        mid = model["model_id"]
        variants = model.get("checkpoint_variants", [])
        variant_by_id = {item["checkpoint_id"]: item for item in variants}
        bundles = {b["bundle_id"]: b for b in model.get("checkpoint_bundles", [])}
        if requested:
            if args.model:
                unknown = [cid for cid in requested if cid not in variant_by_id and cid not in bundles]
                if unknown:
                    raise ValueError(f"unknown checkpoint for {args.model}: {', '.join(unknown)}")
            selected_variants: list[dict] = []
            for req in requested:
                if req in bundles:
                    bundle = bundles[req]
                    for req_id in bundle.get("required_checkpoint_ids", []):
                        if req_id in variant_by_id:
                            if variant_by_id[req_id] not in selected_variants:
                                selected_variants.append(variant_by_id[req_id])
                        else:
                            raise ValueError(f"unknown checkpoint bundle member for {mid}: {req_id}")
                elif req in variant_by_id:
                    if variant_by_id[req] not in selected_variants:
                        selected_variants.append(variant_by_id[req])
            if args.model and not selected_variants:
                raise ValueError(f"unknown checkpoint for {args.model}: {', '.join(requested)}")
            if mid == "limn":
                bundle = next((b for b in model.get("checkpoint_bundles", []) if b.get("bundle_id") == "base_iter0_all_categories"), None)
                if bundle:
                    req_bundle_ids = set(bundle.get("required_checkpoint_ids", []))
                    if any(req == bundle.get("bundle_id") or req in req_bundle_ids for req in requested):
                        unrelated = [v["checkpoint_id"] for v in selected_variants if v["checkpoint_id"] not in req_bundle_ids]
                        if unrelated:
                            raise ValueError(f"unrelated variant in LIMN bundle selection: {', '.join(unrelated)}")
            variants = selected_variants
        else:
            if len(variants) > 1 and mid == "pair" and not (getattr(args, "all", False) or getattr(args, "list", False)):
                print("PAIR has unresolved variants; select --checkpoint pair_b1 or pair_b2.", file=sys.stderr)
                continue
        records.extend((model, checkpoint) for checkpoint in variants)
    if requested and not records:
        raise ValueError(f"unknown checkpoint: {', '.join(requested)}")
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


def _archive_member(checkpoint: dict) -> str | None:
    name = checkpoint.get("archive_member")
    if name is None:
        return None
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name):
        raise ValueError(f"archive_member must be a basename: {name!r}")
    return name


def verify(model: dict, checkpoint: dict, destination: Path) -> bool:
    """Return whether local extracted bytes are intact; provenance is reported separately."""
    member = _archive_member(checkpoint)
    if not destination.is_file():
        print(f"MISSING {destination}")
        return False
    if member and tarfile.is_tarfile(destination):
        print(f"INVALID ARCHIVE CONTAINER (expected extracted member {member}) {destination}")
        return False
    size = destination.stat().st_size
    if size == 0:
        print(f"INVALID EMPTY FILE {destination}")
        return False
    digest = sha256_file(destination)
    expected_size = checkpoint.get("expected_size_bytes")
    size_valid = (type(expected_size) is not int or expected_size <= 0
                  or size == expected_size)
    expected = checkpoint.get("expected_sha256")
    if expected is None:
        status = "LOCAL SHA256 COMPUTED \u2014 official SHA256 unavailable"
        hash_valid = True
    elif digest == expected:
        status = "OFFICIAL SHA256 MATCH"
        hash_valid = True
    else:
        status = "OFFICIAL SHA256 MISMATCH"
        hash_valid = False
    provenance_valid = checkpoint.get("checkpoint_mapping_status") == "VERIFIED_METADATA" and checkpoint.get("download_url") is not None
    if not provenance_valid:
        status += " \u2014 REQUIRED PROVENANCE UNRESOLVED"
    if not size_valid:
        status += f" - EXPECTED SIZE MISMATCH (expected {expected_size}, got {size})"
    print(f"{destination}: {size} bytes sha256={digest} {status}")
    return size_valid and hash_valid and (provenance_valid or member is not None)

#: Upper bound on any single download. For archives, the container is streamed and its declared
#: archive_size_bytes is authoritative; otherwise this ceiling stops runaway responses.
#: For ordinary checkpoints, expected_size_bytes remains the streamed size contract.
_DOWNLOAD_CEILING = 8 * 1024 * 1024 * 1024


def _write_response(response, partial: Path, mode: str, limit: int) -> None:
    """Stream to `partial`, refusing to write more than `limit` bytes IN TOTAL.

    The registry records a checkpoint's exact size. Enforcing it while the bytes arrive means an
    oversized response (a changed upstream file, a redirect to an error page, a hostile mirror) is
    rejected with a clean error instead of being written to disk and only then compared.

    `mode == "ab"` is a resume, so the bytes already on disk count toward the limit. Budgeting only
    the newly received chunks let a resumed transfer finish above the declared size: the earlier
    attempt's bytes were invisible, so a response could add a full `limit` on top of them.
    """
    written = partial.stat().st_size if mode == "ab" and partial.exists() else 0
    if written > limit:
        raise RuntimeError(
            f"resumed file already exceeds the declared size limit of {limit} bytes: {partial}"
        )
    with partial.open(mode) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                return
            written += len(chunk)
            if written > limit:
                raise RuntimeError(
                    f"response exceeded the declared size limit of {limit} bytes; "
                    f"refusing to continue writing {partial}"
                )
            output.write(chunk)


def _valid_content_range(value: str | None, offset: int) -> bool:
    match = re.fullmatch(r"bytes (\d+)-\d+/\d+|bytes (\d+)-\d+/\*", value or "")
    return match is not None and int(match.group(1) or match.group(2)) == offset


def download(url: str, partial: Path, limit: int = _DOWNLOAD_CEILING) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset:
            partial.unlink(missing_ok=True)
            return download(url, partial, limit)
        raise
    with response:
        status = response.getcode()
        if offset and status == 206 and _valid_content_range(response.headers.get("Content-Range"), offset):
            _write_response(response, partial, "ab", limit)
            return
        if status == 200:
            _write_response(response, partial, "wb", limit)
            return
        if offset and status == 206:
            raise RuntimeError(f"server returned incompatible Content-Range for resume: {response.headers.get('Content-Range')!r}")
        raise RuntimeError(f"unexpected HTTP status {status} for checkpoint download")


def _extract_member(partial: Path, member_name: str, temporary: Path) -> None:
    """Copy only the exact regular-file member named by the registry."""
    with tarfile.open(partial, mode="r:*") as archive:
        matches = [member for member in archive.getmembers() if member.name == member_name]
        if len(matches) != 1 or not matches[0].isfile():
            raise RuntimeError(f"archive must contain exactly one regular member named {member_name!r}")
        source = archive.extractfile(matches[0])
        if source is None:
            raise RuntimeError(f"cannot read archive member {member_name!r}")
        with source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list registry checkpoint records")
    group.add_argument("--model", help="select one model")
    group.add_argument("--all", action="store_true", help="select all downloadable model records")
    parser.add_argument("--checkpoint", action="append", help="select checkpoint variant(s); repeatable for multi-checkpoint bundles")
    parser.add_argument("--dry-run", action="store_true", help="print planned operations; never access network")
    parser.add_argument("--verify-only", action="store_true", help="verify local files only; never access network")
    parser.add_argument("--force-redownload", action="store_true", help="replace existing local file after explicit request")
    parser.add_argument("--output-root", type=Path, help="checkpoint destination; defaults to resolved config")
    args = parser.parse_args(argv)
    args.output_root = args.output_root or resolve_config().WORKBENCH_CHECKPOINT_ROOT
    try:
        items = selected(args, load_registry()["models"])
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for model, checkpoint in items:
            describe(model, checkpoint, checkpoint_path(model["model_id"], checkpoint, args.output_root))
            print(f"Status: {checkpoint['status']}\n")
        return 0
    manifest_path = args.output_root / "download_manifest.json"
    manifest = [] if args.verify_only else load_manifest(manifest_path)
    failed = False
    for model, checkpoint in items:
        destination = checkpoint_path(model["model_id"], checkpoint, args.output_root)
        if args.all and checkpoint["checkpoint_mapping_status"] != "VERIFIED_METADATA":
            print(f"SKIPPED: unresolved checkpoint mapping ({model['model_id']} / {checkpoint['checkpoint_id']})")
            continue
        if args.all and checkpoint.get("download_url") is None:
            print(f"SKIPPED: direct official download URL unavailable ({model['model_id']} / {checkpoint['checkpoint_id']})")
            continue
        describe(model, checkpoint, destination)
        if args.verify_only:
            failed |= not verify(model, checkpoint, destination)
            continue
        if checkpoint.get("download_url") is None:
            print("BLOCKED: direct official download URL unavailable; registry remains unmodified.", file=sys.stderr)
            failed = True
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
        archive_member = _archive_member(checkpoint)
        declared_size = checkpoint.get("archive_size_bytes") if archive_member else checkpoint.get("expected_size_bytes")
        limit = declared_size if type(declared_size) is int and declared_size > 0 else _DOWNLOAD_CEILING
        download(checkpoint["download_url"], partial, limit)
        container_size = partial.stat().st_size
        if type(declared_size) is int and declared_size > 0 and container_size != declared_size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"download size mismatch for {destination}: expected {declared_size} bytes, got {container_size}")
        archive_digest = sha256_file(partial) if archive_member else None
        temporary = None
        try:
            if archive_member:
                with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                _extract_member(partial, archive_member, temporary)
                publish_path = temporary
            else:
                publish_path = partial
            size = publish_path.stat().st_size
            expected_size = checkpoint.get("expected_size_bytes")
            if type(expected_size) is int and expected_size > 0 and size != expected_size:
                raise RuntimeError(f"size mismatch for {destination}: expected {expected_size} bytes, got {size}")
            digest = sha256_file(publish_path)
            expected = checkpoint.get("expected_sha256")
            if expected and digest != expected:
                raise RuntimeError(f"sha256 mismatch for {destination}: expected {expected}, got {digest}")
            os.replace(publish_path, destination)
            temporary = None
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            raise
        if archive_member:
            partial.unlink(missing_ok=True)
        manifest = [entry for entry in manifest if not (entry["model_id"] == model["model_id"] and entry["checkpoint_id"] == checkpoint["checkpoint_id"])]
        if archive_member:
            manifest.append({"model_id": model["model_id"], "checkpoint_id": checkpoint["checkpoint_id"],
                             "filename": checkpoint["filename"], "source_url": checkpoint["download_url"],
                             "downloaded_compressed_archive_size_bytes": container_size,
                             "local_compressed_archive_sha256": archive_digest,
                             "extracted_size_bytes": size, "local_extracted_sha256": digest,
                             "expected_extracted_sha256_match": digest == expected if expected else None,
                             "ctime_ns": destination.stat().st_ctime_ns,
                             "downloaded_at": datetime.now(timezone.utc).isoformat()})
        else:
            manifest.append({"model_id": model["model_id"], "checkpoint_id": checkpoint["checkpoint_id"], "filename": checkpoint["filename"], "source_url": checkpoint["download_url"], "downloaded_sha256": digest, "size_bytes": size,
                         # The inode change time cannot be set backwards, so it is the cheap,
                         # un-forgeable half of the presence check's evidence (see
                         # registry.checkpoint_missing_paths).
                         "ctime_ns": destination.stat().st_ctime_ns, "downloaded_at": datetime.now(timezone.utc).isoformat(), "official_hash_available": expected is not None, "official_hash_match": digest == expected if expected else None})
        save_manifest(manifest_path, manifest)
        print(f"DOWNLOADED {destination}: {size} bytes sha256={digest}\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
