#!/usr/bin/env python3
"""Validate a local FashionIQ layout and safely prepare CSMCIR's fixed link."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config

CATEGORIES = ("dress", "shirt", "toptee")


def required_layout(root: Path) -> tuple[Path, ...]:
    return (
        root / "captions",
        root / "image_splits",
        root / "images",
        *(root / "captions" / f"cap.{category}.val.json" for category in CATEGORIES),
        *(root / "image_splits" / f"split.{category}.val.json" for category in CATEGORIES),
    )


def validate_fashioniq(root: Path) -> list[str]:
    if not root.is_dir():
        return [f"dataset root missing: {root}"]
    return [f"required path missing: {path}" for path in required_layout(root) if not (path.is_dir() if path.suffix == "" else path.is_file())]



def prepare_csmcir(root: Path, source: Path, *, check_only: bool, dry_run: bool) -> bool:
    destination = source / "fashionIQ_dataset"
    if not source.is_dir():
        print(f"[BLOCKED] CSMCIR source missing: {source}")
        return False

    # Auxiliary files are runtime prerequisites. Link preparation must stay runnable
    # before their optional acquisition stage; strict doctor checks them before eval.

    if destination.is_symlink():
        try:
            linked_root = destination.resolve()
        except (OSError, RuntimeError) as error:
            print(f"[BLOCKED] CSMCIR dataset link cannot resolve: {destination}: {error}")
            return False
        if linked_root != root:
            print(f"[BLOCKED] CSMCIR dataset link resolves to {linked_root}, expected {root}")
            return False
        print(f"[OK] CSMCIR dataset link: {destination} -> {root}")
        return True
    if os.path.lexists(destination):
        print(f"[BLOCKED] CSMCIR dataset destination already exists: {destination}")
        return False
    if check_only:
        print(f"[WARN] CSMCIR dataset link absent: {destination} (check-only)")
        return True
    if dry_run:
        print(f"[OK] dry-run: would link {destination} -> {root}")
        return True
    destination.symlink_to(root, target_is_directory=True)
    print(f"[OK] CSMCIR dataset link created: {destination} -> {root}")
    return True


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ dataset root; defaults to configured FASHIONIQ_ROOT")
    parser.add_argument("--check-only", action="store_true", help="validate only; never create a link")
    parser.add_argument("--model", help="prepare fixed layout only for csmcir")
    parser.add_argument("--dry-run", action="store_true", help="show link creation without changing files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    config = resolve_config()
    dataset_root = args.dataset_root or config.FASHIONIQ_ROOT
    try:
        root = dataset_root.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        print(f"[BLOCKED] dataset root cannot resolve: {dataset_root}: {error}")
        return 1
    missing = validate_fashioniq(root)
    if missing:
        for message in missing:
            print(f"[BLOCKED] {message}")
        return 1
    print(f"[OK] FashionIQ base layout valid: {root}")

    if args.model == "csmcir":
        source = config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR"
        return 0 if prepare_csmcir(root, source, check_only=args.check_only, dry_run=args.dry_run) else 1
    if args.model:
        print(f"[WARN] {args.model}: layout preparation not verified")
    else:
        print("[WARN] no model selected: check only; layout preparation not verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
