#!/usr/bin/env python3
"""Report CSMCIR auxiliary asset placement; automatic download is unavailable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config

MODEL = "csmcir"
SOURCE = "https://huggingface.co/peng12138/CSMCIR"
CATEGORIES = ("dress", "shirt", "toptee")


def expected_paths(third_party_root: Path, fashioniq_root: Path) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    source = third_party_root / "CSMCIR"
    cot = tuple(source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json" for category in CATEGORIES)
    qwen = tuple(fashioniq_root / "qwen_captions" / f"{category}_cot_val.json" for category in CATEGORIES)
    return source, cot, qwen


def report(third_party_root: Path, fashioniq_root: Path) -> bool:
    source, cot, qwen = expected_paths(third_party_root, fashioniq_root)
    print(f"Model: {MODEL}")
    print(f"Author source: {SOURCE}")
    print(f"CSMCIR source: {source}")
    complete = True
    for label, paths in (("CSMCIR COT", cot), ("FashionIQ Qwen", qwen)):
        for path in paths:
            state = "present" if path.is_file() else "missing"
            print(f"{label}: {path} [{state}]")
            complete = complete and state == "present"
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true")
    selector.add_argument("--model", choices=(MODEL,))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args(argv)
    config = resolve_config()
    complete = report(config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    if args.list:
        return 0
    if args.dry_run:
        print("DRY RUN: would download missing files from the author repository, but exact file URLs are unverified; manual placement required. No network request or filesystem mutation.")
        return 0
    if args.verify_only:
        return 0 if complete else 1
    if not complete:
        print("BLOCKED: exact auxiliary file URLs are unknown; place files manually at paths above. No network request made.", file=sys.stderr)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
