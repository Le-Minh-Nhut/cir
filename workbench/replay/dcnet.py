"""Preflight validation and replay runner for DCNet.

DCNet evaluation relies on the official CLI `test.py --resume <run_dir>`.
This module provides preflight verification of all required artifacts and identifies
the exact missing prerequisites (unpublished Google Drive checkpoint and unseeded
offline GloVe preprocessed pickles).
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import source_pin_status

MODEL = model_by_id("dcnet")
REQUIRED_RUN_FILES = ("config.json", "trained_model.pth")
DATASET_LABEL = "fashioniq"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"DCNet source pin invalid: {reason}"]


def validate_dataset_binding(checkpoint_root: Path, dataset_root: Path) -> list[str]:
    """The evaluator opens the data root its own config names, not the one the caller passed.

    `test.py --resume <run_dir>` reads `<run_dir>/config.json`, and the native loaders take their
    path from `data_loader.args.data_dir` (`parse_config.py:18-28`), which the shipped configs set to
    the relative literal `dataset/fashioniq`. That literal resolves against the evaluator's CWD, so a
    run can pass every other check and still read a *different* tree than the one whose provenance was
    verified — silently, because nothing compares the two.

    This reports the mismatch instead of editing the author's config: the declaration stays as
    published, and the operator is told exactly which path the evaluator will open and what to do.
    """
    blockers: list[str] = []
    config_file = checkpoint_root / "config.json"
    if not config_file.is_file() or config_file.stat().st_size == 0:
        return blockers  # already reported by validate_run_directory
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        blockers.append(f"DCNet config unreadable: {config_file}: {error}")
        return blockers

    declared = (config.get("data_loader") or {}).get("args", {}).get("data_dir")
    if not declared:
        blockers.append(
            f"DCNet config declares no data_loader.args.data_dir: {config_file}. The evaluator "
            "resolves its dataset from that field, so the tree it reads cannot be verified.")
        return blockers

    declared_path = Path(declared)
    if not declared_path.is_absolute():
        # Relative literals resolve against the CWD the evaluator runs in, which is the source
        # checkout (the adapter launches `test.py` from there).
        blockers.append(
            f"DCNet config data_dir is a relative path ({declared!r}), so the evaluator reads it "
            f"relative to its own working directory, not the --dataset-root {dataset_root} the "
            "operator passed. Bind it absolutely (or run the evaluator from the directory the "
            "literal was written for) before trusting which tree was evaluated.")
    return blockers

def validate_run_directory(checkpoint_root: Path) -> list[str]:
    blockers: list[str] = []
    if not checkpoint_root.is_dir():
        blockers.append(f"DCNet run directory missing: {checkpoint_root}")
        return blockers
    for filename in REQUIRED_RUN_FILES:
        target = checkpoint_root / filename
        if not target.is_file():
            blockers.append(f"DCNet run directory requirement missing: {target} (unpublished author checkpoint)")
        elif target.stat().st_size == 0:
            blockers.append(f"DCNet run directory artifact is empty placeholder: {target}")
    return blockers


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    if not (dataset_root / "resized_images").is_dir():
        blockers.append(f"DCNet dataset requirement missing: {dataset_root / 'resized_images'} (expected flat resized_images/ directory)")

    for category in categories:
        split_path = dataset_root / "image_splits" / f"split.{category}.val.json"
        if not split_path.is_file():
            blockers.append(f"DCNet dataset requirement missing: {split_path}")

        glove_path = dataset_root / "captions" / f"cap.{category}.glove.val.pkl"
        if not glove_path.is_file():
            blockers.append(
                f"DCNet preprocessed GloVe feature missing: {glove_path} "
                "(unpublished artifact; raw reconstruction requires deterministic OOV variant)"
            )
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_run_directory(checkpoint_root)
        + validate_dataset_binding(checkpoint_root, dataset_root)
        + validate_dataset(dataset_root, categories)
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Path to run directory containing config.json and trained_model.pth")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--preflight-only", action="store_true", default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1

    print("[READY] DCNet preflight passed; all run directory and dataset artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
