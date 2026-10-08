"""Read-only preflight for the pinned TG-CIR FashionIQ evaluator.

Checkpoint mapping and author hash are unresolved, so this module never loads
checkpoint data, constructs replacement weights, initializes CLIP, or runs inference.
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import source_pin_status


MODEL = model_by_id("tgcir")
RAW_INPUTS = tuple(
    Path(path)
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.train.json",
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)
CORRECTION_DICTIONARIES = tuple(Path(f"captions/correction_dict_{category}.json") for category in CATEGORIES)
RESIZED_IMAGE_DIRS = tuple(Path(f"resized_image/{category}") for category in CATEGORIES)
NATIVE_CACHES = (
    Path("fashion_iq_data.json"),
    *(Path(f"test_{kind}_{category}.pkl") for category in CATEGORIES for kind in ("queries", "targets")),
)


def preflight(source_root: Path, dataset_root: Path, checkpoint_root: Path) -> list[str]:
    """Return all known blockers without importing upstream code or mutating inputs."""
    blockers = []
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    if not ok:
        blockers.append(f"pinned TG-CIR source unavailable: {reason}")

    blockers.extend(f"FashionIQ raw input missing: {dataset_root / path}" for path in RAW_INPUTS if not (dataset_root / path).is_file())
    blockers.extend(
        f"TG-CIR resized-image directory missing: {dataset_root / path}"
        for path in RESIZED_IMAGE_DIRS
        if not (dataset_root / path).is_dir()
    )
    blockers.extend(
        f"pinned-source correction dictionary missing: {dataset_root / path}"
        for path in CORRECTION_DICTIONARIES
        if not (dataset_root / path).is_file()
    )

    missing_caches = [dataset_root / path for path in NATIVE_CACHES if not (dataset_root / path).is_file()]
    if missing_caches:
        blockers.extend(f"TG-CIR native cache missing: {path}" for path in missing_caches)
    else:
        blockers.append(
            "TG-CIR native cache freshness/content is unverified; pinned FashionIQ constructor trusts these files"
        )

    checkpoint = MODEL["checkpoint_variants"][0]
    candidate = checkpoint_root / checkpoint["filename"]
    if not candidate.is_file() or candidate.stat().st_size == 0:
        blockers.append(f"TG-CIR checkpoint missing or empty: {candidate}")
    blockers.append(
        f"checkpoint mapping unresolved: {checkpoint['source_url']} does not verify FashionIQ archive member "
        f"for {candidate}; checkpoint_mapping_status={checkpoint['checkpoint_mapping_status']}"
    )
    blockers.append(
        "author-verified checkpoint SHA-256 is unavailable; whole-model pickle loading is forbidden"
    )
    blockers.append(
        "OpenAI CLIP ViT-B/16 runtime asset/cache provenance and local availability are unverified"
    )
    return blockers


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args(argv)

    blockers = preflight(args.source_root, args.dataset_root, args.checkpoint_root)
    for blocker in blockers:
        print(f"[BLOCKED] {blocker}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
