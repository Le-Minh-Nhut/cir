"""Read-only preflight for the pinned PTHA+MTST FashionIQ evaluator.

Author-published weights on HuggingFace (`yinanzhou1/mtst_ptha`) provide only CIRR
fine-tuned weights (`tuned_mtst_cirr.pt`). No FashionIQ checkpoint is published by the
authors. Therefore, this module acts as a strict preflight reporting the missing
checkpoint without loading untrusted weights or running inference.
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import source_pin_status

MODEL = model_by_id("ptha_mtst")
RAW_INPUTS = tuple(
    Path(path)
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)
HF_REPO = "yinanzhou1/mtst_ptha"
PUBLISHED_CIRR_WEIGHT = "tuned_mtst_cirr.pt"


def preflight(source_root: Path, dataset_root: Path, checkpoint_root: Path) -> list[str]:
    """Return all known blockers without importing upstream code or mutating inputs."""
    blockers: list[str] = []
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned PTHA+MTST source unavailable: {reason}")

    # Pinned entrypoint check
    evaluator_path = source_root / "MTST_PTHA" / "src" / "blip_validate.py"
    if not evaluator_path.is_file():
        blockers.append(f"pinned evaluator missing at {evaluator_path}")

    # Dataset checks
    blockers.extend(
        f"FashionIQ raw input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )

    # Checkpoint checks: HF repo yinanzhou1/mtst_ptha only published CIRR checkpoint
    blockers.append(
        f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
        f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
        f"('{PUBLISHED_CIRR_WEIGHT}'), no FashionIQ checkpoint is published"
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
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
