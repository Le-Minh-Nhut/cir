"""Source-faithful evaluation runner for CVPR25-TME.

Why a wrapper exists: the pinned `precompute_test.py` has a `main()` that reads a local checkpoint
path and prints a metrics dict, but it takes no `--checkpoint-root`, cannot serve more than one
category, and the adapter needs a command it can audit. This module supplies the missing invocation
shell and calls the pinned evaluation unchanged.

Two things about TME's own evaluator that the protocol depends on, both read from the pinned source
rather than assumed:

* The gallery is the per-category validation split in full, and the reference image is ELIGIBLE.
  `precompute_evaluation.py` masks the reference only when `category == 'CIRR'` -- the mask sits inside
  `if category == 'CIRR':` with the comment "only for cirr, delete the reference image from current
  query" -- so on FashionIQ the reference stays a candidate. That is why this model declares
  `fashioniq_val_split_ref_included` and not one of the reference-excluded protocols.
* FashionIQ's two annotations are joined as `"<first> and <second>"` after the pinned
  `strip('.?, ').capitalize()` / `strip('.?, ')` treatment, and the ranking is `argsort` over
  `1 - similarity`, i.e. descending similarity. Both are performed by the pinned code called below.

The native `load_state_dict(..., strict=False)` report is NOT discarded: it is checked, and a load with
missing or unexpected keys fails rather than printing plausible Recall from a partially initialised
model.
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import (
    ReplayResult,
    isolated_cache_root,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

MODEL = model_by_id("tme")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_CHECKPOINT = "fashioniq_0"
PROTOCOL = "fashioniq_val_split_ref_included"
DATASET_LABEL = "FashionIQ"
#: The pinned model name and type the author's own `precompute_test.py` selects for evaluation.
MODEL_NAME = "blip2_cir_image_diff_features"
MODEL_TYPE = "pretrain"

#: Inputs the pinned constructor opens. `data.py` reads `cap.<cat>.<split>.json` and
#: `split.<cat>.<split>.json` from the dataset root, and the gallery path needs the images.
RAW_INPUTS = tuple(
    path
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)


def source_root_for(checkout: Path) -> Path:
    """The directory that holds `src/`, for the vendored layout as delivered."""
    return checkout if (checkout / "src").is_dir() else checkout


def evaluator_path(checkout: Path) -> Path:
    return source_root_for(checkout) / "src" / "precompute_test.py"


def preflight(checkout: Path, dataset_root: Path, checkpoint_root: Path,
              checkpoint_id: str | None = None) -> list[str]:
    """Return blockers without importing upstream code or mutating inputs."""
    blockers: list[str] = []
    ok, reason = source_pin_status(checkout, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned TME source unavailable: {reason}")

    evaluator = evaluator_path(checkout)
    if not evaluator.is_file():
        blockers.append(f"pinned evaluator missing at {evaluator}")

    blockers.extend(
        f"TME dataset input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )

    variant = CHECKPOINTS.get(checkpoint_id or DEFAULT_CHECKPOINT)
    if variant is None:
        blockers.append(f"unsupported TME checkpoint: {checkpoint_id}; valid: {sorted(CHECKPOINTS)}")
        return blockers
    checkpoint = checkpoint_root / MODEL["model_id"] / variant["filename"]
    if not checkpoint.is_file():
        blockers.append(f"TME checkpoint missing: {checkpoint}")
    elif checkpoint.stat().st_size == 0:
        blockers.append(f"TME checkpoint is an empty placeholder: {checkpoint}")
    return blockers


def evaluate_categories(checkout: Path, dataset_root: Path, checkpoint_root: Path,
                        categories: tuple[str, ...] = CATEGORIES,
                        checkpoint_id: str | None = None) -> tuple[list[ReplayResult], dict[str, Any]]:
    """Run the pinned evaluation for each category and collect the native metrics."""
    import torch

    variant = CHECKPOINTS.get(checkpoint_id or DEFAULT_CHECKPOINT)
    if variant is None:
        raise ValueError(f"unsupported TME checkpoint: {checkpoint_id}")
    checkpoint_path = checkpoint_root / MODEL["model_id"] / variant["filename"]

    source = source_root_for(checkout)
    sys.path.insert(0, str(source / "src"))
    # The pinned `data.py` reads these two module-level constants, so the dataset root is pointed at
    # the operator's canonical FashionIQ root rather than a path baked into the vendored file.
    os.environ["MTST_DATA_ROOT"] = str(dataset_root.parent)

    from lavis.models import load_model_and_preprocess
    import data as tme_data
    from precompute_evaluation import evaluate_features
    from utility import targetpad_transform

    tme_data.FashioniqPath = str(dataset_root) + "/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=False, device=device)

    checkpoint_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    report = blip_model.load_state_dict(checkpoint_state, strict=False)
    missing = [key for key in getattr(report, "missing_keys", [])]
    unexpected = [key for key in getattr(report, "unexpected_keys", [])]
    if missing or unexpected:
        # The native call is `strict=False`, which would hide a partially initialised model behind
        # plausible-looking Recall. Refuse instead, and say what it hid.
        raise RuntimeError(
            f"TME checkpoint did not load completely: {len(missing)} missing, {len(unexpected)} "
            f"unexpected (missing: {missing[:3]}, unexpected: {unexpected[:3]}); the native "
            "`strict=False` would have hidden this")
    blip_model.eval()

    preprocess = targetpad_transform(target_ratio=1.25, dim=224)
    results: list[ReplayResult] = []
    with isolated_cache_root(dataset_root, directory_names=("captions", "image_splits")) as cache_root:
        tme_data.FashioniqPath = str(cache_root).rstrip("/") + "/"
        for category in categories:
            dataset = tme_data.get_dataset(DATASET_LABEL, preprocess, "val",
                                           specific_dress_type=category)
            if hasattr(dataset, "load_image_features"):
                dataset.load_image_features()
            metrics = evaluate_features(blip_model, dataset, txt_processors["eval"])
            results.append(ReplayResult(
                model_id="tme",
                checkpoint_id=variant["checkpoint_id"],
                category=category,
                protocol=PROTOCOL,
                r1=metrics.get("recall_at1"),
                r10=metrics.get("recall_at10"),
                r50=metrics.get("recall_at50"),
                complete=all(metrics.get(k) is not None
                             for k in ("recall_at1", "recall_at10", "recall_at50")),
                provenance={
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "source_commit": MODEL["upstream_commit_sha"],
                    "checkpoint_training_noise_pct": variant.get("checkpoint_training_noise_pct"),
                    "reference_policy": "reference_included",
                    "checkpoint_selection_rule": "best_validation_checkpoint_at_sigma_0",
                },
            ))
    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    selected = tuple(args.categories) if args.categories else CATEGORIES
    blockers = preflight(args.source_root, args.dataset_root, args.checkpoint_root, args.checkpoint_id)
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0

    results, aggregate = evaluate_categories(
        args.source_root, args.dataset_root, args.checkpoint_root,
        categories=selected, checkpoint_id=args.checkpoint_id)
    payload = {
        "model_id": "tme",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
