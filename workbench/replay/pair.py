"""Source-faithful replay for the PAIR FashionIQ evaluator.

Why a wrapper exists: the pinned `test.py` defines `test(params, model, testset, category)`
and nothing else — no CLI, no `__main__`, and no checkpoint loader. This module supplies the
missing orchestration shell and calls pinned `test.test(params, model, testset, category)`
without altering ranking, metric arithmetic, gallery construction, or feature extraction.

Exact attributes read by pinned `test.py:6-83`:
- `params.local_rank`: int (lines 26, 43; disables tqdm progress bar when nonzero)
- `params.batch_size`: int (lines 29, 45; batch accumulation threshold)
- `model.eval()`: method (line 7)
- `model.extract_retrieval_compose(imgs, mods)`: method (line 33)
- `model.extract_retrieval_target(imgs)`: method (line 49)
- `testset.test_queries_<cat>` and `testset.test_targets_<cat>`: lists (lines 10-15)
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import (
    ReplayResult,
    isolated_cache_root,
    load_trusted_whole_model,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

MODEL = model_by_id("pair")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_CHECKPOINT = MODEL["checkpoint_variants"][0]
PROTOCOL = "fashioniq_val_split"


def validate_checkpoint(checkpoint_root: Path, checkpoint_id: str | None = None) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(checkpoint_id) if checkpoint_id else DEFAULT_CHECKPOINT
    if variant is None:
        return [f"PAIR checkpoint variant unknown: {checkpoint_id}"]
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"PAIR checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"PAIR checkpoint is an empty placeholder: {path}")
        return blockers
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(
            f"PAIR checkpoint mapping is unverified: {variant['filename']} is not mapped to "
            f"FashionIQ category or split (status={variant.get('checkpoint_mapping_status')})"
        )
    expected_sha = variant.get("expected_sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(
                f"PAIR checkpoint sha256 mismatch: expected {expected_sha}, found {actual_sha}"
            )
    expected_size = variant.get("expected_size_bytes")
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(
            f"PAIR checkpoint size mismatch: expected {expected_size}, found {path.stat().st_size}"
        )
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"PAIR source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"PAIR dataset input missing: {dataset_root / relative}")
        cat_resized = dataset_root / "resized_image" / category
        if not cat_resized.is_dir():
            blockers.append(f"PAIR dataset input missing: {cat_resized}")
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    checkpoint_id: str | None = None,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_checkpoint(checkpoint_root, checkpoint_id)
        + validate_dataset(dataset_root, categories)
    )


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any, Any]:
    import open_clip
    import torch

    for entry in (str(source_root), str(source_root / "data"), str(source_root / "models"), str(source_root / "utils")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    datasets = __import__("datasets")
    test_module = __import__("test")
    return torch, open_clip, datasets, test_module


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    checkpoint_id: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, open_clip, datasets, test_module = _load_replay_modules(source_root)

    variant = CHECKPOINTS.get(checkpoint_id) if checkpoint_id else DEFAULT_CHECKPOINT
    if variant is None:
        raise ValueError(f"unknown checkpoint variant: {checkpoint_id}")
    checkpoint_path = checkpoint_root / variant["filename"]
    model = load_trusted_whole_model(checkpoint_path, variant.get("expected_sha256"), torch)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32")
    transform = [preprocess_train, preprocess_val]

    results: list[ReplayResult] = []
    with isolated_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        dataset = datasets.FashionIQ(path=root_str, transform=transform, split="val-split")
        params = SimpleNamespace(batch_size=batch_size, local_rank=0)

        for category in categories:
            recalls = dict(test_module.test(params, model, dataset, category))
            results.append(
                ReplayResult(
                    model_id="pair",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": variant.get("expected_sha256"),
                        "source_commit": MODEL["upstream_commit_sha"],
                    },
                )
            )

    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    selected_categories = tuple(args.categories) if args.categories else CATEGORIES
    blockers = preflight(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        selected_categories,
        checkpoint_id=args.checkpoint_id,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=selected_categories,
        checkpoint_id=args.checkpoint_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = {
        "model_id": "pair",
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
