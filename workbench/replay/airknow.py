"""Source-faithful evaluation runner for CVPR26-Air-Know.

Why a wrapper exists: `test_BLIP2.py` defines callable routines without `__main__` or a checkpoint
loader, and its auxiliary LaSCo routines call `load_obj("")` with an empty path. This module supplies the
missing invocation shell and runs the native FashionIQ evaluator inside `isolated_cache_root`.

Native model behavior:
`test_BLIP2.py:10` natively sets `model.train()` at the top of `test()`. `model.eval()` is only called
for `fashion200k` (line 13) and `test_lasco` (line 220), leaving FashionIQ evaluation governed by
`model.train()`. An experimental eval-mode variant (`--eval-mode`) is provided and explicitly labelled.
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

MODEL = model_by_id("airknow")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_VARIANT = "fiq_n05"
DATASET_LABEL = "fashioniq"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"

# Verified author HuggingFace LFS metadata (model repository iLearn-Lab/CVPR26-Air-Know)
KNOWN_AUTHOR_SHA256 = {
    "fiq_n05": "f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6",
    "fiq_n08": "756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e",
}
KNOWN_AUTHOR_SIZES = {
    "fiq_n05": 2724056638,
    "fiq_n08": 2724056638,
}


def validate_checkpoint(checkpoint_root: Path, variant_id: str = DEFAULT_VARIANT) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(variant_id)
    if variant is None:
        blockers.append(f"unsupported Air-Know variant: {variant_id}; valid: {list(CHECKPOINTS.keys())}")
        return blockers
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"Air-Know checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"Air-Know checkpoint is empty placeholder: {path}")
        return blockers
    expected_size = variant.get("expected_size_bytes") or KNOWN_AUTHOR_SIZES.get(variant_id)
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(f"Air-Know checkpoint size mismatch: expected {expected_size}, got {path.stat().st_size}")
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"Air-Know checkpoint SHA-256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(f"Air-Know {variant_id} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"Air-Know source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"Air-Know dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"Air-Know dataset requirement missing: {dataset_root / 'resized_image' / category}")
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    variant_id: str = DEFAULT_VARIANT,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_checkpoint(checkpoint_root, variant_id)
        + validate_dataset(dataset_root, categories)
    )


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    for entry in (str(source_root), str(source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    datasets = __import__("datasets")
    data_utils = __import__("data_utils")
    test_blip2 = __import__("test_BLIP2")
    from lavis.models import load_model_and_preprocess

    return torch, datasets, data_utils, test_blip2, load_model_and_preprocess


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    variant_id: str = DEFAULT_VARIANT,
    batch_size: int = 64,
    device: str | None = None,
    eval_mode: bool = False,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, data_utils, test_module, load_model_and_preprocess = _load_replay_modules(source_root)
    variant = CHECKPOINTS[variant_id]
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = checkpoint_root / variant["filename"]
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)

    model = load_trusted_whole_model(model_path, expected_sha, torch)
    model = model.to(target_device)
    if eval_mode:
        model.eval()
        # test_BLIP2.py:10 calls model.train(); neutralize for experimental eval-mode variant
        model.train = lambda *args, **kwargs: model
    else:
        # Native behavior: test_BLIP2.py:10 calls model.train()
        model.train()

    _, _, txt_processors = load_model_and_preprocess(
        name="Blip2QformerCir", model_type="pretrain", is_eval=True, device=target_device
    )

    transform = data_utils.targetpad_transform(1.25, 224)
    params = SimpleNamespace(batch_size=batch_size, dataset=DATASET_LABEL)

    results: list[ReplayResult] = []
    with isolated_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)

        for category in categories:
            recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
            results.append(
                ReplayResult(
                    model_id="airknow",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "native_evaluator": "test_BLIP2.py:test",
                        "native_output": {k: recalls[k] for k in sorted(recalls)},
                        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
                        "evaluation_noise_pct": 0,
                        "model_mode": "eval (experimental variant)" if eval_mode else "train (native behavior)",
                        "checkpoint_sha256": (
                            sha256_file(model_path) if model_path.is_file() and model_path.stat().st_size else None
                        ),
                        "checkpoint_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
                        "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224",
                    },
                )
            )

    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(CHECKPOINTS.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Experimental variant: neutralize model.train() at test_BLIP2.py:10 to evaluate in eval mode",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print(f"[READY] Air-Know preflight passed (variant={args.variant}); no inference executed")
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=categories,
        variant_id=args.variant,
        batch_size=args.batch_size,
        device=args.device,
        eval_mode=args.eval_mode,
    )
    variant = CHECKPOINTS[args.variant]
    output = {
        "model_id": "airknow",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "protocol": PROTOCOL,
        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
        "evaluation_noise_pct": 0,
        "eval_mode_variant": args.eval_mode,
        "category_model_policy": "shared",
        "complete_benchmark": aggregate["complete"],
        "categories": [asdict(r) for r in results],
        "aggregate": aggregate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.executable,
            "source_root": str(args.source_root.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "variant": args.variant,
            "eval_mode": args.eval_mode,
        },
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output:
        if args.output.parent.resolve().name == "reports":
            print("[BLOCKED] --output must not target the official reports directory", file=sys.stderr)
            return 1
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
        print(f"[OK] result written: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
