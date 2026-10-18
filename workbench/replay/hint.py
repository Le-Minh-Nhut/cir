"""Source-faithful replay for the HINT FashionIQ evaluator.

Why a wrapper exists at all: the pinned `test.py` defines `test(params, model, testset, category,
txt_processors)` and nothing else — no `__main__`, no argument parser, no checkpoint loader — so the
published repository has no runnable evaluation entrypoint. This module supplies exactly that
missing shell and calls the pinned `test()` unchanged. It does not re-implement ranking, metric
arithmetic, gallery construction, or feature extraction.

Everything here is read off the pinned source:

- `test.py:11` — `test(params, model, testset, category, txt_processors)`; the only attributes it
  reads from `params` are `params.batch_size` and `params.dataset`.
- `test.py:74-86` — `sims = np.matmul(all_queries, all_imgs).squeeze().max(-1)`, reference written to
  `-10e10`, `nn_result = argsort(-sims)[:50]`; recalls over `[1, 10, 50]`.
- `test.py:14-19` — the FashionIQ dispatch reads `testset.test_queries_<category>` /
  `testset.test_targets_<category>`, which `datasets.FashionIQ.get_test_data` builds as the ordered
  pair union of the validation split (`datasets.py:154-159`). That is the registry protocol
  `fashioniq_val_split`.
- `train.py:117-121` — `load_model_and_preprocess(name="HINT", model_type=backbone, is_eval=True,
  device=...)`, returning `(model, vis_processors, txt_processors)`; only `txt_processors` reaches
  `test()`.
- `train.py:233, 279` — checkpoints are saved with `torch.save(model, ...)` / `torch.save(current_model,
  ...)`, i.e. whole pickled modules, which is why the loader below verifies the author SHA-256 first
  and only then unpickles.
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

MODEL = model_by_id("hint")
CHECKPOINT = MODEL["checkpoint_variants"][0]
# `params.dataset` only reaches the recall label (`test.py:107` `'{}_r{}'.format(name, k)`), so it is
# recorded as the protocol's dataset name rather than being invented.
DATASET_LABEL = "fashioniq"
# FashionIQ evaluation reads the ordered pair union of the validation split.
PROTOCOL = "fashioniq_val_split"


def validate_checkpoint(checkpoint_root: Path) -> list[str]:
    blockers: list[str] = []
    path = checkpoint_root / CHECKPOINT["filename"]
    if not path.is_file():
        blockers.append(f"HINT checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"HINT checkpoint is an empty placeholder: {path}")
        return blockers
    expected = CHECKPOINT.get("expected_sha256")
    if CHECKPOINT.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append("HINT checkpoint mapping is not verified")
    if expected and path.stat().st_size != CHECKPOINT.get("expected_size_bytes"):
        blockers.append(
            f"HINT checkpoint size mismatch: expected {CHECKPOINT['expected_size_bytes']}, "
            f"found {path.stat().st_size}")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"HINT source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    """The inputs `datasets.FashionIQ` opens before `test()` can run.

    `datasets.py:29-31` concatenates the root with `resized_image`, `image_splits` and `captions`, so
    the root must carry a trailing separator; `:137` additionally requires the correction dictionary
    the caption pre-processing reads.
    """
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"HINT dataset input missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"HINT dataset input missing: {dataset_root / 'resized_image' / category}")
    return blockers


def preflight(checkpoint_root: Path, dataset_root: Path, source_root: Path,
              categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    return (validate_source(source_root) + validate_checkpoint(checkpoint_root)
            + validate_dataset(dataset_root, categories))


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any, Any]:
    """Import the pinned modules and the pinned evaluator from the checkout itself."""
    import torch

    for entry in (str(source_root), str(source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    datasets = __import__("datasets")
    data_utils = __import__("data_utils")
    test_module = __import__("test")
    from lavis.models import load_model_and_preprocess

    return torch, datasets, (data_utils, test_module), load_model_and_preprocess


def evaluate_categories(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                        categories: tuple[str, ...] = CATEGORIES,
                        batch_size: int = 64, backbone: str = "pretrain",
                        device: str | None = None) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, pinned, load_model_and_preprocess = _load_replay_modules(source_root)
    data_utils, test_module = pinned

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = checkpoint_root / CHECKPOINT["filename"]
    # Whole pickled module (train.py:233,279): the author digest is verified before unpickling.
    model = load_trusted_whole_model(model_path, CHECKPOINT.get("expected_sha256"), torch)

    _, _, txt_processors = load_model_and_preprocess(
        name="HINT", model_type=backbone, is_eval=True, device=target_device)
    model = model.to(target_device)
    model.eval()

    # `datasets.py:29` concatenates the root with bare subdirectory names.
    transform = data_utils.targetpad_transform(1.25, 224)

    # Only these two attributes are read by the pinned evaluator.
    params = SimpleNamespace(batch_size=batch_size, dataset=DATASET_LABEL)

    results: list[ReplayResult] = []
    with isolated_cache_root(dataset_root) as cache_root:
        root = str(cache_root).rstrip("/") + "/"
        dataset = datasets.FashionIQ(path=root, transform=transform)
        for category in categories:
            recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
            results.append(ReplayResult(
                model_id="hint",
                checkpoint_id=CHECKPOINT["checkpoint_id"],
                category=category,
                protocol=PROTOCOL,
                r1=recalls.get(f"{category}_r1"),
                r10=recalls.get(f"{category}_r10"),
                r50=recalls.get(f"{category}_r50"),
                complete=True,
                provenance={
                    "native_evaluator": "test.py:test",
                    "native_output": {key: recalls[key] for key in sorted(recalls)},
                    "checkpoint_sha256": (sha256_file(model_path)
                                          if model_path.is_file() and model_path.stat().st_size else None),
                    "checkpoint_size_bytes": (model_path.stat().st_size
                                              if model_path.is_file() else None),
                    "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224 (blip_validate.py equivalent path)",
                },
            ))
    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone", default="pretrain", help="'pretrain' selects ViT-g, 'pretrain_vitL' ViT-L")
    parser.add_argument("--device", default=None)
    parser.add_argument("--category", choices=CATEGORIES,
                        help="Single-category diagnostic; never a complete benchmark result")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[READY] HINT preflight passed; no inference executed")
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root, args.dataset_root, args.source_root, categories,
        batch_size=args.batch_size, backbone=args.backbone, device=args.device)
    output = {
        "model_id": "hint",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "protocol": PROTOCOL,
        "evaluation_noise_pct": 0,
        "category_model_policy": "shared",
        "complete_benchmark": aggregate["complete"],
        "categories": [asdict(result) for result in results],
        "aggregate": aggregate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.executable, "source_root": str(args.source_root.resolve()),
                    "dataset_root": str(args.dataset_root.resolve()),
                    "checkpoint_root": str(args.checkpoint_root.resolve()),
                    "backbone": args.backbone},
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output:
        if args.output.parent.resolve().name == "reports":
            print("[BLOCKED] --output must not target the official reports directory", file=sys.stderr)
            return 1
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except FileExistsError:
            print(f"[BLOCKED] output already exists: {args.output}", file=sys.stderr)
            return 1
        print(f"[OK] result written: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
