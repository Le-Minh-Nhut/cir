"""Source-faithful evaluation runner for LIMN base iteration 0."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import ReplayResult, isolated_cache_root, load_trusted_whole_model, macro_aggregate, sha256_file, source_pin_status

MODEL = model_by_id("limn")
LIMN_CHECKPOINTS = {
    category: {
        "checkpoint_id": f"base_iter0_{category}",
        **next(item for item in MODEL["checkpoint_variants"] if item["checkpoint_id"] == f"base_iter0_{category}"),
    }
    for category in CATEGORIES
}



def validate_checkpoint_bundle(checkpoint_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers = []
    for category in categories:
        checkpoint = LIMN_CHECKPOINTS.get(category)
        if checkpoint is None:
            blockers.append(f"unsupported LIMN category: {category}")
            continue
        path = checkpoint_root / checkpoint["filename"]
        if not path.is_file():
            blockers.append(f"LIMN {category} checkpoint missing: {path}")
        elif path.stat().st_size == 0:
            blockers.append(f"LIMN {category} checkpoint is empty placeholder: {path}")
        else:
            actual_sha = sha256_file(path)
            if actual_sha != checkpoint["expected_sha256"]:
                blockers.append(f"LIMN {category} checkpoint SHA-256 mismatch: expected {checkpoint['expected_sha256'][:16]}..., got {actual_sha[:16]}...")
        if checkpoint["checkpoint_mapping_status"] != "VERIFIED_METADATA":
            blockers.append(f"LIMN {category} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"LIMN source pin invalid: {reason}"]



def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
            f"resized_image/{category}",
        ):
            path = dataset_root / relative
            valid = path.is_dir() if relative.startswith("resized_image/") else path.is_file()
            if not valid:
                blockers.append(f"LIMN dataset requirement missing: {path}")
    return blockers


def validate_environment() -> list[str]:
    versions = {"torch": "1.12.1", "torchvision": "0.13.1", "open_clip_torch": "2.20.0"}
    blockers = []
    for package, expected in versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f"LIMN environment package missing: {package}=={expected}")
        else:
            if actual != expected:
                blockers.append(f"LIMN environment version mismatch: expected {package}=={expected}, found {actual}")
    if not blockers:
        import torch
        if not torch.cuda.is_available():
            blockers.append("LIMN native test requires CUDA; torch.cuda.is_available() is false")
    return blockers


def preflight(checkpoint_root: Path, dataset_root: Path, source_root: Path,
              categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    return (validate_source(source_root) + validate_checkpoint_bundle(checkpoint_root, categories)
            + validate_dataset(dataset_root, categories) + validate_environment())


def replay_category(category: str, checkpoint_root: Path, dataset_root: Path,
                    source_root: Path) -> ReplayResult:
    """Load one verified category model and call pinned native test()."""
    if category not in CATEGORIES:
        raise ValueError(f"unsupported LIMN category: {category}")
    checkpoint = LIMN_CHECKPOINTS[category]
    blockers = (validate_checkpoint_bundle(checkpoint_root, (category,))
                + validate_dataset(dataset_root, (category,)) + validate_environment()
                + validate_source(source_root))
    if blockers:
        raise ValueError("; ".join(blockers))

    sys.dont_write_bytecode = True
    source_path = source_root / "LIMN"
    sys.path.insert(0, str(source_path))
    native_model = importlib.import_module("model")
    native_datasets = importlib.import_module("datasets")
    native_test = importlib.import_module("test")
    for module, path in ((native_model, source_path / "model.py"),
                         (native_datasets, source_path / "datasets.py"),
                         (native_test, source_path / "test.py")):
        if Path(module.__file__).resolve() != path.resolve():
            raise ImportError(f"refusing non-pinned LIMN module {module.__name__}: {module.__file__}")

    import torch
    import open_clip

    checkpoint_path = checkpoint_root / checkpoint["filename"]
    model = load_trusted_whole_model(checkpoint_path, checkpoint["expected_sha256"], torch)
    if not isinstance(model, native_model.LIMN):
        raise TypeError("LIMN checkpoint does not deserialize to pinned model.LIMN")
    model.cuda().eval()
    visual = model.backbone.clip.visual
    preprocess_train = open_clip.image_transform(visual.image_size, is_train=True,
                                                 mean=visual.image_mean, std=visual.image_std)
    preprocess_val = open_clip.image_transform(visual.image_size, is_train=False,
                                               mean=visual.image_mean, std=visual.image_std)
    with isolated_cache_root(dataset_root) as cache_root:
        dataset = native_datasets.FashionIQ(
            path=f"{cache_root}/", name=category,
            transform=[preprocess_train, preprocess_val],
        )
        metrics = dict(native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model, dataset, category))
    return ReplayResult(
        model_id="limn", checkpoint_id=checkpoint["checkpoint_id"], category=category,
        protocol="fashioniq_val_split", r1=metrics[f"{category}_r1"],
        r10=metrics[f"{category}_r10"], r50=metrics[f"{category}_r50"], complete=True,
        provenance={"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint["expected_sha256"],
                    "source_commit": MODEL["upstream_commit_sha"]},
    )


def evaluate_categories(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                        categories: tuple[str, ...] = CATEGORIES,
                        replay: Callable[..., ReplayResult] = replay_category) -> tuple[list[ReplayResult], dict[str, Any]]:
    if len(set(categories)) != len(categories) or any(category not in CATEGORIES for category in categories):
        raise ValueError(f"invalid LIMN category selection: {categories}")
    results = [replay(category, checkpoint_root, dataset_root, source_root) for category in categories]
    for category, result in zip(categories, results, strict=True):
        if (result.model_id, result.checkpoint_id, result.category, result.protocol) != (
            "limn", LIMN_CHECKPOINTS[category]["checkpoint_id"], category, "fashioniq_val_split"
        ):
            raise ValueError(f"LIMN replay returned mismatched category/checkpoint result for {category}")
    aggregate = macro_aggregate(results) if categories == CATEGORIES else {
        "macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False
    }
    return results, aggregate


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, help="Single-category diagnostic; never a complete benchmark result")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, help="write aggregate JSON to a new file; refuses overwrite")
    args = parser.parse_args(argv)
    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[READY] LIMN preflight passed; no inference executed")
        return 0
    results, aggregate = evaluate_categories(args.checkpoint_root, args.dataset_root, args.source_root, categories)
    output = {
        "model_id": "limn", "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "protocol": "fashioniq_val_split", "evaluation_noise_pct": 0,
        "category_model_policy": "independent",
        "complete_benchmark": aggregate["complete"], "categories": [asdict(result) for result in results],
        "aggregate": aggregate, "timestamp": datetime.now(UTC).isoformat(),
        "runtime": {"python": sys.executable, "source_root": str(args.source_root.resolve()),
                    "dataset_root": str(args.dataset_root.resolve()),
                    "checkpoint_root": str(args.checkpoint_root.resolve())},
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except FileExistsError:
            print(f"[BLOCKED] output already exists: {args.output}", file=sys.stderr)
            return 1
        except OSError as error:
            print(f"[BLOCKED] cannot write output {args.output}: {error}", file=sys.stderr)
            return 1
        print(f"[OK] result written: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())