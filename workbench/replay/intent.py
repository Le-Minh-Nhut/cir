"""Source-faithful evaluation runner for AAAI26-INTENT.

Why a wrapper exists: the pinned `test.py` defines `test(params, model, testset, category, txt_processors)`
and nothing else — no `__main__`, no CLI parser, and no checkpoint loader. This module supplies the
missing invocation shell and calls the pinned `test()` unchanged inside `isolated_cache_root`.
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
from contextlib import ExitStack, nullcontext
import os
import numpy as np
from workbench.scripts.third_model_native_observer import (
    atomic_json, observe_test_numpy, write_forward_control_diagnostics, write_native_scores,
)
from workbench.replay.conesep import _capture_native_inputs

MODEL = model_by_id("intent")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
# The processor config each `model_type` selects, taken from the pinned model class's own
# `PRETRAINED_MODEL_CONFIG_DICT`
# (`lavis/models/blip2_models/Blip2QformerCir.py`). Kept as the published mapping rather than a path
# invented here. Only `pretrain` is reachable through the adapter, which passes no other
# model_type, but the whole mapping is carried so a divergence is visible.
PROCESSOR_CONFIGS = {
    "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
    "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
    "coco": "configs/models/blip2/blip2_coco.yaml",
}

DEFAULT_VARIANT = "fiq_n02"
DATASET_LABEL = "fashioniq"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"


def validate_checkpoint(checkpoint_root: Path, variant_id: str = DEFAULT_VARIANT) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(variant_id)
    if variant is None:
        blockers.append(f"unsupported INTENT variant: {variant_id}; valid: {list(CHECKPOINTS.keys())}")
        return blockers
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"INTENT checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"INTENT checkpoint is empty placeholder: {path}")
        return blockers
    expected_size = variant.get("expected_size_bytes")
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(f"INTENT checkpoint size mismatch: expected {expected_size}, got {path.stat().st_size}")
    expected_sha = variant.get("expected_sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"INTENT checkpoint SHA-256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(f"INTENT {variant_id} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"INTENT source pin invalid: {reason}"]


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
                blockers.append(f"INTENT dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"INTENT dataset requirement missing: {dataset_root / 'resized_image' / category}")
    # The evaluated subset and the constructor's input set are different questions: the native
    # `datasets.FashionIQ(...)` opens every category's files regardless of `--category`.
    from workbench.replay.common import constructor_input_blockers

    return list(dict.fromkeys(
        blockers + constructor_input_blockers(
            dataset_root, lambda path: f"INTENT dataset requirement missing: {path}")))


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
    test_module = __import__("test")
    from workbench.replay.lavis_processors import text_processors

    txt_processors, boundary = text_processors(source_root, PROCESSOR_CONFIGS["pretrain"])

    return torch, datasets, data_utils, test_module, (txt_processors, boundary)


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    variant_id: str = DEFAULT_VARIANT,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, data_utils, test_module, processors = _load_replay_modules(source_root)
    txt_processors, processor_boundary = processors
    variant = CHECKPOINTS[variant_id]
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = checkpoint_root / variant["filename"]

    model = load_trusted_whole_model(model_path, variant.get("expected_sha256"), torch)
    model = model.to(target_device)
    model.eval()

    transform = data_utils.targetpad_transform(1.25, 224)
    params = SimpleNamespace(batch_size=batch_size, dataset=DATASET_LABEL)

    results: list[ReplayResult] = []
    pair_root = os.environ.get("WORKBENCH_INTENT_ADAPTER_PAIR_OUT")
    native_root = os.environ.get("WORKBENCH_INTENT_NATIVE_RANKINGS_OUT")
    with isolated_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        stack: ExitStack | None = ExitStack() if pair_root else None
        receipts: list[Any] = []
        if stack is None:
            dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
        else:
            from workbench.replay.tensor_spool import disk_backed_images
            counts = {"get_test_img": sum(2 * len(json.loads((dataset_root / "captions" / f"cap.{c}.val.json").read_text())) for c in CATEGORIES),
                      "get_img": sum(len(json.loads((dataset_root / "image_splits" / f"split.{c}.val.json").read_text())) for c in CATEGORIES)}
            receipts = [stack.enter_context(disk_backed_images(dataset_root, cache_root, datasets, method_name=m, split="val", expected_count=n)) for m, n in counts.items()]
            try:
                dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
            except Exception:
                stack.close()
                raise
        for category in categories:
            queries = getattr(dataset, f"test_queries_{category}")
            split = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
            query_ids = [f"{i}:{split[q['source_img_id']]}:{split[q['target_img_id']]}" for i, q in enumerate(queries)]
            observer = None
            original_numpy = getattr(test_module, "np", None)
            if native_root:
                Path(native_root).mkdir(parents=True, exist_ok=True)
                observer = observe_test_numpy(test_module, len(queries), "intent")
            capture = (_capture_native_inputs(model, Path(native_root) / f"{category}.inputs.npz", set(range(min(5, len(queries))))) if native_root else nullcontext())
            try:
                with capture:
                    recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
            finally:
                if observer is not None:
                    test_module.np = original_numpy
            if observer is not None:
                if native_root is None:
                    raise RuntimeError("INTENT native output root missing")
                receipt = write_native_scores(observer, Path(native_root) / f"{category}.scores.npz", model_id="intent", category=category,
                    query_ids=query_ids, gallery_ids=split, reference_positions=[q["source_img_id"] for q in queries], target_positions=[q["target_img_id"] for q in queries])
                atomic_json(Path(native_root) / f"{category}.json", receipt)
            if os.environ.get("WORKBENCH_INTENT_NATIVE_STAGE"):
                if native_root is None:
                    raise RuntimeError("INTENT native-stage output root missing")
                atomic_json(Path(native_root) / f"{category}.queries.json", [
                    {"source_id": split[q["source_img_id"]], "target_id": split[q["target_img_id"]],
                     "text": q["mod"]["str"]} for q in queries])
            if pair_root and not os.environ.get("WORKBENCH_INTENT_NATIVE_STAGE"): 
                if len(categories) != 1 or native_root is None:
                    raise ValueError("INTENT pair requires one category and native output")
                del model
                model = None
                if target_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                pair_path = Path(pair_root)
                from workbench.backend.operator_config import resolve_config
                from workbench.composition.model_intent import Adapter
                def image_path(image_id: str) -> str:
                    prefix, suffix = f"{category}_{image_id}".split("_", 1)
                    return str(Path(dataset.image_dir) / prefix / f"{suffix}.jpg")
                query_paths = [image_path(split[q["source_img_id"]]) for q in queries]
                gallery_paths = [image_path(image_id) for image_id in split]
                adapter = Adapter(resolve_config(), variant["checkpoint_id"])
                adapter.work_root = pair_path / "worker"
                selected = list(range(min(5, len(queries))))
                query_texts = [q["mod"]["str"] for q in queries]
                adapter._capture_spec = {"positions": selected, "output_path": str(pair_path / "adapter_query_inputs.npz")}
                query_features = adapter.encode_queries(query_paths, query_texts, [None] * len(queries))
                adapter._capture_spec = None
                adapter._capture_spec = {"positions": selected, "output_path": str(pair_path / "adapter_gallery_inputs.npz")}
                target_features = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
                adapter._capture_spec = None
                query_repeat = adapter.encode_queries(query_paths, query_texts, [None] * len(queries))
                target_repeat = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
                end_to_end_repeat = bool(np.array_equal(query_features, query_repeat) and np.array_equal(target_features, target_repeat))
                adapter_batch = adapter.batch_size
                adapter.batch_size = 1
                query_b1 = adapter.encode_queries(query_paths[:5], query_texts[:5], [None] * min(5, len(queries)))
                target_b1 = adapter.encode_targets(gallery_paths[:5], [None] * min(5, len(gallery_paths)))
                adapter.batch_size = adapter_batch
                control_flags = write_forward_control_diagnostics(pair_path / "forward_controls.json",
                    {"query_repeat": query_features, "target_repeat": target_features,
                     "query_b1_bn": query_features[:5], "target_b1_bn": target_features[:5]},
                    {"query_repeat": query_repeat, "target_repeat": target_repeat,
                     "query_b1_bn": query_b1, "target_b1_bn": target_b1})
                end_to_end_repeat = control_flags["query_repeat"] and control_flags["target_repeat"]
                b1_equal = control_flags["query_b1_bn"] and control_flags["target_b1_bn"]
                if not end_to_end_repeat or not b1_equal:
                    raise ValueError("INTENT Adapter full-forward repeat or B1/BN control differed")
                adapter._capture_spec = None
                scores = adapter.score(query_features, target_features)
                repeat_scores = adapter.score(query_features, target_features)
                if not np.array_equal(scores, repeat_scores):
                    raise ValueError("INTENT Adapter scorer repeat differed")
                refs = np.asarray([q["source_img_id"] for q in queries], dtype=np.int32)
                scores[np.arange(len(queries)), refs] = -10e10
                orders = np.argsort(-scores, axis=1)
                with np.load(Path(native_root) / f"{category}.scores.npz") as native:
                    for key, values in (("query_ids", np.asarray(query_ids)), ("gallery_ids", np.asarray(split)),
                        ("reference_positions", refs), ("target_positions", np.asarray([q["target_img_id"] for q in queries]))):
                        if not np.array_equal(native[key].astype(str) if key.endswith("ids") else native[key], values.astype(str) if key.endswith("ids") else values):
                            raise ValueError(f"INTENT native/Adapter {key} differs")
                    np.savez(pair_path / "adapter_pair_scores.npz", scores=scores, orders=orders, query_ids=query_ids,
                        gallery_ids=split, reference_positions=refs, target_positions=[q["target_img_id"] for q in queries],
                        adapter_forward_repeat_equal=np.asarray([end_to_end_repeat]), adapter_b1_bn_equal=np.asarray([b1_equal]))
                    np.savez(pair_path / "comparison.npz", abs_score_delta=np.abs(native["scores"] - scores),
                        rank_equal=np.asarray([np.count_nonzero(native["orders"] == orders)]),
                        top10_equal=np.asarray([np.count_nonzero(np.sort(native["orders"][:, -10:], axis=1) == np.sort(orders[:, -10:], axis=1))]),
                        adapter_forward_repeat_equal=np.asarray([end_to_end_repeat]), adapter_b1_bn_equal=np.asarray([b1_equal]))
                with np.load(Path(native_root) / f"{category}.inputs.npz") as native, np.load(pair_path / "adapter_query_inputs.npz") as query, np.load(pair_path / "adapter_gallery_inputs.npz") as gallery:
                    checks = {"query_pixels": np.array_equal(native["query_pixels"], query["pixels"]), "query_positions": np.array_equal(native["query_positions"], query["positions"]),
                        "token_positions": np.array_equal(native["token_positions"], query["token_positions"]), "input_ids": np.array_equal(native["input_ids"], query["input_ids"]),
                        "attention_masks": np.array_equal(native["attention_masks"], query["attention_masks"]), "token_texts": np.array_equal(native["token_texts"], query["token_texts"]),
                        "gallery_pixels": np.array_equal(native["gallery_pixels"], gallery["pixels"]), "gallery_positions": np.array_equal(native["gallery_positions"], gallery["positions"])}
                (pair_path / "input_comparison.json").write_text(json.dumps(checks, indent=2) + "\n")
                if not all(checks.values()):
                    raise ValueError(f"INTENT selected actual inputs differ: {checks}")
                adapter.close()
                if stack is not None:
                    stack.close()
            results.append(
                ReplayResult(
                    model_id="intent",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "native_evaluator": "test.py:test",
                        "native_output": {k: recalls[k] for k in sorted(recalls)},
                        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
                        "evaluation_noise_pct": 0,
                        "checkpoint_sha256": (
                            sha256_file(model_path) if model_path.is_file() and model_path.stat().st_size else None
                        ),
                        "checkpoint_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
                        "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224",
                        "text_processor": {
                            "config": PROCESSOR_CONFIGS["pretrain"],
                            "loaded_via": processor_boundary,
                            "note": "the pinned BlipCaptionProcessor from the model class's own "
                                    "config; no second BLIP-2 model is constructed and no "
                                    "/root/bert-base-uncased asset is read",
                        },
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
        print(f"[READY] INTENT preflight passed (variant={args.variant}); no inference executed")
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=categories,
        variant_id=args.variant,
        batch_size=args.batch_size,
        device=args.device,
    )
    variant = CHECKPOINTS[args.variant]
    output = {
        "model_id": "intent",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "protocol": PROTOCOL,
        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
        "evaluation_noise_pct": 0,
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
