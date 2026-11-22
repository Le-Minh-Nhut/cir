"""Source-faithful evaluation runner for CVPR26-ConeSep.

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
from contextlib import ExitStack, contextmanager, nullcontext
import os
import numpy as np
from workbench.scripts.third_model_native_observer import atomic_json, atomic_npz, observe_test_numpy, write_native_scores

MODEL = model_by_id("conesep")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
# The processor config each `model_type` selects, taken from the pinned model class's own
# `PRETRAINED_MODEL_CONFIG_DICT`
# (`lavis/models/blip2_models/ConeSep.py`). Kept as the published mapping rather than a path
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

# Verified author HuggingFace LFS metadata (model repository iLearn-Lab/CVPR26-ConeSep)
KNOWN_AUTHOR_SHA256 = {
    "fiq_n02": "5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2",
    "fiq_n05": "1953ec2f9269a5cb24d61bdf89d05ed8430fb3bdc6dd18cef08fe9e3dfbe1e26",
    "fiq_n08": "7c2bd0fbaea423a1a64809956b3fda72e191a6f7e5ed6ecfa708f1e3a8730eae",
}
KNOWN_AUTHOR_SIZES = {
    "fiq_n02": 2721428966,
    "fiq_n05": 2721428966,
    "fiq_n08": 2721428966,
}


def validate_checkpoint(checkpoint_root: Path, variant_id: str = DEFAULT_VARIANT) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(variant_id)
    if variant is None:
        blockers.append(f"unsupported ConeSep variant: {variant_id}; valid: {list(CHECKPOINTS.keys())}")
        return blockers
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"ConeSep checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"ConeSep checkpoint is empty placeholder: {path}")
        return blockers
    expected_size = variant.get("expected_size_bytes") or KNOWN_AUTHOR_SIZES.get(variant_id)
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(f"ConeSep checkpoint size mismatch: expected {expected_size}, got {path.stat().st_size}")
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"ConeSep checkpoint SHA-256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(f"ConeSep {variant_id} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"ConeSep source pin invalid: {reason}"]


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
                blockers.append(f"ConeSep dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"ConeSep dataset requirement missing: {dataset_root / 'resized_image' / category}")
    # The evaluated subset and the constructor's input set are different questions: the native
    # `datasets.FashionIQ(...)` opens every category's files regardless of `--category`.
    from workbench.replay.common import constructor_input_blockers

    return list(dict.fromkeys(
        blockers + constructor_input_blockers(
            dataset_root, lambda path: f"ConeSep dataset requirement missing: {path}")))


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


@contextmanager
def _capture_native_inputs(model: Any, output: Path, selected: set[int]):
    captured: dict[str, list[Any]] = {"query_pixels": [], "gallery_pixels": [], "tokens": []}
    query_start = 0
    gallery_start = 0
    original_query = model.extract_retrieval_compose
    original_target = model.extract_retrieval_target
    original_tokenizer = model.tokenizer

    def query_forward(pixels: Any, texts: list[str]):
        nonlocal query_start
        for offset in range(len(texts)):
            position = query_start + offset
            if position in selected:
                captured["query_pixels"].append((position, pixels[offset].detach().cpu().numpy().copy()))
        query_start += len(texts)
        return original_query(pixels, texts)

    def target_forward(pixels: Any):
        nonlocal gallery_start
        for offset in range(len(pixels)):
            position = gallery_start + offset
            if position in selected:
                captured["gallery_pixels"].append((position, pixels[offset].detach().cpu().numpy().copy()))
        gallery_start += len(pixels)
        return original_target(pixels)

    class TokenizerCapture:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_tokenizer, name)

        def __call__(self, texts: Any, *args: Any, **kwargs: Any):
            encoded = original_tokenizer(texts, *args, **kwargs)
            for offset, text in enumerate(texts):
                position = query_start - len(texts) + offset
                if position in selected:
                    captured["tokens"].append((position, str(text), encoded.input_ids[offset].detach().cpu().numpy().copy(),
                                               encoded.attention_mask[offset].detach().cpu().numpy().copy()))
            return encoded

    model.extract_retrieval_compose = query_forward
    model.extract_retrieval_target = target_forward
    model.tokenizer = TokenizerCapture()
    try:
        yield
    finally:
        model.extract_retrieval_compose = original_query
        model.extract_retrieval_target = original_target
        model.tokenizer = original_tokenizer
        atomic_npz(output,
            query_positions=np.asarray([r[0] for r in captured["query_pixels"]], dtype=np.int32),
            query_pixels=np.stack([r[1] for r in captured["query_pixels"]]),
            gallery_positions=np.asarray([r[0] for r in captured["gallery_pixels"]], dtype=np.int32),
            gallery_pixels=np.stack([r[1] for r in captured["gallery_pixels"]]),
            token_positions=np.asarray([r[0] for r in captured["tokens"]], dtype=np.int32),
            token_texts=np.asarray([r[1] for r in captured["tokens"]]),
            input_ids=np.stack([r[2] for r in captured["tokens"]]),
            attention_masks=np.stack([r[3] for r in captured["tokens"]]))

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
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)
    model: Any | None = load_trusted_whole_model(model_path, expected_sha, torch).to(target_device).eval()
    transform = data_utils.targetpad_transform(1.25, 224)
    params = SimpleNamespace(batch_size=batch_size, dataset=DATASET_LABEL)
    results: list[ReplayResult] = []
    pair_root = os.environ.get("WORKBENCH_CONESEP_ADAPTER_PAIR_OUT")
    native_root = os.environ.get("WORKBENCH_CONESEP_NATIVE_RANKINGS_OUT")
    with isolated_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        if pair_root:
            from workbench.replay.tensor_spool import disk_backed_images
            image_counts: dict[str, int] = {"get_test_img": 0, "get_img": 0}
            for spool_category in CATEGORIES:
                image_counts["get_test_img"] += 2 * len(json.loads((dataset_root / "captions" / f"cap.{spool_category}.val.json").read_text()))
                image_counts["get_img"] += len(json.loads((dataset_root / "image_splits" / f"split.{spool_category}.val.json").read_text()))
            stack = ExitStack()
            receipts = [stack.enter_context(disk_backed_images(dataset_root, cache_root, datasets,
                         method_name=method, split="val", expected_count=count)) for method, count in image_counts.items()]
            try:
                dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)
            except Exception:
                stack.close()
                raise
        else:
            stack, receipts = None, []
            dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)
        for category in categories:
            queries = getattr(dataset, f"test_queries_{category}")
            targets = getattr(dataset, f"test_targets_{category}")
            if pair_root and (len(categories) != 1 or not native_root):
                raise ValueError("ConeSep pair requires exactly one category and native output path")
            observer = None
            original_numpy = getattr(test_module, "np", None)
            if native_root:
                Path(native_root).mkdir(parents=True, exist_ok=True)
                observer = observe_test_numpy(test_module, len(queries), "conesep")
            if model is None:
                raise RuntimeError("ConeSep native model unavailable for category evaluation")
            input_capture = (_capture_native_inputs(model, Path(native_root) / f"{category}.inputs.npz", set(range(min(5, len(queries)))))
                             if native_root else nullcontext())
            try:
                with input_capture:
                    recalls: dict[str, Any] = dict(test_module.test(params, model, dataset, category, txt_processors))
            finally:
                if observer is not None:
                    test_module.np = original_numpy
            split = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
            query_ids = [f"{i}:{split[q['source_img_id']]}:{split[q['target_img_id']]}" for i, q in enumerate(queries)]
            if observer is not None:
                if native_root is None:
                    raise RuntimeError("ConeSep native output root missing")
                native_path = Path(native_root) / f"{category}.scores.npz"
                receipt = write_native_scores(observer, native_path, model_id="conesep", category=category,
                    query_ids=query_ids, gallery_ids=split,
                    reference_positions=[q["source_img_id"] for q in queries],
                    target_positions=[q["target_img_id"] for q in queries])
                atomic_json(Path(native_root) / f"{category}.json", receipt)
            if os.environ.get("WORKBENCH_CONESEP_NATIVE_STAGE"):
                if native_root is None:
                    raise RuntimeError("ConeSep native-stage output root missing")
                atomic_json(Path(native_root) / f"{category}.queries.json", [
                    {"source_id": split[q["source_img_id"]], "target_id": split[q["target_img_id"]],
                     "text": q["mod"]["str"]} for q in queries])
            if pair_root and not os.environ.get("WORKBENCH_CONESEP_NATIVE_STAGE"):
                import numpy as np
                from workbench.backend.operator_config import resolve_config
                from workbench.composition.model_conesep import Adapter
                if observer is None:
                    raise RuntimeError("ConeSep native rank observer required for paired run")
                del model
                model = None
                if target_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                pair_path = Path(pair_root)
                pair_path.mkdir(parents=True, exist_ok=True)
                def image_path(image_id: str) -> str:
                    prefix, suffix = f"{category}_{image_id}".split("_", 1)
                    return str(Path(dataset.image_dir) / prefix / f"{suffix}.jpg")
                query_paths = [image_path(split[q["source_img_id"]]) for q in queries]
                gallery_paths = [image_path(image_id) for image_id in split]
                adapter = Adapter(resolve_config(), variant["checkpoint_id"])
                adapter.work_root = pair_path / "worker"
                selected = list(range(min(5, len(queries))))
                adapter._capture_spec = {"positions": selected, "output_path": str(pair_path / "adapter_query_inputs.npz")}
                query_texts = [q["mod"]["str"] for q in queries]
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
                b1_equal = bool(np.array_equal(query_features[:5], query_b1) and np.array_equal(target_features[:5], target_b1))
                if not end_to_end_repeat or not b1_equal:
                    raise ValueError("ConeSep Adapter full-forward repeat or B1/BN control differed")
                adapter._capture_spec = None
                if native_root is None:
                    raise RuntimeError("ConeSep native output root missing")
                scores = adapter.score(query_features, target_features)
                repeat_scores = adapter.score(query_features, target_features)
                if not np.array_equal(scores, repeat_scores):
                    raise ValueError("ConeSep Adapter scorer repeat differed")
                refs = np.asarray([q["source_img_id"] for q in queries], dtype=np.int32)
                scores[np.arange(len(queries)), refs] = -10e10
                orders = np.argsort(-scores, axis=1)
                native_path = Path(native_root) / f"{category}.scores.npz"
                with np.load(native_path) as native:
                    native_scores, native_orders = native["scores"], native["orders"]
                    if not np.array_equal(native["query_ids"].astype(str), np.asarray(query_ids)) or not np.array_equal(native["gallery_ids"].astype(str), np.asarray(split)):
                        raise ValueError("ConeSep native/Adapter population identity mismatch")
                    np.savez(pair_path / "adapter_pair_scores.npz", scores=scores, orders=orders,
                        query_ids=np.asarray(query_ids), gallery_ids=np.asarray(split), reference_positions=refs,
                        target_positions=np.asarray([q["target_img_id"] for q in queries], dtype=np.int32),
                        adapter_forward_repeat_equal=np.asarray([end_to_end_repeat]), adapter_b1_bn_equal=np.asarray([b1_equal]))
                    np.savez(pair_path / "comparison.npz", native_scores=native_scores, adapter_scores=scores,
                        native_orders=native_orders, adapter_orders=orders, abs_score_delta=np.abs(native_scores - scores),
                        rank_equal=np.asarray([np.count_nonzero(native_orders == orders)]),
                        top10_membership_equal=np.asarray([np.count_nonzero(np.sort(native_orders[:, -10:], axis=1) == np.sort(orders[:, -10:], axis=1))]),
                        adapter_forward_repeat_equal=np.asarray([end_to_end_repeat]), adapter_b1_bn_equal=np.asarray([b1_equal]))
                with np.load(Path(native_root) / f"{category}.inputs.npz") as native_inputs, \
                     np.load(pair_path / "adapter_query_inputs.npz") as adapter_query, \
                     np.load(pair_path / "adapter_gallery_inputs.npz") as adapter_gallery:
                    input_checks = {
                        "query_positions": np.array_equal(native_inputs["query_positions"], adapter_query["positions"]),
                        "query_pixels": np.array_equal(native_inputs["query_pixels"], adapter_query["pixels"]),
                        "token_positions": np.array_equal(native_inputs["token_positions"], adapter_query["token_positions"]),
                        "input_ids": np.array_equal(native_inputs["input_ids"], adapter_query["input_ids"]),
                        "attention_masks": np.array_equal(native_inputs["attention_masks"], adapter_query["attention_masks"]),
                        "token_texts": np.array_equal(native_inputs["token_texts"], adapter_query["token_texts"]),
                        "gallery_positions": np.array_equal(native_inputs["gallery_positions"], adapter_gallery["positions"]),
                        "gallery_pixels": np.array_equal(native_inputs["gallery_pixels"], adapter_gallery["pixels"]),
                    }
                (pair_path / "input_comparison.json").write_text(json.dumps(input_checks, indent=2) + "\n")
                if not all(input_checks.values()):
                    raise ValueError(f"ConeSep native/Adapter selected forward input mismatch: {input_checks}")
                try:
                    adapter.close()
                finally:
                    if stack is not None:
                        stack.close()
            results.append(ReplayResult(model_id="conesep", checkpoint_id=variant["checkpoint_id"],
                category=category, protocol=PROTOCOL, r1=recalls.get(f"{category}_r1"),
                r10=recalls.get(f"{category}_r10"), r50=recalls.get(f"{category}_r50"), complete=True,
                provenance={"native_evaluator": "test.py:test", "native_output": {k: recalls[k] for k in sorted(recalls)},
                    "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"], "evaluation_noise_pct": 0,
                    "checkpoint_sha256": sha256_file(model_path) if model_path.is_file() else None,
                    "checkpoint_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
                    "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224",
                    "text_processor": {"config": PROCESSOR_CONFIGS["pretrain"], "loaded_via": processor_boundary}}))
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
        print(f"[READY] ConeSep preflight passed (variant={args.variant}); no inference executed")
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
        "model_id": "conesep",
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
