"""Bounded native-versus-registered-Adapter CLIP4Cir FashionIQ score diagnostic.

Each phase is a separate invocation. Native writes the source distances/order; adapter writes the
production Adapter's equivalent cosine scores. Compare binds outputs to exact ordered IDs and checks
score/rank parity. No inference runs unless an explicit phase is invoked.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CATEGORIES = ("dress", "shirt", "toptee")

def _categories(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.category,) if args.category else CATEGORIES
MODEL_VARIANTS = {
    "combiner_rn50x4_noft": ("rn50x4_noft_unresolved", "fiq_comb_RN50x4_noft.pt", None),
    "clip4cir_rn50x4_fullft": (
        "rn50x4_fullft_unresolved", "fiq_comb_RN50x4_fullft.pt", "fiq_clip_RN50x4_fullft.pt"),
}


def _source_dataset(root: Path, category: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        captions = json.loads((root / "captions" / f"cap.{category}.val.json").read_text(encoding="utf-8"))
        gallery = json.loads((root / "image_splits" / f"split.{category}.val.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{category}: cannot read FashionIQ validation population: {error}") from error
    if not captions or not gallery or len(set(gallery)) != len(gallery):
        raise ValueError(f"{category}: empty or duplicate FashionIQ source population")
    return captions, gallery


def _caption(row: dict[str, Any]) -> str:
    captions = row["captions"]
    if len(captions) != 2:
        raise ValueError("CLIP4Cir source query must contain exactly two captions")
    return f"{captions[0].strip('.?, ').capitalize()} and {captions[1].strip('.?, ')}"


def _write_source(args: argparse.Namespace, dataset_root: Path) -> None:
    from workbench.replay.clip4cir_score_capture import bind_fashioniq_rows, run_native_capture

    checkpoint_root = args.checkpoint_root.resolve()
    source_root = args.source_root.resolve()
    base_cache = Path(args.base_weights).expanduser().resolve()
    if not base_cache.is_file():
        raise FileNotFoundError(f"OpenAI RN50x4 base weights missing: {base_cache}")
    native_root = args.native_data_root.resolve()
    native_root.mkdir(parents=True, exist_ok=False)
    for name in ("captions", "image_splits"):
        (native_root / name).symlink_to((dataset_root / name).resolve(), target_is_directory=True)
    images = native_root / "images"
    images.mkdir()
    categories = _categories(args)
    for category in categories:
        rows, gallery = _source_dataset(dataset_root, category)
        names = {image_id for row in rows for image_id in (row["candidate"], row["target"])} | set(gallery)
        for image_id in names:
            source_image = dataset_root / "images" / f"{image_id}.png"
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            destination = images / f"{image_id}.jpg"
            if not destination.exists():
                destination.symlink_to(source_image.resolve())

    model_id = args.model_id
    _, combiner_file, clip_file = MODEL_VARIANTS[model_id]
    combiner = checkpoint_root / "clip4cir" / "FashionIQ" / (
        "RN50x4_noft" if clip_file is None else "RN50x4_fullft") / combiner_file
    argv = ["--dataset", "FashionIQ", "--combining-function", "combiner", "--combiner-path", str(combiner),
            "--projection-dim", "2560", "--hidden-dim", "5120", "--clip-model-name", "RN50x4",
            "--target-ratio", "1.25", "--transform", "targetpad"]
    if clip_file is not None:
        argv.extend(("--clip-model-path", str(combiner.parent / clip_file)))

    def capture(dataset: Any, distances: Any, order: Any, gallery_ids: Any,
                sort_args: tuple[Any, ...], sort_kwargs: dict[str, Any]) -> None:
        if sort_args or sort_kwargs.get("dim") != -1:
            raise ValueError("unexpected CLIP4Cir native argsort invocation")
        rows = bind_fashioniq_rows(dataset, distances, order, gallery_ids)
        category = dataset.dress_types[0]
        output = args.output / f"{category}.native.npz"
        if output.exists():
            raise FileExistsError(output)
        np.savez_compressed(
            output,
            query_ids=np.asarray([row["query_id"] for row in rows]),
            target_ids=np.asarray([row["target_id"] for row in rows]),
            reference_ids=np.asarray([row["reference_id"] for row in rows]),
            gallery_ids=np.asarray(gallery_ids),
            distances=np.stack([row["distances"] for row in rows]).astype(np.float32, copy=False),
            native_order=np.stack([row["native_order"] for row in rows]).astype(np.int32, copy=False),
        )

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    original_cwd = Path.cwd()
    try:
        os.chdir(source_root)
        if args.category:
            import importlib

            source_dir = source_root / "src"
            sys.path.insert(0, str(source_dir))
            validate = None
            original_retrieval = None
            original_cpu_count = None
            try:
                validate = importlib.import_module("validate")
                original_retrieval = validate.fashioniq_val_retrieval
                original_cpu_count = validate.multiprocessing.cpu_count
                setattr(validate, "fashioniq_val_retrieval", lambda category, *rest: (
                    original_retrieval(category, *rest) if category == args.category else (0, 0)))
                validate.multiprocessing.cpu_count = lambda: 4
                run_native_capture(source_root, argv, capture, dataset_root=native_root)
            finally:
                if validate is not None:
                    if original_cpu_count is not None:
                        validate.multiprocessing.cpu_count = original_cpu_count
                    if original_retrieval is not None:
                        setattr(validate, "fashioniq_val_retrieval", original_retrieval)
                    sys.modules.pop("validate", None)
                sys.path.remove(str(source_dir))
        else:
            run_native_capture(source_root, argv, capture, dataset_root=native_root)
    finally:
        os.chdir(original_cwd)
    missing = [category for category in categories if not (args.output / f"{category}.native.npz").is_file()]


def _write_adapter(args: argparse.Namespace, dataset_root: Path) -> None:
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_clip4cir_rn50x4_fullft import Adapter as FullFtAdapter
    from workbench.composition.model_combiner_rn50x4_noft import Adapter as NoFtAdapter

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model_id = args.model_id
    checkpoint_id = MODEL_VARIANTS[model_id][0]
    adapter = (NoFtAdapter if model_id == "combiner_rn50x4_noft" else FullFtAdapter)(
        resolve_config(), checkpoint_id=checkpoint_id)
    adapter.work_root = args.work_root.resolve()
    try:
        for category in _categories(args):
            rows, gallery = _source_dataset(dataset_root, category)
            image_root = dataset_root / "images"
            query_paths = [image_root / f"{row['candidate']}.png" for row in rows]
            gallery_paths = [image_root / f"{image_id}.png" for image_id in gallery]
            query_texts = [_caption(row) for row in rows]
            queries = adapter.encode_queries([str(path) for path in query_paths], query_texts, [None] * len(rows))
            targets = adapter.encode_targets([str(path) for path in gallery_paths], [None] * len(gallery))
            scores = adapter.score(queries, targets)
            np.savez_compressed(
                args.output / f"{category}.adapter.npz",
                query_ids=np.asarray([f"{category}#row{i}:{row['candidate']}:{row['target']}"
                                      for i, row in enumerate(rows)]),
                target_ids=np.asarray([row["target"] for row in rows]),
                reference_ids=np.asarray([row["candidate"] for row in rows]),
                gallery_ids=np.asarray(gallery),
                scores=scores.astype(np.float32, copy=False),
            )
    finally:
        adapter.close()


def _compare(args: argparse.Namespace) -> None:
    summaries = []
    for category in _categories(args):
        with np.load(args.native_dir / f"{category}.native.npz", allow_pickle=False) as native, \
                np.load(args.adapter_dir / f"{category}.adapter.npz", allow_pickle=False) as adapter:
            for key in ("query_ids", "target_ids", "reference_ids", "gallery_ids"):
                if not np.array_equal(native[key], adapter[key]):
                    raise ValueError(f"{category}: native and Adapter {key} populations/order differ")
            native_scores = -native["distances"]
            adapter_scores = adapter["scores"]
            if native_scores.shape != adapter_scores.shape:
                raise ValueError(f"{category}: score matrix shapes differ")
            native_order = native["native_order"]
            adapter_order = np.argsort(-adapter_scores, axis=1, kind="stable")
            gallery_positions = {image_id: index for index, image_id in enumerate(native["gallery_ids"])}
            positive_at = np.asarray([gallery_positions[target] for target in native["target_ids"]])
            native_rank = np.take_along_axis(np.argsort(native_order, axis=1),
                                             positive_at[:, None], axis=1).ravel()
            adapter_rank = np.take_along_axis(np.argsort(adapter_order, axis=1),
                                              positive_at[:, None], axis=1).ravel()
            summaries.append({
                "category": category, "queries": len(native["query_ids"]), "gallery": len(native["gallery_ids"]),
                "max_abs_score_error": float(np.max(np.abs(native_scores - adapter_scores))),
                "rank_mismatch_queries": int(np.count_nonzero(native_rank != adapter_rank)),
                "native_positive_rank_r1": int(np.count_nonzero(native_rank == 0)),
                "adapter_positive_rank_r1": int(np.count_nonzero(adapter_rank == 0)),
            })
    print(json.dumps({"model_id": args.model_id, "protocol": "fashioniq_original_split",
                      "reference_eligible": True, "categories": summaries}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("native", "adapter", "compare"))
    parser.add_argument("--model-id", choices=tuple(MODEL_VARIANTS), required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/FashionIQ")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints")
    parser.add_argument("--source-root", type=Path, default=ROOT / "workbench/third_party/CLIP4Cir")
    parser.add_argument("--base-weights", type=Path, default=Path.home() / ".cache/clip/RN50x4.pt")
    parser.add_argument("--native-data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--category", choices=CATEGORIES,
                        help="run one FashionIQ category; default is all three")
    parser.add_argument("--adapter-dir", type=Path)
    args = parser.parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if args.phase == "native":
        if args.native_data_root is None or args.output is None:
            parser.error("native phase requires --native-data-root and --output")
        _write_source(args, dataset_root)
    elif args.phase == "adapter":
        if args.output is None or args.work_root is None:
            parser.error("adapter phase requires --output and --work-root")
        _write_adapter(args, dataset_root)
    else:
        if args.native_dir is None or args.adapter_dir is None:
            parser.error("compare phase requires --native-dir and --adapter-dir")
        _compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
