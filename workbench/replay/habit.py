"""Source-faithful evaluation runner for AAAI26-HABIT.

Why a wrapper exists: the pinned `test.py` defines `test(params, model, testset, category, txt_processors)`
and nothing else — no `__main__`, no CLI parser, and no checkpoint loader. This module supplies the
missing invocation shell and calls the pinned `test()` unchanged inside `isolated_cache_root`.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from contextlib import contextmanager
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

from workbench.scripts.third_model_native_observer import atomic_npz

MODEL = model_by_id("habit")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
# The processor config each `model_type` selects, taken from the pinned model class's own
# `PRETRAINED_MODEL_CONFIG_DICT`
# (`lavis/models/blip2_models/HABIT.py`). Kept as the published mapping rather than a path
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

NATIVE_RANKINGS_ENV = "WORKBENCH_HABIT_NATIVE_RANKINGS_OUT"
NATIVE_TOP_K = 200


class _ArgsortCapture:
    def __init__(self, numpy_module: Any, expected: int) -> None:
        self._numpy = numpy_module
        self.expected = expected
        self.rows: list[tuple[Any, Any]] = []
        self.calls = 0
        self.matmul_calls = 0
        self.matmul_shape: tuple[int, ...] | None = None
        self.identical = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._numpy, name)

    def matmul(self, left: Any, right: Any) -> Any:
        result = self._numpy.matmul(left, right)
        self.matmul_calls += 1
        self.matmul_shape = tuple(result.shape)
        return result

    def argsort(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        order = self._numpy.argsort(values, *args, **kwargs)
        if self.calls < self.expected:
            self.rows.append((values.copy(), order.copy()))
        else:
            first_values, first_order = self.rows[self.calls - self.expected]
            self.identical &= self._numpy.array_equal(first_values, values)
            self.identical &= self._numpy.array_equal(first_order, order)
        self.calls += 1
        return order

@contextmanager
def _capture_native_inputs(model: Any, output: Path, selected: set[int]):
    import numpy as np

    captured: dict[str, list[Any]] = {"query_pixels": [], "gallery_pixels": [], "tokens": []}
    original_query = model.extract_retrieval_compose
    original_target = model.extract_retrieval_target
    original_tokenizer = model.tokenizer
    query_start = 0
    gallery_start = 0

    def query_forward(pixels: Any, texts: list[str]):
        nonlocal query_start
        for offset, text in enumerate(texts):
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
                    captured["tokens"].append((position, str(text),
                        encoded.input_ids[offset].detach().cpu().numpy().copy(),
                        encoded.attention_mask[offset].detach().cpu().numpy().copy()))
            return encoded

    model.extract_retrieval_compose = query_forward
    model.extract_retrieval_target = target_forward
    model.tokenizer = TokenizerCapture()
    try:
        yield captured
    finally:
        model.extract_retrieval_compose = original_query
        model.extract_retrieval_target = original_target
        model.tokenizer = original_tokenizer
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(output,
                 query_positions=np.asarray([row[0] for row in captured["query_pixels"]], dtype=np.int32),
                 query_pixels=np.stack([row[1] for row in captured["query_pixels"]]),
                 gallery_positions=np.asarray([row[0] for row in captured["gallery_pixels"]], dtype=np.int32),
                 gallery_pixels=np.stack([row[1] for row in captured["gallery_pixels"]]),
                 token_positions=np.asarray([row[0] for row in captured["tokens"]], dtype=np.int32),
                 token_texts=np.asarray([row[1] for row in captured["tokens"]]),
                 input_ids=np.stack([row[2] for row in captured["tokens"]]),
                 attention_masks=np.stack([row[3] for row in captured["tokens"]]))


@contextmanager
def _capture_native_rankings(test_module: Any, dataset: Any, category: str, output: Path):
    """Observe pinned test.py's actual masked sort calls without rerunning inference."""
    import numpy as np

    capture_error = None
    original_numpy = getattr(test_module, "np", None)
    capture = None
    try:
        if original_numpy is None:
            raise ValueError("pinned test module has no NumPy binding")
        queries = getattr(dataset, f"test_queries_{category}")
        targets = getattr(dataset, f"test_targets_{category}")
        image_ids = json.loads((Path(dataset.path) / "image_splits" / f"split.{category}.val.json").read_text())
        if len(set(image_ids)) != len(image_ids) or not queries or not targets:
            raise ValueError(f"{category}: invalid ordered gallery or empty native query/gallery")
        if any(t.get("target_img_id") != position for position, t in enumerate(targets)):
            raise ValueError(f"{category}: native gallery order differs from dataset split")
        if any(not isinstance(q.get("source_img_id"), int) or not isinstance(q.get("target_img_id"), int)
               or not 0 <= q["source_img_id"] < len(image_ids) or not 0 <= q["target_img_id"] < len(image_ids)
               for q in queries):
            raise ValueError(f"{category}: native query has invalid image positions")
        capture = _ArgsortCapture(original_numpy, len(queries))
        test_module.np = capture
    except Exception as error:
        capture_error = f"{type(error).__name__}: {error}"
        image_ids, queries, targets = [], [], []
    try:
        yield
    finally:
        if original_numpy is not None:
            test_module.np = original_numpy
        if capture_error is not None:
            receipt = {"schema": "habit-native-rankings/v1", "category": category,
                       "protocol": PROTOCOL, "capture_error": capture_error}
        else:
            try:
                expected = len(queries)
                assert capture is not None
                if capture.calls != expected * 2 or not capture.identical:
                    raise ValueError(f"native argsort passes differ or call count invalid: {capture.calls}")
                records = []
                metric_hits = {1: 0, 10: 0, 50: 0}
                for index, ((first_values, first_order), query) in enumerate(
                        zip(capture.rows, queries)):
                    reference, target = query["source_img_id"], query["target_img_id"]
                    if first_values.ndim != 1 or len(first_values) != len(targets) or first_order.shape != first_values.shape:
                        raise ValueError(f"native ranking has invalid shape at query {index}")
                    if not np.all(np.isfinite(first_values)) or first_values[reference] != first_values.dtype.type(10e10):
                        raise ValueError(f"native reference mask/finite sort inputs invalid at query {index}")
                    if int(first_order[-1]) != reference:
                        raise ValueError(f"masked reference is not last at query {index}")
                    rank = int(np.flatnonzero(first_order == target)[0]) + 1
                    for cutoff in metric_hits:
                        metric_hits[cutoff] += rank <= cutoff
                    records.append({
                        "annotation_index": index, "reference_id": image_ids[reference],
                        "target_id": image_ids[target], "model_input_text": query["mod"]["str"],
                        "target_rank": rank,
                        "top_results": [
                            {"rank": position, "image_id": image_ids[int(gallery)], "score": float(-first_values[gallery])}
                            for position, gallery in enumerate(first_order[:min(NATIVE_TOP_K, len(first_order) - 1)], 1)
                        ],
                    })
                if capture.matmul_calls != 1:
                    raise ValueError(f"native score matmul count invalid: {capture.matmul_calls}")
                score_matrix = np.stack([-row[0] for row in capture.rows])
                score_path = output.with_suffix(".scores.npz")
                atomic_npz(score_path, scores=score_matrix,
                         orders=np.stack([row[1] for row in capture.rows]),
                         query_ids=np.asarray([f"{index}:{image_ids[q['source_img_id']]}:{image_ids[q['target_img_id']]}"
                                               for index, q in enumerate(queries)]),
                         gallery_ids=np.asarray(image_ids),
                         reference_positions=np.asarray([q["source_img_id"] for q in queries], dtype=np.int32),
                         target_positions=np.asarray([q["target_img_id"] for q in queries], dtype=np.int32))
                receipt = {"schema": "habit-native-rankings/v1", "category": category,
                           "protocol": PROTOCOL, "capture": "pinned_test.py_argsort_return",
                           "native_matmul_calls": capture.matmul_calls,
                           "native_score_artifact": score_path.name,
                           "native_score_shape": list(score_matrix.shape),
                           "argsort_calls": capture.calls, "verified_identical_passes": expected,
                           "query_ids": [f"{index}:{image_ids[q['source_img_id']]}:{image_ids[q['target_img_id']]}"
                                         for index, q in enumerate(queries)],
                           "gallery_ids": image_ids, "queries": records,
                           "native_hit_counts": {f"r{k}": count for k, count in metric_hits.items()}}
            except Exception as error:
                receipt = {"schema": "habit-native-rankings/v1", "category": category,
                           "protocol": PROTOCOL, "capture_error": f"{type(error).__name__}: {error}"}
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        except OSError as error:
            print(f"[BLOCKED] HABIT native ranking receipt unavailable: {error}", file=sys.stderr)

def _verify_native_receipt(path: Path, category: str, recalls: dict[str, Any]) -> bool:
    try:
        receipt = json.loads(path.read_text())
        if receipt.get("capture_error"):
            return False
        counts = receipt["native_hit_counts"]
        total = len(receipt["queries"])
        if total == 0:
            raise ValueError("native ranking receipt has no queries")
        for cutoff in (1, 10, 50):
            key = f"{category}_r{cutoff}"
            expected = 100 * counts[f"r{cutoff}"] / total
            if key not in recalls or float(recalls[key]) != expected:
                raise ValueError(f"{key} does not match captured native ranks")
        return True
    except Exception as error:
        print(f"[BLOCKED] HABIT native ranking receipt invalid: {error}", file=sys.stderr)
        return False


def validate_checkpoint(checkpoint_root: Path, variant_id: str = DEFAULT_VARIANT) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(variant_id)
    if variant is None:
        blockers.append(f"unsupported HABIT variant: {variant_id}; valid: {list(CHECKPOINTS.keys())}")
        return blockers
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"HABIT checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"HABIT checkpoint is empty placeholder: {path}")
        return blockers
    expected_size = variant.get("expected_size_bytes")
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(f"HABIT checkpoint size mismatch: expected {expected_size}, got {path.stat().st_size}")
    expected_sha = variant.get("expected_sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"HABIT checkpoint SHA-256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(f"HABIT {variant_id} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"HABIT source pin invalid: {reason}"]


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
                blockers.append(f"HABIT dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"HABIT dataset requirement missing: {dataset_root / 'resized_image' / category}")
    # The evaluated subset and the constructor's input set are different questions: the native
    # `datasets.FashionIQ(...)` opens every category's files regardless of `--category`.
    from workbench.replay.common import constructor_input_blockers

    return list(dict.fromkeys(
        blockers + constructor_input_blockers(
            dataset_root, lambda path: f"HABIT dataset requirement missing: {path}")))


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
def _habit_category_dataset(datasets: Any, category: str):
    """Keep pinned HABIT construction semantics for one evaluated category, without unused caches."""
    cls = datasets.FashionIQ
    original_get_test_data = cls.get_test_data
    original_save_obj = datasets.save_obj

    def get_test_data(instance: Any, requested: str):
        if requested != category:
            return [], []
        return original_get_test_data(instance, requested)

    cls.get_test_data = get_test_data
    datasets.save_obj = lambda _obj, _path: None
    try:
        yield
    finally:
        cls.get_test_data = original_get_test_data
        datasets.save_obj = original_save_obj


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
    tensor_spool_provenance: dict[str, Any] | None = None
    with isolated_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        if os.environ.get("WORKBENCH_HABIT_ADAPTER_PAIR_OUT"):
            from contextlib import ExitStack
            from workbench.replay.tensor_spool import disk_backed_images

            spool_categories = categories if len(categories) == 1 else CATEGORIES
            image_counts = {"get_test_img": 0, "get_img": 0}
            for spool_category in spool_categories:
                captions = json.loads((dataset_root / "captions" / f"cap.{spool_category}.val.json").read_text())
                split = json.loads((dataset_root / "image_splits" / f"split.{spool_category}.val.json").read_text())
                image_counts["get_test_img"] += 2 * len(captions)
                image_counts["get_img"] += len(split)
            spool_stack = ExitStack()
            try:
                spool_receipts = [
                    spool_stack.enter_context(disk_backed_images(dataset_root, cache_root, datasets,
                        method_name=method_name, split="val", expected_count=expected_count))
                    for method_name, expected_count in image_counts.items()
                ]
                if len(categories) == 1:
                    with _habit_category_dataset(datasets, categories[0]):
                        dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
                else:
                    dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
            except Exception:
                spool_stack.close()
                raise
            for receipt in spool_receipts:
                receipt.evict()
        else:
            dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0, split="original-split")
        for category in categories:
            ranking_path: Path | None = None
            ranking_root = os.environ.get(NATIVE_RANKINGS_ENV)
            if ranking_root and len(categories) != 1:
                raise ValueError(f"{NATIVE_RANKINGS_ENV} requires exactly one category per invocation")
            if ranking_root:
                ranking_path = Path(ranking_root) / f"{category}.json"
                pair_root = os.environ.get("WORKBENCH_HABIT_ADAPTER_PAIR_OUT")
                if pair_root and category == "dress":
                    selected = set(range(min(5, len(dataset.test_queries_dress))))
                    with _capture_native_inputs(model, Path(pair_root) / "native_inputs.npz", selected):
                        with _capture_native_rankings(test_module, dataset, category, ranking_path):
                            recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
                else:
                    with _capture_native_rankings(test_module, dataset, category, ranking_path):
                        recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
                native_rankings_verified = _verify_native_receipt(ranking_path, category, recalls)
                pair_root = os.environ.get("WORKBENCH_HABIT_ADAPTER_PAIR_OUT")
                if pair_root and not os.environ.get("WORKBENCH_HABIT_NATIVE_STAGE"):
                    if category != "dress" or len(categories) != 1:
                        raise ValueError("HABIT paired Adapter evidence requires one dress invocation")
                    import numpy as np
                    from workbench.backend.operator_config import resolve_config
                    from workbench.composition.model_habit import Adapter

                    del model
                    if target_device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    pair_path = Path(pair_root)
                    pair_path.mkdir(parents=True, exist_ok=True)
                    query_rows = dataset.test_queries_dress
                    split = json.loads((Path(dataset.path) / "image_splits" / "split.dress.val.json").read_text())
                    query_positions = [int(row["source_img_id"]) for row in query_rows]
                    query_ids = [f"{index}:{split[row['source_img_id']]}:{split[row['target_img_id']]}"
                                 for index, row in enumerate(query_rows)]
                    def image_path(image_id: str) -> str:
                        prefixed_id = f"{category}_{image_id}"
                        prefix, suffix = prefixed_id.split("_", 1)
                        return str(Path(dataset.image_dir) / prefix / f"{suffix}.jpg")
                    query_paths = [image_path(split[position]) for position in query_positions]
                    gallery_paths = [image_path(item) for item in split]
                    adapter = Adapter(config=resolve_config(), checkpoint_id=variant["checkpoint_id"])
                    adapter.work_root = pair_path / "worker"
                    selected = list(dict.fromkeys(range(min(5, len(query_rows)))))
                    adapter._capture_spec = {"positions": selected, "output_path": str(pair_path / "adapter_query_inputs.npz")}
                    adapter_queries = adapter.encode_queries(query_paths,
                        [row["mod"]["str"] for row in query_rows], [None] * len(query_rows))
                    adapter._capture_spec = {"positions": selected, "output_path": str(pair_path / "adapter_gallery_inputs.npz")}
                    adapter_targets = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
                    adapter._capture_spec = None
                    adapter_scores = adapter.score(adapter_queries, adapter_targets)
                    repeated_adapter_scores = adapter.score(adapter_queries, adapter_targets)
                    if not np.array_equal(adapter_scores, repeated_adapter_scores):
                        raise ValueError("repeated production Adapter scores differ")
                    ref_positions = np.asarray(query_positions, dtype=np.int32)
                    adapter_scores[np.arange(len(query_rows)), ref_positions] = -10e10
                    np.savez(pair_path / "adapter_pair_scores.npz", scores=adapter_scores,
                             query_ids=np.asarray(query_ids), gallery_ids=np.asarray(split),
                             reference_positions=ref_positions,
                             target_positions=np.asarray([row["target_img_id"] for row in query_rows], dtype=np.int32),
                             repeated_scores_identical=np.asarray(True))
                    native_path = Path(ranking_root) / "dress.scores.npz"
                    with np.load(native_path, allow_pickle=False) as native:
                        native_scores = native["scores"]
                        native_orders = native["orders"]
                    adapter_orders = np.argsort(-adapter_scores, axis=1)
                    target_positions = np.asarray([row["target_img_id"] for row in query_rows], dtype=np.int32)
                    adapter_ranks = np.argmax(adapter_orders == target_positions[:, None], axis=1) + 1
                    native_ranks = np.argmax(native_orders == target_positions[:, None], axis=1) + 1
                    k = min(10, len(split))
                    topk_overlap = np.mean([
                        len(set(native_orders[row, :k]) & set(adapter_orders[row, :k])) / k
                        for row in range(len(query_rows))
                    ])
                    native_hits = {f"r{cutoff}": native_ranks <= cutoff for cutoff in (1, 10, 50)}
                    adapter_hits = {f"r{cutoff}": adapter_ranks <= cutoff for cutoff in (1, 10, 50)}
                    score_delta = adapter_scores - native_scores
                    topk_membership_equal = np.all(np.sort(native_orders[:, :k], axis=1) ==
                                                   np.sort(adapter_orders[:, :k], axis=1), axis=1)
                    topk_order_equal = np.all(native_orders[:, :k] == adapter_orders[:, :k], axis=1)
                    np.savez(pair_path / "paired_rank_metrics.npz",
                             native_ranks=native_ranks, adapter_ranks=adapter_ranks,
                             native_topk=native_orders[:, :k], adapter_topk=adapter_orders[:, :k],
                             score_delta=score_delta, rank_delta=adapter_ranks - native_ranks,
                             topk_membership_equal=topk_membership_equal,
                             topk_order_equal=topk_order_equal,
                             **{f"native_{name}_hits": values for name, values in native_hits.items()},
                             **{f"adapter_{name}_hits": values for name, values in adapter_hits.items()})
                    (pair_path / "paired_summary.json").write_text(json.dumps({
                        "schema": "habit-native-adapter-pair/v1", "category": "dress",
                        "query_count": len(query_rows), "gallery_count": len(split),
                        "denominator": len(query_rows), "batch_size": batch_size, "precision": "float32",
                        "native_score_shape": list(native_scores.shape),
                        "adapter_score_shape": list(adapter_scores.shape),
                        "adapter_score_repeats": 2, "adapter_score_repeats_identical": True,
                        "max_abs_score_delta": float(np.max(np.abs(score_delta))),
                        "mean_abs_score_delta": float(np.mean(np.abs(score_delta))),
                        "native_hit_counts": {name: int(values.sum()) for name, values in native_hits.items()},
                        "adapter_hit_counts": {name: int(values.sum()) for name, values in adapter_hits.items()},
                        "hit_flip_counts": {name: int(np.count_nonzero(native_hits[name] != adapter_hits[name]))
                                            for name in native_hits},
                        "top10_membership_mismatch_count": int(np.count_nonzero(~topk_membership_equal)),
                        "top10_order_mismatch_count": int(np.count_nonzero(~topk_order_equal)),
                        "top10_overlap_fraction": float(topk_overlap),
                        "rank_exact_match_count": int(np.sum(native_ranks == adapter_ranks)),
                    }, indent=2) + "\n", encoding="utf-8")
                    spool_stack.close()
                    tensor_spool_provenance = {method: dict(receipt) for method, receipt in zip(image_counts, spool_receipts)}
                    adapter.close()
            else:
                native_rankings_verified = False
                recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
            results.append(
                ReplayResult(
                    model_id="habit",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    provenance={
                        "native_rankings": {
                            "status": "VERIFIED" if native_rankings_verified else "BLOCKED",
                            "path": str(ranking_path) if ranking_root else None,
                        },
                        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
                        "evaluation_noise_pct": 0,
                        "checkpoint_sha256": (
                            sha256_file(model_path) if model_path.is_file() and model_path.stat().st_size else None
                        ),
                        "checkpoint_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
                        "preprocessing": "targetpad transform, target_ratio 1.25, input_dim 224",
                        "tensor_spool": tensor_spool_provenance if os.environ.get("WORKBENCH_HABIT_ADAPTER_PAIR_OUT") else None,
                        "text_processor": {
                            "config": PROCESSOR_CONFIGS["pretrain"],
                            "loaded_via": processor_boundary,
                            "note": "the pinned BlipCaptionProcessor from the model class's own "
                                    "config; no second BLIP-2 model is constructed and no "
                                    "/root/bert-base-uncased asset is read",
                        },
                    },
                    per_query_published=native_rankings_verified,
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
        print(f"[READY] HABIT preflight passed (variant={args.variant}); no inference executed")
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
        "model_id": "habit",
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
