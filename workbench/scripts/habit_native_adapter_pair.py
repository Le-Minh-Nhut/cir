"""Run one HABIT dress source-evaluator/production-Adapter evidence pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.replay.habit import evaluate_categories


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


import subprocess

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import require_verified_model_interpreter
from workbench.composition.model_habit import Adapter
import numpy as np
from workbench.scripts.third_model_native_observer import atomic_json
from workbench.scripts.third_model_native_observer import atomic_npz



def _record_adapter_controls(pair_root: Path, arrays: dict[str, np.ndarray],
                             controls: dict[str, Any]) -> None:
    atomic_npz(pair_root / "adapter_control_deltas.npz", **arrays)
    atomic_json(pair_root / "adapter_control_comparison.json", controls)
    failed = [key for key, value in (("full_forward_repeat", controls["full_forward_repeat_equal"]),
                                     ("B1_BN", controls["b1_bn_equal"])) if not value]
    if failed:
        raise ValueError(f"HABIT Adapter control failed: {failed}; measurements=adapter_control_comparison.json")

def _native_stage(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                  output: Path, batch_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / "HABIT-FIQ_N0.2.pt"
    source_files = [source_root / "test.py", source_root / "lavis/models/blip2_models/HABIT.py"]
    population_files = [dataset_root / "image_splits" / f"split.{category}.val.json"
                        for category in ("dress", "shirt", "toptee")]
    population_files += [dataset_root / "captions" / f"cap.{category}.val.json"
                         for category in ("dress", "shirt", "toptee")]
    manifest = {"role": "fresh_native_primary", "category": "dress", "seed": None,
        "partition": "FashionIQ original-split dress.val", "batch_size": batch_size,
        "precision": "float32", "runtime_python": sys.executable,
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
        "replay_sha256": sha256(Path(__file__).resolve().parents[1] / "replay/habit.py"),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "adapter_sha256": sha256(Path(__file__).resolve().parents[1] / "composition/model_habit.py"),
        "population_files_sha256": {str(path.relative_to(dataset_root)): sha256(path) for path in population_files}}
    atomic_json(output / "native/manifest.json", manifest)
    os.environ["WORKBENCH_HABIT_NATIVE_STAGE"] = "1"
    os.environ["WORKBENCH_HABIT_NATIVE_RANKINGS_OUT"] = str(output / "native")
    os.environ["WORKBENCH_HABIT_ADAPTER_PAIR_OUT"] = str(output / "adapter")
    results, _ = evaluate_categories(checkpoint_root, dataset_root, source_root,
        categories=("dress",), variant_id="fiq_n02", batch_size=batch_size, device="cuda")
    atomic_json(output / "native/native-stage.json", {
        "native_pid": os.getpid(), "result": results[0].__dict__,
    })


def _controller_adapter(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                        output: Path, batch_size: int, native_root: Path | None = None) -> dict[str, Any]:
    native_root = native_root or output / "native"
    pair_root = output / "adapter"
    receipt = json.loads((native_root / "dress.json").read_text(encoding="utf-8"))
    if (receipt.get("capture_error") or receipt.get("native_matmul_calls") != 1
            or receipt.get("argsort_calls") != 2 * len(receipt.get("queries", []))
            or receipt.get("verified_identical_passes") != len(receipt.get("queries", []))):
        raise RuntimeError(f"HABIT native stage incomplete: {receipt.get('capture_error', receipt)}")
    with np.load(native_root / "dress.scores.npz", allow_pickle=False) as native:
        query_ids = native["query_ids"].astype(str).tolist()
        gallery_ids = native["gallery_ids"].astype(str).tolist()
        refs, targets = native["reference_positions"].copy(), native["target_positions"].copy()
        native_scores, native_orders = native["scores"].copy(), native["orders"].copy()
    query_count, gallery_count = len(query_ids), len(gallery_ids)
    if (receipt.get("query_ids") != query_ids or query_count != len(receipt["queries"])
            or query_ids != [f"{index}:{row['reference_id']}:{row['target_id']}"
                             for index, row in enumerate(receipt["queries"])]
            or refs.shape != (query_count,) or targets.shape != (query_count,)
            or native_scores.shape != (query_count, gallery_count)
            or native_orders.shape != (query_count, gallery_count)
            or not np.isfinite(native_scores).all()
            or not np.array_equal(np.sort(native_orders, axis=1),
                                  np.broadcast_to(np.arange(gallery_count), native_orders.shape))
            or not np.array_equal(refs, [gallery_ids.index(row["reference_id"]) for row in receipt["queries"]])
            or not np.array_equal(targets, [gallery_ids.index(row["target_id"]) for row in receipt["queries"]])
            or not np.array_equal(native_orders[np.arange(query_count), -1], refs)):
        raise ValueError("HABIT native receipt/score identities, shapes, or positions differ")
    def image_path(image_id: str) -> str:
        prefix, suffix = f"dress_{image_id}".split("_", 1)
        return str(dataset_root / "resized_image" / prefix / f"{suffix}.jpg")
    query_paths = [image_path(q["reference_id"]) for q in receipt["queries"]]
    gallery_paths = [image_path(image_id) for image_id in gallery_ids]
    texts = [q["model_input_text"] for q in receipt["queries"]]
    os.environ["WORKBENCH_COMPOSITION_BATCH_SIZE"] = str(batch_size)
    adapter = Adapter(config=resolve_config(), checkpoint_id="fiq_n02")
    adapter.work_root = pair_root / "worker"
    selected = list(range(min(5, len(query_paths))))
    try:
        adapter._capture_spec = {"positions": selected, "output_path": str(pair_root / "adapter_query_inputs.npz")}
        queries = adapter.encode_queries(query_paths, texts, [None] * len(query_paths))
        adapter._capture_spec = {"positions": selected, "output_path": str(pair_root / "adapter_gallery_inputs.npz")}
        targets_features = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        adapter._capture_spec = None
        query_repeat = adapter.encode_queries(query_paths, texts, [None] * len(query_paths))
        target_repeat = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        query_repeat_equal = np.array_equal(queries, query_repeat)
        target_repeat_equal = np.array_equal(targets_features, target_repeat)
        forward_repeat = query_repeat_equal and target_repeat_equal
        query_repeat_delta = np.abs(queries - query_repeat)
        target_repeat_delta = np.abs(targets_features - target_repeat)
        old_batch = adapter.batch_size
        try:
            adapter.batch_size = 1
            b1_query = adapter.encode_queries(query_paths[:5], texts[:5], [None] * min(5, len(query_paths)))
            b1_target = adapter.encode_targets(gallery_paths[:5], [None] * min(5, len(gallery_paths)))
        finally:
            adapter.batch_size = old_batch
        b1_query_delta = np.abs(queries[:5] - b1_query)
        b1_target_delta = np.abs(targets_features[:5] - b1_target)
        b1_query_equal = np.array_equal(queries[:5], b1_query)
        b1_target_equal = np.array_equal(targets_features[:5], b1_target)
        b1_bn = b1_query_equal and b1_target_equal
        arrays = {
            "query_repeat_delta": query_repeat_delta, "target_repeat_delta": target_repeat_delta,
            "b1_query_delta": b1_query_delta, "b1_target_delta": b1_target_delta,
            "query_repeat_expected": queries, "query_repeat_actual": query_repeat,
            "target_repeat_expected": targets_features, "target_repeat_actual": target_repeat,
            "b1_query_expected": queries[:5], "b1_query_actual": b1_query,
            "b1_target_expected": targets_features[:5], "b1_target_actual": b1_target}
        controls = {
            "query_repeat_equal": bool(query_repeat_equal), "target_repeat_equal": bool(target_repeat_equal),
            "query_repeat_max_abs_delta": float(np.max(query_repeat_delta)),
            "query_repeat_mean_abs_delta": float(np.mean(query_repeat_delta)),
            "target_repeat_max_abs_delta": float(np.max(target_repeat_delta)),
            "target_repeat_mean_abs_delta": float(np.mean(target_repeat_delta)),
            "b1_query_equal": bool(b1_query_equal), "b1_target_equal": bool(b1_target_equal),
            "b1_query_max_abs_delta": float(np.max(b1_query_delta)),
            "b1_query_mean_abs_delta": float(np.mean(b1_query_delta)),
            "b1_target_max_abs_delta": float(np.max(b1_target_delta)),
            "b1_target_mean_abs_delta": float(np.mean(b1_target_delta)),
            "full_forward_repeat_equal": bool(forward_repeat), "b1_bn_equal": bool(b1_bn),
            "tolerance": {"comparison": "exact array equality", "atol": 0, "rtol": 0}}
        _record_adapter_controls(pair_root, arrays, controls)
        scores = adapter.score(queries, targets_features)
        scores[np.arange(len(refs)), refs] = -10e10
        orders = np.argsort(-scores, axis=1)
        if not forward_repeat or not b1_bn:
            failed = [key for key, value in (("full_forward_repeat", forward_repeat), ("B1_BN", b1_bn))
                      if not value]
            raise ValueError(f"HABIT Adapter control failed: {failed}; measurements=adapter_control_comparison.json")
    finally:
        adapter.close()
    atomic_npz(pair_root / "adapter_pair_scores.npz", scores=scores, orders=orders,
        query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids), reference_positions=refs,
        target_positions=targets, adapter_forward_repeat_equal=np.asarray([forward_repeat]),
        adapter_b1_bn_equal=np.asarray([b1_bn]))
    native_input_path = native_root.parent / "adapter/native_inputs.npz"
    with np.load(native_input_path) as native_inputs, \
         np.load(pair_root / "adapter_query_inputs.npz") as query_inputs, \
         np.load(pair_root / "adapter_gallery_inputs.npz") as gallery_inputs:
        checks = {"query_pixels": np.array_equal(native_inputs["query_pixels"], query_inputs["pixels"]),
            "query_positions": np.array_equal(native_inputs["query_positions"], query_inputs["positions"]),
            "input_ids": np.array_equal(native_inputs["input_ids"], query_inputs["input_ids"]),
            "attention_masks": np.array_equal(native_inputs["attention_masks"], query_inputs["attention_masks"]),
            "token_texts": np.array_equal(native_inputs["token_texts"], query_inputs["token_texts"]),
            "gallery_pixels": np.array_equal(native_inputs["gallery_pixels"], gallery_inputs["pixels"]),
            "gallery_positions": np.array_equal(native_inputs["gallery_positions"], gallery_inputs["positions"])}
    atomic_json(pair_root / "input_comparison.json", checks)
    if not all(checks.values()):
        raise ValueError(f"HABIT source/Adapter actual inputs differ: {checks}")
    native_ranks = np.argmax(native_orders == targets[:, None], axis=1) + 1
    adapter_ranks = np.argmax(orders == targets[:, None], axis=1) + 1
    native_top10, adapter_top10 = native_orders[:, :10], orders[:, :10]
    return {"schema": "habit-native-adapter-pair/v2", "query_count": len(query_ids),
        "gallery_count": len(gallery_ids), "denominator": len(query_ids), "batch_size": batch_size,
        "precision": "float32", "adapter_forward_repeat_equal": bool(forward_repeat),
        "adapter_b1_bn_equal": bool(b1_bn), "native_hit_counts": {str(k): int(np.count_nonzero(native_ranks <= k)) for k in (1,10,50)},
        "adapter_hit_counts": {str(k): int(np.count_nonzero(adapter_ranks <= k)) for k in (1,10,50)},
        "hit_flip_counts": {str(k): int(np.count_nonzero((native_ranks <= k) != (adapter_ranks <= k))) for k in (1,10,50)},
        "rank_exact_match_count": int(np.count_nonzero(native_ranks == adapter_ranks)),
        "top10_membership_equal_count": int(sum(set(a.tolist()) == set(b.tolist()) for a,b in zip(native_top10, adapter_top10))),
        "top10_order_equal_count": int(np.count_nonzero(np.all(native_top10 == adapter_top10, axis=1))),
        "max_abs_score_delta": float(np.max(np.abs(native_scores-scores))),
        "mean_abs_score_delta": float(np.mean(np.abs(native_scores-scores)))}
def run(checkpoint_root: Path, dataset_root: Path, source_root: Path, output: Path,
        batch_size: int = 64, resume_native_root: Path | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    interpreter = require_verified_model_interpreter(model_by_id("habit"), config=resolve_config())
    if resume_native_root is None:
        command = [interpreter, "-m", "workbench.scripts.habit_native_adapter_pair", "--native-stage",
            "--checkpoint-root", str(checkpoint_root), "--dataset-root", str(dataset_root),
            "--source-root", str(source_root), "--output", str(output), "--batch-size", str(batch_size)]
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
        native_root = output / "native"
        stage = json.loads((native_root / "native-stage.json").read_text(encoding="utf-8"))
    else:
        native_root = resume_native_root.resolve()
        required = (native_root / "manifest.json", native_root / "native-stage.json",
                    native_root / "dress.scores.npz", native_root / "dress.json")
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"HABIT resume archive incomplete: {native_root}")
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        stage = json.loads(required[1].read_text(encoding="utf-8"))
        if (manifest.get("role") != "fresh_native_primary"
                or manifest.get("partition") != "FashionIQ original-split dress.val"
                or manifest.get("checkpoint_sha256") != sha256(checkpoint_root / "HABIT-FIQ_N0.2.pt")
                or manifest.get("source_sha256", {}).get("test.py") != sha256(source_root / "test.py")
                or manifest.get("source_sha256", {}).get("lavis/models/blip2_models/HABIT.py")
                   != sha256(source_root / "lavis/models/blip2_models/HABIT.py")
                or manifest.get("population_files_sha256") != {
                    str(path.relative_to(dataset_root)): sha256(path)
                    for path in ([dataset_root / "image_splits" / f"split.{category}.val.json"
                                  for category in ("dress", "shirt", "toptee")]
                                 + [dataset_root / "captions" / f"cap.{category}.val.json"
                                    for category in ("dress", "shirt", "toptee")])}
                or manifest.get("batch_size") != 32 or manifest.get("precision") != "float32"
                or stage.get("native_pid") == os.getpid()):
            raise RuntimeError("HABIT resume archive provenance differs from current source/checkpoint/population")
    if stage.get("native_pid") == os.getpid():
        raise RuntimeError("HABIT native source stage shared the Adapter controller process")
    native = _controller_adapter(checkpoint_root, dataset_root, source_root, output, batch_size,
                                 native_root=native_root)
    receipt_path, scores_path = native_root / "dress.json", native_root / "dress.scores.npz"
    adapter_path = output / "adapter/adapter_pair_scores.npz"
    provenance = {
        "schema": "habit-native-adapter-pair-seal/v2", "model_id": "habit",
        "checkpoint_id": "fiq_n02", "category": "dress", "native_stage": stage,
        "native_archive": str(native_root),
        "native_source_manifest_sha256": sha256(native_root / "manifest.json"),
        "source_root": str(source_root.resolve()), "dataset_root": str(dataset_root.resolve()),
        "checkpoint": str((checkpoint_root / "HABIT-FIQ_N0.2.pt").resolve()),
        "checkpoint_sha256": sha256(checkpoint_root / "HABIT-FIQ_N0.2.pt"),
        "source_files_sha256": {"test.py": sha256(source_root / "test.py"),
            "HABIT.py": sha256(source_root / "lavis/models/blip2_models/HABIT.py")},
        "controller_python": sys.executable, "native_python": interpreter,
        "native": native, "native_receipt_sha256": sha256(receipt_path),
        "native_scores_sha256": sha256(scores_path),
        "adapter_scores": str(adapter_path.relative_to(output)),
        "artifact_sha256": {str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "seal.json"},
    }
    atomic_json(output / "seal.json", provenance)
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-native-root", type=Path,
                        help="Reuse verified native archive; skip native source stage")
    parser.add_argument("--native-stage", action="store_true")
    args = parser.parse_args(argv)
    if args.native_stage:
        _native_stage(args.checkpoint_root, args.dataset_root, args.source_root, args.output, args.batch_size)
        return 0
    result = run(args.checkpoint_root, args.dataset_root, args.source_root, args.output,
                 args.batch_size, resume_native_root=args.resume_native_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


