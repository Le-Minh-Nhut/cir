"""Run one source-evaluator/production-Adapter ConeSep FashionIQ pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.replay.conesep import CHECKPOINTS, evaluate_categories


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


import subprocess

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import require_verified_model_interpreter
from workbench.composition.model_conesep import Adapter

from workbench.scripts.third_model_native_observer import atomic_json, atomic_npz, write_forward_control_diagnostics



def _native_stage(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                  output: Path, category: str, batch_size: int) -> None:
    checkpoint = checkpoint_root / CHECKPOINTS["fiq_n02"]["filename"]
    source_files = [source_root / "test.py", source_root / "lavis/models/blip2_models/ConeSep.py"]
    split = dataset_root / "image_splits" / f"split.{category}.val.json"
    population_files = [split, dataset_root / "captions" / f"cap.{category}.val.json"]
    manifest = {"role": "fresh_native_primary", "category": category,
        "partition": f"FashionIQ original-split {category}.val", "seed": None,
        "batch_size": batch_size, "precision": "float32", "runtime_python": sys.executable,
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
        "replay_sha256": sha256(Path(__file__).resolve().parents[1] / "replay/conesep.py"),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "adapter_sha256": sha256(Path(__file__).resolve().parents[1] / "composition/model_conesep.py"),
        "population_files_sha256": {str(path.relative_to(dataset_root)): sha256(path) for path in population_files}}
    native_root = output / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    atomic_json(native_root / "manifest.json", manifest)
    os.environ["WORKBENCH_CONESEP_NATIVE_STAGE"] = "1"
    os.environ["WORKBENCH_CONESEP_NATIVE_RANKINGS_OUT"] = str(native_root)
    os.environ["WORKBENCH_CONESEP_ADAPTER_PAIR_OUT"] = str(output / "adapter")
    results, _ = evaluate_categories(checkpoint_root, dataset_root, source_root,
        categories=(category,), variant_id="fiq_n02", batch_size=batch_size, device="cuda")
    atomic_json(native_root / "native-stage.json", {
        "native_pid": os.getpid(), "result": results[0].__dict__,
    })


def _controller_adapter(checkpoint_root: Path, dataset_root: Path, source_root: Path,
                        output: Path, category: str, batch_size: int) -> dict[str, Any]:
    native_root, adapter_root = output / "native", output / "adapter"
    receipt = json.loads((native_root / f"{category}.json").read_text(encoding="utf-8"))
    with np.load(native_root / f"{category}.scores.npz") as source, \
         np.load(native_root / f"{category}.inputs.npz") as source_inputs:
        query_ids, gallery_ids = source["query_ids"].astype(str).tolist(), source["gallery_ids"].astype(str).tolist()
        refs, targets = source["reference_positions"].copy(), source["target_positions"].copy()
        native_scores, native_orders = source["scores"].copy(), source["orders"].copy()
        native_input_values = {key: source_inputs[key].copy() for key in source_inputs.files}
    queries = json.loads((native_root / f"{category}.queries.json").read_text(encoding="utf-8"))
    if receipt.get("matmul_calls") != 1 or not receipt.get("identical_repeat_passes"):
        raise RuntimeError(f"ConeSep native source capture incomplete: {receipt}")
    expected_query_ids = [f"{index}:{row['source_id']}:{row['target_id']}" for index, row in enumerate(queries)]
    if (len(queries) != len(query_ids) or receipt.get("query_count") != len(query_ids)
            or query_ids != expected_query_ids
            or not np.array_equal(refs, [gallery_ids.index(row["source_id"]) for row in queries])
            or not np.array_equal(targets, [gallery_ids.index(row["target_id"]) for row in queries])):
        raise ValueError("ConeSep native query identities/count/positions differ")
    def image_path(image_id: str) -> str:
        prefix, suffix = f"{category}_{image_id}".split("_", 1)
        return str(dataset_root / "resized_image" / prefix / f"{suffix}.jpg")
    query_paths = [image_path(row["source_id"]) for row in queries]
    gallery_paths = [image_path(image_id) for image_id in gallery_ids]
    texts = [row["text"] for row in queries]
    os.environ["WORKBENCH_COMPOSITION_BATCH_SIZE"] = str(batch_size)
    adapter = Adapter(resolve_config(), "fiq_n02")
    adapter.work_root = adapter_root / "worker"
    selected = list(range(min(5, len(query_paths))))
    try:
        adapter._capture_spec = {"positions": selected, "output_path": str(adapter_root / "adapter_query_inputs.npz")}
        query_features = adapter.encode_queries(query_paths, texts, [None] * len(query_paths))
        adapter._capture_spec = {"positions": selected, "output_path": str(adapter_root / "adapter_gallery_inputs.npz")}
        target_features = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        adapter._capture_spec = None
        query_repeat = adapter.encode_queries(query_paths, texts, [None] * len(texts))
        target_repeat = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        forward_repeat = bool(np.array_equal(query_features, query_repeat) and np.array_equal(target_features, target_repeat))
        full_batch = adapter.batch_size
        try:
            adapter.batch_size = 1
            query_b1 = adapter.encode_queries(query_paths[:5], texts[:5], [None] * min(5, len(texts)))
            target_b1 = adapter.encode_targets(gallery_paths[:5], [None] * min(5, len(gallery_paths)))
        finally:
            adapter.batch_size = full_batch
        control_flags = write_forward_control_diagnostics(adapter_root / "forward_controls.json",
            {"query_repeat": query_features, "target_repeat": target_features,
             "query_b1_bn": query_features[:5], "target_b1_bn": target_features[:5]},
            {"query_repeat": query_repeat, "target_repeat": target_repeat,
             "query_b1_bn": query_b1, "target_b1_bn": target_b1})
        forward_repeat = control_flags["query_repeat"] and control_flags["target_repeat"]
        b1_equal = control_flags["query_b1_bn"] and control_flags["target_b1_bn"]
        scores = adapter.score(query_features, target_features)
        if not forward_repeat or not b1_equal:
            raise ValueError("ConeSep Adapter forward repeat/B1-BN control failed")
    finally:
        adapter._capture_spec = None
        adapter.close()
    scores[np.arange(len(refs)), refs] = -10e10
    orders = np.argsort(-scores, axis=1)
    atomic_npz(adapter_root / "adapter_pair_scores.npz", scores=scores, orders=orders,
        query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids), reference_positions=refs,
        target_positions=targets, adapter_forward_repeat_equal=np.asarray([forward_repeat]),
        adapter_b1_bn_equal=np.asarray([b1_equal]))
    with np.load(adapter_root / "adapter_query_inputs.npz") as query_inputs, \
         np.load(adapter_root / "adapter_gallery_inputs.npz") as gallery_inputs:
        checks = {"query_pixels": np.array_equal(native_input_values["query_pixels"], query_inputs["pixels"]),
            "query_positions": np.array_equal(native_input_values["query_positions"], query_inputs["positions"]),
            "input_ids": np.array_equal(native_input_values["input_ids"], query_inputs["input_ids"]),
            "attention_masks": np.array_equal(native_input_values["attention_masks"], query_inputs["attention_masks"]),
            "token_texts": np.array_equal(native_input_values["token_texts"], query_inputs["token_texts"]),
            "gallery_pixels": np.array_equal(native_input_values["gallery_pixels"], gallery_inputs["pixels"]),
            "gallery_positions": np.array_equal(native_input_values["gallery_positions"], gallery_inputs["positions"])}
    atomic_json(adapter_root / "input_comparison.json", checks)
    if not all(checks.values()):
        raise ValueError(f"ConeSep source/Adapter selected inputs differ: {checks}")
    native_rank = np.argmax(native_orders == targets[:, None], axis=1) + 1
    adapter_rank = np.argmax(orders == targets[:, None], axis=1) + 1
    native_top10, adapter_top10 = native_orders[:, :10], orders[:, :10]
    return {"query_count": len(query_ids), "gallery_count": len(gallery_ids), "denominator": len(query_ids),
        "adapter_batch_size": batch_size, "precision": "float32", "adapter_forward_repeat_equal": forward_repeat,
        "adapter_b1_bn_equal": b1_equal, "native_hit_counts": {str(k): int(np.count_nonzero(native_rank <= k)) for k in (1,10,50)},
        "adapter_hit_counts": {str(k): int(np.count_nonzero(adapter_rank <= k)) for k in (1,10,50)},
        "hit_flip_counts": {str(k): int(np.count_nonzero((native_rank <= k) != (adapter_rank <= k))) for k in (1,10,50)},
        "rank_exact_match_count": int(np.count_nonzero(native_rank == adapter_rank)),
        "top10_membership_equal_count": int(sum(set(a.tolist()) == set(b.tolist()) for a,b in zip(native_top10, adapter_top10))),
        "top10_order_equal_count": int(np.count_nonzero(np.all(native_top10 == adapter_top10, axis=1))),
        "max_abs_score_delta": float(np.max(np.abs(native_scores - scores))),
        "mean_abs_score_delta": float(np.mean(np.abs(native_scores - scores)))}
def run(checkpoint_root: Path, dataset_root: Path, source_root: Path, output: Path,
        category: str = "dress", batch_size: int = 64) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    interpreter = require_verified_model_interpreter(model_by_id("conesep"), config=resolve_config())
    command = [interpreter, "-m", "workbench.scripts.conesep_native_adapter_pair", "--native-stage",
        "--checkpoint-root", str(checkpoint_root), "--dataset-root", str(dataset_root),
        "--source-root", str(source_root), "--output", str(output), "--category", category,
        "--batch-size", str(batch_size)]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    try:
        stage = json.loads((output / "native/native-stage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("ConeSep native stage receipt unreadable") from error
    if stage.get("native_pid") == os.getpid():
        raise RuntimeError("ConeSep native source stage shared the Adapter controller process")
    native = _controller_adapter(checkpoint_root, dataset_root, source_root, output, category, batch_size)
    variant = CHECKPOINTS["fiq_n02"]
    checkpoint = checkpoint_root / variant["filename"]
    sources = [source_root / "test.py", source_root / "lavis/models/blip2_models/ConeSep.py"]
    provenance = {"model_id": "conesep", "checkpoint_id": "fiq_n02", "category": category,
        "source_root": str(source_root.resolve()), "dataset_root": str(dataset_root.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in sources},
        "controller_python": sys.executable, "native_python": interpreter,
        "native_stage": stage, "metrics": native,
        "artifact_sha256": {str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "seal.json"}}
    atomic_json(output / "seal.json", provenance)
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="dress")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--native-stage", action="store_true")
    args = parser.parse_args(argv)
    if args.native_stage:
        _native_stage(args.checkpoint_root, args.dataset_root, args.source_root,
            args.output, args.category, args.batch_size)
        return 0
    print(json.dumps(run(args.checkpoint_root, args.dataset_root, args.source_root,
        args.output, args.category, args.batch_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
