"""Resume DCNet adapter scoring from sealed native score archives without rerunning native GPU evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.composition.model_dcnet import Adapter


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


import sys


def _require_distinct_controller(controller_executable: str, model_interpreter: str | None) -> None:
    if model_interpreter is None:
        raise RuntimeError("DCNet verified model interpreter is unresolved")
    if Path(controller_executable).resolve() == Path(model_interpreter).resolve():
        raise RuntimeError("DCNet adapter controller must not run in the model interpreter")


def _load_native_category(native_root: Path, category: str, identity: dict[str, Any],
                          expected_archives: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                     list[str], list[dict[str, Any]]]:
    archive_path = native_root / identity["native_final_score_file"]
    component_path = native_root / f"native-{category}-scores.npz"
    for label, path in (("final", archive_path), ("components", component_path)):
        if _file_sha256(path) != expected_archives[label]:
            raise ValueError(f"{category}: {label} native archive differs from sealed recovery plan")
    if expected_archives["final"] != identity["native_final_score_sha256"]:
        raise ValueError(f"{category}: native receipt final archive hash mismatch")
    with np.load(component_path, allow_pickle=False) as archive:
        gallery_ids = archive["gallery_ids"].tolist()
        query_meta = json.loads(archive["query_metadata_json"].item())
    with np.load(archive_path, allow_pickle=False) as archive:
        scores = archive["scores"]
        top50 = archive["top50"]
        derived_order = archive["derived_order"]
    if scores.shape != (identity["query_count"], identity["gallery_count"]):
        raise ValueError(f"{category}: native final score dimensions mismatch")
    if top50.shape != (identity["query_count"], min(50, identity["gallery_count"])):
        raise ValueError(f"{category}: native observed top50 dimensions mismatch")
    if derived_order.shape != scores.shape or any(
            sorted(row.tolist()) != list(range(scores.shape[1])) for row in derived_order):
        raise ValueError(f"{category}: native derived full order is not a complete permutation")
    if len(query_meta) != identity["query_count"] or len(gallery_ids) != identity["gallery_count"]:
        raise ValueError(f"{category}: native ID/archive dimensions mismatch")
    if _digest(gallery_ids) != identity["gallery_order_sha256"] or _digest(query_meta) != identity["query_order_sha256"]:
        raise ValueError(f"{category}: native source ID ordering mismatch")
    return scores, top50, derived_order, gallery_ids, query_meta

def _hit_flips(native: np.ndarray, adapted: np.ndarray, targets: np.ndarray,
               cutoffs: tuple[int, ...] = (10, 50)) -> dict[int, dict[str, list[int]]]:
    if native.shape != adapted.shape or native.ndim != 2 or targets.shape != (len(native),):
        raise ValueError("native, adapted, and target score dimensions must match")
    native_order = np.argsort(-native, axis=1, kind="stable")
    adapter_order = np.argsort(-adapted, axis=1, kind="stable")
    flips = {}
    for k in cutoffs:
        if not isinstance(k, int) or not 1 <= k <= native.shape[1]:
            raise ValueError(f"hit cutoff must be between 1 and gallery size: {k}")
        native_hits = np.any(native_order[:, :k] == targets[:, None], axis=1)
        adapter_hits = np.any(adapter_order[:, :k] == targets[:, None], axis=1)
        flips[k] = {"gained": np.flatnonzero(~native_hits & adapter_hits).tolist(),
                    "lost": np.flatnonzero(native_hits & ~adapter_hits).tolist()}
    return flips



def recover(native_root: Path, checkpoint: Path, output: Path,
            archive_hashes: dict[str, dict[str, str]]) -> dict[str, Any]:
    native_root, checkpoint, output = native_root.resolve(), checkpoint.resolve(), output.resolve()
    receipt_path = native_root / "native-source-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    digest = hashlib.sha256()
    with (checkpoint / "trained_model.pth").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    expected_checkpoint = digest.hexdigest()
    if receipt["status"] != "native_source_scores_complete" or receipt["checkpoint_sha256"] != expected_checkpoint:
        raise ValueError("native receipt/checkpoint identity mismatch")
    config = resolve_config()
    adapter = Adapter(config, checkpoint_id="fashioniq_run_directory")
    adapter.prepare_identity()
    verified_interpreter = adapter.require_verified_interpreter()
    _require_distinct_controller(sys.executable, verified_interpreter)
    adapter.work_root = output.parent / "adapter-work"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "adapter_scores_complete", "recovered_from": str(native_root),
                              "native_receipt_sha256": hashlib.sha256((native_root / "native-source-receipt.json").read_bytes()).hexdigest(),
                              "checkpoint_sha256": expected_checkpoint, "categories": {}}
    try:
        for category, identity in receipt["categories"].items():
            native, native_top, native_order, gallery_ids, query_meta = _load_native_category(
                native_root, category, identity, archive_hashes[category])
            caption_file = Path(config.FASHIONIQ_ROOT) / "captions" / f"cap.{category}.glove.val.pkl"
            with caption_file.open("rb") as stream:
                source_rows = pickle.load(stream)
            if len(source_rows) != len(query_meta):
                raise ValueError(f"{category}: source query count mismatch")
            for index, (source_row, archived) in enumerate(zip(source_rows, query_meta)):
                if archived != {"category": category, "row": index, "candidate": source_row["candidate"],
                                "target": source_row["target"], "reference": source_row["candidate"]}:
                    raise ValueError(f"{category}: source row {index} does not match native archive")
            query_ids = [row["candidate"] for row in source_rows]
            query_texts = [" and ".join(row["captions"]) for row in source_rows]
            root = Path(config.FASHIONIQ_ROOT) / "resized_images"
            query_paths = [str(root / f"{image_id}.jpg") for image_id in query_ids]
            gallery_paths = [str(root / f"{image_id}.jpg") for image_id in gallery_ids]
            gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
            reference_indices = [gallery_index[image_id] for image_id in query_ids]
            adapter.batch_size = 8
            queries = adapter.encode_queries(query_paths, query_texts, [None] * len(query_paths))
            adapter.batch_size = 100
            targets = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
            adapter.batch_size = 8
            adapted = adapter.score(queries, targets, reference_indices=reference_indices)
            if native.shape != adapted.shape:
                raise ValueError(f"{category}: score matrix shape mismatch {native.shape} != {adapted.shape}")
            adapter_order = np.argsort(-adapted, axis=1, kind="stable")
            adapter_top = adapter_order[:, :native_top.shape[1]]
            expected = np.asarray([gallery_index[row["target"]] for row in source_rows])
            flips = _hit_flips(native, adapted, expected)
            adapter_ranks = adapter._ranks_from_scores(adapted, [[int(index)] for index in expected])
            hits = lambda top: [int(np.count_nonzero(np.any(top[:, :k] == expected[:, None], axis=1))) for k in (10, 50)]
            rank_hits = lambda ranks: [int(np.count_nonzero(np.asarray(ranks) <= k)) for k in (10, 50)]
            score_file = output.with_name(f"{output.stem}-{category}-scores.npz")
            np.savez_compressed(score_file, native_observed_scores=native, native_observed_top50=native_top,
                                native_derived_order=native_order, adapter_scores=adapted,
                                adapter_derived_order=adapter_order, adapter_top50=adapter_top,
                                query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                                reference_ids=np.asarray(query_ids),
                                target_ids=np.asarray([row["target"] for row in source_rows]))
            result["categories"][category] = {
                "queries": len(source_rows), "gallery": len(gallery_ids),
                "native_observed_hits_at_10_50": hits(native_top),
                "adapter_exact_rank_hits_at_10_50": rank_hits(adapter_ranks),
                "adapter_derived_order_hits_at_10_50": hits(adapter_top),
                "hit_flips": {str(k): {direction: rows for direction, rows in directions.items()}
                              for k, directions in flips.items()},
                "recall_equal": rank_hits(adapter_ranks) == hits(native_top),
                "native_observed_top50_vs_adapter_derived_top50_exact_query_count": int(
                    np.count_nonzero(np.all(native_top == adapter_top, axis=1))),
                "native_derived_order_vs_adapter_derived_order_exact_query_count": int(
                    np.count_nonzero(np.all(native_order == adapter_order, axis=1))),
                "native_top50_observed": True, "native_full_order_derived_from_observed_scores": True,
                "adapter_order_derived_from_full_score_matrix": True,
                "max_abs_native_score_error": float(np.max(np.abs(native - adapted))),
                "mean_abs_native_score_error": float(np.mean(np.abs(native - adapted))),
                "query_order_sha256": identity["query_order_sha256"],
                "gallery_order_sha256": identity["gallery_order_sha256"],
                "score_file": str(score_file)}
        result["adapter_worker_identity"] = adapter.worker_identity
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    finally:
        adapter.close()
    return result




def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sealed recovery plan: {exc}") from exc
    native_root = args.native_root.resolve()
    if native_root != Path(plan["native_root"]).resolve():
        raise ValueError("native root differs from sealed recovery plan")
    receipt_path = native_root / "native-source-receipt.json"
    if _file_sha256(receipt_path) != plan["native_receipt_sha256"]:
        raise ValueError("native receipt differs from sealed recovery plan")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["checkpoint_sha256"] != plan["checkpoint_sha256"] or receipt["source_commit"] != plan["source_commit"]:
        raise ValueError("native source/checkpoint identity differs from sealed recovery plan")
    archive_hashes = plan["native_archives"]
    if set(archive_hashes) != {"dress", "shirt", "toptee"}:
        raise ValueError("sealed recovery plan must name all three category archives")
    for category, files in archive_hashes.items():
        identity = receipt["categories"][category]
        if set(files) != {"final", "components"}:
            raise ValueError(f"{category}: sealed plan must hash final and component archives")
        if files["final"]["file"] != identity["native_final_score_file"]:
            raise ValueError(f"{category}: final archive path differs from native receipt")
        if files["final"]["sha256"] != identity["native_final_score_sha256"]:
            raise ValueError(f"{category}: final archive digest differs from native receipt")
        if files["components"]["file"] != f"native-{category}-scores.npz":
            raise ValueError(f"{category}: component archive path differs from sealed plan")
        for label, entry in files.items():
            if _file_sha256(native_root / entry["file"]) != entry["sha256"]:
                raise ValueError(f"{category}: {label} native archive differs from sealed recovery plan")
    config = resolve_config()
    if set(plan["caption_files"]) != {"dress", "shirt", "toptee"}:
        raise ValueError("sealed recovery plan must hash all category captions")
    for category, filename in plan["caption_files"].items():
        if filename != f"cap.{category}.glove.val.pkl":
            raise ValueError(f"{category}: caption file differs from expected FashionIQ validation split")
        path = Path(config.FASHIONIQ_ROOT) / "captions" / filename
        if _file_sha256(path) != plan["caption_sha256"][category]:
            raise ValueError(f"{category}: FashionIQ caption source differs from sealed recovery plan")
    if args.preflight_only:
        adapter = Adapter(config, checkpoint_id="fashioniq_run_directory")
        adapter.prepare_identity()
        verified_interpreter = adapter.require_verified_interpreter()
        _require_distinct_controller(sys.executable, verified_interpreter)
        print(json.dumps({"status": "adapter_recovery_preflight_passed", "controller": sys.executable,
                          "model_interpreter": verified_interpreter,
                          "native_receipt_sha256": plan["native_receipt_sha256"],
                          "native_archives": archive_hashes}, sort_keys=True))
        return 0
    print(json.dumps(recover(args.native_root, args.checkpoint, args.output,
                             {category: {label: entry["sha256"] for label, entry in files.items()}
                              for category, files in archive_hashes.items()}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
