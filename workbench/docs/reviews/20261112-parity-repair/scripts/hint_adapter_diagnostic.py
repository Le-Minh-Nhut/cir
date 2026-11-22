"""Fresh HINT Adapter comparison against sealed pinned dress evaluator arrays; never writes V3 receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
from typing import Any, Sequence

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(source_scores: Any, source_orders: Any, adapter_scores: Any,
            gallery_ids: Sequence[str], reference_ids: Sequence[str],
            target_ids: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    source = np.asarray(source_scores)
    adapter = np.asarray(adapter_scores)
    orders = np.asarray(source_orders)
    if (source.dtype != np.float32 or adapter.dtype != np.float32 or orders.dtype != np.int32 or
            source.shape != adapter.shape or orders.shape != source.shape):
        raise ValueError("HINT source/adapter score and order shapes or dtypes differ")
    if not np.isfinite(source).all() or not np.isfinite(adapter).all():
        raise ValueError("HINT source or Adapter scores contain non-finite values")
    if source.shape != (len(reference_ids), len(gallery_ids)) or len(target_ids) != len(reference_ids):
        raise ValueError("HINT score matrices differ from recorded cohort")
    positions = {str(image_id): i for i, image_id in enumerate(gallery_ids)}
    if len(positions) != len(gallery_ids):
        raise ValueError("HINT gallery IDs are not unique")
    refs = np.asarray([positions[str(value)] for value in reference_ids], dtype=np.int32)
    targets = np.asarray([positions[str(value)] for value in target_ids], dtype=np.int32)
    rows = np.arange(len(refs))
    sentinel = np.asarray(-10e10, dtype=np.float32).item()
    if not np.all(source[rows, refs] == sentinel):
        raise ValueError("pinned HINT source scores do not contain reference exclusion sentinel")
    masked = adapter.copy()
    masked[rows, refs] = sentinel
    adapter_orders = np.argsort(-masked, axis=1)
    source_ranks = np.empty(len(refs), dtype=np.int32)
    adapter_ranks = np.empty(len(refs), dtype=np.int32)
    for row, target in enumerate(targets):
        if not np.array_equal(np.sort(orders[row]), np.arange(len(gallery_ids))):
            raise ValueError(f"pinned HINT order is not a gallery permutation at query {row}")
        source_ranks[row] = np.flatnonzero(orders[row] == target)[0] + 1
        adapter_ranks[row] = np.flatnonzero(adapter_orders[row] == target)[0] + 1
    cutoffs = (1, 10, 50)
    source_hits = {f"r{k}": int(np.count_nonzero(source_ranks <= k)) for k in cutoffs}
    adapter_hits = {f"r{k}": int(np.count_nonzero(adapter_ranks <= k)) for k in cutoffs}
    return {
        "max_abs_score_delta": float(np.max(np.abs(source - masked))),
        "source_hit_counts": source_hits,
        "adapter_hit_counts": adapter_hits,
        "target_rank_flip_query_indices": np.flatnonzero(source_ranks != adapter_ranks).tolist(),
        "source_ranks": source_ranks,
        "adapter_ranks": adapter_ranks,
        "source_target_topk_hits": source_ranks[:, None] <= np.asarray(cutoffs)[None, :],
        "adapter_target_topk_hits": adapter_ranks[:, None] <= np.asarray(cutoffs)[None, :],
        "target_hit_flip_query_indices": {
            f"k{k}": np.flatnonzero((source_ranks <= k) != (adapter_ranks <= k)).tolist()
            for k in cutoffs},
        "topk_candidate_membership_flip_query_indices": {
            f"k{k}": [row for row in range(len(refs)) if not np.array_equal(
                np.sort(orders[row, :k]), np.sort(adapter_orders[row, :k]))]
            for k in cutoffs},
        "adapter_masked_scores": masked,
        "adapter_orders": adapter_orders.astype(np.int32, copy=False),
    }


def load_sealed(native_root: Path, receipt_path: Path):
    import numpy as np

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("identity", {}).get("queries") != 2017 or receipt["identity"].get("gallery") != 2628:
        raise ValueError("native receipt identity is not the sealed dress cohort")
    artifact = receipt["artifacts"]
    files = {
        "npz": (receipt_path.parent / artifact["native_npz"], artifact["native_npz_sha256"]),
        "scores": (receipt_path.parent / artifact["scores"], artifact["scores_sha256"]),
        "orders": (receipt_path.parent / artifact["orders"], artifact["orders_sha256"]),
    }
    for name, (path, expected) in files.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"sealed native {name} hash mismatch: {path}")
    if files["npz"][0].stat().st_size != artifact["native_npz_size_bytes"]:
        raise ValueError("sealed native NPZ byte length mismatch")
    inputs = json.loads((native_root / "raw/sample_hint_dress_inputs.json").read_text())
    identity = json.loads((native_root / "raw/identity_hint_dress.json").read_text())
    with np.load(files["npz"][0], allow_pickle=False) as archive:
        native = {key: archive[key] for key in archive.files}
    score_rows = np.load(files["scores"][0], mmap_mode="r", allow_pickle=False)
    order_rows = np.load(files["orders"][0], mmap_mode="r", allow_pickle=False)
    if score_rows.shape != (2017, 2628) or score_rows.dtype != np.float32 or order_rows.shape != score_rows.shape or order_rows.dtype != np.int32:
        raise ValueError("sealed native source score/order layout mismatch")
    scores = score_rows
    orders = order_rows
    required = {
        "population_native_queries", "population_native_gallery", "pinned_source_ranks",
        "pinned_source_topk_target_hits", "pinned_source_hit_cutoffs", "pinned_source_hit_counts",
        "pinned_source_gallery_ids", "pinned_source_reference_ids", "pinned_source_target_ids",
        "population_native_query_paths", "population_native_gallery_paths", "native_query_ids",
        "native_target_ids", "native_query_paths", "native_target_paths", "native_gallery_sample_ids",
        "native_gallery_sample_paths", "native_query_pixels", "native_query_token_ids",
        "native_query_token_mask", "native_target_pixels", "native_gallery_sample_pixels",
    }
    missing = required.difference(native)
    if missing:
        raise ValueError(f"sealed native NPZ missing required fields: {sorted(missing)}")
    if (native["population_native_queries"].shape != (2017, 256) or
            native["population_native_gallery"].shape != (2628, 32, 256) or
            native["population_native_queries"].dtype != np.float32 or
            native["population_native_gallery"].dtype != np.float32 or
            native["pinned_source_ranks"].shape != (2017,) or
            native["pinned_source_ranks"].dtype != np.int32 or
            native["pinned_source_topk_target_hits"].shape != (2017, 3) or
            native["native_query_pixels"].shape != (5, 3, 224, 224) or
            native["native_target_pixels"].shape != (5, 3, 224, 224) or
            native["native_gallery_sample_pixels"].shape != (20, 3, 224, 224)):
        raise ValueError("sealed native feature/sample shapes or dtypes mismatch")
    checks = (
        ("pinned_source_gallery_ids", inputs["gallery_ids"], "gallery IDs"),
        ("pinned_source_reference_ids", inputs["full_reference_ids"], "reference IDs"),
        ("pinned_source_target_ids", inputs["full_target_ids"], "target IDs"),
        ("population_native_query_paths", inputs["full_query_paths"], "query paths"),
        ("population_native_gallery_paths", inputs["gallery_paths"], "gallery paths"),
    )
    for key, expected, label in checks:
        if native[key].tolist() != expected:
            raise ValueError(f"sealed native {label} mismatch")
    if (identity["query_count_native"] != 2017 or identity["gallery_count_native"] != 2628 or
            not identity["gallery_order_identical"] or not identity["reference_ids_identical"] or
            not identity["target_ids_identical"] or
            inputs["full_reference_ids"] != [row["reference_id"] for row in identity["query_annotations"]] or
            inputs["full_target_ids"] != [row["target_id"] for row in identity["query_annotations"]] or
            inputs["gallery_ids"] != identity["gallery_native_order"]):
        raise ValueError("sealed native identity record mismatch")
    sample_indices = inputs["sample_indices"]
    if (len(sample_indices) != len(native["native_query_ids"]) or
            [inputs["full_reference_ids"][i] for i in sample_indices] != native["native_query_ids"].tolist() or
            [inputs["full_target_ids"][i] for i in sample_indices] != native["native_target_ids"].tolist()):
        raise ValueError("sealed native G1 sample rows differ from selected full-query rows")
    positions = {image_id: i for i, image_id in enumerate(inputs["gallery_ids"])}
    sentinel = np.asarray(-10e10, dtype=np.float32).item()
    for row, (reference_id, target_id) in enumerate(zip(inputs["full_reference_ids"], inputs["full_target_ids"])):
        if scores[row, positions[reference_id]] != sentinel:
            raise ValueError(f"sealed native reference mask invalid at query {row}")
        rank = int(np.flatnonzero(orders[row] == positions[target_id])[0]) + 1
        if rank != int(native["pinned_source_ranks"][row]):
            raise ValueError(f"sealed native source order/rank mismatch at query {row}")
    return receipt, inputs, identity, native, scores, orders


def run(native_root: Path, receipt_path: Path, output_root: Path, *, root_claimed: bool = False) -> dict[str, Any]:
    import numpy as np
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_hint import Adapter

    native_root, receipt_path, output_root = (path.expanduser().resolve()
                                               for path in (native_root, receipt_path, output_root))
    if root_claimed:
        if not output_root.is_dir():
            raise FileNotFoundError(f"claimed diagnostic root missing: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    for name in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HF_HOME",
                 "TRANSFORMERS_CACHE", "TORCH_HOME", "TMPDIR"):
        value = os.environ.get(name)
        if value:
            path = Path(value)
            if not path.is_absolute() or not path.is_relative_to(output_root):
                raise ValueError(f"{name} must be private under diagnostic root: {path}")
            path.mkdir(parents=True, exist_ok=True)
    for name in ("WORKBENCH_ARTIFACTS_ROOT", "WORKBENCH_RESULTS_ROOT"):
        value = os.environ.get(name)
        if value:
            path = Path(value)
            if not path.is_absolute() or not path.is_relative_to(output_root):
                raise ValueError(f"{name} must be private under diagnostic root: {path}")
            path.mkdir(parents=True, exist_ok=True)
    receipt, inputs, identity, native, source_scores, source_orders = load_sealed(native_root, receipt_path)
    capture_root = output_root / "captures"
    capture_root.mkdir()
    adapter = Adapter(resolve_config())
    adapter.work_root = output_root / "adapter-work"
    sample_indices = inputs["sample_indices"]
    query_ids = inputs["full_reference_ids"]
    positional_query_ids = [str(index) for index in range(len(query_ids))]
    query_capture_ids = [positional_query_ids[index] for index in sample_indices]
    selected_target_ids = list(dict.fromkeys(inputs["target_ids"]))
    selected_gallery_ids = list(dict.fromkeys(inputs["gallery_sample_ids"]))
    selected_gallery_capture_ids = list(dict.fromkeys(selected_target_ids + selected_gallery_ids))
    try:
        queries = adapter.encode_queries(inputs["full_query_paths"], inputs["full_query_texts"],
            [None] * len(query_ids), capture_path=capture_root / "queries.npz",
            capture_ids=positional_query_ids, selected_capture_ids=query_capture_ids)
        gallery = adapter.encode_targets(inputs["gallery_paths"], [None] * len(inputs["gallery_ids"]),
            capture_path=capture_root / "gallery.npz", capture_ids=inputs["gallery_ids"],
            selected_capture_ids=selected_gallery_capture_ids)
        adapter_scores = adapter.score(queries, gallery)
    finally:
        adapter.close()
    np.savez(output_root / "adapter-population-raw.npz", queries=queries, gallery=gallery,
        score_matrix=adapter_scores)
    with np.load(capture_root / "queries.npz", allow_pickle=False) as captured_queries, \
            np.load(capture_root / "gallery.npz", allow_pickle=False) as captured_gallery:
        query_capture_positions = {str(value): row for row, value in enumerate(captured_queries["capture_ids"])}
        gallery_capture_positions = {str(value): row for row, value in enumerate(captured_gallery["capture_ids"])}
        query_rows = [query_capture_positions[positional_query_ids[index]] for index in sample_indices]
        target_rows = [gallery_capture_positions[value] for value in inputs["target_ids"]]
        gallery_rows = [gallery_capture_positions[value] for value in inputs["gallery_sample_ids"]]
        adapter_g1 = {
            "query_pixels": captured_queries["pixels"][query_rows],
            "query_token_ids": captured_queries["token_ids"][query_rows],
            "query_token_mask": captured_queries["token_mask"][query_rows],
            "processed_texts": captured_queries["processed_texts"][query_rows],
            "query_paths": captured_queries["image_paths"][query_rows],
            "query_ids": np.asarray([inputs["reference_ids"][row] for row in range(len(sample_indices))]),
            "query_capture_ids": captured_queries["capture_ids"][query_rows],
            "target_pixels": captured_gallery["pixels"][target_rows],
            "target_paths": captured_gallery["image_paths"][target_rows],
            "target_ids": captured_gallery["capture_ids"][target_rows],
            "gallery_pixels": captured_gallery["pixels"][gallery_rows],
            "gallery_paths": captured_gallery["image_paths"][gallery_rows],
            "gallery_ids": captured_gallery["capture_ids"][gallery_rows],
        }
    expected_gallery_paths = [inputs["gallery_paths"][inputs["gallery_ids"].index(value)]
                              for value in inputs["gallery_sample_ids"]]
    identity = {
        "query_ids_equal": adapter_g1["query_ids"].tolist() == native["native_query_ids"].tolist(),
        "target_ids_equal": adapter_g1["target_ids"].tolist() == native["native_target_ids"].tolist(),
        "gallery_ids_equal": adapter_g1["gallery_ids"].tolist() == native["native_gallery_sample_ids"].tolist(),
        "query_paths_equal": adapter_g1["query_paths"].tolist() == native["native_query_paths"].tolist(),
        "target_paths_equal": adapter_g1["target_paths"].tolist() == native["native_target_paths"].tolist(),
        "gallery_paths_equal": adapter_g1["gallery_paths"].tolist() == native["native_gallery_sample_paths"].tolist() == expected_gallery_paths,
    }
    dtype_identity = {}
    pixel_identity = {}
    for field, native_field in (("query_pixels", "native_query_pixels"),
                                ("query_token_ids", "native_query_token_ids"),
                                ("query_token_mask", "native_query_token_mask"),
                                ("target_pixels", "native_target_pixels"),
                                ("gallery_pixels", "native_gallery_sample_pixels")):
        dtype_identity[f"{field}_dtype_equal"] = adapter_g1[field].dtype == native[native_field].dtype
        pixel_identity[f"{field}_equal"] = bool(np.array_equal(native[native_field], adapter_g1[field]))
    np.savez(capture_root / "adapter-g1-inputs.npz", **adapter_g1)
    try:
        comparison = compare(source_scores, source_orders, adapter_scores,
            inputs["gallery_ids"], query_ids, inputs["full_target_ids"])
        comparison_error = None
    except (ValueError, IndexError, KeyError) as error:
        comparison = None
        comparison_error = f"{type(error).__name__}: {error}"
    if comparison is not None:
        np.savez(output_root / "adapter-population-ranking.npz",
            masked_score_matrix=comparison["adapter_masked_scores"],
            order_matrix=comparison["adapter_orders"], ranks=comparison["adapter_ranks"])
    g1 = {
        **identity,
        **dtype_identity,
        **pixel_identity,
        "strict_pixel_token_mask_identity": all(identity.values()) and all(dtype_identity.values()) and all(pixel_identity.values()),
        "native_processed_texts_present": "native_query_processed_texts" in native,
        "native_processed_text_status": "UNMEASURED_NOT_CAPTURED_IN_SEALED_NATIVE_NPZ",
        "adapter_processed_query_texts": adapter_g1["processed_texts"].tolist(),
        "complete_text_equality": False,
    }
    report = {
        "status": ("MEASURED_ADAPTER_COMPARISON_INVALID" if comparison is None else
                   "MEASURED_ADAPTER_G1_MISMATCH" if not g1["strict_pixel_token_mask_identity"] else
                   "MEASURED_ADAPTER_ONLY"),
        "comparison_error": comparison_error,
        "category": "dress",
        "native_root": str(native_root), "native_receipt": str(receipt_path),
        "native_artifact_hashes": {name: expected for name, (_, expected) in {
            "npz": (receipt_path.parent / receipt["artifacts"]["native_npz"], receipt["artifacts"]["native_npz_sha256"]),
            "scores": (receipt_path.parent / receipt["artifacts"]["scores"], receipt["artifacts"]["scores_sha256"]),
            "orders": (receipt_path.parent / receipt["artifacts"]["orders"], receipt["artifacts"]["orders_sha256"])}.items()},
        "query_order_sha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(),
        "gallery_order_sha256": hashlib.sha256(("\n".join(inputs["gallery_ids"]) + "\n").encode()).hexdigest(),
        "query_count": len(query_ids), "gallery_count": len(inputs["gallery_ids"]),
        "sample_indices": sample_indices,
        "reference_exclusion": "float32 -10e10 applied to Adapter scores to match pinned source",
        "g1": g1,
        "comparison": ({key: value for key, value in comparison.items() if key not in {
            "source_ranks", "adapter_ranks", "source_target_topk_hits", "adapter_target_topk_hits",
            "adapter_masked_scores", "adapter_orders"}} if comparison is not None else None),
        "source_ranks": comparison["source_ranks"].tolist() if comparison is not None else None,
        "adapter_ranks": comparison["adapter_ranks"].tolist() if comparison is not None else None,
        "adapter_score_shape": list(adapter_scores.shape), "adapter_score_dtype": str(adapter_scores.dtype),
        "adapter_query_shape": list(queries.shape), "adapter_query_dtype": str(queries.dtype),
        "adapter_gallery_shape": list(gallery.shape), "adapter_gallery_dtype": str(gallery.dtype),
        "native_model_compute": receipt["precision_note"]["native_model_compute"],
        "v3_receipt_written": False,
    }
    (output_root / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        failure = {"status": "MEASURED_ADAPTER_DIAGNOSTIC_ERROR",
                   "error": f"{type(error).__name__}: {error}",
                   "output_root_claimed": False, "v3_receipt_written": False}
        print(json.dumps(failure, indent=2))
        return 1
    private = {
        "HOME": output_root / "private/home",
        "XDG_CACHE_HOME": output_root / "private/cache/xdg",
        "XDG_CONFIG_HOME": output_root / "private/config",
        "XDG_DATA_HOME": output_root / "private/data",
        "HF_HOME": output_root / "private/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": output_root / "private/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": output_root / "private/cache/transformers",
        "TORCH_HOME": output_root / "private/cache/torch",
        "TMPDIR": output_root / "private/tmp",
        "WORKBENCH_ARTIFACTS_ROOT": output_root / "private/artifacts",
        "WORKBENCH_RESULTS_ROOT": output_root / "private/results",
    }
    for name, path in private.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    try:
        report = run(args.native_root, args.native_receipt, output_root, root_claimed=True)
    except Exception as error:
        failure = {"status": "MEASURED_ADAPTER_DIAGNOSTIC_ERROR",
                   "error": f"{type(error).__name__}: {error}",
                   "raw_population_saved": (output_root / "adapter-population-raw.npz").is_file(),
                   "output_root_claimed": True, "private_environment": {k: str(v) for k, v in private.items()},
                   "v3_receipt_written": False}
        (output_root / "diagnostic.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["status"] != "MEASURED_ADAPTER_ONLY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
