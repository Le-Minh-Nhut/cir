"""Task 3: native LIMN FashionIQ evaluation over the three category checkpoints.

Invokes the pinned upstream entrypoint (`test.py::test`) through `workbench.replay.limn`, and
instruments the SAME loaded objects with `workbench.replay.limn_observer` so per-query ranks exist
and the reconstructed hit counts are cross-checked against the native aggregate EXACTLY.

Everything written here is derived from files: checkpoint sha256 is recomputed, query/gallery counts
come from the pinned manifests, and nothing is copied from a paper or a previous report.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/home/heheboiz/data/cir")

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.operator_config import resolve_config
from workbench.replay import limn as replay
from workbench.replay.limn_observer import cross_check_with_native, observe_category

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn")
TOP_K = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_counts(dataset_root: Path, category: str) -> dict:
    """Query and gallery counts the native constructor will build, taken from the files it reads."""
    split = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
    captions = json.loads((dataset_root / "captions" / f"cap.{category}.val.json").read_text())
    ordered: list[str] = []
    seen: set[str] = set()
    for row in captions:
        for key in ("candidate", "target"):
            if row[key] not in seen:
                seen.add(row[key])
                ordered.append(row[key])
    missing = [name for name in ordered if name not in set(split)]
    return {
        "val_split_images": len(split),
        "caption_rows": len(captions),
        "native_query_count": len(captions),
        "native_gallery_count": len(ordered),
        "gallery_derivation": "ordered unique union of candidate and target ids over cap.<cat>.val.json, first appearance order",
        "gallery_ids_absent_from_val_split": len(missing),
        "manifest_source": {
            "split": str(dataset_root / "image_splits" / f"split.{category}.val.json"),
            "captions": str(dataset_root / "captions" / f"cap.{category}.val.json"),
            "correction_dict": str(dataset_root / "captions" / f"correction_dict_{category}.json"),
        },
    }


def aggregate_only() -> int:
    """Combine the per-category evidence files into the one report, refusing an incomplete set."""
    paths = {category: OUT / f"limn_native.{category}.json" for category in CATEGORIES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print(f"[BLOCKED] missing per-category evidence: {missing}", file=sys.stderr)
        return 1
    merged: dict = {}
    invocation = None
    categories: dict = {}
    for category, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        invocation = invocation or payload["invocation"]
        categories[category] = payload["categories"][category]
    merged.update({
        "artifact_type": "limn_native_evaluation",
        "schema_version": 1,
        "protocol": "fashioniq_val_split",
        "coverage_note": ("Native LIMN coverage is independent of CIRCUS coverage. LIMN's native "
                          "protocol evaluates each category checkpoint on its own category's "
                          "validation annotations; the two CIRCUS routing failures are a property of "
                          "the released M-BEIR task-7 cohort and do not exist here."),
        "invocation": invocation,
        "categories": categories,
        "aggregate": {
            "macro_r1": sum(categories[c]["native_metrics"][f"{c}_r1"] for c in CATEGORIES) / 3,
            "macro_r10": sum(categories[c]["native_metrics"][f"{c}_r10"] for c in CATEGORIES) / 3,
            "macro_r50": sum(categories[c]["native_metrics"][f"{c}_r50"] for c in CATEGORIES) / 3,
            "macro_definition": "unweighted mean over the three category results (each category weighted 1)",
            "categories_averaged": list(CATEGORIES),
        },
        "all_parity_verified": all(categories[c]["instrumentation_parity"] == "VERIFIED"
                                   for c in CATEGORIES),
        "source_files": {category: str(path) for category, path in paths.items()},
    })
    (OUT / "limn_native.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": merged["aggregate"],
                      "parity": merged["all_parity_verified"],
                      "metrics": {c: categories[c]["native_metrics"] for c in CATEGORIES}}, indent=1))
    return 0 if merged["all_parity_verified"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = None
    if "--aggregate-only" in argv:
        return aggregate_only()

    if "--category" in argv:
        only = argv[argv.index("--category") + 1]
        if only not in CATEGORIES:
            raise SystemExit(f"unknown category {only!r}")
    config = resolve_config()
    checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "LIMN"
    started = time.time()

    problems = replay.preflight(checkpoint_root, dataset_root, source_root, CATEGORIES)
    report: dict = {
        "artifact_type": "limn_native_evaluation",
        "schema_version": 1,
        "protocol": "fashioniq_val_split",
        "coverage_note": ("Native LIMN coverage is independent of CIRCUS coverage. LIMN's native "
                          "protocol evaluates each category checkpoint on its own category's "
                          "validation annotations; the two CIRCUS routing failures are a property of "
                          "the released M-BEIR task-7 cohort and do not exist here."),
        "invocation": {
            "entrypoint": "workbench/third_party/LIMN/LIMN/test.py::test",
            "runner": "workbench.replay.limn.replay_category -> native test.test(params, model, testset, category)",
            "runner_module": str(Path(replay.__file__).resolve()),
            "observer": "workbench.replay.limn_observer.observe_category + cross_check_with_native",
            "interpreter": sys.executable,
            "interpreter_version": sys.version,
            "platform": platform.platform(),
            "source_root": str(source_root),
            "dataset_root": str(dataset_root),
            "checkpoint_root": str(checkpoint_root),
            "upstream_commit": replay.MODEL["upstream_commit_sha"],
            "batch_size": replay.resolved_batch_size(),
            "batch_size_env": {replay.LIMN_BATCH_SIZE_ENV: os.environ.get(replay.LIMN_BATCH_SIZE_ENV)},
            "preflight_blockers": problems,
            "environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "WORKBENCH_DECODE_WORKERS", "WORKBENCH_RUNTIME_ADAPTATION", "WORKBENCH_LIMN_BATCH_SIZE",
                "WORKBENCH_LIMN_PYTHON", "HF_HOME")},
        },
        "categories": {},
    }
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        (OUT / "raw" / "t3_native_blocked.json").write_text(json.dumps(report, indent=2) + "\n")
        return 1

    import torch
    report["invocation"]["torch"] = torch.__version__
    report["invocation"]["torch_cuda"] = torch.version.cuda
    report["invocation"]["gpu"] = torch.cuda.get_device_name(0)

    macro_holder: dict = {}
    for category in ((only,) if only else CATEGORIES):
        print(f"# {category}: loading pinned checkpoint + native dataset", flush=True)
        began = time.time()
        counts = manifest_counts(dataset_root, category)
        with replay.native_category(category, checkpoint_root, dataset_root, source_root) as loaded:
            checkpoint = loaded["checkpoint"]
            checkpoint_path = loaded["checkpoint_path"]
            measured = sha256_file(checkpoint_path)
            if measured != checkpoint["expected_sha256"]:
                raise SystemExit(f"REFUSING {checkpoint_path}: sha256 {measured} != {checkpoint['expected_sha256']}")
            model, dataset, native_test = loaded["model"], loaded["dataset"], loaded["native_test"]
            if not isinstance(model, loaded["native_model"].LIMN):
                raise SystemExit("native model identity check failed")
            if model.training:
                raise SystemExit("native model was not in eval() mode")
            params = SimpleNamespace(local_rank=0, batch_size=replay.resolved_batch_size())
            observed_native = dataset.test_queries
            observed_gallery = dataset.test_targets
            metrics = dict(native_test.test(params, model, dataset, category))
            native_seconds = time.time() - began
            print(f"# {category}: native test done in {native_seconds:.1f}s -> {metrics}", flush=True)

            began = time.time()
            observation = observe_category(category, model, dataset, dataset_root,
                                           batch_size=replay.resolved_batch_size(), top_k=TOP_K)
            observe_seconds = time.time() - began
        mismatches = cross_check_with_native(observation, metrics)
        ranks = [query.target_rank for query in observation.queries]
        ranks_sorted = sorted(ranks)
        report["categories"][category] = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_filename": checkpoint["filename"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": measured,
            "checkpoint_sha256_source": "recomputed over the file on disk; equals the registry expected_sha256",
            "checkpoint_mapping_status": checkpoint["checkpoint_mapping_status"],
            "dataset_type_index": {"dress": 0, "shirt": 1, "toptee": 2}[category],
            "manifests": counts,
            "observed_query_count": len(observed_native),
            "observed_gallery_count": len(observed_gallery),
            "observer_gallery_size": observation.gallery_size,
            "observer_eligible_count": observation.eligible_count,
            "native_metrics": {k: v for k, v in metrics.items()},
            "native_metrics_keys": sorted(metrics),
            "recall_from_observed_ranks": {
                f"r{k}": observation.recall(k) for k in (1, 10, 50)},
            "instrumentation_parity": "VERIFIED" if not mismatches else "MISMATCH",
            "instrumentation_mismatches": list(mismatches),
            "target_rank_summary": {
                "count": len(ranks),
                "min": ranks_sorted[0],
                "median": ranks_sorted[len(ranks_sorted) // 2],
                "max": ranks_sorted[-1],
                "equal_to_1": sum(1 for r in ranks if r == 1),
            },
            "timing_seconds": {"native_test": round(native_seconds, 2),
                              "observation": round(observe_seconds, 2)},
        }
        out_jsonl = OUT / f"limn_native.{category}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for query in observation.queries:
                handle.write(json.dumps({
                    "category": category,
                    "annotation_index": query.annotation_index,
                    "reference_id": query.reference_id,
                    "target_id": query.target_id,
                    "raw_captions": list(query.raw_captions),
                    "model_input_text": query.model_input_text,
                    "target_rank": query.target_rank,
                    "checkpoint_sha256": measured,
                    "candidates": [{"rank": rank, "image_id": image_id, "score": score}
                                   for rank, image_id, score in query.top_results],
                }, separators=(",", ":")) + "\n")
        report["categories"][category]["per_query_export"] = str(out_jsonl)
        macro_holder[category] = {
            "r10": metrics[f"{category}_r10"], "r50": metrics[f"{category}_r50"],
            "r1": metrics[f"{category}_r1"]}
        print(f"# {category}: parity={report['categories'][category]['instrumentation_parity']} "
              f"mismatches={mismatches}", flush=True)

    if only:
        # Per-category invocation: write just the one category's evidence, so a long run is
        # resumable without re-doing categories that already finished.
        report["partial_categories"] = [only]
        (OUT / f"limn_native.{only}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        entry = report["categories"][only]
        print(json.dumps({"category": only, "metrics": entry["native_metrics"],
                          "parity": entry["instrumentation_parity"],
                          "mismatches": entry["instrumentation_mismatches"],
                          "target_rank_summary": entry["target_rank_summary"],
                          "timing": entry["timing_seconds"]}, indent=1))
        return 0 if entry["instrumentation_parity"] == "VERIFIED" else 1

    macro = {
        "macro_r1": sum(v["r1"] for v in macro_holder.values()) / len(macro_holder),
        "macro_r10": sum(v["r10"] for v in macro_holder.values()) / len(macro_holder),
        "macro_r50": sum(v["r50"] for v in macro_holder.values()) / len(macro_holder),
        "macro_definition": "unweighted mean over the three category results (each category weighted 1)",
        "categories_averaged": sorted(macro_holder),
    }
    report["aggregate"] = macro
    report["all_parity_verified"] = all(
        entry["instrumentation_parity"] == "VERIFIED" for entry in report["categories"].values())
    report["elapsed_seconds"] = round(time.time() - started, 2)
    (OUT / "limn_native.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": macro_holder, "macro": macro,
                      "parity": report["all_parity_verified"],
                      "seconds": report["elapsed_seconds"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
