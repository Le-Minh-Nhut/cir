"""Dedicated TG-CIR FashionIQ V3 preregistration and role evidence consumer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from workbench.replay.tgcir_pair import g1_sample, scoring_query_ids

ROLE_SIDES = {
    "native_repeat_1": "native", "native_repeat_2": "native",
    "adapter_repeat_1": "adapter", "adapter_repeat_2": "adapter",
    "native_adapter_native": "native", "native_adapter_adapter": "adapter",
    "native_batch_1": "native", "native_batch_n": "native",
    "adapter_batch_1": "adapter", "adapter_batch_n": "adapter",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def shared_v3_helper():
    path = Path(__file__).resolve().parents[1] / "docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
    spec = importlib.util.spec_from_file_location("cir_t4_tgcir_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared V3 evidence contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ordered_sha(values: list[str]) -> str:
    return _sha(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _scope(coverage: str, population: int, ids: list[str], cohort: str, dimension: int = 512) -> dict[str, Any]:
    shape = [len(ids), dimension]
    return {"coverage": coverage, "population_count": population,
            "tensor_shape": shape, "component_shape": shape,
            "ordered_ids_sha256": _ordered_sha(ids), "cohort_id": cohort}


def build_plan(*, dataset_root: Path, category: str, checkpoint_sha256: str,
               configuration_sha256: str, environment: str, protocol: str,
               code_sha256: dict[str, str], output_root: Path, cohort_id: str,
               seed: int | str = "unset-by-pinned-evaluator") -> dict[str, Any]:
    population = g1_sample(dataset_root, category, limit=None)
    query_ids = scoring_query_ids(population["query_image_names"])
    gallery_ids = population["gallery_image_names"]
    refs, targets = population["reference_indices"], population["target_indices"]
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("TG-CIR V3 population IDs must be unique and ordered")
    if not (len(query_ids) == len(refs) == len(targets)):
        raise ValueError("TG-CIR V3 source label counts differ from query population")
    labels = {"query_targets": [[q, gallery_ids[t]] for q, t in zip(query_ids, targets)],
              "reference_exclusions": [[q, [gallery_ids[r]]] for q, r in zip(query_ids, refs)]}
    primary = {"checkpoint": checkpoint_sha256, "configuration": configuration_sha256,
               "environment": environment, "protocol": protocol, "category": category,
               "cohort_id": cohort_id, "precision": "float32", "seed": seed,
               "query_order_sha256": _ordered_sha(query_ids),
               "gallery_order_sha256": _ordered_sha(gallery_ids),
               "reference_exclusions_sha256": _sha(json.dumps(labels, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")),
               "query_count": len(query_ids), "gallery_count": len(gallery_ids)}
    definitions = {
        "native_repeat_1": ("native", 32), "native_repeat_2": ("native", 32),
        "adapter_repeat_1": ("adapter", 32), "adapter_repeat_2": ("adapter", 32),
        "native_adapter_native": ("native", 32), "native_adapter_adapter": ("adapter", 32),
        "native_batch_1": ("native", 1), "native_batch_n": ("native", 32),
        "adapter_batch_1": ("adapter", 1), "adapter_batch_n": ("adapter", 32)}
    partition_sha = _ordered_sha(query_ids)
    gallery_partition_sha = _ordered_sha(gallery_ids)
    runs = {}
    for role, (side, batch) in definitions.items():
        runs[role] = {"run_id": f"tgcir:{category}:{role}:{os.urandom(16).hex()}",
                      "source_id": f"tgcir-{side}-source-{os.urandom(16).hex()}",
                      "identity": dict(primary),
                      "execution": {"partition_id": f"tgcir-{category}-{cohort_id}-full",
                                    "query_batch_size": batch, "gallery_batch_size": batch,
                                    "query_partition_sha256": partition_sha,
                                    "gallery_partition_sha256": gallery_partition_sha}}
    defs = {"native_repeat": ("native_repeat_1", "native_repeat_2", "native"),
            "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2", "adapter"),
            "native_adapter": ("native_adapter_native", "native_adapter_adapter", "native"),
            "native_batch_1_vs_n": ("native_batch_1", "native_batch_n", "native"),
            "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n", "adapter")}
    controls = {}
    for name, (reference, actual, side) in defs.items():
        is_batch = "batch" in name
        scopes = {"query": _scope("full", len(query_ids), query_ids, cohort_id),
                  "gallery": _scope("full", len(gallery_ids), gallery_ids, cohort_id)}
        controls[name] = {"reference_run_id": runs[reference]["run_id"],
                          "actual_run_id": runs[actual]["run_id"], "reference_side": side,
                          "dimension": "batch_parity" if is_batch else "forward_parity", "dtype": "float32",
                          "tolerance": {"atol": 1e-4 if is_batch else 1e-5,
                                        "rtol": 1e-3 if is_batch else 1e-4}, "scopes": scopes}
    return {"standard": "parity-absolute-relative/v3", "model_id": "tgcir", "category": category,
            "code_sha256": code_sha256, "identity": primary, "runs": runs, "controls": controls,
            "scoring_labels": labels,
            "score_order_conventions": {"native": "fp32_descending_score", "adapter": "fp32_descending_score"},
            "scoring_proof_path": f"g1_g2.tgcir.{category}.native.score-ranks.jsonl",
            "metric_cutoffs": [1, 10, 50],
            "supported_batch_policy": {"supported": True, "batch_invariant": False, "batch_sizes": [1, 32]},
            "population_scope": {"coverage": "full", "query_count": len(query_ids),
                                 "gallery_count": len(gallery_ids), "cohort_id": cohort_id,
                                 "query_ids": query_ids, "source_query_ids": population["query_image_names"],
                                 "gallery_ids": gallery_ids, "reference_ids": [gallery_ids[i] for i in refs],
                                 "target_ids": [gallery_ids[i] for i in targets]},
            "reference_policy": "native and Adapter mask reference gallery candidate with -10e10"}


def preregister(path: Path, plan: dict[str, Any]) -> str:
    """Persist immutable role plan through shared contract before any role starts."""
    if set(plan.get("runs", {})) != set(ROLE_SIDES):
        raise ValueError("TG-CIR V3 preregistration must contain all ten physical roles")
    if plan.get("population_scope", {}).get("coverage") != "full":
        raise ValueError("TG-CIR V3 preregistration requires full population")
    return shared_v3_helper().write_v3_plan(path, plan)

def validate_native_g1_sample(*, plan: dict[str, Any], role: str,
                               native_g1_receipt: Path, g1: dict[str, Any],
                               paired_receipt: dict[str, Any], native_receipt_path: Path) -> dict[str, Any]:
    """Check Adapter G1 sample against preregistered native evidence and bind receipts."""
    from workbench.replay.common import sha256_file

    expected_role = plan["runs"]["native_adapter_native" if role == "native_adapter_adapter" else "native_repeat_1"]
    expected_role_name = "native_adapter_native" if role == "native_adapter_adapter" else "native_repeat_1"
    if (paired_receipt.get("role") != expected_role_name
            or paired_receipt.get("run_id") != expected_role["run_id"]
            or paired_receipt.get("source_id") != expected_role["source_id"]):
        raise ValueError("TG-CIR paired native G1 receipt does not match preregistered source role")
    with np.load(native_g1_receipt, allow_pickle=False) as native_g1:
        for name, value in (("query_pixels", g1["query_pixels"]),
                            ("gallery_pixels", g1["gallery_pixels"])):
            if not np.array_equal(native_g1[f"g1_{name}"], value):
                raise ValueError(f"TG-CIR native and Adapter G1 {name} differ")
        if (native_g1["g1_query_image_names"].astype(str).tolist() != g1["query_image_names"]
                or native_g1["g1_gallery_image_names"].astype(str).tolist() != g1["gallery_image_names"]
                or native_g1["g1_raw_query_texts"].astype(str).tolist() != g1["raw_query_texts"]):
            raise ValueError("TG-CIR native role receipt G1 IDs or raw texts differ from Adapter sample")
    g1["native_role"] = {"role": paired_receipt["role"], "run_id": paired_receipt["run_id"],
                          "source_id": paired_receipt["source_id"],
                          "role_receipt_sha256": sha256_file(native_receipt_path)}
    g1["native_role_artifact_sha256"] = sha256_file(native_g1_receipt)
    return g1


def run_role(*, plan_path: Path, role: str, dataset_root: Path, checkpoint_root: Path,
             source_root: Path, output: Path, native_g1_receipt: Path | None = None) -> dict[str, Any]:
    """Execute one predeclared native source or production Adapter role."""
    from dataclasses import asdict
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import checkpoint_path as resolve_checkpoint_path
    from workbench.composition.model_tgcir import Adapter
    from workbench.replay.common import sha256_file
    from workbench.replay.tgcir import CHECKPOINT
    from workbench.replay.tgcir_pair import _run_native_stage

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"TG-CIR V3 plan is unreadable or invalid: {plan_path}: {error}") from error
    if role not in ROLE_SIDES or role not in plan.get("runs", {}):
        raise ValueError(f"TG-CIR V3 role not preregistered: {role}")
    started = datetime.now(timezone.utc).isoformat()
    declared = plan["runs"][role]
    batch = int(declared["execution"]["query_batch_size"])
    population = g1_sample(dataset_root, plan["category"], None)
    sample = g1_sample(dataset_root, plan["category"], limit=5)
    query_ids = scoring_query_ids(population["query_image_names"])
    gallery_ids = population["gallery_image_names"]
    if (query_ids != plan["population_scope"]["query_ids"]
            or gallery_ids != plan["population_scope"]["gallery_ids"]
            or [gallery_ids[i] for i in population["reference_indices"]] != plan["population_scope"]["reference_ids"]
            or [gallery_ids[i] for i in population["target_indices"]] != plan["population_scope"]["target_ids"]):
        raise ValueError("TG-CIR V3 source population differs from preregistered order or labels")
    config = resolve_config()
    checkpoint_path = resolve_checkpoint_path("tgcir", CHECKPOINT, checkpoint_root)
    if sha256_file(checkpoint_path) != plan["identity"]["checkpoint"]:
        raise ValueError("TG-CIR V3 checkpoint digest differs from preregistration")
    code_files = [source_root / name for name in ("datasets.py", "test.py", "model.py")]
    code_files += [Path(__file__).resolve(), Path(__file__).with_name("tgcir_pair.py"),
                   Path(__file__).with_name("tensor_spool.py"),
                   Path(__file__).resolve().parents[1] / "composition/model_tgcir.py"]
    if {str(path): sha256_file(path) for path in code_files} != plan["code_sha256"]:
        raise ValueError("TG-CIR V3 producer/source code differs from preregistration")
    config_sha = _sha(json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8"))
    if config_sha != plan["identity"]["configuration"]:
        raise ValueError("TG-CIR V3 operator configuration differs from preregistration")

    native_runtime: dict[str, Any] | None = None
    adapter_interpreter: str | None = None
    production_ranks: np.ndarray | None = None
    stage: dict[str, Any] | None = None
    if ROLE_SIDES[role] == "native":
        stage_root = output.with_name(output.name + "-native-stage")
        stage_root.mkdir(parents=True, exist_ok=False)
        stage = _run_native_stage(dataset_root, checkpoint_root, source_root,
                                  stage_root, batch, plan["category"])
        native_runtime = stage["native_runtime"]
        with np.load(stage_root / "native-arrays.npz", allow_pickle=False) as values:
            query, gallery = values["queries"].copy(), values["targets"].copy()
            scores, ranks = values["scores"].copy(), values["ranks"].copy()
            token_ids = values["g1_token_ids"].copy()
        native_meta_path = stage_root / "native-stage.json"
        try:
            native_meta = json.loads(native_meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"TG-CIR native stage metadata is unreadable: {native_meta_path}: {error}") from error
        pixels_path = Path(native_meta["source_receipt"]["pixel_artifact"])
        with np.load(pixels_path, allow_pickle=False) as values:
            try:
                query_pixels = np.stack([values[name.replace("/", "_")]
                                         for name in sample["query_image_names"]])
                gallery_pixels = np.stack([values[name.replace("/", "_")]
                                           for name in sample["gallery_image_names"]])
            except KeyError as error:
                raise ValueError("TG-CIR native pixel receipt omits declared G1 image") from error
        if token_ids.shape[0] != len(sample["query_texts"]):
            raise ValueError("TG-CIR native G1 token capture count differs from declared query sample")
        g1 = {"query_pixels": query_pixels, "gallery_pixels": gallery_pixels,
              "query_texts": native_meta["g1_text"], "raw_query_texts": sample["query_texts"],
              "token_ids": token_ids, "token_mask": token_ids != 0,
              "query_paths": sample["query_paths"], "gallery_paths": sample["gallery_paths"],
              "query_image_names": sample["query_image_names"],
              "gallery_image_names": sample["gallery_image_names"],
              "native_pixel_artifact_sha256": sha256_file(pixels_path)}
        reference_indices = population["reference_indices"]
        if scores.shape != (len(query_ids), len(gallery_ids)) or ranks.shape != scores.shape:
            raise ValueError("TG-CIR native scores/ranks differ from full population")
        if not np.isfinite(scores).all():
            raise ValueError("TG-CIR native scores contain non-finite values")
        if not np.array_equal(np.sort(ranks, axis=1), np.broadcast_to(np.arange(len(gallery_ids)), ranks.shape)):
            raise ValueError("TG-CIR native observed ranking is not a gallery permutation")
        if not np.all(scores[np.arange(len(query_ids)), reference_indices] == -10e10):
            raise ValueError("TG-CIR native source scores do not contain reference exclusion")
        source_order = np.stack([np.argsort(-row) for row in scores])
        if not np.array_equal(ranks, source_order):
            raise ValueError("TG-CIR captured native production order differs from source score rows")

        adapter_identity = "native source evaluator; pinned subprocess=" + json.dumps(native_runtime, sort_keys=True)
    else:
        adapter = Adapter(config)
        adapter.work_root = output.with_name(output.name + "-adapter-work")
        adapter.batch_size = batch
        g1_root = output.with_name(output.name + "-g1")
        try:
            query = adapter.encode_queries(population["query_paths"], population["query_texts"],
                                           [None] * len(query_ids))
            gallery = adapter.encode_targets(population["gallery_paths"], [None] * len(gallery_ids))
            query_capture_path, gallery_capture_path = g1_root / "queries.npz", g1_root / "targets.npz"
            adapter.encode_queries(sample["query_paths"], sample["query_texts"],
                                   [None] * len(sample["query_paths"]), capture_path=query_capture_path)
            adapter.encode_targets(sample["gallery_paths"], [None] * len(sample["gallery_paths"]),
                                   capture_path=gallery_capture_path)
            with np.load(query_capture_path, allow_pickle=False) as qcap, \
                    np.load(gallery_capture_path, allow_pickle=False) as gcap:
                if (qcap["image_paths"].astype(str).tolist() != sample["query_paths"]
                        or gcap["image_paths"].astype(str).tolist() != sample["gallery_paths"]):
                    raise ValueError("TG-CIR Adapter G1 capture order differs from declared source sample")
                g1 = {"query_pixels": qcap["pixels"].copy(), "gallery_pixels": gcap["pixels"].copy(),
                      "query_texts": qcap["corrected_texts"].astype(str).tolist(),
                      "raw_query_texts": sample["query_texts"],
                      "token_ids": qcap["token_ids"].copy(), "token_mask": qcap["token_mask"].copy(),
                      "query_paths": qcap["image_paths"].astype(str).tolist(),
                      "gallery_paths": gcap["image_paths"].astype(str).tolist(),
                      "query_image_names": sample["query_image_names"],
                      "gallery_image_names": sample["gallery_image_names"]}
            positions = [[int(index)] for index in population["target_indices"]]
            reference_indices = population["reference_indices"]
            observed: list[np.ndarray] = []
            eligibility_scores: list[np.ndarray] = []
            original_score = adapter.score
            def observe_score(*args: Any, **kwargs: Any) -> np.ndarray:
                value = np.asarray(original_score(*args, **kwargs), dtype=np.float32).copy()
                eligible = value.copy()
                eligible[np.arange(len(reference_indices)), reference_indices] = -10e10
                observed.append(value)
                eligibility_scores.append(eligible)
                return eligible
            adapter.score = observe_score
            try:
                production_ranks, production_topk = adapter.rank_and_topk(
                    query, gallery, positions, [None] * len(query_ids), depth=min(50, len(gallery_ids)))
            finally:
                adapter.score = original_score
            if len(observed) != 1:
                raise ValueError(f"expected one production Adapter score call, observed {len(observed)}")
            scores = eligibility_scores[0]
            ranks = np.argsort(-scores, axis=1, kind="stable")
            production_ranks = np.asarray(production_ranks, dtype=np.int64)
            if production_ranks.shape != (len(query_ids),):
                raise ValueError("production Adapter rank count differs from full population")
            for row, candidates in enumerate(production_topk):
                expected = [(int(i), float(scores[row, i])) for i in ranks[row, :len(candidates)]]
                if [(int(i), float(score)) for i, score in candidates] != expected:
                    raise ValueError("production Adapter TopK differs from masked captured score ordering")
            for row, positions_row in enumerate(positions):
                expected_rank = min(1 + int(np.count_nonzero(scores[row] > scores[row, pos]))
                                    + int(np.count_nonzero(scores[row, :pos] == scores[row, pos]))
                                    for pos in positions_row)
                if production_ranks[row] != expected_rank:
                    raise ValueError("production Adapter metric rank differs from masked complete scores")
            adapter_interpreter = str(adapter.interpreter)
            adapter_identity = adapter_interpreter
        finally:
            adapter.close()
        if native_g1_receipt is None or not native_g1_receipt.is_file():
            raise ValueError("TG-CIR Adapter role requires paired native role receipt for independent G1 comparison")
        native_receipt_path = native_g1_receipt.with_name("role-receipt.json")
        try:
            paired_receipt = json.loads(native_receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"TG-CIR paired native G1 receipt is unreadable: {native_receipt_path}: {error}") from error
        g1 = validate_native_g1_sample(plan=plan, role=role,
                                       native_g1_receipt=native_g1_receipt, g1=g1,
                                       paired_receipt=paired_receipt,
                                       native_receipt_path=native_receipt_path)

    labels = np.asarray(population["target_indices"], dtype=np.int64)
    reference_indices = population["reference_indices"]
    if ROLE_SIDES[role] == "native":
        production_ranks = np.asarray([int(np.flatnonzero(row == target)[0]) + 1
                                       for row, target in zip(ranks, labels)], dtype=np.int64)
        if stage is None:
            raise ValueError("TG-CIR native source stage receipt is missing")
        for cutoff in (1, 10, 50):
            observed_hits = int(np.count_nonzero(production_ranks <= cutoff))
            expected_hits = round(float(stage["metrics"][f"r{cutoff}"]) * len(query_ids) / 100)
            if observed_hits != expected_hits:
                raise ValueError(f"TG-CIR native captured order does not reproduce source R@{cutoff}")
    arrays = {"query_features": np.asarray(query), "gallery_features": np.asarray(gallery),
              "scores": np.asarray(scores), "ranks": np.asarray(ranks),
              "production_ranks": production_ranks, "labels": labels}
    ordered_ids = {name: query_ids for name in ("query_features", "scores", "ranks", "production_ranks", "labels")}
    ordered_ids["gallery_features"] = gallery_ids
    execution = dict(declared["execution"])
    environment = plan["identity"]["environment"]
    if ROLE_SIDES[role] == "adapter" and (adapter_interpreter is None or adapter_interpreter not in environment):
        raise ValueError("TG-CIR V3 Adapter interpreter is absent from preregistered environment")
    if ROLE_SIDES[role] == "native" and (native_runtime is None or any(
            value not in environment for value in (native_runtime["python"].split()[0], native_runtime["torch"]))):
        raise ValueError("TG-CIR V3 native runtime differs from preregistered environment")
    return capture_role(plan=plan, plan_sha256=_sha(plan_path.read_bytes()), role=role,
                        started_at=started, execution=execution, arrays=arrays,
                        ordered_ids=ordered_ids, output=output,
                        runtime={"identity": adapter_identity, "native": native_runtime,
                                 "adapter_interpreter": adapter_interpreter}, g1=g1)

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(ROLE_SIDES), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-g1-receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = run_role(plan_path=args.plan, role=args.role, dataset_root=args.dataset_root,
                       checkpoint_root=args.checkpoint_root, source_root=args.source_root,
                       output=args.output, native_g1_receipt=args.native_g1_receipt)
    print(json.dumps({"role": receipt["role"], "run_id": receipt["run_id"],
                      "receipt": str(args.output / "role-receipt.json")}, sort_keys=True))
    return 0




def capture_role(*, plan: dict[str, Any], plan_sha256: str, role: str,
                 started_at: str, execution: dict[str, Any], arrays: dict[str, np.ndarray],
                 ordered_ids: dict[str, list[str]], output: Path,
                 runtime: dict[str, Any] | None = None,
                 g1: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist actual model outputs and a shared-contract receipt for one started role."""
    required = {"query_features", "gallery_features", "scores", "ranks", "production_ranks", "labels"}
    if role not in ROLE_SIDES or set(arrays) != required or set(ordered_ids) != required:
        raise ValueError("TG-CIR V3 role arrays or role name invalid")
    population = plan["population_scope"]
    qids, gids = population["query_ids"], population["gallery_ids"]
    expected_labels = np.asarray([gids.index(target) for target in population["target_ids"]], dtype=np.int64)
    expected_ids = {"query_features": qids, "gallery_features": gids,
                    "scores": qids, "ranks": qids, "production_ranks": qids, "labels": qids}
    if ordered_ids != expected_ids or not np.array_equal(arrays["labels"], expected_labels):
        raise ValueError("TG-CIR V3 role population order or positive labels mismatch")
    qn, gn = len(qids), len(gids)
    if (arrays["query_features"].shape[0] != qn or arrays["gallery_features"].shape[0] != gn
            or arrays["scores"].shape != (qn, gn) or arrays["ranks"].shape != (qn, gn)
            or arrays["production_ranks"].shape != (qn,)):
        raise ValueError("TG-CIR V3 role feature, score or rank shape differs from full population")
    g1 = g1 or {}
    g1_arrays = {f"g1_{name}": np.asarray(value) for name, value in g1.items()
                 if isinstance(value, np.ndarray)}
    if not {"g1_query_pixels", "g1_gallery_pixels", "g1_token_ids", "g1_token_mask"}.issubset(g1_arrays):
        raise ValueError("TG-CIR V3 role G1 capture is incomplete")
    g1_metadata = {key: value for key, value in g1.items() if not isinstance(value, np.ndarray)}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "role-arrays.npz", **arrays, **g1_arrays,
                        g1_query_texts=np.asarray(g1_metadata["query_texts"]),
                        g1_raw_query_texts=np.asarray(g1_metadata["raw_query_texts"]),
                        g1_query_image_names=np.asarray(g1_metadata["query_image_names"]),
                        g1_gallery_image_names=np.asarray(g1_metadata["gallery_image_names"]),
                        query_ids=np.asarray(qids), gallery_ids=np.asarray(gids))
    captured = {name: (arrays[name], ordered_ids[name]) for name in required}
    for control_name, control in plan["controls"].items():
        if plan["runs"][role]["run_id"] in (control["reference_run_id"], control["actual_run_id"]):
            captured[f"{control_name}.query"] = (arrays["query_features"], qids)
            captured[f"{control_name}.gallery"] = (arrays["gallery_features"], gids)
    receipt = shared_v3_helper().v3_run_receipt(
        run_id=plan["runs"][role]["run_id"], side=ROLE_SIDES[role],
        source_id=plan["runs"][role]["source_id"], plan_sha256=plan_sha256,
        started_at=started_at, arrays=captured, execution=execution,
        identity=plan["runs"][role]["identity"])
    receipt["role"] = role
    g1_counts = {"query": plan["identity"]["query_count"],
                 "gallery": plan["identity"]["gallery_count"]}
    g1_samples = {"query": int(g1_arrays["g1_query_pixels"].shape[0]),
                  "gallery": int(g1_arrays["g1_gallery_pixels"].shape[0])}
    receipt["g1"] = {"query_texts": g1_metadata["query_texts"],
                     "raw_query_texts": g1_metadata["raw_query_texts"],
                     "query_paths": g1_metadata["query_paths"],
                     "gallery_paths": g1_metadata["gallery_paths"],
                     "query_image_names": g1_metadata["query_image_names"],
                     "gallery_image_names": g1_metadata["gallery_image_names"],
                     "coverage": {scope: {"captured_count": g1_samples[scope],
                                          "population_count": g1_counts[scope],
                                          "scope": "full" if g1_samples[scope] == g1_counts[scope] else "sample"}
                                  for scope in ("query", "gallery")},
                     "native_role": g1_metadata.get("native_role"),
                     "native_role_artifact_sha256": g1_metadata.get("native_role_artifact_sha256"),
                     "native_pixel_artifact_sha256": g1_metadata.get("native_pixel_artifact_sha256"),
                     "reference_policy": "native FashionIQ benchmark wrapper excludes source image; Adapter.score remains unmasked for CIRCUS"}
    receipt["g1"]["arrays"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype),
               "sha256": _sha(np.ascontiguousarray(value).tobytes())}
        for name, value in g1_arrays.items()}
    receipt["artifact"] = {"path": "role-arrays.npz",
                           "sha256": _sha((output / "role-arrays.npz").read_bytes())}
    (output / "role-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                                               encoding="utf-8")
    return receipt


def aggregate(plan_path: Path, receipt_paths: dict[str, Path], output: Path) -> dict[str, Any]:
    """Validate ten captures and emit native/adapter V3 gate artifacts plus scorer proof."""
    try:
        plan_bytes = plan_path.read_bytes()
        plan = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"TG-CIR V3 plan is unreadable or invalid: {plan_path}: {error}") from error
    if set(plan.get("runs", {})) != set(ROLE_SIDES) or set(receipt_paths) != set(ROLE_SIDES):
        raise ValueError("TG-CIR V3 aggregate requires exactly ten preregistered role receipts")
    helper = shared_v3_helper()
    plan_sha = _sha(plan_bytes)
    arrays_by_role: dict[str, dict[str, np.ndarray]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for role, path in receipt_paths.items():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"TG-CIR V3 receipt is unreadable or invalid: {path}: {error}") from error
        declared = plan["runs"][role]
        if (receipt.get("role") != role or receipt.get("run_id") != declared["run_id"]
                or receipt.get("source_id") != declared["source_id"]
                or receipt.get("side") != ROLE_SIDES[role] or receipt.get("plan_sha256") != plan_sha
                or receipt.get("identity") != declared["identity"]):
            raise ValueError(f"TG-CIR V3 receipt does not match preregistered role: {role}")
        artifact = path.parent / receipt.get("artifact", {}).get("path", "")
        if not artifact.is_file() or _sha(artifact.read_bytes()) != receipt.get("artifact", {}).get("sha256"):
            raise ValueError(f"TG-CIR V3 role artifact missing or digest mismatch: {role}")
        if receipt.get("execution") != declared["execution"] or not receipt.get("started_at") or not receipt.get("captured_at"):
            raise ValueError(f"TG-CIR V3 receipt execution or timestamps invalid: {role}")
        with np.load(artifact, allow_pickle=False) as values:
            qids, gids = values["query_ids"].astype(str).tolist(), values["gallery_ids"].astype(str).tolist()
            if qids != plan["population_scope"]["query_ids"] or gids != plan["population_scope"]["gallery_ids"]:
                raise ValueError(f"TG-CIR V3 role artifact population order mismatch: {role}")
            arrays = {name: values[name].copy() for name in values.files if name not in ("query_ids", "gallery_ids")}
        for name, record in receipt["arrays"].items():
            if "." in name:
                value = arrays["query_features" if name.endswith(".query") else "gallery_features"]
                ids = qids if name.endswith(".query") else gids
            else:
                value = arrays[name]
                ids = qids if name != "gallery_features" else gids
            if (list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]
                    or helper._component_digest(value) != record["component_sha256"]):
                raise ValueError(f"TG-CIR V3 role array receipt mismatch: {role}:{name}")
            if _ordered_sha(ids) != record["ordered_ids_sha256"]:
                raise ValueError(f"TG-CIR V3 role ordered IDs mismatch: {role}:{name}")
        if not {"g1_query_pixels", "g1_gallery_pixels", "g1_token_ids", "g1_token_mask",
                "g1_query_texts", "g1_raw_query_texts"}.issubset(arrays):
            raise ValueError(f"TG-CIR V3 G1 evidence incomplete: {role}")
        g1 = receipt.get("g1", {})
        if (arrays["g1_query_texts"].astype(str).tolist() != g1.get("query_texts")
                or arrays["g1_raw_query_texts"].astype(str).tolist() != g1.get("raw_query_texts")):
            raise ValueError(f"TG-CIR V3 G1 text receipt mismatch: {role}")
        for name, record in g1.get("arrays", {}).items():
            value = arrays.get(name)
            if (value is None or list(value.shape) != record["shape"]
                    or str(value.dtype) != record["dtype"]
                    or _sha(np.ascontiguousarray(value).tobytes()) != record["sha256"]):
                raise ValueError(f"TG-CIR V3 G1 array receipt mismatch: {role}:{name}")
        arrays_by_role[role], receipts[role] = arrays, receipt
    for role, side in ROLE_SIDES.items():
        if side != "adapter":
            continue
        source_role = "native_adapter_native" if role == "native_adapter_adapter" else "native_repeat_1"
        source_receipt = receipts[source_role]
        source_artifact = receipt_paths[source_role].parent / source_receipt["artifact"]["path"]
        expected_binding = {"role": source_role, "run_id": source_receipt["run_id"],
                            "source_id": source_receipt["source_id"],
                            "role_receipt_sha256": _sha(receipt_paths[source_role].read_bytes())}
        if (receipts[role].get("g1", {}).get("native_role") != expected_binding
                or receipts[role].get("g1", {}).get("native_role_artifact_sha256") != _sha(source_artifact.read_bytes())):
            raise ValueError(f"TG-CIR V3 paired native G1 source binding mismatch: {role}")
    for adapter_role in (role for role, side in ROLE_SIDES.items() if side == "adapter"):
        native_role = "native_adapter_native" if adapter_role == "native_adapter_adapter" else "native_repeat_1"
        native, adapter = arrays_by_role[native_role], arrays_by_role[adapter_role]
        for name in ("g1_query_pixels", "g1_gallery_pixels", "g1_token_ids", "g1_token_mask",
                     "g1_query_texts", "g1_raw_query_texts", "g1_query_image_names",
                     "g1_gallery_image_names"):
            if not np.array_equal(native[name], adapter[name]):
                raise ValueError(f"TG-CIR V3 paired G1 sample differs: {native_role}/{adapter_role}:{name}")
        for scope, name in (("query", "g1_query_image_names"), ("gallery", "g1_gallery_image_names")):
            native_names = native[name].astype(str).tolist()
            if len(native_names) != receipts[native_role]["g1"]["coverage"][scope]["captured_count"]:
                raise ValueError(f"TG-CIR V3 G1 sample names/count mismatch: {native_role}:{scope}")
    scopes: dict[str, dict[str, Any]] = {}
    component_failures = []
    for name, control in plan["controls"].items():
        reference = next(role for role, run in plan["runs"].items() if run["run_id"] == control["reference_run_id"])
        actual = next(role for role, run in plan["runs"].items() if run["run_id"] == control["actual_run_id"])
        scopes[name] = {}
        for tensor, key in (("query", "query_features"), ("gallery", "gallery_features")):
            ref = arrays_by_role[reference][key]
            act = arrays_by_role[actual][key]
            expected = control["scopes"][tensor]
            record = helper.component_scope(name, control["dimension"], act, ref,
                                            atol=control["tolerance"]["atol"],
                                            rtol=control["tolerance"]["rtol"])
            if list(act.shape) != expected["component_shape"]:
                raise ValueError(f"TG-CIR V3 {name}.{tensor} shape differs from preregistration")
            measured_coverage = record["coverage"]
            record.update({"reference_run_id": control["reference_run_id"],
                           "actual_run_id": control["actual_run_id"], "dtype": control["dtype"],
                           **expected})
            record["coverage"] = measured_coverage
            scopes[name][tensor] = record
            if record["mismatch_count"]:
                component_failures.append(f"{name}.{tensor}: {record['mismatch_count']} mismatches")
    native_role, adapter_role = "native_adapter_native", "native_adapter_adapter"
    native, adapter = arrays_by_role[native_role], arrays_by_role[adapter_role]
    qids, gids = plan["population_scope"]["query_ids"], plan["population_scope"]["gallery_ids"]
    labels = native["labels"].astype(np.int64)
    if (not np.array_equal(labels, adapter["labels"]) or
            not np.array_equal(np.sort(native["ranks"], axis=1), np.broadcast_to(np.arange(len(gids)), native["ranks"].shape))):
        raise ValueError("TG-CIR V3 full score/rank/label arrays disagree")
    for side, arrays in (("native", native), ("adapter", adapter)):
        ranks = arrays["ranks"]
        if not np.array_equal(np.sort(ranks, axis=1), np.broadcast_to(np.arange(len(gids)), ranks.shape)):
            raise ValueError(f"TG-CIR V3 {side} rankings are not complete permutations")
        if not np.array_equal(arrays["production_ranks"],
                              np.asarray([int(np.flatnonzero(row == label)[0]) + 1
                                          for row, label in zip(ranks, labels)])):
            raise ValueError(f"TG-CIR V3 {side} production ranks differ from captured order")
    proof_path = output / plan["scoring_proof_path"]
    if proof_path.exists():
        raise FileExistsError(f"refusing to replace TG-CIR score-rank proof: {proof_path}")
    score_receipts = {}
    for role, side in ((native_role, "native"), (adapter_role, "adapter")):
        hasher = hashlib.sha256()
        for index, query_id in enumerate(qids):
            values = arrays_by_role[role]
            row = {"query_id": query_id, "scores": values["scores"][index].astype(float).tolist(),
                   "order": values["ranks"][index].astype(int).tolist()}
            hasher.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        score_receipts[role] = {"run_id": receipts[role]["run_id"], "source_id": receipts[role]["source_id"],
                                "side": side, "identity": receipts[role]["identity"],
                                "score_order_sha256": hasher.hexdigest()}
    with proof_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "cir.score-ranks/v1", "identity": plan["identity"],
                                 "query_ids": qids, "gallery_ids": gids, **plan["scoring_labels"]},
                                ensure_ascii=False, separators=(",", ":")) + "\n")
        for index, query_id in enumerate(qids):
            target = plan["scoring_labels"]["query_targets"][index][1]
            exclusion = plan["scoring_labels"]["reference_exclusions"][index][1]
            stream.write(json.dumps({"query_id": query_id, "target_id": target,
                "target_position": gids.index(target), "excluded_gallery_ids": sorted(exclusion),
                "native_scores": native["scores"][index].tolist(),
                "adapter_scores": adapter["scores"][index].tolist(),
                "native_order": native["ranks"][index].tolist(),
                "adapter_order": adapter["ranks"][index].tolist()},
                ensure_ascii=False, separators=(",", ":")) + "\n")
    metrics = {}
    for side, arrays in (("native", native), ("adapter", adapter)):
        positive = np.asarray([np.flatnonzero(row == label)[0] + 1
                               for row, label in zip(arrays["ranks"], labels)], dtype=np.int64)
        metrics[side] = {"query_count": len(qids),
                         "hits": {f"r{k}": int(np.count_nonzero(positive <= k)) for k in (1, 10, 50)}}
    proof_sha = _sha(proof_path.read_bytes())
    plan_artifact_name = f"g1_g2.tgcir.{plan['category']}.v3-plan.json"
    block = helper.v3_artifact_block(Path(plan_artifact_name), plan_sha, scopes=scopes, receipts=receipts)
    block.update({"scoring_proof_path": proof_path.name, "scoring_proof_sha256": proof_sha,
                  "scoring_run_receipts": score_receipts})
    identity = plan["identity"]
    g1_observation = {
        role: {"coverage": receipts[role]["g1"]["coverage"],
               "query_image_names_sha256": _ordered_sha(receipts[role]["g1"]["query_image_names"]),
               "gallery_image_names_sha256": _ordered_sha(receipts[role]["g1"]["gallery_image_names"]),
               "raw_query_texts_sha256": _ordered_sha(receipts[role]["g1"]["raw_query_texts"])}
        for role in ROLE_SIDES
    }
    native_artifact = {"model_id": "tgcir", "category": plan["category"], "standard": plan["standard"],
        "identity": identity, "checkpoint_sha256": identity["checkpoint"], "checkpoint_id": identity["checkpoint"],
        "protocol": identity["protocol"], "protocol_id": identity["protocol"],
        "query_order_sha256": identity["query_order_sha256"], "gallery_order_sha256": identity["gallery_order_sha256"],
        "tolerances": {"score_max_abs": 0.0, "tie_epsilon": 0.0}, "parity_v3": block,
        "g1_observation": g1_observation,
        "native": {"layer1_identity": {"query_count_native": len(qids), "query_count_annotations": len(qids),
            "gallery_count_native": len(gids), "gallery_count_rebuilt_from_annotations": len(gids)}},
        "native_metrics_parity_detail": {"cutoffs": [1, 10, 50], "native": metrics["native"],
            "adapter": metrics["adapter"], "eligible_query_set_identical": True,
            "eligible_query_set_evidence": "Both score rows use preregistered query and gallery order."}}
    adapter_artifact = {"model_id": "tgcir", "category": plan["category"], "standard": plan["standard"],
        "identity": identity, "checkpoint_sha256": identity["checkpoint"], "checkpoint_id": identity["checkpoint"],
        "protocol": identity["protocol"], "protocol_id": identity["protocol"],
        "query_order_sha256": identity["query_order_sha256"], "gallery_order_sha256": identity["gallery_order_sha256"],
        "tolerances": {"score_max_abs": 0.0, "tie_epsilon": 0.0}, "adapter": {"tgcir_v3_capture": True},
        "g1_observation": g1_observation}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"g1_g2.tgcir.{plan['category']}.native.json").write_text(json.dumps(native_artifact, sort_keys=True) + "\n")
    (output / f"g1_g2.tgcir.{plan['category']}.adapter.json").write_text(json.dumps(adapter_artifact, sort_keys=True) + "\n")
    for path, name in ((plan_path, plan_artifact_name), (proof_path, proof_path.name)):
        destination = output / name
        if path.resolve() == destination.resolve():
            continue
        if path == plan_path:
            os.link(path, destination)
        else:
            destination.write_bytes(path.read_bytes())
    return {"model_id": "tgcir", "category": plan["category"], "plan_path": str(plan_path),
            "scoring_proof_path": str(proof_path), "scoring_proof_sha256": proof_sha,
            "controls": scopes, "component_failures": component_failures,
            "status": "complete_gate_artifacts"}

if __name__ == "__main__":
    raise SystemExit(main())
