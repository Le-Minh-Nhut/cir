from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T4_PATH = ROOT / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
ROLE_ORDER = ("native_adapter_native", "native_adapter_adapter", "native_batch_1", "adapter_batch_1",
              "native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
              "native_batch_n", "adapter_batch_n")


def _t4() -> Any:
    spec = importlib.util.spec_from_file_location("cir_fashioniq_v3_t4", T4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V3 component helper: {T4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_receipts(paths: list[Path], plan_sha: str) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            receipt = item.get("v3_receipt") if isinstance(item, dict) else None
            role = item.get("role") if isinstance(item, dict) else None
            if isinstance(receipt, dict) and role in ROLE_ORDER and receipt.get("plan_sha256") == plan_sha:
                if role in found:
                    raise ValueError(f"duplicate receipt for role {role}")
                found[role] = receipt
    return found


def _component(t4: Any, name: str, actual: np.ndarray, reference: np.ndarray,
               atol: float, rtol: float) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(f"{name}: shape mismatch {actual.shape} != {reference.shape}")
    return t4.component_scope(name, "feature_components", actual, reference, atol=atol, rtol=rtol)


def _metrics(blob: Any) -> dict[str, Any]:
    scores = blob["scores"]
    ranks = blob["ranks"]
    labels = blob["labels"]
    ids = blob["query_ids"].astype(str).tolist()
    gallery = blob["gallery_ids"].astype(str).tolist()
    if scores.ndim != 2 or ranks.shape != scores.shape or labels.shape != (scores.shape[0],):
        raise ValueError("scores/ranks/labels dimensions do not align")
    if len(ids) != len(labels) or scores.shape[1] != len(gallery):
        raise ValueError("query/gallery IDs do not align with score matrix")
    if not np.array_equal(np.sort(ranks, axis=1), np.broadcast_to(np.arange(scores.shape[1]), ranks.shape)):
        raise ValueError("rank rows are not complete permutations")
    target_ids = np.asarray([query_id.rsplit(":", 1)[-1] for query_id in ids])
    if any(gallery[int(label)] != target for label, target in zip(labels, target_ids)):
        raise ValueError("integer positive-label indices do not resolve to query targets")
    true_ranks = np.empty(len(ids), dtype=np.int64)
    for i in range(len(ids)):
        positions = np.flatnonzero(ranks[i] == labels[i])
        if len(positions) != 1:
            raise ValueError(f"query {ids[i]} has non-unique positive rank")
        true_ranks[i] = positions[0] + 1
    hit_counts = {f"r{k}": int(np.count_nonzero(true_ranks <= k)) for k in (1, 10, 50)}
    return {"query_count": len(ids), "gallery_count": len(gallery),
            "scores_shape": list(scores.shape), "scores_dtype": str(scores.dtype),
            "scores_sha256": hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest(),
            "ranks_shape": list(ranks.shape), "ranks_dtype": str(ranks.dtype),
            "ranks_sha256": hashlib.sha256(np.ascontiguousarray(ranks).tobytes()).hexdigest(),
            "positive_gallery_indices_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
            "positive_ranks_sha256": hashlib.sha256(np.ascontiguousarray(true_ranks).tobytes()).hexdigest(),
            "positive_rank_min": int(true_ranks.min()), "positive_rank_max": int(true_ranks.max()),
            "topk": {str(k): int(np.count_nonzero(true_ranks <= k)) for k in (1, 5, 10, 20, 50)},
            "integer_hits": hit_counts}


def _input_capture(receipt: dict[str, Any], role: str, scope: str) -> tuple[dict[str, np.ndarray], list[str]]:
    captures = receipt.get("input_capture")
    manifest = captures.get(scope) if isinstance(captures, dict) else None
    if not isinstance(manifest, dict):
        raise ValueError(f"{role} receipt lacks bounded {scope} input capture")
    path = Path(manifest["path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"{role} {scope} input capture digest mismatch")
    with np.load(path, allow_pickle=False) as blob:
        ids = blob["ordered_ids"].astype(str).tolist()
        arrays = {key: blob[key].copy() for key in blob.files if key != "ordered_ids"}
    if scope == "query" and "raw_text" in arrays:
        arrays["raw_composition_text"] = arrays.pop("raw_text")
    digests = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
               for key, value in arrays.items()}
    expected_digests = manifest.get("array_sha256", {})
    if scope == "query" and "raw_text" in expected_digests:
        expected_digests = {**expected_digests, "raw_composition_text": expected_digests["raw_text"]}
        expected_digests.pop("raw_text")
    expected_arrays = manifest.get("arrays", {})
    if scope == "query" and "raw_text" in expected_arrays:
        expected_arrays = {**expected_arrays, "raw_composition_text": expected_arrays["raw_text"]}
        expected_arrays.pop("raw_text")
    if (ids != manifest.get("ordered_ids") or digests != expected_digests
            or {key: list(value.shape) for key, value in arrays.items()} != expected_arrays):
        raise ValueError(f"{role} {scope} input capture arrays/IDs differ from immutable receipt")
    return arrays, ids
def _paired_input_capture(receipts: dict[str, dict[str, Any]], plan: dict[str, Any], root: Path) -> dict[str, Any]:
    captures = {}
    for side, role in (("native", "native_adapter_native"), ("adapter", "native_adapter_adapter")):
        captures[side] = {}
        for scope in ("query", "gallery"):
            arrays, ids = _input_capture(receipts[role], role, scope)
            selected = plan["input_capture"][f"{scope}_ids"]
            population_ids = plan["population_scope"]["query_ids" if scope == "query" else "gallery_ids"]
            if ids != [item for item in population_ids if item in selected]:
                raise ValueError(f"{role} {scope} IDs differ from preregistered capture sample")
            captures[side][scope] = (arrays, ids)
    result = {}
    for scope in ("query", "gallery"):
        native, native_ids = captures["native"][scope]
        adapter, adapter_ids = captures["adapter"][scope]
        fields = sorted(set(native) | set(adapter))
        if set(native) != set(adapter):
            field_diagnostics = {name: {"present_native": name in native,
                                        "present_adapter": name in adapter}
                                 for name in fields}
        else:
            field_diagnostics = {name: {"exact_equal": bool(np.array_equal(native[name], adapter[name])),
                                        "native_shape": list(native[name].shape), "adapter_shape": list(adapter[name].shape),
                                        "native_sha256": hashlib.sha256(np.ascontiguousarray(native[name]).tobytes()).hexdigest(),
                                        "adapter_sha256": hashlib.sha256(np.ascontiguousarray(adapter[name]).tobytes()).hexdigest()}
                                 for name in fields}
        result[scope] = {"coverage": "sample", "sample_count": len(native_ids),
                         "sampled_ids_identical": native_ids == adapter_ids,
                         "fields": field_diagnostics}
        if (not result[scope]["sampled_ids_identical"] or
                set(native) != set(adapter) or any(not item.get("exact_equal", False) for item in field_diagnostics.values())):
            diagnostic = {"schema": "cir.fashioniq-v3-input-capture-mismatch/v1",
                          "scope": scope, "sampled_ids_identical": native_ids == adapter_ids,
                          "native_ids": native_ids, "adapter_ids": adapter_ids,
                          "native_capture": receipts["native_adapter_native"]["input_capture"][scope],
                          "adapter_capture": receipts["native_adapter_adapter"]["input_capture"][scope],
                          "fields": field_diagnostics}
            diagnostic_path = root / (Path(plan["scoring_proof_path"]).stem + f".{scope}.capture-mismatch.json")
            if diagnostic_path.exists():
                raise FileExistsError(f"refusing to overwrite input capture diagnostic: {diagnostic_path}")
            diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n",
                                       encoding="utf-8")
            raise ValueError(f"native/Adapter sampled {scope} runtime inputs differ; diagnostic={diagnostic_path}")

        raw = [receipts[role]["input_capture"][scope].get("raw_composition_text")
               for role in ("native_adapter_native", "native_adapter_adapter")]
        if any(item is not None for item in raw):
            result[scope]["raw_composition_text_identical"] = raw[0] == raw[1]
    return result




def aggregate(plan_path: Path, log_paths: list[Path]) -> dict[str, Any]:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    root = plan_path.parent
    receipts = _log_receipts(log_paths, plan_sha)
    missing = sorted(set(ROLE_ORDER) - set(receipts))
    if missing:
        raise ValueError(f"missing actual role receipts: {missing}")
    role_paths = {
        role: root / f"{plan_path.stem}.{role}.{plan['runs'][role]['source_id'].split(':', 1)[0]}.npz"
        for role in ROLE_ORDER
    }
    for role, path in role_paths.items():
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"missing nonempty role artifact: {role} {path}")
    control_paths = role_paths
    if plan["model_id"] == "tme":
        control_paths = {role: root / "control-captures" / role / f"{plan['category']}.npz"
                         for role in ROLE_ORDER}
        for role, path in control_paths.items():
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"missing nonempty TME control artifact: {role} {path}")
            control_receipt_path = path.with_suffix(".run.json")
            control_receipt = json.loads(control_receipt_path.read_text(encoding="utf-8"))
            expected = plan["runs"][role]
            if (control_receipt.get("model_id") != "tme"
                    or control_receipt.get("category") != plan["category"]
                    or control_receipt.get("role") != role
                    or control_receipt.get("side") != expected["source_id"].split(":", 1)[0]
                    or control_receipt.get("run_id") != expected["run_id"]
                    or control_receipt.get("source_id") != expected["source_id"]
                    or control_receipt.get("plan_sha256") != plan_sha
                    or control_receipt.get("identity") != expected["identity"]
                    or control_receipt.get("execution") != expected["execution"]
                    or control_receipt.get("capture_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"TME control receipt identity differs from plan or capture: {role}")
            with np.load(path, allow_pickle=False) as control_artifact:
                identity = expected["identity"]
                for ids_key, count_key, digest_key in (
                        ("query_ids", "query_count", "query_order_sha256"),
                        ("gallery_ids", "gallery_count", "gallery_order_sha256")):
                    ids = control_artifact[ids_key].astype(str).tolist()
                    if (len(ids) != identity[count_key]
                            or hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                separators=(",", ":")).encode()).hexdigest() != identity[digest_key]):
                        raise ValueError(f"TME control capture IDs differ from preregistered identity: {role}/{ids_key}")
                for key in ("query_ids", "gallery_ids", "query_features", "gallery_features"):
                    array = control_artifact[key]
                    recorded = control_receipt.get("arrays", {}).get(key, {})
                    if (recorded.get("dtype") != str(array.dtype)
                            or recorded.get("shape") != list(array.shape)
                            or recorded.get("sha256") != hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()):
                        raise ValueError(f"TME control receipt array differs from capture: {role}/{key}")

    for role in ROLE_ORDER:
        receipt = receipts[role]
        expected = plan["runs"][role]
        if receipt.get("run_id") != expected["run_id"]:
            raise ValueError(f"role receipt run ID differs from plan: {role}")
        if receipt.get("source_id") != expected["source_id"]:
            raise ValueError(f"role receipt source differs from plan: {role}")
        expected_side = expected["source_id"].split(":", 1)[0]
        if receipt.get("side") != expected_side:
            raise ValueError(f"role receipt side differs from plan: {role}")
        role_path = role_paths[role]
        with np.load(role_path, allow_pickle=False) as artifact:
            for key in ("query_features", "target_features", "scores", "ranks", "labels"):
                recorded = receipt["arrays"][key]
                array = artifact[key]
                if key == "target_features":
                    array = array.reshape(array.shape[0], -1)
                id_key = "gallery_ids" if key == "target_features" else "query_ids"
                ordered_ids = artifact[id_key].astype(str).tolist()
                id_digest = hashlib.sha256(json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
                if recorded.get("shape") != list(array.shape) or recorded.get("dtype") != str(array.dtype):
                    raise ValueError(f"role receipt array metadata differs from output: {role}/{key}")
                if recorded.get("component_sha256") != _t4()._component_digest(array):
                    raise ValueError(f"role receipt component digest differs from output: {role}/{key}")
                if recorded.get("ordered_ids_sha256") != id_digest:
                    raise ValueError(f"role receipt ordered-ID digest differs from output: {role}/{key}")
    t4 = _t4()
    control_results: dict[str, Any] = {}
    component_failures: list[str] = []
    feature_tensor_component_failures: list[str] = []
    for name, control in plan["controls"].items():
        ref_role = next(role for role, item in plan["runs"].items()
                        if item["run_id"] == control["reference_run_id"])
        actual_role = next(role for role, item in plan["runs"].items()
                           if item["run_id"] == control["actual_run_id"])
        scopes = {}
        gallery_key = "gallery_features" if plan["model_id"] == "tme" else "target_features"
        control_tensors = (("query", "query_features", "query_ids"),
                           ("gallery", gallery_key, "gallery_ids"))
        for tensor, key, ids_key in control_tensors:
            with np.load(control_paths[ref_role], allow_pickle=False) as ref, np.load(control_paths[actual_role], allow_pickle=False) as actual:
                ref_arr = ref[key]
                act_arr = actual[key]
                ref_ids = ref[ids_key].astype(str).tolist()
                act_ids = actual[ids_key].astype(str).tolist()
            if tensor == "gallery":
                ref_arr = ref_arr.reshape(ref_arr.shape[0], -1)
                act_arr = act_arr.reshape(act_arr.shape[0], -1)
            if ref_ids != act_ids:
                raise ValueError(f"{name}.{tensor} ordered identity mismatch")
            component_tol = control["tolerance"]
            record = _component(t4, name, act_arr, ref_arr, component_tol["atol"], component_tol["rtol"])
            record["dimension"] = control["dimension"]
            expected = control["scopes"][tensor]
            if list(act_arr.shape) != expected["component_shape"]:
                raise ValueError(f"{name}.{tensor} shape differs from preregistered component shape")
            record.update({"reference_run_id": control["reference_run_id"],
                           "actual_run_id": control["actual_run_id"],
                           "population_coverage": expected["coverage"], "population_count": expected["population_count"],
                           "sample_count": expected.get("sample_count"), "ordered_ids_sha256":
                           hashlib.sha256(json.dumps(act_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                           "atol": component_tol["atol"], "rtol": component_tol["rtol"]})
            scopes[tensor] = record
            if record["mismatch_count"]:
                failure = f"{name}.{tensor}: {record['mismatch_count']} mismatches, max_abs={record['max_abs']}"
                component_failures.append(failure)
                feature_tensor_component_failures.append(failure)
        control_results[name] = scopes
        with np.load(role_paths[ref_role], allow_pickle=False) as ref, np.load(role_paths[actual_role], allow_pickle=False) as actual:
            if not np.array_equal(ref["query_ids"], actual["query_ids"]) or not np.array_equal(ref["gallery_ids"], actual["gallery_ids"]):
                raise ValueError(f"{name} score query/gallery identity mismatch")
            score_delta = np.abs(actual["scores"].astype(np.float64) - ref["scores"].astype(np.float64))
            score_max_abs = float(score_delta.max(initial=0.0))
            score_mismatch_count = int(np.count_nonzero(score_delta > 1e-6))
            score_record = {"within_absolute_1e-6": score_mismatch_count == 0,
                             "mismatch_count": score_mismatch_count,
                            "max_abs": score_max_abs,
                            "absolute_tolerance": 1e-6,
                             "reference_sha256": hashlib.sha256(np.ascontiguousarray(ref["scores"]).tobytes()).hexdigest(),
                             "actual_sha256": hashlib.sha256(np.ascontiguousarray(actual["scores"]).tobytes()).hexdigest()}
            rank_equal = np.array_equal(actual["ranks"], ref["ranks"])
            label_equal = np.array_equal(actual["labels"], ref["labels"])
            actual_metric = _metrics(actual)
            reference_metric = _metrics(ref)
            ref_positive = np.argmax(ref["ranks"] == ref["labels"][:, None], axis=1).astype(int)
            actual_positive = np.argmax(actual["ranks"] == actual["labels"][:, None], axis=1).astype(int)
        scopes["scores"] = score_record
        scopes["ranks"] = {"exact_equal": rank_equal,
                            "positive_rank_positions_0based_equal": bool(np.array_equal(actual_positive, ref_positive)),
                            "per_query_reference_positive_rank_0based": ref_positive.tolist(),
                            "per_query_actual_positive_rank_0based": actual_positive.tolist()}
        scopes["integer_labels"] = {"exact_equal": label_equal}
        scopes["integer_hits"] = {"reference": reference_metric["integer_hits"],
                                   "actual": actual_metric["integer_hits"],
                                   "per_query_reference_hit_flags": {str(k): (ref_positive < k).astype(int).tolist() for k in (1, 5, 10, 20, 50)},
                                   "per_query_actual_hit_flags": {str(k): (actual_positive < k).astype(int).tolist() for k in (1, 5, 10, 20, 50)}}
        if score_record["mismatch_count"]:
            component_failures.append(f"{name}.scores: {score_record['mismatch_count']} outside declared tolerance, max_abs={score_record['max_abs']}")
        if not rank_equal:
            component_failures.append(f"{name}.ranks: exact sorted-index rows differ")
        if not label_equal:
            component_failures.append(f"{name}.integer_labels: positive gallery indices differ")
        if not np.array_equal(actual_positive, ref_positive):
            component_failures.append(f"{name}.positive_ranks: per-query target ranks differ")

    full_role = {"native": "native_adapter_native", "adapter": "native_adapter_adapter"}
    metrics = {}
    side_data = {}
    for side, role in full_role.items():
        with np.load(role_paths[role], allow_pickle=False) as blob:
            metrics[side] = _metrics(blob)
            side_data[side] = {"query_features_shape": list(blob["query_features"].shape),
                               "gallery_features_shape": list(blob["target_features"].shape),
                               "query_order_sha256": hashlib.sha256(json.dumps(blob["query_ids"].astype(str).tolist(), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                               "gallery_order_sha256": hashlib.sha256(json.dumps(blob["gallery_ids"].astype(str).tolist(), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
    integer_hit_comparison = {cutoff: metrics["native"]["integer_hits"][cutoff] == metrics["adapter"]["integer_hits"][cutoff]
                              for cutoff in ("r10", "r50")}
    metric_pass = all(integer_hit_comparison.values())
    roles = {}
    for role, path in role_paths.items():
        with np.load(path, allow_pickle=False) as blob:
            roles[role] = {"path": str(path), "bytes": path.stat().st_size,
                           "run_id": receipts[role]["run_id"], "side": receipts[role]["side"],
                           "plan_sha256": receipts[role]["plan_sha256"],
                           "started_at": receipts[role]["started_at"], "captured_at": receipts[role]["captured_at"],
                           "query_count": int(blob["query_features"].shape[0]),
                           "gallery_count": int(blob["target_features"].shape[0]),
                           "scores_sha256": hashlib.sha256(np.ascontiguousarray(blob["scores"]).tobytes()).hexdigest(),
                           "ranks_sha256": hashlib.sha256(np.ascontiguousarray(blob["ranks"]).tobytes()).hexdigest()}
    proof_name = plan.get("scoring_proof_path")
    if not isinstance(proof_name, str) or Path(proof_name).name != proof_name or not proof_name.endswith(".score-ranks.jsonl"):
        raise ValueError("plan lacks a deterministic sibling scoring proof path")
    query_targets = plan.get("scoring_labels", {}).get("query_targets")
    exclusions = plan.get("scoring_labels", {}).get("reference_exclusions")
    if not isinstance(query_targets, list) or not isinstance(exclusions, list):
        raise ValueError("plan lacks preregistered target/exclusion labels")
    proof_path = plan_path.parent / proof_name
    if proof_path.exists():
        raise FileExistsError(f"refusing to replace scoring proof: {proof_path}")
    full_native, full_adapter = ("native_adapter_native", "native_adapter_adapter")
    scoring_receipts = {}
    for role, side in ((full_native, "native"), (full_adapter, "adapter")):
        receipt = receipts[role]
        if (not isinstance(receipt.get("score_order_sha256"), str)
                or receipt.get("identity") != plan["runs"][role]["identity"]):
            raise ValueError(f"full-role receipt lacks capture-time score proof binding: {role}")
        scoring_receipts[role] = {"run_id": receipt["run_id"], "source_id": receipt["source_id"],
                                  "side": side, "identity": receipt["identity"],
                                  "score_order_sha256": receipt["score_order_sha256"]}
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(role_paths[full_native], allow_pickle=False) as native, np.load(role_paths[full_adapter], allow_pickle=False) as adapter:
        query_ids = native["query_ids"].astype(str).tolist()
        gallery_ids = native["gallery_ids"].astype(str).tolist()
        native_scores, adapter_scores = native["scores"], adapter["scores"]
        native_order, adapter_order = native["ranks"], adapter["ranks"]
        label_bytes = json.dumps(plan["scoring_labels"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(label_bytes).hexdigest() != plan["identity"]["reference_exclusions_sha256"]:
            raise ValueError("scoring labels differ from preregistered identity digest")
        if (query_ids != adapter["query_ids"].astype(str).tolist()
                or gallery_ids != adapter["gallery_ids"].astype(str).tolist()
                or native_scores.shape != adapter_scores.shape
                or native_order.shape != native_scores.shape
                or adapter_order.shape != native_scores.shape
                or len(query_ids) != len(query_targets) or len(query_ids) != len(exclusions)):
            raise ValueError("full-role scoring proof populations/labels differ")
        gallery_positions = {gallery_id: index for index, gallery_id in enumerate(gallery_ids)}
        normalized_labels = []
        for index, query_id in enumerate(query_ids):
            target, exclusion = query_targets[index], exclusions[index]
            if (not isinstance(target, list) or len(target) != 2 or target[0] != query_id
                    or target[1] not in gallery_positions or not isinstance(exclusion, list)
                    or len(exclusion) != 2 or exclusion[0] != query_id
                    or not isinstance(exclusion[1], list)
                    or len(set(exclusion[1])) != len(exclusion[1])
                    or any(value not in gallery_positions for value in exclusion[1])
                    or target[1] in exclusion[1]):
                raise ValueError(f"invalid preregistered target/exclusions for {query_id}")
            normalized_labels.append((target[1], gallery_positions[target[1]], sorted(exclusion[1])))
        conventions = plan.get("score_order_conventions")
        expected_conventions = {"native": "fp32_one_minus_score_ascending",
                                "adapter": "fp32_descending_score"}
        if conventions != expected_conventions:
            raise ValueError(f"plan score_order_conventions must declare observed FashionIQ sorters: {expected_conventions}")
        for side, scores, orders in (("native", native_scores, native_order),
                                      ("adapter", adapter_scores, adapter_order)):
            ascending = conventions[side] == "fp32_one_minus_score_ascending"
            keys = np.subtract(np.float32(1.0), scores, dtype=np.float32) if ascending else scores
            if (not np.isfinite(scores).all()
                    or not np.array_equal(np.sort(orders, axis=1),
                                          np.broadcast_to(np.arange(len(gallery_ids)), orders.shape))
                    or any(np.any(keys[row, orders[row, :-1]] > keys[row, orders[row, 1:]]) if ascending
                           else np.any(keys[row, orders[row, :-1]] < keys[row, orders[row, 1:]])
                           for row in range(len(query_ids)))):
                raise ValueError(f"full-role {side} scores/orders contradict {conventions[side]}")
        score_hashes = {full_native: hashlib.sha256(), full_adapter: hashlib.sha256()}
        header = {"schema": "cir.score-ranks/v1", "identity": plan["identity"],
                  "query_ids": query_ids, "gallery_ids": gallery_ids,
                  **plan["scoring_labels"]}
        with proof_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
            for index, query_id in enumerate(query_ids):
                target_id, target_position, excluded = normalized_labels[index]
                row = {"query_id": query_id, "target_id": target_id,
                       "target_position": target_position, "excluded_gallery_ids": excluded,
                       "native_scores": native_scores[index].tolist(),
                       "adapter_scores": adapter_scores[index].tolist(),
                       "native_order": native_order[index].tolist(),
                       "adapter_order": adapter_order[index].tolist()}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                for role, scores, order in ((full_native, native_scores, native_order),
                                            (full_adapter, adapter_scores, adapter_order)):
                    captured = {"query_id": query_id, "scores": scores[index].tolist(),
                                "order": order[index].tolist()}
                    score_hashes[role].update(json.dumps(captured, ensure_ascii=False,
                        separators=(",", ":")).encode("utf-8") + b"\n")
        for role, hasher in score_hashes.items():
            if hasher.hexdigest() != scoring_receipts[role]["score_order_sha256"]:
                proof_path.unlink()
                raise ValueError(f"{role} NPZ arrays differ from capture-time score/order receipt")
    proof_sha256 = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    t4 = _t4()
    scopes = {}
    for name, control_scopes in control_results.items():
        scopes[name] = {}
        control = plan["controls"][name]
        gallery_key = "gallery_features" if plan["model_id"] == "tme" else "target_features"
        control_tensors = (("query", "query_features", "query_ids"),
                           ("gallery", gallery_key, "gallery_ids"))
        for tensor, key, ids_key in control_tensors:
            record = control_scopes[tensor]
            record["dtype"] = control["dtype"]
            expected = control["scopes"][tensor]
            reference_role = next(role for role, item in plan["runs"].items()
                                  if item["run_id"] == control["reference_run_id"])
            record.update({"population_coverage": expected["coverage"],
                           "population_count": expected["population_count"],
                           "sample_count": expected["sample_count"],
                           "tensor_shape": expected["tensor_shape"],
                           "component_shape": expected["component_shape"],
                           "cohort_id": plan["runs"][reference_role]["identity"]["cohort_id"]})
            scopes[name][tensor] = record
            roles_for_control = [role for role, item in plan["runs"].items()
                                 if item["run_id"] in (control["reference_run_id"], control["actual_run_id"])]
            for role in roles_for_control:
                with np.load(control_paths[role], allow_pickle=False) as blob:
                    raw_array = blob[key]
                    array = raw_array.reshape(raw_array.shape[0], -1) if tensor == "gallery" else raw_array
                    ids = blob[ids_key].astype(str).tolist()
                receipts[role].setdefault("arrays", {})[f"{name}.{tensor}"] = {
                    "dtype": str(array.dtype), "shape": list(raw_array.shape),
                    "component_sha256": t4._component_digest(array),
                    "ordered_ids_sha256": hashlib.sha256(json.dumps(
                        ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}
    block = t4.v3_artifact_block(Path(plan_path.name), plan_sha, scopes=scopes, receipts=receipts)
    block.update({"scoring_proof_path": proof_name,
                  "scoring_proof_sha256": proof_sha256,
                  "scoring_run_receipts": scoring_receipts})
    identity = plan["identity"]
    native_identity = receipts[full_native].get("identity_evidence")
    adapter_identity = receipts[full_adapter].get("identity_evidence")
    if not isinstance(native_identity, dict) or not isinstance(adapter_identity, dict):
        raise ValueError("full-role receipts lack actual scorer identity evidence")
    sampled_inputs = _paired_input_capture(receipts, plan, root)
    population = plan["population_scope"]
    expected_references = population["reference_ids"]
    expected_targets = population["target_ids"]
    expected_gallery = population["gallery_ids"]
    pair_rows = json.loads(Path(population["query_source"]).read_text(encoding="utf-8"))
    expected_texts = [f"{row['captions'][0].strip('.?, ').capitalize()} and {row['captions'][1].strip('.?, ')}"
                      for row in pair_rows]

    identity_matches = {
        "gallery_order_identical": (native_identity.get("gallery_order_ids") == expected_gallery
                                    and adapter_identity.get("gallery_order_ids") == expected_gallery),
        "reference_ids_identical": (native_identity.get("reference_ids") == expected_references
                                    and adapter_identity.get("reference_ids") == expected_references),
        "target_ids_identical": (native_identity.get("target_ids") == expected_targets
                                 and adapter_identity.get("target_ids") == expected_targets),
        "model_text_identical_to_author_composition": (
            native_identity.get("model_texts") == expected_texts
            and adapter_identity.get("model_texts") == expected_texts),
        "sampled_runtime_query_inputs_identical": all(
            field["exact_equal"] for field in sampled_inputs["query"]["fields"].values()),
    }
    layer1_identity = {**identity_matches,
        "query_count_native": identity["query_count"],
        "query_count_annotations": len(pair_rows),
        "gallery_count_native": identity["gallery_count"],
        "gallery_count_rebuilt_from_annotations": len(expected_gallery),
        "gallery_has_duplicates": len(set(expected_gallery)) != len(expected_gallery),
        "sampled_runtime_inputs": sampled_inputs}
    metric_detail = {"cutoffs": [10, 50], "native": {"query_count": identity["query_count"],
                     "hits": {f"r{k}": metrics["native"]["integer_hits"][f"r{k}"] for k in (10, 50)}},
                     "adapter": {"query_count": identity["query_count"],
                       "hits": {f"r{k}": metrics["adapter"]["integer_hits"][f"r{k}"] for k in (10, 50)}},
                     "eligible_query_set_identical": (
                         native_identity.get("reference_ids") == adapter_identity.get("reference_ids")
                         and native_identity.get("target_ids") == adapter_identity.get("target_ids")),
                     "eligible_query_set_evidence": "Independent native and production Adapter receipts bind full scorer reference/target identities; references remain gallery eligible."}
    common = {"model_id": plan["model_id"], "category": plan["category"],
              "standard": plan["standard"], "identity": identity,
              "checkpoint_sha256": identity["checkpoint"], "checkpoint_id": identity["checkpoint"],
              "protocol": identity["protocol"], "protocol_id": identity["protocol"],
              "query_order_sha256": identity["query_order_sha256"],
              "gallery_order_sha256": identity["gallery_order_sha256"],
              "tolerances": plan["tolerances"]}
    native_artifact = {**common, "parity_v3": block,
        "native": {"layer1_identity": layer1_identity},
        "native_metrics_parity_detail": metric_detail,
        "native_metrics_match_feature_rerank": all(
            metric_detail["native"]["hits"][f"r{k}"] == metric_detail["adapter"]["hits"][f"r{k}"]
            for k in (10, 50))}
    adapter_artifact = {**common, "adapter": {"fashioniq_v3_capture": True}}
    native_path = root / f"g1_g2.{plan['model_id']}.{plan['category']}.native.json"
    adapter_path = root / f"g1_g2.{plan['model_id']}.{plan['category']}.adapter.json"
    for artifact_path, document in ((native_path, native_artifact), (adapter_path, adapter_artifact)):
        if artifact_path.exists():
            raise FileExistsError(f"refusing to replace V3 gate artifact: {artifact_path}")
        artifact_path.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
                                 encoding="utf-8")
    controls_pass = not component_failures
    return {"model_id": plan["model_id"], "category": plan["category"], "plan_path": str(plan_path),
            "plan_sha256": plan_sha, "scoring_proof_path": str(proof_path),
            "scoring_proof_sha256": proof_sha256, "scoring_run_receipts": scoring_receipts,
            "query_count_full": plan["identity"]["query_count"],
            "gallery_count_full": plan["identity"]["gallery_count"], "roles": roles,
            "full_g2": {"side_shapes": side_data, "metrics": metrics,
                        "integer_hit_equality_by_cutoff": integer_hit_comparison,
                        "integer_hit_equality": metric_pass},
            "controls": control_results, "checks": {
                "all_five_controls_componentwise_within_preregistered_tolerance": controls_pass,
                "all_five_control_feature_tensors_within_preregistered_tolerance": not feature_tensor_component_failures,
                "full_native_adapter_integer_hits_equal": metric_pass,
                "component_failures": component_failures,
                "legacy_mixed_diagnostic_failures": component_failures,
                "feature_tensor_component_failures": feature_tensor_component_failures,
                "score_order_label_diagnostic_failures": [failure for failure in component_failures
                    if failure not in feature_tensor_component_failures]},
            "status": "ARTIFACTS_ASSEMBLED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--log", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.plan, args.log)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"], "checks": result["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
