"""Capture native CLVC/TME roles and run preregistered direct production Adapter roles.

Native source replay remains the source-faithful reference; this producer binds both paths to
their immutable V3 roles before publishing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import tempfile

import uuid
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "workbench/artifacts/parity/v3/clvc-tme"
CATEGORIES = ("dress", "shirt", "toptee")
MODELS = ("clvc_net", "tme")
STANDARD = "parity-absolute-relative/v3"


def _score_order_digest(query_ids: Any, scores: Any, orders: Any) -> str:
    rows = ({"query_id": str(query_id), "scores": scores[index].tolist(),
             "order": orders[index].tolist()} for index, query_id in enumerate(query_ids))
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _t4() -> Any:
    path = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
    spec = importlib.util.spec_from_file_location("cir_t4_parity_clvc_tme", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import V3 publisher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(path: Path) -> tuple[dict[str, Any], str]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read immutable V3 plan {path}: {error}") from error
    if plan.get("standard") != STANDARD or plan.get("model_id") not in MODELS:
        raise ValueError(f"unsupported V3 plan: {path}")
    if plan.get("category") not in CATEGORIES:
        raise ValueError(f"unsupported V3 plan category: {path}")
    return plan, _sha(path)


def _role(plan: dict[str, Any], role_name: str) -> dict[str, Any]:
    role = plan.get("runs", {}).get(role_name)
    if not isinstance(role, dict) or role.get("side") != "native":
        raise ValueError(f"role is not a preregistered native role: {role_name}")
    identity = role.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"native role lacks frozen identity: {role_name}")
    return role


def _metrics(path: Path, override: Path | None) -> dict[str, Any]:
    import numpy as np
    if override is not None:
        value = json.loads(override.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("metrics JSON must contain an object")
        if "results" in value:
            for result in value["results"]:
                if isinstance(result, dict) and result.get("category") and path.stem == result["category"]:
                    return dict(result.get("metrics", {}))
        return value
    with np.load(path, allow_pickle=False) as capture:
        if "metadata" not in capture.files:
            raise ValueError("native capture has no metrics metadata; pass --metrics-json")
        raw = capture["metadata"].item()
    value = json.loads(str(raw))
    metrics = value.get("metrics") if isinstance(value, dict) else None
    if not isinstance(metrics, dict) and isinstance(value, dict):
        metrics = value
    if not isinstance(metrics, dict):
        raise ValueError("native capture metadata has no metrics object; pass --metrics-json")
    if isinstance(metrics.get(path.stem), dict):
        metrics = metrics[path.stem]
    return metrics


def convert_native(plan_path: Path, capture_path: Path, output_root: Path,
                   role_name: str, metrics_path: Path | None) -> Path:
    plan, plan_sha = _plan(plan_path)
    role = _role(plan, role_name)
    if not capture_path.is_file():
        raise FileNotFoundError(f"native capture missing: {capture_path}")
    receipt_path = capture_path.with_suffix(".run.json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"native capture has no valid role binding: {receipt_path}: {error}") from error
    import numpy as np
    with np.load(capture_path, allow_pickle=False) as archive:
        score_order_sha256 = _score_order_digest(
            archive["query_ids"], archive["scores"], archive["native_order"])
    if (receipt.get("standard") != STANDARD or receipt.get("model_id") != plan["model_id"] or
            receipt.get("category") != plan["category"] or receipt.get("role") != role_name or
            receipt.get("side") != "native" or receipt.get("plan_sha256") != plan_sha or
            receipt.get("run_id") != role["run_id"] or receipt.get("source_id") != role["source_id"] or
            receipt.get("identity") != plan["identity"] or receipt.get("execution") != role["execution"] or
            receipt.get("score_order_sha256") != score_order_sha256 or
            receipt.get("capture_sha256") != _sha(capture_path)):
        raise ValueError("native capture binding differs from immutable role plan")
    from workbench.replay.clvc_tme_parity import RANK_POLICY, native_artifact_from_capture
    native_artifact_from_capture(
        capture_path=capture_path, root=output_root, model_id=plan["model_id"],
        category=plan["category"], condition=role_name, run_id=role["run_id"],
        plan_path=Path(plan_path.name), plan_sha256=plan_sha,
        started_at=datetime.now(timezone.utc).isoformat(), identity=plan["identity"],
        source_id=role["source_id"], metrics=_metrics(capture_path, metrics_path),
        execution=role["execution"], rank_policy=RANK_POLICY)
    return output_root / "native" / plan["model_id"] / f"{plan['category']}.{role_name}.json"


def bind_native(plan_path: Path, capture_path: Path, role_name: str) -> Path:
    plan, plan_sha = _plan(plan_path)
    role = _role(plan, role_name)
    if role.get("side") != "native" or not capture_path.is_file():
        raise ValueError("native capture binding requires a preregistered native role")
    import numpy as np
    with np.load(capture_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        invocation_id = metadata.get("invocation_id")
        score_order_sha256 = _score_order_digest(
            archive["query_ids"], archive["scores"], archive["native_order"])
    if not isinstance(invocation_id, str) or not invocation_id:
        raise ValueError("native capture lacks its pre-run invocation ID")
    receipt = {"standard": STANDARD, "model_id": plan["model_id"],
               "category": plan["category"], "role": role_name, "side": "native",
               "plan_sha256": plan_sha, "run_id": role["run_id"],
               "source_id": role["source_id"], "identity": plan["identity"],
               "execution": role["execution"], "invocation_id": invocation_id,
               "score_order_sha256": score_order_sha256,
               "capture_sha256": _sha(capture_path)}
    path = capture_path.with_suffix(".run.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return path
from workbench.replay.prepare_clvc_tme_v3_plans import _tme_rows

def bind_tme_family(family_path: Path, capture_root: Path, role_name: str) -> list[Path]:
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if (family.get("schema") != "cir-tme-family-plan-v1" or
            set(family.get("categories", {})) != set(CATEGORIES) or
            role_name not in family.get("roles", {})):
        raise ValueError("invalid TME family role plan")
    family_sha = _sha(family_path)
    family_role = family["roles"][role_name]
    result_path = capture_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_invocation_id = result.get("invocation_id")
    if not isinstance(result_invocation_id, str) or not result_invocation_id:
        raise ValueError("TME family result lacks its pre-run invocation ID")
    receipts = []
    import numpy as np
    invocation_ids = set()
    for category in CATEGORIES:
        member = family["categories"][category]
        plan_path = Path(member["plan_path"])
        plan, plan_sha = _plan(plan_path)
        if plan_sha != member["plan_sha256"] or plan["identity"] != member["identity"]:
            raise ValueError(f"TME family member plan changed: {category}")
        role = _role(plan, role_name)
        expected_queries = family_role["query_ids"][category]
        planned_queries = role["execution"].get("sample_query_ids", member["query_ids"])
        if expected_queries != planned_queries:
            raise ValueError(f"TME family query map differs from category role plan: {category}")
        capture = capture_root / f"{category}.npz"
        if not capture.is_file():
            raise FileNotFoundError(f"TME family capture missing: {capture}")
        with np.load(capture, allow_pickle=False) as archive:
            source_queries = archive["query_ids"].astype(str).tolist()
            if (not np.array_equal(archive["gallery_ids"], np.asarray(_tme_rows(category)[1])) or
                    len(source_queries) != len(set(source_queries)) or
                    not set(expected_queries) <= set(source_queries)):
                raise ValueError(f"TME family capture population differs from plan: {category}")
            metadata = json.loads(str(archive["metadata"].item()))
            invocation_id = metadata.get("invocation_id")
            started_at = metadata.get("started_at")
            score_order_sha256 = _score_order_digest(
                archive["query_ids"], archive["scores"], archive["native_order"])
            if metadata.get("model_id") != "tme" or metadata.get("category") != category:
                raise ValueError(f"TME family capture source identity differs: {category}")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError(f"TME capture lacks pre-run invocation ID: {category}")
        invocation_ids.add(invocation_id)
        receipts.append((capture, member, plan, plan_sha, role, score_order_sha256,
                         invocation_id, started_at))
    if len(invocation_ids) != 1 or invocation_ids != {result_invocation_id}:
        raise ValueError("TME family result and captures do not share one native invocation")
    for capture, member, plan, plan_sha, role, score_order_sha256, invocation_id, started_at in receipts:
        receipt = {"standard": STANDARD, "model_id": "tme", "category": plan["category"],
                   "role": role_name, "side": "native", "plan_sha256": plan_sha,
                   "family_plan_sha256": family_sha, "run_id": role["run_id"],
                   "source_id": role["source_id"], "identity": role["identity"],
                   "execution": role["execution"], "invocation_id": invocation_id,
                   "started_at": started_at, "score_order_sha256": score_order_sha256,
                   "capture_sha256": _sha(capture)
        }
        path = capture.with_suffix(".run.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        control_path = (family_path.parent / "control-captures" / role_name /
                        f"{plan['category']}.npz")
        control_path.parent.mkdir(parents=True, exist_ok=True)
        write_tme_control_capture(
            capture, control_path, role=role, role_name=role_name, plan_sha=plan_sha,
            category=plan["category"], query_ids=family_role["query_ids"][plan["category"]],
            gallery_ids=_tme_rows(plan["category"])[1], side="native",
            invocation_id=result_invocation_id, started_at=started_at)

    return [capture.with_suffix(".run.json") for capture, *_ in receipts]
def write_tme_control_capture(source_path: Path, output_path: Path, *, role: dict[str, Any],
                              role_name: str, plan_sha: str, category: str,
                              query_ids: list[str], gallery_ids: list[str],
                              side: str, invocation_id: str, started_at: str) -> Path:
    """Select ordered query rows and bind unmodified scorer tensors to one frozen role."""
    import numpy as np

    with np.load(source_path, allow_pickle=False) as archive:
        source_queries = archive["query_ids"].astype(str).tolist()
        source_gallery = archive["gallery_ids"].astype(str).tolist()
        source_query_count = len(source_queries)
        source_gallery_count = len(source_gallery)
        if source_gallery != gallery_ids or len(source_queries) != len(set(source_queries)):
            raise ValueError(f"TME control source population differs: {role_name}/{category}")
        positions = {query_id: index for index, query_id in enumerate(source_queries)}
        if (len(query_ids) != len(set(query_ids)) or not set(query_ids) <= positions.keys() or
                query_ids != [query_id for query_id in source_queries if query_id in set(query_ids)]):
            raise ValueError(f"TME control query selection differs from source chronology: {role_name}/{category}")
        query_at = np.asarray([positions[query_id] for query_id in query_ids], dtype=np.int64)
        query_features = archive["query_features"][query_at]
        gallery_features = archive["gallery_features"]
        if (query_features.shape != (len(query_ids), 256) or
                gallery_features.shape != (len(gallery_ids), 32, 256)):
            raise ValueError(f"TME scorer tensor shape differs from frozen contract: {role_name}/{category}")
        identity = role["identity"]
        if (identity["query_count"] != len(query_ids) or identity["gallery_count"] != len(gallery_ids) or
                identity["query_order_sha256"] != _ordered_ids_sha256(query_ids) or
                identity["gallery_order_sha256"] != _ordered_ids_sha256(gallery_ids)):
            raise ValueError(f"TME role identity differs from selected scorer tensors: {role_name}/{category}")
        payload = {"query_ids": np.asarray(query_ids), "gallery_ids": np.asarray(gallery_ids),
                   "query_features": query_features, "gallery_features": gallery_features}
        with output_path.open("xb") as stream:
            np.savez(stream, **payload)
    receipt = {"model_id": "tme", "category": category, "role": role_name,
               "side": side, "run_id": role["run_id"], "source_id": role["source_id"],
               "plan_sha256": plan_sha, "identity": identity,
               "execution": role["execution"], "invocation_id": invocation_id,
               "started_at": started_at, "source_capture_sha256": _sha(source_path),
               "source_population": {"query_count": source_query_count,
                                     "gallery_count": source_gallery_count,
                                     "evaluation_scope": ("full-source-evaluator" if side == "native"
                                                         else "adapter-role-query-forward"),
                                     "selected_query_count": len(query_ids)},
               "capture_sha256": _sha(output_path), "arrays": {
                   name: {"dtype": str(value.dtype), "shape": list(value.shape),
                          "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                   for name, value in payload.items()}}
    receipt_path = output_path.with_suffix(".run.json")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return receipt_path


def run_adapter(plan_path: Path, role_name: str, output_path: Path,
                sample_queries_path: Path | None = None) -> Path:
    """Run the registered production Adapter and persist raw + V3 receipts."""
    import numpy as np
    from workbench.backend.operator_config import resolve_config
    from workbench.replay.prepare_clvc_tme_v3_plans import _clvc_rows, _tme_rows

    plan, plan_sha = _plan(plan_path)
    role = plan.get("runs", {}).get(role_name)
    if not isinstance(role, dict) or role.get("side") != "adapter":
        raise ValueError(f"role is not a preregistered adapter execution: {role_name}")
    model_id, category = plan["model_id"], plan["category"]
    if model_id == "clvc_net":
        from workbench.composition.model_clvc_net import Adapter

        query_ids, gallery_ids = _clvc_rows(category)
        checkpoint_id = None
    else:
        from workbench.composition.model_tme import Adapter

        query_ids, gallery_ids = _tme_rows(category)
        checkpoint_id = "fashioniq_0"
    config = resolve_config()
    dataset_root = Path(config.FASHIONIQ_ROOT)
    caption_path = dataset_root / "captions" / f"cap.{category}.val.json"
    rows = json.loads(caption_path.read_text(encoding="utf-8"))
    if len(query_ids) != len(rows):
        raise ValueError("frozen query IDs differ from source caption rows")
    sample_query_ids = set(role["execution"].get("sample_query_ids", []))
    if role["execution"].get("sample_query_count"):
        if sample_queries_path is None:
            raise ValueError("sample Adapter role requires its sealed query-selection file")
        declared_samples = json.loads(sample_queries_path.read_text(encoding="utf-8"))
        if (declared_samples != role["execution"]["sample_query_ids"] or
                _sha(sample_queries_path) != hashlib.sha256(
                    (json.dumps(declared_samples, separators=(",", ":")) + "\n").encode()).hexdigest()):
            raise ValueError("sample Adapter query file differs from frozen role selection")
        if sample_query_ids != set(declared_samples):
            raise ValueError("sample Adapter role IDs differ from its query file")
    elif sample_queries_path is not None:
        raise ValueError("full Adapter role must not use a sample query file")
    if sample_query_ids:
        keep = [index for index, query_id in enumerate(query_ids) if query_id in sample_query_ids]
        if len(keep) != len(sample_query_ids):
            raise ValueError("sample query IDs differ from frozen category population")
        query_ids = [query_ids[index] for index in keep]
        rows = [rows[index] for index in keep]
    elif role["execution"].get("sample_query_count"):
        raise ValueError("sample execution has no query IDs")
    invocation_id = str(uuid.uuid4())
    adapter = Adapter(config, checkpoint_id=checkpoint_id)
    worker_interpreter = adapter.require_verified_interpreter()
    work_root = Path(tempfile.mkdtemp(prefix=f"{model_id}-parity-"))
    adapter.work_root = work_root
    adapter.batch_size = int(role["execution"]["query_batch_size"])
    if model_id == "clvc_net":
        image_root = dataset_root / "resized_image" / category
        query_paths = [image_root / f"{row['candidate']}.jpg" for row in rows]
        gallery_paths = [image_root / f"{image_id}.jpg" for image_id in gallery_ids]
        texts = [f"<BOS> {row['captions'][0]} <AND> {row['captions'][1]} <EOS>" for row in rows]
        query_categories = [category] * len(rows)
        gallery_categories = [category] * len(gallery_ids)
    else:
        image_root = dataset_root / "images"
        query_paths = [image_root / f"{row['candidate']}.jpg" for row in rows]
        gallery_paths = [image_root / f"{image_id}.jpg" for image_id in gallery_ids]
        texts = [f"{row['captions'][0].strip('.?, ').capitalize()} and {row['captions'][1].strip('.?, ')}"
                 for row in rows]
        query_categories = [None] * len(rows)
        gallery_categories = [None] * len(gallery_ids)
    refs = [row["candidate"] for row in rows]
    targets = [row["target"] for row in rows]
    if any(not path.is_file() for path in query_paths + gallery_paths):
        adapter.close()
        shutil.rmtree(work_root, ignore_errors=True)
        raise FileNotFoundError(f"missing native-layout image under {image_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        adapter.close()
        shutil.rmtree(work_root, ignore_errors=True)
        raise FileExistsError(output_path)
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        queries = adapter.encode_queries(query_paths, texts, query_categories)
        adapter.batch_size = int(role["execution"]["gallery_batch_size"])
        gallery = adapter.encode_targets(gallery_paths, gallery_categories)
        index = {image_id: i for i, image_id in enumerate(gallery_ids)}
        targets_at = np.asarray([index[image_id] for image_id in targets], dtype=np.int64)
        refs_at = np.asarray([index[image_id] for image_id in refs], dtype=np.int64)
        ranks = np.empty(len(query_ids), dtype=np.int64)
        orders = np.empty((len(query_ids), len(gallery_ids)), dtype=np.int64)
        scores = np.empty((len(query_ids), len(gallery_ids)), dtype=np.float32)
        for row_index, query in enumerate(queries):
            row_scores = adapter.score(query[None, :], gallery)[0].astype(np.float32, copy=False)
            if model_id == "clvc_net":
                row_scores[refs_at[row_index]] = -10e10
                orders[row_index] = np.argsort(-row_scores)
            else:
                import torch

                orders[row_index] = torch.argsort(1 - torch.as_tensor(row_scores)).cpu().numpy()
            ranks[row_index] = int(np.flatnonzero(
                orders[row_index] == targets_at[row_index])[0] + 1)
            scores[row_index] = row_scores
        if model_id == "clvc_net":
            metrics = {f"{category}_r{k}": float(np.mean(ranks <= k) * 100) for k in (1, 10, 50)}
        else:
            metrics = {f"recall_at{k}": float(np.mean(ranks <= k) * 100) for k in (1, 5, 10, 50)}
            metrics["acc"] = (metrics["recall_at10"] + metrics["recall_at50"]) / 2
            metrics["mean"] = float(np.mean([metrics["recall_at10"], metrics["recall_at50"]]))
        with output_path.open("xb") as stream:
            np.savez_compressed(stream, query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                                query_features=queries, gallery_features=gallery, scores=scores,
                                native_order=orders, ranks=ranks, reference_ids=np.asarray(refs),
                                target_ids=np.asarray(targets),
                                metrics=np.asarray(json.dumps(metrics, sort_keys=True)),
                                plan_sha256=np.asarray(plan_sha), role=np.asarray(role_name),
                                invocation_id=np.asarray(invocation_id))
        score_order_sha256 = _score_order_digest(query_ids, scores, orders)
        receipt = {"standard": STANDARD, "model_id": model_id, "category": category,
                   "role": role_name, "side": "adapter", "plan_sha256": plan_sha,
                   "run_id": role["run_id"], "source_id": role["source_id"],
                   "identity": plan["identity"], "execution": role["execution"],
                   "invocation_id": invocation_id, "started_at": started_at,
                   "worker_interpreter": str(worker_interpreter),
                   "score_order_sha256": score_order_sha256,
                   "capture_sha256": _sha(output_path)}
        receipt_path = output_path.with_suffix(".run.json")
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        if model_id == "tme":
            control_path = (plan_path.parent / "control-captures" / role_name /
                            f"{category}.npz")
            control_path.parent.mkdir(parents=True, exist_ok=True)
            write_tme_control_capture(
                output_path, control_path, role=role, role_name=role_name, plan_sha=plan_sha,
                category=category, query_ids=query_ids, gallery_ids=gallery_ids, side="adapter",
                invocation_id=invocation_id, started_at=started_at)
        if role_name == "native_adapter_adapter":
            from workbench.replay.clvc_tme_parity import AdapterArtifactSink

            sink_role = dict(role, plan_sha256=plan_sha, plan_path=str(plan_path.resolve()))
            sink = AdapterArtifactSink(plan_path.parent, parity_run_for=lambda *_: sink_role)
            sink(model_id=model_id, category=category, condition=role_name,
                 run_id=role["run_id"], plan_path=plan_path, plan_sha256=plan_sha,
                 started_at=started_at, identity=plan["identity"],
                 source_id=role["source_id"], query_ids=query_ids, gallery_ids=gallery_ids,
                 query_features=queries, gallery_features=gallery,
                 ranks=dict(zip(query_ids, ranks.tolist(), strict=True)), metrics=metrics,
                 execution=role["execution"])
    finally:
        adapter.close()
        shutil.rmtree(work_root, ignore_errors=True)
    return output_path




def compare_pair(plan_path: Path, native_path: Path, adapter_path: Path, output_path: Path) -> Path:
    """Compare full native-source and direct-Adapter score/rank/metric arrays."""
    import numpy as np
    from workbench.replay.prepare_clvc_tme_v3_plans import _clvc_rows, _tme_rows
    plan, plan_sha = _plan(plan_path)
    expected_queries, expected_gallery = (_clvc_rows(plan["category"])
        if plan["model_id"] == "clvc_net" else _tme_rows(plan["category"]))
    native_role = plan["runs"]["native_adapter_native"]
    if (native_role.get("side") != "native" or
            native_role.get("identity") != plan.get("identity") or
            plan["runs"]["native_adapter_adapter"].get("identity") != plan.get("identity")):
        raise ValueError("paired roles differ from immutable plan identity")
    with np.load(native_path, allow_pickle=False) as native, np.load(adapter_path, allow_pickle=False) as adapter:
        if not np.array_equal(native["query_ids"], np.asarray(expected_queries)):
            raise ValueError("native capture query order differs from immutable plan population")
        if not np.array_equal(native["gallery_ids"], np.asarray(expected_gallery)):
            raise ValueError("native capture gallery order differs from immutable plan population")
        if str(adapter["role"].item()) != "native_adapter_adapter":
            raise ValueError("adapter capture is not the preregistered paired role")
        adapter_receipt_path = adapter_path.with_suffix(".run.json")
        try:
            adapter_receipt = json.loads(adapter_receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"adapter capture has no valid role binding: {adapter_receipt_path}: {error}") from error
        adapter_role = plan["runs"]["native_adapter_adapter"]
        if (str(adapter["plan_sha256"].item()) != plan_sha or
                adapter_receipt.get("standard") != STANDARD or
                adapter_receipt.get("model_id") != plan["model_id"] or
                adapter_receipt.get("category") != plan["category"] or
                adapter_receipt.get("role") != "native_adapter_adapter" or
                adapter_receipt.get("plan_sha256") != plan_sha or
                adapter_receipt.get("run_id") != adapter_role["run_id"] or
                adapter_receipt.get("source_id") != adapter_role["source_id"] or
                adapter_receipt.get("identity") != plan["identity"] or
                adapter_receipt.get("execution") != adapter_role["execution"] or
                adapter_receipt.get("side") != "adapter" or
                adapter_receipt.get("score_order_sha256") != _score_order_digest(
                    adapter["query_ids"], adapter["scores"], adapter["native_order"]) or
                adapter_receipt.get("invocation_id") != str(adapter["invocation_id"].item()) or
                adapter_receipt.get("capture_sha256") != _sha(adapter_path)):
            raise ValueError("adapter capture binding differs from immutable paired role")
        native_receipt_path = native_path.with_suffix(".run.json")
        try:
            native_receipt = json.loads(native_receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"native capture has no valid role binding: {native_receipt_path}: {error}") from error
        with np.load(native_path, allow_pickle=False) as native_metadata_archive:
            native_invocation_id = json.loads(
                str(native_metadata_archive["metadata"].item())).get("invocation_id")
        if (native_receipt.get("standard") != STANDARD or
                native_receipt.get("model_id") != plan["model_id"] or
                native_receipt.get("category") != plan["category"] or
                native_receipt.get("role") != "native_adapter_native" or
                native_receipt.get("plan_sha256") != plan_sha or
                native_receipt.get("run_id") != native_role["run_id"] or
                native_receipt.get("source_id") != native_role["source_id"] or
                native_receipt.get("identity") != plan["identity"] or
                native_receipt.get("execution") != native_role["execution"] or
                native_receipt.get("side") != "native" or
                native_receipt.get("score_order_sha256") != _score_order_digest(
                    native["query_ids"], native["scores"], native["native_order"]) or
                native_receipt.get("invocation_id") != native_invocation_id or
                native_receipt.get("capture_sha256") != _sha(native_path)):
            raise ValueError("native capture binding differs from immutable paired role")
        if not np.array_equal(native["query_ids"], adapter["query_ids"]):
            raise ValueError("native/adapter query populations differ")
        if not np.array_equal(native["gallery_ids"], adapter["gallery_ids"]):
            raise ValueError("native/adapter gallery populations differ")
        for key in ("reference_ids", "target_ids"):
            if key in native.files and not np.array_equal(native[key], adapter[key]):
                raise ValueError(f"native/adapter population differs: {key}")
        source_scores = np.asarray(native["scores"], dtype=np.float32)
        adapter_scores = np.asarray(adapter["scores"], dtype=np.float32)
        source_ranks = np.asarray(native["ranks"], dtype=np.int64)
        adapter_ranks = np.asarray(adapter["ranks"], dtype=np.int64)
        if source_scores.shape != adapter_scores.shape or source_ranks.shape != adapter_ranks.shape:
            raise ValueError("native/adapter score or rank shapes differ")
        if (source_scores.shape != (len(expected_queries), len(expected_gallery)) or
                source_ranks.shape != (len(expected_queries),) or
                not np.isfinite(source_scores).all() or not np.isfinite(adapter_scores).all()):
            raise ValueError("native/adapter paired score arrays are incomplete or non-finite")
        expected_targets = [target for _, target in plan["scoring_labels"]["query_targets"]]
        expected_references = ([excluded[0] for _, excluded in plan["scoring_labels"]["reference_exclusions"]]
                              if plan["model_id"] == "clvc_net" else [])
        if (native["target_ids"].tolist() != expected_targets or
                (plan["model_id"] == "clvc_net" and
                 native["reference_ids"].tolist() != expected_references)):
            raise ValueError("native capture labels differ from sealed scoring labels")
        for side, capture in (("native", native), ("adapter", adapter)):
            order = np.asarray(capture["native_order"])
            scores = source_scores if side == "native" else adapter_scores
            if (order.shape != scores.shape or
                    any(not np.array_equal(np.sort(row), np.arange(len(expected_gallery)))
                        for row in order)):
                raise ValueError(f"{side} captured source sort is not a complete gallery permutation")
        errors = np.abs(source_scores - adapter_scores)
        native_metrics = _metrics(native_path, None)
        adapter_metrics = json.loads(str(adapter["metrics"].item()))
        proof_path = plan_path.parent / plan["scoring_proof_path"]
        if proof_path.exists():
            raise FileExistsError(proof_path)
        proof_header = {
            "schema": "cir.score-ranks/v1", "identity": plan["identity"],
            "query_ids": native["query_ids"].tolist(),
            "gallery_ids": native["gallery_ids"].tolist(),
            **plan["scoring_labels"],
        }
        score_receipts = {}
        for side, role_name, score_array, order_array in (
                ("native", "native_adapter_native", source_scores, native["native_order"]),
                ("adapter", "native_adapter_adapter", adapter_scores, adapter["native_order"])):
            rows = [{"query_id": str(query_id), "scores": score_array[index].tolist(),
                     "order": order_array[index].tolist()}
                    for index, query_id in enumerate(native["query_ids"])]
            digest = hashlib.sha256("".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows).encode("utf-8")).hexdigest()
            role = plan["runs"][role_name]
            score_receipts[role_name] = {"run_id": role["run_id"], "source_id": role["source_id"],
                "side": side, "identity": plan["identity"], "score_order_sha256": digest}
        with proof_path.open("x", encoding="utf-8") as proof:
            proof.write(json.dumps(proof_header, ensure_ascii=False,
                                   separators=(",", ":")) + "\n")
            for index, query_id in enumerate(native["query_ids"]):
                proof.write(json.dumps({
                    "query_id": str(query_id),
                    "target_id": str(native["target_ids"][index]),
                    "target_position": int(np.flatnonzero(
                        native["gallery_ids"] == native["target_ids"][index])[0]),
                    "excluded_gallery_ids": ([str(native["reference_ids"][index])]
                        if plan["model_id"] == "clvc_net" else []),
                    "native_scores": source_scores[index].tolist(),
                    "adapter_scores": adapter_scores[index].tolist(),
                    "native_order": native["native_order"][index].tolist(),
                    "adapter_order": adapter["native_order"][index].tolist(),
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
        proof_sha = _sha(proof_path)
        scoring_proof = {"scoring_proof_path": plan["scoring_proof_path"],
                         "scoring_proof_sha256": proof_sha,
                         "scoring_run_receipts": score_receipts}
        artifact_path = (plan_path.parent / "native" / plan["model_id"] /
                         f"{plan['category']}.native_adapter_native.json")
        if artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["parity_v3"].update(scoring_proof)
            artifact["protocol"] = plan["identity"]["protocol"]
            artifact["protocol_id"] = plan["identity"]["protocol"]
            artifact["checkpoint_id"] = plan["identity"]["checkpoint"]
            artifact["checkpoint_sha256"] = plan["identity"]["checkpoint"]
            artifact["query_order_sha256"] = plan["identity"]["query_order_sha256"]
            artifact["native"] = {"layer1_identity": {
                "query_count_native": len(native["query_ids"]),
                "query_count_annotations": len(plan["scoring_labels"]["query_targets"]),
                "gallery_count_native": len(native["gallery_ids"]),
                "gallery_count_rebuilt_from_annotations": len(expected_gallery),
                "gallery_order_identical": bool(np.array_equal(native["gallery_ids"], adapter["gallery_ids"])),
                "target_ids_identical": bool(np.array_equal(native["target_ids"], adapter["target_ids"])),
                "reference_eligible": plan["model_id"] == "tme",
                "gallery_has_duplicates": len(set(native["gallery_ids"].tolist())) != len(native["gallery_ids"])}}
            cutoffs = [1, 10, 50]
            native_hits = {f"r{k}": int(np.count_nonzero(source_ranks <= k)) for k in cutoffs}
            adapter_hits = {f"r{k}": int(np.count_nonzero(adapter_ranks <= k)) for k in cutoffs}
            detail = {"cutoffs": cutoffs,
                "native": {"hits": native_hits, "query_count": len(source_ranks)},
                "adapter": {"hits": adapter_hits, "query_count": len(adapter_ranks)},
                "eligible_query_set_identical": bool(np.array_equal(native["query_ids"], adapter["query_ids"])),
                "equality": {key: native_hits[key] == adapter_hits[key] for key in native_hits}}
            artifact["native_metrics_parity_detail"] = detail
            artifact["native_metrics_match_feature_rerank"] = all(detail["equality"].values())
            artifact_path.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
        adapter_manifest = (plan_path.parent / "adapter" / plan["model_id"] /
                            f"{plan['category']}.native_adapter_adapter.json")
        if adapter_manifest.is_file():
            adapter_artifact = json.loads(adapter_manifest.read_text(encoding="utf-8"))
            adapter_artifact.pop("parity_v3", None)
            adapter_manifest.write_text(json.dumps(adapter_artifact, sort_keys=True, indent=2) + "\n",
                                        encoding="utf-8")
        metrics_equal = native_metrics == adapter_metrics
        scores_close = bool(np.allclose(source_scores, adapter_scores, rtol=1e-5, atol=1e-5))
        ranks_equal = bool(np.array_equal(source_ranks, adapter_ranks))
        report = {
            "standard": STANDARD, "model_id": plan["model_id"], "category": plan["category"],
            "plan_sha256": plan_sha, "native_source_id": native_role["source_id"],
            "adapter_source_id": plan["runs"]["native_adapter_adapter"]["source_id"],
            "native_capture": str(native_path), "adapter_capture": str(adapter_path),
            "query_count": len(source_ranks), "gallery_count": int(source_scores.shape[1]),
            "score_mae": float(errors.mean()), "score_max_abs_error": float(errors.max()),
            "score_close_at_1e-5": scores_close,
            "rank_equal_count": int(np.count_nonzero(source_ranks == adapter_ranks)),
            "rank_different_count": int(np.count_nonzero(source_ranks != adapter_ranks)),
            "metrics_equal": metrics_equal, "native_metrics": native_metrics,
            "adapter_metrics": adapter_metrics,
            "status": "PASS" if scores_close and ranks_equal and metrics_equal else "FAIL",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
    return output_path




def publish(output_root: Path, publication_root: Path, model_id: str) -> Path:
    """Publish only when every category has both native and adapter V3 manifests and plans."""
    artifacts: dict[str, dict[str, Any]] = {}
    proof_paths: dict[str, Path] = {}
    plans: dict[str, Path] = {}
    for category in CATEGORIES:
        plan_path = output_root / f"g1_g2.{model_id}.{category}.v3-plan.json"
        plan, _ = _plan(plan_path)
        proof_path = output_root / plan["scoring_proof_path"]
        if not proof_path.is_file():
            raise FileNotFoundError(f"missing scoring proof: {proof_path}")
        proof_paths[proof_path.name] = proof_path
        plans[plan_path.name] = plan_path
        for side, role_name in (("native", "native_adapter_native"),
                                ("adapter", "native_adapter_adapter")):
            source = output_root / side / model_id / f"{category}.{role_name}.json"
            if not source.is_file():
                raise FileNotFoundError(f"missing complete-role {side} artifact: {source}")
            artifact = json.loads(source.read_text(encoding="utf-8"))
            if (artifact.get("standard") != STANDARD or artifact.get("model_id") != model_id or
                    artifact.get("category") != category or artifact.get("plan_sha256") != _sha(plan_path)):
                raise ValueError(f"invalid {side} artifact: {source}")
            artifacts[f"g1_g2.{model_id}.{category}.{side}.json"] = artifact
    generation = f"{model_id}-v3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    staging = output_root / "publication-staging" / generation
    staging.mkdir(parents=True, exist_ok=False)
    for name, path in plans.items():
        shutil.copy2(path, staging / name)
    for name, artifact in artifacts.items():
        (staging / name).write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return _t4().publish_v3_generation(publication_root, generation, artifacts,
                                       plan_paths=plans, proof_paths=proof_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "bind-native", "bind-tme-family", "adapter", "compare", "publish"), required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--family-plan", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--role")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--output")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--sample-queries", type=Path)
    parser.add_argument("--publication-root", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "native":
        if not args.plan or not args.capture or not args.role:
            parser.error("native phase needs --plan, --capture, and --role")
        print(convert_native(args.plan, args.capture, args.output_root, args.role, args.metrics_json))
    elif args.phase == "bind-native":
        if not args.plan or not args.capture or not args.role:
            parser.error("bind-native phase needs --plan, --capture, and --role")
        print(bind_native(args.plan, args.capture, args.role))
    elif args.phase == "bind-tme-family":
        if not args.family_plan or not args.capture or not args.role:
            parser.error("bind-tme-family needs --family-plan, --capture, and --role")
        print(json.dumps([str(path) for path in bind_tme_family(args.family_plan, args.capture, args.role)]))
    elif args.phase == "adapter":
        if not args.plan or not args.role or not args.output:
            parser.error("adapter phase needs --plan, --role, and --output")
        print(run_adapter(args.plan, args.role, Path(args.output), args.sample_queries))
    elif args.phase == "compare":
        if not args.plan or not args.native or not args.adapter or not args.output:
            parser.error("compare phase needs --plan, --native, --adapter, and --output")
        report_path = compare_pair(args.plan, args.native, args.adapter, Path(args.output))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(report_path)
        if report.get("status") != "PASS":
            return 2
    else:
        if not args.model or not args.publication_root:
            parser.error("publish phase needs --model and --publication-root")
        print(publish(args.output_root, args.publication_root, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
