"""PTHA+MTST preflight and explicitly labelled CIRR-to-FashionIQ transfer replay."""
from __future__ import annotations

import importlib
import json
import os
import tempfile
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import isolated_cache_root, source_pin_status, sha256_file

MODEL = model_by_id("ptha_mtst")
RAW_INPUTS = tuple(
    Path(path)
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)
HF_REPO = "yinanzhou1/mtst_ptha"
CHECKPOINT_IDS = tuple(checkpoint["checkpoint_id"] for checkpoint in MODEL["checkpoint_variants"])
DEFAULT_CHECKPOINT_ID = CHECKPOINT_IDS[0]
PUBLISHED_CIRR_WEIGHT = next(
    checkpoint["filename"] for checkpoint in MODEL["checkpoint_variants"]
    if checkpoint.get("transfer_condition") == "cross_dataset_transfer"
)
TRANSFER_LABEL = "PTHA+MTST | CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"
V3_STANDARD = "parity-absolute-relative/v3"
V3_TOLERANCES = {
    "fp32_forward": {"atol": 1e-5, "rtol": 1e-4},
    "batch": {"atol": 1e-4, "rtol": 1e-3},
    "score": {"atol": 1e-6, "rtol": 0.0},
    "tie": {"atol": 1e-6, "rtol": 0.0},
}
V3_BATCHES = {"native": (16, 64), "adapter": (16, 64)}
V3_ROLE_DEFINITIONS = {
    "native_repeat_1": ("native", 16, 64), "native_repeat_2": ("native", 16, 64),
    "adapter_repeat_1": ("adapter", 16, 64), "adapter_repeat_2": ("adapter", 16, 64),
    "native_adapter_native": ("native", 16, 64), "native_adapter_adapter": ("adapter", 16, 64),
    "native_batch_1": ("native", 1, 1), "native_batch_n": ("native", 16, 64),
    "adapter_batch_1": ("adapter", 1, 1), "adapter_batch_n": ("adapter", 16, 64),
}
V3_ROLES = {role: side for role, (side, _query_batch, _gallery_batch)
            in V3_ROLE_DEFINITIONS.items()}
V3_CONTROLS = {
    "native_repeat": ("native_repeat_1", "native_repeat_2", "native"),
    "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2", "adapter"),
    "native_adapter": ("native_adapter_native", "native_adapter_adapter", "native"),
    "native_batch_1_vs_n": ("native_batch_1", "native_batch_n", "native"),
    "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n", "adapter"),
}


def _shared_v3_helper():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / (
        "docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py")
    spec = importlib.util.spec_from_file_location("cir_t4_ptha_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared V3 evidence contract: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def transfer_v3_plan(identity: dict[str, Any], query_ids: list[str], gallery_ids: list[str],
                     target_ids: list[str]) -> dict[str, Any]:
    from datetime import datetime, timezone

    if identity["category"] not in CATEGORIES:
        raise ValueError(f"PTHA V3 category is not registered FashionIQ scope: {identity['category']}")
    if (len(query_ids) != identity["query_count"] or len(target_ids) != len(query_ids)
            or len(gallery_ids) != identity["gallery_count"]):
        raise ValueError("PTHA V3 transfer plan population does not match identity")
    labels = {"query_targets": [[query_id, target_id]
                                for query_id, target_id in zip(query_ids, target_ids, strict=True)],
              "reference_exclusions": [[query_id, []] for query_id in query_ids]}
    if _canonical_sha256(labels) != identity["reference_exclusions_sha256"]:
        raise ValueError("PTHA V3 transfer labels differ from identity")
    partition = _ordered_ids_sha256(query_ids)
    gallery_partition = _ordered_ids_sha256(gallery_ids)
    runs: dict[str, dict[str, Any]] = {}
    for role, (side, query_batch, gallery_batch) in V3_ROLE_DEFINITIONS.items():
        now = datetime.now(timezone.utc).isoformat()
        runs[role] = {
            "run_id": f"ptha_mtst:{identity['category']}:{role}:{uuid.uuid4().hex}",
            "source_id": f"{side}:ptha_mtst:{uuid.uuid4().hex}",
            "identity": dict(identity),
            "execution": {"partition_id": f"ptha_mtst-{identity['category']}-full",
                          "query_batch_size": query_batch, "gallery_batch_size": gallery_batch,
                          "query_partition_sha256": partition,
                          "gallery_partition_sha256": gallery_partition,
                          "predeclared_at": now},
        }
    controls = {}
    for name, (reference, actual, side) in V3_CONTROLS.items():
        scopes = {key: {"coverage": "full", "population_count": count,
                        "component_shape": shape}
                  for key, count, shape in (("query", len(query_ids), [len(query_ids), 256]),
                                            ("gallery", len(gallery_ids), [len(gallery_ids), 8192]))}
        controls[name] = {"reference_run_id": runs[reference]["run_id"],
                          "actual_run_id": runs[actual]["run_id"], "reference_side": side,
                          "dimension": "batch_parity" if "batch" in name else "forward_parity",
                          "dtype": "float32",
                          "tolerance": V3_TOLERANCES["batch" if "batch" in name else "fp32_forward"],
                          "scopes": scopes}
    return {"standard": V3_STANDARD, "model_id": "ptha_mtst", "category": identity["category"],
            "identity": dict(identity), "runs": runs, "controls": controls,
            "scoring_labels": labels,
            "score_order_conventions": {"native": "fp32_one_minus_score_ascending",
                                         "adapter": "fp32_descending_score"},
            "scoring_proof_path": f"g1_g2.ptha_mtst.{identity['category']}.native.score-ranks.jsonl",
            "metric_cutoffs": [10, 50],
            "supported_batch_policy": {"supported": True, "batch_invariant": False,
                                        "native_query_batch": 16, "native_gallery_batch": 64,
                                        "adapter_query_batch": 16, "adapter_gallery_batch": 64,
                                        "batch_sizes": [1, 16, 64], "control_batch": 1},
            "population_scope": {"coverage": "full", "query_count": len(query_ids),
                                 "gallery_count": len(gallery_ids), "cohort_id": identity["cohort_id"],
                                 "query_ids": query_ids, "gallery_ids": gallery_ids,
                                 "reference_ids": query_ids, "target_ids": target_ids},
            "reference_policy": "FashionIQ native gallery; reference candidates remain eligible"}



def _canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _runtime_fingerprint(interpreter: str) -> dict[str, str]:
    code = ("import json,platform,sys,torch,numpy,torchvision; "
            "available=torch.cuda.is_available(); "
            "capability=torch.cuda.get_device_capability() if available else None; "
            "print(json.dumps({'python':sys.executable,'version':platform.python_version(),"
            "'torch':torch.__version__,'torchvision':torchvision.__version__,"
            "'numpy':numpy.__version__,'cuda':str(torch.version.cuda),"
            "'cuda_available':str(available),'device_capability':str(capability)},sort_keys=True))")
    result = subprocess.run([interpreter, "-c", code], check=True, capture_output=True,
                            text=True, timeout=60)
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        raise ValueError(f"malformed PTHA runtime identity from {interpreter}")
    return value


def _native_checkpoint_model_key(interpreter: str, checkpoint: Path) -> str:
    code = ("import json,torch,sys; "
            "obj=torch.load(sys.argv[1],map_location='cpu',weights_only=True,mmap=True); "
            "keys=[key for key in ('Blip2QformerCirCat','Blip2Qformer4TGIRCESIM') if key in obj]; "
            "print(json.dumps(keys))")
    result = subprocess.run([interpreter, "-c", code, str(checkpoint)], check=True,
                            capture_output=True, text=True, timeout=180)
    keys = json.loads(result.stdout)
    if len(keys) != 1:
        raise ValueError(f"checkpoint must select exactly one native model class; observed {keys}")
    return keys[0]


def build_transfer_v3_identities(source_root: Path, dataset_root: Path,
                                 checkpoint: Path, checkpoint_id: str) -> dict[str, dict[str, Any]]:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.runtime import resolve_model_interpreter

    config = resolve_config()
    native_python = resolve_model_interpreter(MODEL, config)
    if not native_python or not Path(native_python).is_file():
        raise RuntimeError("verified PTHA native interpreter is unavailable")
    adapter_python = _verify_adapter_runtime(checkpoint_id)
    checkpoint_key = _native_checkpoint_model_key(native_python, checkpoint)
    model_name, model_type = {
        "Blip2QformerCirCat": ("blip2_cir_cat", "pretrain"),
        "Blip2Qformer4TGIRCESIM": ("blip2_tgir_ce_sim", "pretrain_mtst"),
    }[checkpoint_key]
    runtime = {"native": _runtime_fingerprint(native_python),
               "adapter": _runtime_fingerprint(adapter_python)}
    runtime_digest = _canonical_sha256(runtime)
    source = source_tree(source_root) / "src"
    source_files = (
        "validate_blip.py", "utils.py", "data_utils.py",
        "lavis/models/blip2_models/blip2_qformer_cir_cat.py",
        "lavis/models/blip2_models/blip2_qformer_tgir_concatembedding_sim.py",
        "lavis/configs/models/blip2/blip2_pretrain.yaml",
        "lavis/configs/models/blip2/blip2_pretrain_mtst.yaml",
        "lavis/processors/blip_processors.py",
    )
    source_hashes = {name: sha256_file(source / name) for name in source_files}
    source_hashes["selected_native_model"] = _canonical_sha256({
        "checkpoint_key": checkpoint_key, "model_name": model_name, "model_type": model_type,
        "default_config_path": f"lavis/configs/models/blip2/blip2_{model_type}.yaml",
        "default_config_sha256": source_hashes[f"lavis/configs/models/blip2/blip2_{model_type}.yaml"],
    })
    adapter_file = Path(__file__).resolve().parents[1] / "composition" / "model_ptha_mtst.py"
    source_hashes[str(adapter_file.relative_to(Path(__file__).resolve().parents[2]))] = sha256_file(adapter_file)
    checkpoint_sha = sha256_file(checkpoint)
    fashioniq_root = (dataset_root / "fashionIQ_dataset"
                      if (dataset_root / "fashionIQ_dataset").is_dir() else dataset_root)
    images_root = fashioniq_root / "images"
    result: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        annotation = fashioniq_root / "captions" / f"cap.{category}.val.json"
        gallery_file = fashioniq_root / "image_splits" / f"split.{category}.val.json"
        pairs = json.loads(annotation.read_text(encoding="utf-8"))
        split_gallery = json.loads(gallery_file.read_text(encoding="utf-8"))
        if not isinstance(pairs, list) or not isinstance(split_gallery, list):
            raise ValueError(f"{category}: malformed FashionIQ annotations or gallery split")
        result[category] = _transfer_v3_category_identity(
            category, pairs, split_gallery, images_root,
            annotation_sha256=sha256_file(annotation), gallery_sha256=sha256_file(gallery_file),
            checkpoint_sha256=checkpoint_sha, checkpoint_id=checkpoint_id,
            checkpoint_bytes=checkpoint.stat().st_size, source_hashes=source_hashes,
            runtime_digest=runtime_digest, checkpoint_model_key=checkpoint_key,
            model_name=model_name, model_type=model_type,
        )
    return result


def _transfer_v3_category_identity(category: str, pairs: list[dict[str, Any]],
                                    split_gallery: list[str], images_root: Path, *,
                                    annotation_sha256: str, gallery_sha256: str,
                                    checkpoint_sha256: str, checkpoint_id: str,
                                    checkpoint_bytes: int, source_hashes: dict[str, str],
                                    runtime_digest: str, checkpoint_model_key: str,
                                    model_name: str, model_type: str) -> dict[str, Any]:
    source_query_ids = [str(row["candidate"].replace("jpg", "png")) for row in pairs]
    query_ids = [f"{query_id}#row-{index}" for index, query_id in enumerate(source_query_ids)]
    target_ids = [str(row["target"].replace("jpg", "png")) for row in pairs]
    gallery_ids = [str(image_id) for image_id in split_gallery
                   if (images_root / f"{image_id}.png").is_file()]
    if not query_ids or not gallery_ids or len(query_ids) != len(target_ids):
        raise ValueError(f"{category}: empty or inconsistent FashionIQ transfer population")
    labels = {"query_targets": [[query_id, target_id]
                                for query_id, target_id in zip(query_ids, target_ids, strict=True)],
              "reference_exclusions": [[query_id, []] for query_id in query_ids]}
    configuration = {"source_commit": MODEL["upstream_commit_sha"],
                     "source_files": source_hashes, "annotation_sha256": annotation_sha256,
                     "gallery_split_sha256": gallery_sha256, "checkpoint_id": checkpoint_id,
                     "checkpoint_bytes": checkpoint_bytes, "checkpoint_sha256": checkpoint_sha256,
                     "checkpoint_model_key": checkpoint_model_key,
                     "model_name": model_name, "model_type": model_type,
                     "adapter_scoring": "CompositionAdapter.score cosine FP32; max target-token cosine",
                     "native_protocol": MODEL["native_protocol"],
                     "resource_intervention": "G64/Q16 unchanged; replay-only CPU memmap placement"}
    return {
        "checkpoint": checkpoint_sha256,
        "configuration": _canonical_sha256(configuration),
        "environment": runtime_digest,
        "protocol": "FashionIQ original val relative/classic; native maybe_autocast; full gallery; references eligible",
        "category": category,
        "cohort_id": f"fashioniq-val-full-ptha_mtst-{category}-cirr-transfer-v3",
        "precision": "native maybe_autocast; FP32 captured query/gallery scores; Adapter FP32; TF32 unchanged",
        "seed": 0,
        "query_order_sha256": _ordered_ids_sha256(query_ids),
        "gallery_order_sha256": _ordered_ids_sha256(gallery_ids),
        "reference_exclusions_sha256": _canonical_sha256(labels),
        "query_count": len(query_ids), "gallery_count": len(gallery_ids),
    }


def publish_transfer_v3_plan(path: Path, identity: dict[str, Any], query_ids: list[str],
                              gallery_ids: list[str], target_ids: list[str]) -> str:
    plan = transfer_v3_plan(identity, query_ids, gallery_ids, target_ids)
    return _shared_v3_helper().write_v3_plan(path, plan)

def transfer_v3_role_receipt(plan_path: Path, role: str, stage: str) -> dict[str, Any]:
    try:
        raw_plan = plan_path.read_bytes()
        plan = json.loads(raw_plan)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read PTHA V3 plan {plan_path}: {error}") from error
    plan_digest = __import__("hashlib").sha256(raw_plan).hexdigest()
    if role not in V3_ROLE_DEFINITIONS:
        raise ValueError(f"PTHA V3 role {role!r} has no role-specific capture implementation")
    if V3_ROLES[role] != stage:
        raise ValueError(f"PTHA V3 role {role!r} does not bind capture stage {stage!r}")
    run = plan.get("runs", {}).get(role)
    if not isinstance(run, dict) or run.get("identity") != plan.get("identity"):
        raise ValueError(f"PTHA V3 role {role!r} is not preregistered in this plan")
    execution = run.get("execution")
    expected = {"query_batch_size": V3_ROLE_DEFINITIONS[role][1],
                "gallery_batch_size": V3_ROLE_DEFINITIONS[role][2]}
    if not isinstance(execution, dict) or any(execution.get(key) != value
                                              for key, value in expected.items()):
        raise ValueError(f"PTHA V3 role {role!r} plan batch settings mismatch: {execution}")
    return {"standard": V3_STANDARD, "plan_sha256": plan_digest, "role": role,
            "stage": stage, "identity": plan["identity"], "execution": execution,
            "score_order_conventions": plan["score_order_conventions"]}

def _ordered_ids_sha256(ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _v3_capture_receipt(plan_path: Path | None, role: str | None, stage: str,
                        category: str, query_ids: list[str], gallery_ids: list[str],
                        target_ids: list[str] | None = None,
                        reference_ids: list[str] | None = None) -> dict[str, Any] | None:
    if plan_path is None and role is None:
        return None
    if plan_path is None or role is None:
        raise ValueError("PTHA V3 capture requires both --v3-plan and --v3-role")
    receipt = transfer_v3_role_receipt(plan_path, role, stage)
    if receipt["score_order_conventions"].get(stage) != {
            "native": "fp32_one_minus_score_ascending",
            "adapter": "fp32_descending_score"}.get(stage):
        raise ValueError(f"PTHA V3 plan has unsupported {stage} score-order convention")
    identity = receipt["identity"]
    observed = {"category": category, "query_count": len(query_ids), "gallery_count": len(gallery_ids),
                "query_order_sha256": _ordered_ids_sha256(query_ids),
                "gallery_order_sha256": _ordered_ids_sha256(gallery_ids)}
    if target_ids is not None or reference_ids is not None:
        if target_ids is None or reference_ids is None or len(target_ids) != len(query_ids) \
                or len(reference_ids) != len(query_ids):
            raise ValueError(f"PTHA V3 captured labels or references differ for {category}")
        labels = {"query_targets": [[query_id, target_id]
                                    for query_id, target_id in zip(query_ids, target_ids, strict=True)],
                  "reference_exclusions": [[query_id, []] for query_id in query_ids]}
        observed["reference_exclusions_sha256"] = _canonical_sha256(labels)
    if target_ids is not None:
        positions = {image_id: index for index, image_id in enumerate(gallery_ids)}
        try:
            receipt["positive_positions"] = [positions[target_id] for target_id in target_ids]
        except KeyError as error:
            raise ValueError(f"PTHA V3 target is absent from gallery: {error.args[0]}") from error
    if any(identity[key] != value for key, value in observed.items()):
        raise ValueError(f"PTHA V3 plan identity does not match captured {category} inputs: {observed}")
    return receipt


def _score_proof(receipt: dict[str, Any] | None, scores: Any, ranks: Any) -> dict[str, Any] | None:
    if receipt is None:
        return None
    import numpy as np

    role, stage = receipt["role"], receipt["stage"]
    allowed = {"native": {"native_repeat_1", "native_repeat_2", "native_adapter_native",
                          "native_batch_1", "native_batch_n"},
               "adapter": {"adapter_repeat_1", "adapter_repeat_2", "native_adapter_adapter",
                           "adapter_batch_1", "adapter_batch_n"}}
    if role not in allowed[stage]:
        raise ValueError(f"score proof role mismatch: {role!r} is not {stage}")
    scores, ranks = np.asarray(scores), np.asarray(ranks)
    identity = receipt["identity"]
    if scores.shape != (identity["query_count"], identity["gallery_count"]):
        raise ValueError(f"score proof shape does not match {role} identity")
    expected_rank_shape = scores.shape if stage == "native" else (scores.shape[0],)
    if (ranks.shape != expected_rank_shape or scores.dtype != np.float32
            or not np.isfinite(scores).all()):
        raise ValueError(f"score proof arrays are incomplete or non-finite for {role}")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"rank proof must use integer indices for {role}")
    convention = receipt["score_order_conventions"][stage]
    if stage == "native":
        if (np.any(ranks < 0) or np.any(ranks >= scores.shape[1])
                or any(len(set(row.tolist())) != scores.shape[1] for row in ranks)):
            raise ValueError(f"native sorter proof is not a full gallery permutation for {role}")
        keys = np.subtract(np.float32(1.0), scores, dtype=np.float32)
        if any(np.any(keys[row, ranks[row, :-1]] > keys[row, ranks[row, 1:]])
               for row in range(scores.shape[0])):
            raise ValueError(f"native rank proof contradicts FP32 torch.argsort(1 - pred_sim) for {role}")
    else:
        if np.any(ranks < 1) or np.any(ranks > scores.shape[1]):
            raise ValueError(f"Adapter positive-rank proof is invalid for {role}")
        targets = receipt.get("positive_positions")
        if (not isinstance(targets, list) or len(targets) != scores.shape[0]
                or any(isinstance(position, bool) or not isinstance(position, int)
                       or position < 0 or position >= scores.shape[1] for position in targets)):
            raise ValueError(f"Adapter score proof lacks valid captured target positions for {role}")
        expected = np.asarray([
            1 + np.count_nonzero(row > row[position]) + np.count_nonzero(row[:position] == row[position])
            for row, position in zip(scores, targets, strict=True)], dtype=np.int64)
        if not np.array_equal(ranks, expected):
            raise ValueError(f"Adapter ranks contradict actual descending score convention for {role}")
    return {"role": role, "stage": stage,
            "scorer": "native source model.inference" if stage == "native" else "actual CompositionAdapter.score",
            "score_order_convention": receipt["score_order_conventions"][stage],
            "score_shape": list(scores.shape), "score_dtype": str(scores.dtype),
            "rank_semantics": "full_native_gallery_order" if stage == "native" else "positive_rank_1based",
            "rank_shape": list(ranks.shape), "rank_dtype": str(ranks.dtype),
            "scores_sha256": _canonical_sha256(scores.tolist()),
            "ranks_sha256": _canonical_sha256(ranks.tolist())}


def _publish_v3_role_receipt(root: Path, category: str, receipt: dict[str, Any] | None,
                             artifact: Path, score_proof: dict[str, Any] | None = None) -> None:
    if receipt is None:
        return
    role = receipt["role"]
    if (receipt["identity"].get("category") != category
            or V3_ROLES.get(role) != receipt["stage"]):
        raise ValueError("PTHA V3 receipt category/stage does not bind its registered role")
    if score_proof is None or score_proof["role"] != role:
        raise ValueError("PTHA V3 score proof must bind the registered capture role")
    if score_proof.get("score_order_convention") != receipt["score_order_conventions"].get(receipt["stage"]):
        raise ValueError("PTHA V3 score proof convention differs from sealed role plan")
    output = {**receipt, "artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
              "score_proof": score_proof}
    path = root / f"{category}.{receipt['role']}.v3-role.json"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ValueError(f"PTHA V3 role receipt is immutable and already exists: {path}") from error
def validate_transfer_v3_plans(directory: Path, role: str, stage: str,
                               expected_identities: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
    hashes = {}
    for category in CATEGORIES:
        receipt = transfer_v3_role_receipt(directory / f"{category}.json", role, stage)
        if receipt["identity"]["category"] != category:
            raise ValueError(f"PTHA V3 plan category mismatch for {category}")
        if expected_identities is not None and receipt["identity"] != expected_identities[category]:
            raise ValueError(f"PTHA V3 {category} plan identity differs from current inputs/runtime")
        hashes[category] = receipt["plan_sha256"]
    return hashes


def publish_transfer_v3_plans(directory: Path, identities: dict[str, dict[str, Any]],
                              dataset_root: Path) -> dict[str, str]:
    if set(identities) != set(CATEGORIES):
        raise ValueError(f"PTHA V3 preregistration requires categories {list(CATEGORIES)}")
    fashioniq_root = (dataset_root / "fashionIQ_dataset"
                      if (dataset_root / "fashionIQ_dataset").is_dir() else dataset_root)
    images_root = fashioniq_root / "images"
    digests = {}
    for category in CATEGORIES:
        pairs = json.loads((fashioniq_root / "captions" / f"cap.{category}.val.json").read_text())
        gallery = json.loads((fashioniq_root / "image_splits" / f"split.{category}.val.json").read_text())
        source_ids = [str(row["candidate"].replace("jpg", "png")) for row in pairs]
        query_ids = [f"{value}#row-{index}" for index, value in enumerate(source_ids)]
        target_ids = [str(row["target"].replace("jpg", "png")) for row in pairs]
        gallery_ids = [str(value) for value in gallery if (images_root / f"{value}.png").is_file()]
        digests[category] = publish_transfer_v3_plan(
            directory / f"{category}.json", identities[category], query_ids, gallery_ids, target_ids)
    return digests




def _fashioniq_image_path(images_root: Path, image_id: str) -> Path:
    path = images_root / image_id
    if path.is_file():
        return path
    image_path = Path(image_id)
    filename = image_path.with_suffix(".png").name if image_path.suffix else f"{image_id}.png"
    return images_root / filename


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        import numpy as np
        np.savez(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)



def source_tree(source_root: Path) -> Path:
    """The directory that actually holds the upstream tree's `src/`.

    The vendored checkout keeps the repository's own top-level directory inside the checkout, so the
    code lives at `<source_root>/MTST_PTHA/MTST_PTHA/src` while `requirements.txt` and `provenance.json`
    sit at `<source_root>/MTST_PTHA`. Resolving it in ONE place means the evaluator path and the
    import path cannot disagree, which is what they did: both named `<source_root>/MTST_PTHA/src`, a
    directory that does not exist, so every preflight reported the evaluator missing and the import
    would have failed on `data_utils`.
    """
    outer = source_root / "MTST_PTHA"
    nested = outer / "MTST_PTHA"
    return nested if (nested / "src").is_dir() else outer


def evaluator_path(source_root: Path) -> Path:
    return source_tree(source_root) / "src" / "validate_blip.py"


def preflight(
    source_root: Path,
    dataset_root: Path,
    checkpoint_root: Path,
    *,
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
    transfer_cirr_to_fashioniq: bool = False,
) -> list[str]:
    """Return blockers without importing upstream code or mutating inputs."""
    blockers: list[str] = []
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned PTHA+MTST source unavailable: {reason}")

    evaluator = evaluator_path(source_root)
    if not evaluator.is_file():
        blockers.append(f"pinned evaluator missing at {evaluator}")

    if not (dataset_root / "images").is_dir():
        blockers.append(f"FashionIQ raw input missing: {dataset_root / 'images'}")
    blockers.extend(
        f"FashionIQ raw input missing: {dataset_root / path}"
        for path in RAW_INPUTS
        if not (dataset_root / path).is_file()
    )
    from workbench.replay.common import fashioniq_image_binding_blockers
    blockers.extend(fashioniq_image_binding_blockers(dataset_root, "val", "images", ".png"))

    checkpoint = next(
        (item for item in MODEL["checkpoint_variants"] if item["checkpoint_id"] == checkpoint_id),
        None,
    )
    if checkpoint is None:
        blockers.append(
            f"unknown checkpoint id {checkpoint_id!r}; valid checkpoint ids: {', '.join(CHECKPOINT_IDS)}"
        )
        return blockers

    is_cirr_transfer = (
        checkpoint.get("transfer_condition") == "cross_dataset_transfer"
        and checkpoint.get("checkpoint_training_dataset") == "CIRR"
        and checkpoint.get("checkpoint_forwards_dataset") == "FashionIQ"
    )
    if transfer_cirr_to_fashioniq and not is_cirr_transfer:
        blockers.append(
            f"checkpoint {checkpoint_id!r} is not registered as a CIRR-to-FashionIQ transfer"
        )
    elif is_cirr_transfer and transfer_cirr_to_fashioniq:
        path = checkpoint_root / checkpoint["filename"]
        if not path.is_file():
            blockers.append(f"PTHA+MTST CIRR checkpoint missing: {path}")
        elif path.stat().st_size == 0:
            blockers.append(f"PTHA+MTST CIRR checkpoint is an empty placeholder: {path}")
        from workbench.backend.registry import external_asset_blockers
        blockers.extend(external_asset_blockers(MODEL))
    elif is_cirr_transfer:
        blockers.append(
            f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
            f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
            f"('{checkpoint['filename']}'); CIRR weights require explicit transfer opt-in"
        )
    else:
        blockers.append(
            f"PTHA+MTST FashionIQ fine-tuned checkpoint missing: author repository "
            f"https://huggingface.co/{HF_REPO} publishes only CIRR weights "
            f"('{PUBLISHED_CIRR_WEIGHT}'); CIRR weights require explicit transfer opt-in"
        )
    return blockers


@contextmanager
def _fashioniq_dataset_binding(dataset_root: Path) -> Iterator[Any]:
    """Bind MTST's literal dataset path to this requested root for one evaluation."""
    previous_env = os.environ.get("MTST_DATA_ROOT")
    previous_module = sys.modules.get("data_utils")
    previous_base_path = getattr(previous_module, "base_path", None)
    with ExitStack() as stack:
        if dataset_root.name == "fashionIQ_dataset":
            data_root = dataset_root.parent
        else:
            data_root = stack.enter_context(isolated_cache_root(dataset_root, directory_names=()))
            (data_root / "fashionIQ_dataset").symlink_to(
                dataset_root.resolve(), target_is_directory=True
            )
        os.environ["MTST_DATA_ROOT"] = str(data_root)
        try:
            data_utils = importlib.import_module("data_utils")
            restore_base_path = (
                previous_base_path if data_utils is previous_module
                else Path(previous_env) if previous_env else Path("path_to_data_dir")
            )
            data_utils.base_path = data_root
            try:
                yield data_utils
            finally:
                data_utils.base_path = restore_base_path
        finally:
            if previous_env is None:
                os.environ.pop("MTST_DATA_ROOT", None)
            else:
                os.environ["MTST_DATA_ROOT"] = previous_env


#: Which side of the boundary owns the `vision_proj` step, decided by the SELECTED AND LOADED class.
#:
#: Both supported classes call `extract_target_features(image, mode="mean")` and both keep the
#: projection commented out, so both hand the boundary raw Q-Former states `[N, 32, 768]`. They
#: differ in where the projection belongs:
#:
#: ``cir_cat``  -- `blip2_qformer_cir_cat.py` defines `extract_target_features` TWICE (lines 453 and
#:   483), Python keeps the second one, and its `inference()` (line 287) matmuls the 256-wide query
#:   side against the target tensor it is handed. It never calls `vision_proj` on targets itself, so
#:   the boundary MUST: the width cannot otherwise meet, and the checkpoint carries the trained
#:   parameters (`vision_proj` 256x768 = 196 864).
#:
#: ``tgir_ce_sim`` -- `blip2_qformer_tgir_concatembedding_sim.py:429` runs
#:   `F.normalize(self.vision_proj(target_embeds), dim=-1)` INSIDE `inference()`, and its
#:   `extract_target_features` (line 501) also returns the raw states. The boundary must therefore
#:   pass raw states through and leave the projection to the model.
#:
#: Choosing this by class rather than by tensor width or filename is what the earlier revision got
#: wrong: applying the projection unconditionally to both classes double-projected the TGIR branch.
#: A shape check would not have caught that either, because projecting raw tgir states first yields a
#: tensor that this same `inference` then projects again -- a wrong-but-shaped-correct result.
TARGET_FEATURE_CONTRACTS: dict[str, str] = {
    "Blip2QformerCirCat": "project_at_boundary",
    "Blip2Qformer4TGIRCESIM": "project_inside_inference",
}



@contextmanager
def _bounded_native_gallery_store(model: Any, utils_module: Any, count: int,
                                 directory: Path) -> Iterator[dict[str, Any]]:
    """Run pinned extraction unchanged while its actual output batches live in NPY maps."""
    import numpy as np

    original_extract = model.extract_target_features
    original_torch = utils_module["torch"] if isinstance(utils_module, dict) else utils_module.torch
    stores: list[Any] = []
    views: list[list[Any]] = [[], []]
    actual: dict[str, Any] = {"batch_shapes": [], "batch_dtypes": [], "batch_strides": [],
                              "batch_count": 0, "feature_shapes": [], "feature_dtypes": [],
                              "feature_nbytes": []}
    directory.mkdir(parents=True, exist_ok=False)

    def set_torch(value: Any) -> None:
        if isinstance(utils_module, dict):
            utils_module["torch"] = value
        else:
            utils_module.torch = value

    def extract_target_features(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        outputs = original_extract(*args, **kwargs)
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError("native extract_target_features did not return expected feature pair")
        batch = int(outputs[0].shape[0])
        if outputs[1].shape[0] != batch or actual["batch_count"] + batch > count:
            raise ValueError("native gallery feature batch size/order exceeds declared dataset length")
        if any(tensor.device.type != "cpu" or tensor.dtype != original_torch.float32
               for tensor in outputs):
            raise ValueError("native gallery storage requires actual CPU FP32 feature tensors")
        views_for_batch: list[Any] = []
        for index, tensor in enumerate(outputs):
            if index == len(stores):
                stores.append(np.lib.format.open_memmap(
                    directory / f"gallery-{index}.npy", mode="w+", dtype=tensor.numpy().dtype,
                    shape=(count, *tensor.shape[1:])))
            store = stores[index]
            if tensor.shape[1:] != store.shape[1:] or tensor.numpy().dtype != store.dtype:
                raise ValueError("native gallery output shape or dtype changed between batches")
            start = actual["batch_count"]
            store[start:start + batch] = tensor.numpy()
            store.flush()
            import mmap
            store._mmap.madvise(mmap.MADV_DONTNEED)
            view = original_torch.from_numpy(store[start:start + batch])
            views[index].append(view)
            views_for_batch.append(view)
        actual["batch_shapes"].append([list(value.shape) for value in views_for_batch])
        actual["batch_strides"].append([list(value.stride()) for value in views_for_batch])
        actual["batch_dtypes"].append([str(value.dtype) for value in views_for_batch])
        if not actual["feature_shapes"]:
            actual["feature_shapes"] = [list(store.shape[1:]) for store in stores]
            actual["feature_dtypes"] = [str(original_torch.from_numpy(store).dtype) for store in stores]
            actual["feature_nbytes"] = [int(store.nbytes) for store in stores]
        actual["batch_count"] += batch
        return (views_for_batch[0], views_for_batch[1])

    class TorchGalleryStore:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_torch, name)

        def vstack(self, tensors: list[Any]) -> Any:
            if not stores or len(views) != 2 or actual["batch_count"] != count:
                raise ValueError("native gallery storage did not observe complete two-feature coverage")
            index = actual.setdefault("stack_count", 0)
            expected_views = views[index] if index < 2 else []
            if index >= 2 or len(tensors) != len(expected_views) or any(
                    tensor is not expected for tensor, expected in zip(tensors, expected_views)):
                raise ValueError("unexpected native vstack call or batch order; refusing fallback allocation")
            start = 0
            for tensor in tensors:
                if (tensor.device.type != "cpu" or not tensor.is_contiguous()
                        or tensor.data_ptr() != stores[index].ctypes.data + start * stores[index].strides[0]):
                    raise ValueError("native gallery batch views do not tile memmap in order")
                start += len(tensor)
            if start != count or not stores[index].flags.c_contiguous:
                raise ValueError("native gallery views do not cover contiguous full memmap")
            actual["stack_count"] += 1
            return original_torch.from_numpy(stores[index][:actual["batch_count"]])
    def cleanup() -> None:
        views[0].clear()
        views[1].clear()
        for store in stores:
            store.flush()
            store._mmap.close()
        stores.clear()
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()

    actual["cleanup"] = cleanup

    setattr(model, "extract_target_features", extract_target_features)
    set_torch(TorchGalleryStore())
    try:
        yield actual
        if actual.get("stack_count") != 2:
            raise ValueError("native extractor did not consume both verified gallery feature stacks")
        for store in stores:
            store.flush()
    finally:
        setattr(model, "extract_target_features", original_extract)
        set_torch(original_torch)



def _target_feature_contract(model: Any, index_features: Any, torch: Any,
                             contract: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Give `inference()` the target representation THIS loaded class expects, exactly once.

    `index_features` is the native `extract_index_blip_features` output: a 2-tuple whose members are
    the gallery target features and the frozen vision embeddings. Both members are preserved
    unchanged in type and order, because `compute_fiq_val_metrics` reaches for `index_features[0]`
    to score against and the reference side is taken from `index_features[-1]`.

    Raises instead of guessing when the contract is not one this workbench declares, or when the
    declared contract does not match the loaded class's geometry.
    """
    contract = contract if contract is not None else TARGET_FEATURE_CONTRACTS.get(type(model).__name__)
    class_name = type(model).__name__
    if contract is None:
        raise ValueError(
            f"PTHA+MTST target-feature contract is undeclared for loaded class {class_name!r}; "
            f"declared classes are {sorted(TARGET_FEATURE_CONTRACTS)}. Refusing to guess whether "
            f"this class projects target embeddings inside `inference()`, because guessing wrong "
            f"either double-projects or fails to project at all.")
    if contract not in TARGET_FEATURE_CONTRACTS.values():
        raise ValueError(
            f"unknown PTHA+MTST target-feature contract {contract!r} for class {class_name!r}; "
            f"expected one of {sorted(set(TARGET_FEATURE_CONTRACTS.values()))}")

    if not (isinstance(index_features, tuple) and len(index_features) == 2):
        raise ValueError(
            "PTHA+MTST index features are not the native (target_features, raw_vision_embeds) pair: "
            f"got {type(index_features).__name__} of length "
            f"{len(index_features) if hasattr(index_features, '__len__') else 'n/a'}")
    raw_targets, frozen_vision = index_features
    report: dict[str, Any] = {
        "contract": contract,
        "loaded_class": class_name,
        "extracted_shape": list(raw_targets.shape),
        "new_parameters_introduced": 0,
        "frozen_vision_preserved": True,
    }

    if contract == "project_inside_inference":
        # The model projects these itself at `..._concatembedding_sim.py:429`. Any projection here
        # would be applied a second time inside `inference()`, which no shape check would flag.
        report.update(
            action="passed_through_unprojected",
            reason="this class applies vision_proj to target embeddings inside inference(); "
                   "projecting at the boundary would double-project",
            output_shape=list(raw_targets.shape),
        )
        return (raw_targets, frozen_vision), report

    hidden = getattr(model.vision_proj, "in_features", None)
    embed = getattr(model.vision_proj, "out_features", None)
    # Width is validated BEFORE projecting: otherwise a changed extraction path surfaces as an
    # opaque `mat1 and mat2 shapes cannot be multiplied` from inside `nn.Linear` instead of as the
    # declared-contract diagnostic below.
    if hidden is not None and raw_targets.shape[-1] != hidden:
        raise ValueError(
            f"PTHA+MTST contract {contract!r} for class {class_name!r} expects unprojected targets of "
            f"width {hidden}, but the extracted features are width {raw_targets.shape[-1]}. "
            f"Either the extraction path changed or the wrong contract was dispatched.")
    projected = model.inference_target(raw_targets)
    if embed is not None and projected.shape[-1] != embed:
        raise ValueError(
            f"PTHA+MTST boundary projection produced width {projected.shape[-1]}, but this class's "
            f"inference() consumes width {embed}")
    report.update(
        action="projected_at_boundary_via_inference_target",
        reason="effective extract_target_features returns unprojected Q-Former states while "
               "inference() does not project targets itself",
        projection_parameters_reused=sum(p.numel() for p in model.vision_proj.parameters()),
        projection_in_features=hidden,
        projection_out_features=embed,
        output_shape=list(projected.shape),
    )
    # Equivalence is checked against the author's own operation with the SAME weights, on the SAME
    # device, so this stays meaningful on CPU-only test hosts too.
    device = model.vision_proj.weight.device
    with torch.no_grad():
        direct = torch.nn.functional.normalize(model.vision_proj(raw_targets.to(device)), dim=-1)
    report["equivalence_vs_direct_projection_max_abs_diff"] = float(
        (direct.to(projected.device) - projected).abs().max())
    return (projected, frozen_vision), report


def _capture_stem(category: str, role: str | None, stage: str) -> str:
    return f"{category}.{role}.{stage}" if role is not None else f"{category}.{stage}"


def _native_role_for_adapter(role: str | None) -> str | None:
    if role is None:
        return None
    return {"native_adapter_adapter": "native_adapter_native",
            "adapter_repeat_1": "native_repeat_1",
            "adapter_repeat_2": "native_repeat_2",
            "adapter_batch_1": "native_batch_1",
            "adapter_batch_n": "native_batch_n"}[role]
def _state_dict_report(model_state: dict[str, Any], checkpoint_state: dict[str, Any]) -> dict[str, Any]:
    expected = set(model_state)
    supplied = set(checkpoint_state)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    mismatched_shapes = sorted(
        (key, tuple(checkpoint_state[key].shape), tuple(model_state[key].shape))
        for key in expected & supplied
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": [
            {"key": key, "checkpoint": source, "model": target}
            for key, source, target in mismatched_shapes
        ],
        "strict_false_would_hide": {"missing_keys": missing, "unexpected_keys": unexpected},
        "strict_false_shape_errors": [key for key, _, _ in mismatched_shapes],
    }




def _expected_batch_lengths(population: int, batch_size: int) -> list[int]:
    if population < 0 or batch_size < 1:
        raise ValueError("batch population must be nonnegative and batch size positive")
    return [min(batch_size, population - first) for first in range(0, population, batch_size)]


def _validate_native_batch_observations(observed: dict[str, list[int]], query_count: int,
                                        gallery_count: int, query_batch: int,
                                        gallery_batch: int) -> None:
    expected = {"query": _expected_batch_lengths(query_count, query_batch),
                "gallery": _expected_batch_lengths(gallery_count, gallery_batch)}
    if observed != expected:
        raise ValueError(f"native observed batch partitions differ from sealed role: {observed} != {expected}")


def _validate_adapter_batch_receipt(receipt: dict[str, Any] | None, population: int,
                                    batch_size: int, phase: str) -> None:
    expected = {"batch_size": batch_size,
                "batch_lengths": _expected_batch_lengths(population, batch_size)}
    if receipt != expected:
        raise ValueError(f"Adapter {phase} batch receipt mismatch: {receipt}")


def _validate_adapter_score_batches(observed: list[int], query_count: int,
                                    query_batch: int) -> None:
    expected = _expected_batch_lengths(query_count, query_batch)
    if observed != expected:
        raise ValueError(f"Adapter score query chunks differ from sealed role: {observed} != {expected}")
def _validate_native_capture_metadata(category: str, metadata: dict[str, Any]) -> None:
    required = ("query_sha256", "gallery_sha256", "source")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"{category}: native capture receipt missing required fields: {missing}")
    if not isinstance(metadata["query_sha256"], list) or not isinstance(metadata["gallery_sha256"], list):
        raise ValueError(f"{category}: native capture image hashes must be lists")
    if not isinstance(metadata["source"], dict) or not metadata["source"].get("commit"):
        raise ValueError(f"{category}: native capture source receipt lacks commit")

def _publish_native_capture(root: Path, category: str, arrays: dict[str, Any],
                            metadata: dict[str, Any], role: str | None = None) -> None:
    _validate_native_capture_metadata(category, metadata)
    stem = _capture_stem(category, role, "native")
    if role is not None and any((root / f"{stem}{suffix}").exists() for suffix in (".npz", ".json")):
        raise FileExistsError(f"PTHA V3 role capture is immutable and already exists: {stem}")
    _atomic_savez(root / f"{stem}.npz", **arrays)
    _atomic_json(root / f"{stem}.json", metadata)


def _load_native_capture(root: Path, category: str, role: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    stem = _capture_stem(category, role, "native")
    metadata = json.loads((root / f"{stem}.json").read_text(encoding="utf-8"))
    _validate_native_capture_metadata(category, metadata)
    with np.load(root / f"{stem}.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def _publish_paired_capture(root: Path, category: str, arrays: dict[str, Any],
                            metadata: dict[str, Any], role: str | None = None) -> Path:
    stem = _capture_stem(category, role, "paired")
    metadata["artifact"] = f"{stem}.npz"
    artifact = root / f"{stem}.npz"
    if role is not None and any((root / f"{stem}{suffix}").exists() for suffix in (".npz", ".json")):
        raise FileExistsError(f"PTHA V3 role capture is immutable and already exists: {stem}")
    _atomic_savez(artifact, **arrays)
    _atomic_json(root / f"{stem}.json", metadata)
    return artifact

def _adapter_rank_and_topk(adapter_scores: Any, positive_positions: list[list[int]],
                           native_rank: Any, rank_scores: Any) -> tuple[Any, Any, Any, dict[str, int], bool]:
    import numpy as np
    from workbench.composition.topk import topk_from_scores

    adapter_ranks = rank_scores(adapter_scores, positive_positions)
    adapter_positive_ranks = np.asarray(adapter_ranks, dtype=np.int64)
    adapter_top_rows = [topk_from_scores(row, 50) for row in adapter_scores]
    adapter_top = np.asarray([[position for position, _score in row]
                              for row in adapter_top_rows], dtype=np.int64)
    native_positive_ranks = np.asarray([
        np.flatnonzero(row == position)[0] + 1
        for row, positions in zip(native_rank, positive_positions)
        for position in positions[:1]], dtype=np.int64)
    hits = {str(k): int(sum(rank is not None and rank <= k
                            for rank in adapter_ranks)) for k in (10, 50)}
    return (adapter_positive_ranks, adapter_top, native_positive_ranks, hits,
            bool(np.array_equal(native_positive_ranks, adapter_positive_ranks)))


def _evaluate_adapter_capture(capture_directory: Path, dataset_root: Path,
                              checkpoint_id: str, v3_plan_directory: Path | None = None,
                              v3_role: str | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_ptha_mtst import Adapter

    root = capture_directory.resolve()
    image_root = (dataset_root / "fashionIQ_dataset" / "images"
                  if (dataset_root / "fashionIQ_dataset" / "images").is_dir()
                  else dataset_root / "images")
    adapter = Adapter(resolve_config(), checkpoint_id=checkpoint_id)
    adapter.work_root = root / "adapter-work"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    comparisons: dict[str, dict[str, Any]] = {}
    try:
        native_role = _native_role_for_adapter(v3_role)
        for category in CATEGORIES:
            native_meta, native = _load_native_capture(root, category, native_role)
            query_ids, gallery_ids = native["query_ids"].tolist(), native["gallery_ids"].tolist()
            v3_query_ids = [f"{query_id}#row-{index}"
                            for index, query_id in enumerate(query_ids)]
            v3_receipt = _v3_capture_receipt(
                v3_plan_directory / f"{category}.json" if v3_plan_directory else None,
                v3_role, "adapter", category, v3_query_ids, gallery_ids,
                native["target_ids"].tolist(), v3_query_ids)
            if v3_receipt is not None:
                expected_native_role = _native_role_for_adapter(v3_role)
                native_batch_receipt = native_meta.get("native_batch_receipt", {})
                if (native_batch_receipt.get("role") != expected_native_role
                        or native_batch_receipt.get("plan_sha256") != v3_receipt["plan_sha256"]):
                    raise ValueError(f"{category}: native capture is not bound to paired role {expected_native_role}")
            query_paths = [_fashioniq_image_path(image_root, name) for name in query_ids]
            gallery_paths = [_fashioniq_image_path(image_root, name) for name in gallery_ids]
            if [sha256_file(path) for path in query_paths] != native_meta["query_sha256"]:
                raise ValueError(f"{category}: query image inputs changed after native capture")
            if [sha256_file(path) for path in gallery_paths] != native_meta["gallery_sha256"]:
                raise ValueError(f"{category}: gallery image inputs changed after native capture")
            query_paths = [str(path) for path in query_paths]
            gallery_paths = [str(path) for path in gallery_paths]
            execution = v3_receipt["execution"] if v3_receipt is not None else {
                "query_batch_size": 16, "gallery_batch_size": 64}
            query_batch = execution["query_batch_size"]
            gallery_batch = execution["gallery_batch_size"]
            adapter.batch_size = query_batch
            query_features = adapter.encode_queries(query_paths, native["query_texts"].tolist(),
                                                    [None] * len(query_paths))
            query_receipt = adapter._last_batch_receipt
            _validate_adapter_batch_receipt(query_receipt, len(query_paths), query_batch, "query")
            adapter.batch_size = gallery_batch
            target_features = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
            gallery_receipt = adapter._last_batch_receipt
            _validate_adapter_batch_receipt(gallery_receipt, len(gallery_paths), gallery_batch, "gallery")
            native_scores = native["native_scores"]
            adapter_scores = np.empty_like(native_scores, dtype=np.float32)
            score_batches = []
            for first in range(0, len(query_features), query_batch):
                stop = min(first + query_batch, len(query_features))
                adapter_scores[first:stop] = adapter.score(query_features[first:stop], target_features)
                score_batches.append(stop - first)
            _validate_adapter_score_batches(score_batches, len(query_features), query_batch)
            native_rank = native["native_ranks"]
            gallery_index = {name: index for index, name in enumerate(gallery_ids)}
            positive_positions = [[gallery_index[name]] for name in native["target_ids"].tolist()]
            adapter_rank, adapter_top, native_positive_ranks, adapter_hits, rank_parity = (
                _adapter_rank_and_topk(adapter_scores, positive_positions, native_rank,
                                       adapter._ranks_from_scores))
            score_max_abs_error = float(np.max(np.abs(native_scores - adapter_scores)))
            score_parity = bool(np.allclose(native_scores, adapter_scores, rtol=1e-4, atol=1e-5))
            adapter_score_proof = _score_proof(v3_receipt, adapter_scores, adapter_rank)
            metric_detail = {
                "cutoffs": [10, 50],
                "native": {"hits": {f"r{k}": native_meta["native_hits"][str(k)] for k in (10, 50)},
                           "query_count": len(query_ids)},
                "adapter": {"hits": {f"r{k}": adapter_hits[str(k)] for k in (10, 50)},
                            "query_count": len(query_ids)},
                "eligible_query_set_identical": True,
                "eligible_query_set_evidence": (
                    "Both scores use native-captured ordered query/gallery IDs; reference gallery entries remain eligible."),
                "equality": {f"r{k}": native_meta["native_hits"][str(k)] == adapter_hits[str(k)]
                             for k in (10, 50)},
            }
            metric_equal = all(metric_detail["equality"].values())
            adapter_batch_receipt = {"role": v3_role,
                                     "plan_sha256": v3_receipt["plan_sha256"] if v3_receipt else None,
                                     "query": query_receipt, "gallery": gallery_receipt,
                                     "score_query_lengths": score_batches,
                                     "score_query_batch_size": query_batch}
            comparison = {
                "query_count": len(query_ids), "gallery_count": len(gallery_ids),
                "native_batch_sizes": native_meta["native_batch_sizes"],
                "native_num_workers": native_meta["num_workers"],
                "adapter_batch_receipts": adapter_batch_receipt,
                "denominator": len(query_ids), "native_metric": native_meta["native_metric"],
                "adapter_metric": {str(k): {"hits": adapter_hits[str(k)],
                                             "denominator": len(query_ids)} for k in (10, 50)},
                "native_metrics_parity_detail": metric_detail,
                "native_metrics_match_feature_rerank": metric_equal,
                "score_max_abs_error": score_max_abs_error, "score_parity": score_parity,
                "positive_rank_equal": rank_parity, "native_sort": native_meta["native_sort"],
                "native_source_batch_receipt": native_meta.get("native_batch_receipt"),
                "adapter_topk_policy": "actual topk_from_scores over Adapter.score rows; no derived full permutation",
                "reference_eligibility": "all gallery entries; native source does not mask reference",
            }
            paired_arrays = dict(
                query_ids=native["query_ids"], gallery_ids=native["gallery_ids"],
                target_ids=native["target_ids"], reference_ids=native["reference_ids"],
                query_texts=native["query_texts"], processed_texts=native["processed_texts"],
                native_g1_pixels=native["native_g1_pixels"],
                native_g1_pixel_gallery_id=native["native_g1_pixel_gallery_id"],
                native_g1_gallery_features=native["native_g1_gallery_features"],
                native_query_reference_embeds=native["native_query_reference_embeds"],
                native_gallery_features=native["native_gallery_features"],
                native_query_features=native["native_query_features"],
                native_g1_input_ids=native["native_g1_input_ids"],
                native_g1_attention_masks=native["native_g1_attention_masks"],
                query_features=query_features, target_features=target_features,
                native_scores=native_scores, adapter_scores=adapter_scores,
                native_ranks=native_rank, adapter_positive_ranks=adapter_rank,
                native_positive_ranks=native_positive_ranks,
                native_top10=native["native_top10"], adapter_top10=adapter_top[:, :10],
                native_top50=native["native_top50"], adapter_top50=adapter_top)
            paired_artifact = _publish_paired_capture(root, category, paired_arrays, comparison, v3_role)
            _publish_v3_role_receipt(root, category, v3_receipt, paired_artifact,
                                     adapter_score_proof)
            comparisons[category] = comparison
    finally:
        adapter.close()
        torch.cuda.empty_cache()
    return {"label": TRANSFER_LABEL, "checkpoint_id": checkpoint_id,
            "capture_stage": "adapter", "paired_capture": comparisons}
def _verify_adapter_runtime(checkpoint_id: str) -> str:
    """Use the Adapter's normal security gate before running expensive native evaluation."""
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_ptha_mtst import Adapter

    adapter = Adapter(resolve_config(), checkpoint_id=checkpoint_id)
    try:
        return adapter.require_verified_interpreter()
    finally:
        adapter.close()


def _cgroup_memory_snapshot(phase: str) -> dict[str, Any]:
    group = next(line.split(":", 2)[2].strip("/\n") for line in
                 Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / group
    stat = {key: int(value) for key, value in
            (line.split() for line in (root / "memory.stat").read_text().splitlines())}
    return {"phase": phase, "current": int((root / "memory.current").read_text()),
            "peak": int((root / "memory.peak").read_text()),
            "anon": stat.get("anon"), "file": stat.get("file")}


def evaluate_transfer(
    source_root: Path, dataset_root: Path, checkpoint: Path, checkpoint_id: str,
    capture_directory: Path | None = None, capture_stage: str = "combined",
    gallery_probe_batches: int = 0, v3_plan_directory: Path | None = None,
    v3_role: str | None = None,
) -> dict[str, Any]:
    if capture_stage not in {"combined", "native", "adapter"}:
        raise ValueError(f"unknown capture stage: {capture_stage}")
    if capture_stage == "native" and capture_directory is None:
        raise ValueError("native capture stage requires --capture-directory")
    if capture_stage == "combined" and capture_directory is not None:
        raise ValueError("paired capture must use separate native and adapter processes")
    if (v3_plan_directory is None) != (v3_role is None):
        raise ValueError("PTHA V3 capture requires both plan directory and role")
    import torch
    if capture_stage == "adapter":
        assert capture_directory is not None
        return _evaluate_adapter_capture(capture_directory, dataset_root, checkpoint_id,
                                        v3_plan_directory, v3_role)

    src = source_tree(source_root) / "src"
    sys.path.insert(0, str(src))

    checkpoint_sha256 = sha256_file(checkpoint)
    if gallery_probe_batches not in {0, 1, 2}:
        raise ValueError("gallery probe batch count must be 0, 1, or 2")
    if gallery_probe_batches and (capture_directory is None or capture_stage != "native"):
        raise ValueError("bounded gallery diagnostic requires native stage and capture directory")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    probe_memory: list[dict[str, Any]] = []
    if gallery_probe_batches:
        probe_memory.append(_cgroup_memory_snapshot("checkpoint_loaded"))
        print(json.dumps(probe_memory[-1]), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PTHA native inference uses CUDA-only `.cuda()` calls; CUDA is unavailable")

    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess

    checkpoint_models = {
        "Blip2QformerCirCat": ("blip2_cir_cat", "pretrain"),
        "Blip2Qformer4TGIRCESIM": ("blip2_tgir_ce_sim", "pretrain_mtst"),
    }
    checkpoint_key = next((key for key in checkpoint_models if key in payload), None) if isinstance(payload, dict) else None
    if checkpoint_key is None:
        keys = sorted(map(str, payload)) if isinstance(payload, dict) else [type(payload).__name__]
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; expected one of {list(checkpoint_models)}; "
            f"observed keys={keys}"
        )
    state = payload[checkpoint_key]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; {checkpoint_key!r} is not a tensor-only state dictionary"
        )

    model_name, model_type = checkpoint_models[checkpoint_key]
    model_class = registry.get_model_class(model_name)
    config = OmegaConf.load(model_class.default_config_path(model_type))
    # The state dict must not cause the constructor to fetch an implicit BLIP-2 base checkpoint.
    config.model.load_pretrained = False
    model = model_class.from_config(config.model)
    report = _state_dict_report(model.state_dict(), state)
    if report["missing_keys"] or report["unexpected_keys"] or report["shape_mismatches"]:
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256} incompatible with "
            f"{model_name}/{model_type}: {json.dumps(report)}"
        )
    model.load_state_dict(state, strict=True)
    report["strict_false_hid_anything"] = False
    del payload, state
    import gc
    gc.collect()
    model = model.cuda().eval()
    if gallery_probe_batches:
        probe_memory.append(_cgroup_memory_snapshot("model_cuda_loaded"))
        print(json.dumps(probe_memory[-1]), flush=True)
    # Resolved from the INSTANCE that was actually constructed and loaded, not from the checkpoint
    # key and not from a tensor width: the same payload key could in principle build a different
    # class, and the contract belongs to the class whose `inference()` will run.
    target_contract = TARGET_FEATURE_CONTRACTS.get(type(model).__name__)
    if target_contract is None:
        raise ValueError(
            f"checkpoint sha256={checkpoint_sha256}; checkpoint key {checkpoint_key!r} built class "
            f"{type(model).__name__!r}, which has no declared target-feature contract "
            f"({sorted(TARGET_FEATURE_CONTRACTS)}); refusing to guess a projection")

    _vis_processors, text_processors = load_preprocess(config.preprocess)
    with _fashioniq_dataset_binding(dataset_root) as data_utils:
        from utils import extract_index_blip_features
        from validate_blip import compute_fiq_val_metrics
        validate_blip = importlib.import_module("validate_blip")

        FashionIQDataset = data_utils.FashionIQDataset
        targetpad_transform = data_utils.targetpad_transform
        transform = targetpad_transform(1.25, 224)
        category_metrics: dict[str, dict[str, float]] = {}
        pair_metrics: dict[str, dict[str, Any]] = {}
        contract_reports: dict[str, dict[str, Any]] = {}
        native_batch_observations: dict[str, list[int]] = {"query": [], "gallery": []}
        v3_execution = None
        if v3_plan_directory is not None and v3_role is not None:
            first_receipt = transfer_v3_role_receipt(v3_plan_directory / f"{CATEGORIES[0]}.json",
                                                     v3_role, "native")
            v3_execution = first_receipt["execution"]
        utils_module = importlib.import_module("utils")
        validate_blip = importlib.import_module("validate_blip")
        if capture_directory is not None:
            from torch.utils.data import DataLoader as NativeDataLoader

            def bounded_dataloader(*args: Any, **kwargs: Any):
                kwargs["num_workers"] = 0
                if v3_execution is not None:
                    dataset = kwargs.get("dataset", args[0] if args else None)
                    side = "gallery" if getattr(dataset, "mode", None) == "classic" else "query"
                    kwargs["batch_size"] = v3_execution[f"{side}_batch_size"]
                loader = NativeDataLoader(*args, **kwargs)

                def observed_batches():
                    side = "gallery" if getattr(loader.dataset, "mode", None) == "classic" else "query"
                    for batch in loader:
                        value = batch[0] if isinstance(batch, (tuple, list)) else batch
                        native_batch_observations[side].append(len(value))
                        yield batch
                return observed_batches()

            setattr(utils_module, "DataLoader", bounded_dataloader)
            setattr(validate_blip, "DataLoader", bounded_dataloader)
        capture_root = capture_directory.resolve() if capture_directory is not None else None
        active_gallery_store: dict[str, Any] | None = None
        try:
            for category in CATEGORIES:
                relative = FashionIQDataset("val", [category], "relative", transform)
                classic = FashionIQDataset("val", [category], "classic", transform)
                native_batch_observations["query"].clear()
                native_batch_observations["gallery"].clear()
                if v3_plan_directory is not None and v3_role is not None:
                    role_receipt = transfer_v3_role_receipt(v3_plan_directory / f"{category}.json",
                                                            v3_role, "native")
                    if v3_execution is None or any(
                            role_receipt["execution"].get(key) != v3_execution.get(key)
                            for key in ("query_batch_size", "gallery_batch_size")):
                        raise ValueError(f"{category}: native role batch settings differ across category plans")
                if gallery_probe_batches:

                    probe_rows = 64 * gallery_probe_batches
                    if len(classic) < probe_rows:
                        raise ValueError(f"{category}: gallery too short for requested native batches")
                    classic.image_names = classic.image_names[:probe_rows]
                    if capture_root is None or utils_module is None:
                        raise ValueError("bounded gallery diagnostic requires native capture directory")
                    gallery_storage = capture_root / ".probe.gallery"
                    with _bounded_native_gallery_store(model, utils_module, probe_rows,
                                                       gallery_storage) as gallery_store:
                        active_gallery_store = gallery_store
                        original_extract = model.extract_target_features
                        observed_batches: list[dict[str, Any]] = []

                        def observe_extract(*args: Any, **kwargs: Any):
                            outputs = original_extract(*args, **kwargs)
                            record = {"batch": len(observed_batches) + 1,
                                      "features": [{"shape": list(value.shape), "dtype": str(value.dtype),
                                                    "storage_bytes": value.untyped_storage().nbytes()}
                                                   for value in outputs],
                                      **_cgroup_memory_snapshot(
                                          f"gallery_batch_{len(observed_batches) + 1}")}
                            observed_batches.append(record)
                            print(json.dumps(record), flush=True)
                            return outputs

                        setattr(model, "extract_target_features", observe_extract)
                        try:
                            features, names = extract_index_blip_features(classic, model, save_memory=True)
                        finally:
                            setattr(model, "extract_target_features", original_extract)
                        feature_report = {
                            "feature_shapes": [list(value.shape) for value in features],
                            "feature_dtypes": [str(value.dtype) for value in features],
                            "feature_storage_bytes": [value.untyped_storage().nbytes() for value in features],
                            "storage": {key: value for key, value in gallery_store.items()
                                        if key != "cleanup"},
                        }
                        del features
                    gallery_store["cleanup"]()
                    active_gallery_store = None
                    result = {"diagnostic_only": True, "category": category,
                              "requested_batches": gallery_probe_batches,
                              "gallery_batch_size": 64, "rows": len(names),
                              "first_ids": names[:8], "last_ids": names[-8:],
                              **feature_report, "batches": observed_batches,
                              "storage_cleaned": not gallery_storage.exists(),
                              "memory": probe_memory + [_cgroup_memory_snapshot("gallery_probe_complete")]}
                    print(json.dumps(result, indent=2), flush=True)
                    return result
                capture: dict[str, Any] = {"native_forward": [], "native_tokens": [],
                                           "native_pixels": [], "native_query_features": []}
                visual_hook = None
                if capture_root is not None:
                    visual_hook = model.visual_encoder.register_forward_pre_hook(
                        lambda _module, inputs: capture["native_pixels"].append(
                            inputs[0][:1].detach().float().cpu().numpy()) if not capture["native_pixels"] else None)
                try:
                    if capture_root is None:
                        index_features, index_names = extract_index_blip_features(
                            classic, model, save_memory=False)
                        gallery_store = None
                    else:
                        gallery_storage = capture_root / f".{category}.gallery"
                        with _bounded_native_gallery_store(model, utils_module, len(classic),
                                                           gallery_storage) as gallery_store:
                            active_gallery_store = gallery_store
                            index_features, index_names = extract_index_blip_features(
                                classic, model, save_memory=True)
                finally:
                    if visual_hook is not None:
                        visual_hook.remove()
                if capture_root is not None and len(index_names) != len(classic):
                    raise ValueError(f"{category}: native gallery extractor returned incomplete ordered IDs")
                index_features, contract_report = _target_feature_contract(
                    model, index_features, torch, target_contract)
                contract_reports[category] = contract_report
                if capture_root is None:
                    r10, r50 = compute_fiq_val_metrics(relative, model, index_features, index_names, text_processors)
                    category_metrics[category] = {"r10": r10, "r50": r50}
                else:
                    capture["native_forward"].clear()
                    capture["native_tokens"].clear()
                    capture["native_query_features"].clear()
                    original_generator = validate_blip.generate_fiq_val_predictions
                    original_inference = model.inference
                    text_proj_hook = model.text_proj.register_forward_hook(
                        lambda _module, _inputs, output: capture["native_query_features"].append(
                            output.detach().float().cpu().numpy()))
                    original_tokenizer = model.tokenizer

                    def capture_generator(*args: Any, **kwargs: Any):
                        output = original_generator(*args, **kwargs)
                        nonlocal native_score_tensor
                        native_score_tensor = output[0]
                        capture["native_scores"] = output[0].detach().float().cpu().numpy()
                        capture["target_ids"] = list(output[1])
                        capture["reference_ids"] = list(output[2])
                        capture["processed_texts"] = list(output[3])
                        return output

                    def capture_inference(*args: Any, **kwargs: Any):
                        result = original_inference(*args, **kwargs)
                        if not capture["native_forward"]:
                            capture["native_forward"].append(args[0].detach().float().cpu().numpy())
                        return result
                    def capture_tokenizer(*args: Any, **kwargs: Any):
                        tokens = original_tokenizer(*args, **kwargs)
                        capture["native_tokens"].append({
                            "input_ids": tokens.input_ids.detach().cpu().numpy(),
                            "attention_mask": tokens.attention_mask.detach().cpu().numpy(),
                        })
                        return tokens

                    setattr(validate_blip, "generate_fiq_val_predictions", capture_generator)
                    model.inference = capture_inference
                    model.tokenizer = capture_tokenizer
                    original_argsort = torch.argsort
                    native_rank_capture: list[dict[str, Any]] = []
                    native_score_tensor: Any = None

                    def capture_argsort(input_tensor: Any, *args: Any, **kwargs: Any):
                        result = original_argsort(input_tensor, *args, **kwargs)
                        dim = kwargs.get("dim", args[0] if args else -1)
                        if (native_score_tensor is not None and input_tensor.ndim == 2
                                and tuple(input_tensor.shape) == tuple(native_score_tensor.shape)
                                and dim == -1
                                and input_tensor.device == native_score_tensor.device
                                and input_tensor.dtype == native_score_tensor.dtype
                                and torch.equal(input_tensor, 1 - native_score_tensor)):
                            native_rank_capture.append({
                                "ranks": result.detach().cpu().numpy(),
                                "device": str(input_tensor.device),
                                "dtype": str(input_tensor.dtype),
                                "shape": list(input_tensor.shape),
                                "dim": dim,
                                "key": "validate_blip.compute_fiq_val_metrics: torch.argsort(1 - pred_sim, dim=-1)"})
                        return result

                    torch.argsort = capture_argsort
                    try:
                        r10, r50 = compute_fiq_val_metrics(
                            relative, model, index_features, index_names, text_processors,
                            save_memory=True)
                    finally:
                        torch.argsort = original_argsort
                        setattr(validate_blip, "generate_fiq_val_predictions", original_generator)
                        model.inference = original_inference
                        model.tokenizer = original_tokenizer
                        text_proj_hook.remove()
                    category_metrics[category] = {"r10": r10, "r50": r50}
                    import numpy as np
                    gallery_ids = [str(name) for name in index_names]
                    images_root = data_utils.base_path / "fashionIQ_dataset" / "images"
                    gallery_paths = [_fashioniq_image_path(images_root, name) for name in gallery_ids]
                    rows = list(relative.triplets)
                    native_reference_ids = capture["reference_ids"]
                    query_ids = list(native_reference_ids)
                    row_reference_ids = [row["candidate"].replace("jpg", "png") for row in rows]
                    row_target_ids = [row["target"].replace("jpg", "png") for row in rows]
                    if query_ids != row_reference_ids or capture["target_ids"] != row_target_ids:
                        raise ValueError(f"{category}: native query/target ID order differs from val dataset rows")
                    query_texts = [
                        f"{row['captions'][0].strip('.?, ').capitalize()} and {row['captions'][1].strip('.?, ')}"
                        for row in rows]
                    if [text_processors["eval"](text) for text in query_texts] != capture["processed_texts"]:
                        raise ValueError(f"{category}: native query text/order differs from adapter inputs")
                    v3_query_ids = [f"{query_id}#row-{index}"
                                    for index, query_id in enumerate(query_ids)]
                    v3_receipt = _v3_capture_receipt(
                        v3_plan_directory / f"{category}.json" if v3_plan_directory else None,
                        v3_role, "native", category, v3_query_ids, gallery_ids,
                        row_target_ids, v3_query_ids)
                    if capture_stage == "native":
                        native_scores = capture["native_scores"]
                        if len(native_rank_capture) != 1:
                            raise ValueError(f"{category}: expected one native metric score sort, got {len(native_rank_capture)}")
                        native_sort = native_rank_capture[0]
                        native_rank = native_sort["ranks"]
                        if native_sort["shape"] != list(native_scores.shape):
                            raise ValueError(f"{category}: native sorter returned wrong rank matrix shape")
                        gallery_index = {name: index for index, name in enumerate(gallery_ids)}
                        positives = np.asarray([gallery_index[name] for name in capture["target_ids"]], dtype=np.int64)
                        native_hits = {str(k): int(np.count_nonzero(
                            np.any(native_rank[:, :k] == positives[:, None], axis=1))) for k in (10, 50)}
                        native_ids = np.concatenate([row["input_ids"] for row in capture["native_tokens"]])
                        native_masks = np.concatenate([row["attention_mask"] for row in capture["native_tokens"]])
                        native_query_features = np.concatenate(capture["native_query_features"])
                        native_query_features = torch.nn.functional.normalize(
                            torch.from_numpy(native_query_features), dim=-1).numpy() / float(model.temp)
                        native_arrays = dict(
                            query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                            target_ids=np.asarray(capture["target_ids"]),
                            reference_ids=np.asarray(capture["reference_ids"]),
                            query_texts=np.asarray(query_texts),
                            processed_texts=np.asarray(capture["processed_texts"]),
                            native_g1_pixels=capture["native_pixels"][0],
                            native_g1_pixel_gallery_id=np.asarray(gallery_ids[0]),
                            native_g1_gallery_features=index_features[0][0].detach().float().cpu().numpy(),
                            native_query_reference_embeds=capture["native_forward"][0],
                            native_gallery_features=index_features[0].detach().float().cpu().numpy(),
                            native_query_features=native_query_features,
                            native_g1_input_ids=native_ids, native_g1_attention_masks=native_masks,
                            native_scores=native_scores, native_ranks=native_rank,
                            native_top10=native_rank[:, :10], native_top50=native_rank[:, :50])
                        native_score_proof = _score_proof(v3_receipt, native_scores, native_rank)
                        if gallery_store is None:
                            raise ValueError("native capture has no bounded gallery storage receipt")
                        query_batch = (v3_execution["query_batch_size"] if v3_execution else 16)
                        gallery_batch = (v3_execution["gallery_batch_size"] if v3_execution else 64)
                        if v3_receipt is not None:
                            _validate_native_batch_observations(native_batch_observations,
                                                               len(query_ids), len(gallery_ids),
                                                               query_batch, gallery_batch)
                        native_batch_receipt = {
                            "role": v3_role, "plan_sha256": v3_receipt["plan_sha256"] if v3_receipt else None,
                            "configured": {"query": query_batch, "gallery": gallery_batch},
                            "observed_lengths": {key: list(value)
                                                 for key, value in native_batch_observations.items()}}
                        comparison = {
                            "query_count": len(query_ids), "gallery_count": len(gallery_ids),
                            "native_batch_sizes": {"query": query_batch, "gallery": gallery_batch},
                            "native_batch_receipt": native_batch_receipt,
                            "num_workers": {"query": 0, "gallery": 0},
                            "source_original_num_workers": {"query": 4, "gallery": 2},
                            "native_hits": native_hits, "denominator": len(query_ids),
                            "native_metric": {str(k): {"hits": native_hits[str(k)],
                                                       "denominator": len(query_ids)} for k in (10, 50)},
                            "query_sha256": [sha256_file(_fashioniq_image_path(images_root, name)) for name in query_ids],
                            "gallery_sha256": [sha256_file(path) for path in gallery_paths],
                            "source": {"repository": "MTST_PTHA", "commit": MODEL["upstream_commit_sha"],
                                       "extractor": "utils.extract_index_blip_features",
                                       "prediction": "validate_blip.generate_fiq_val_predictions",
                                       "scoring": "model.inference",
                                       "sort": "validate_blip.compute_fiq_val_metrics torch.argsort"},
                            "native_sort": {key: value for key, value in native_sort.items() if key != "ranks"},
                            "native_gallery_storage": {
                                key: value for key, value in gallery_store.items() if key != "cleanup"},
                            "resource_intervention": (
                                "replay-scoped storage only: actual bound extract_target_features outputs "
                                "copied bit-exactly to CPU NPY memmaps; source utils.torch.vstack returns "
                                "the verified contiguous full memmap view; native extractor/model math unchanged"),
                            "artifact": f"{_capture_stem(category, v3_role, 'native')}.npz"}
                        _publish_native_capture(capture_root, category, native_arrays, comparison, v3_role)
                        native_artifact = capture_root / comparison["artifact"]
                        _publish_v3_role_receipt(capture_root, category, v3_receipt,
                                                 native_artifact, native_score_proof)
                        pair_metrics[category] = comparison
                        del index_features
                        gallery_store["cleanup"]()
                        active_gallery_store = None
        finally:
            if active_gallery_store is not None:
                features = None
                index_features = None
                import gc
                gc.collect()
                active_gallery_store["cleanup"]()
            torch.cuda.empty_cache()


    if capture_stage == "native":
        return {"label": TRANSFER_LABEL, "train_dataset": "CIRR", "evaluation_dataset": "FashionIQ",
                "checkpoint_id": checkpoint_id, "checkpoint_sha256": checkpoint_sha256,
                "capture_stage": "native", "native_capture": pair_metrics}
    r10 = sum(values["r10"] for values in category_metrics.values()) / len(category_metrics)
    r50 = sum(values["r50"] for values in category_metrics.values()) / len(category_metrics)
    return {
        "model_id": MODEL["model_id"],
        "protocol_id": MODEL["native_protocol"],
        "label": TRANSFER_LABEL,
        "train_dataset": "CIRR",
        "evaluation_dataset": "FashionIQ",
        "checkpoint_training_noise": "unknown",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_top_level_model_key": checkpoint_key,
        "serialization_inspection": "torch.load(weights_only=True)",
        "model_name": model_name,
        "backbone_config": f"{model_type} (EVA-CLIP ViT-g; 224 px, 32 Q-Former tokens)",
        "load_report": {**report, "strict_load": True},
        "target_feature_contract": contract_reports,
        "paired_capture": pair_metrics,
        "paper_fashioniq_context_only": {
            "training_condition": "FashionIQ_MTST",
            "r10": 54.99,
            "r50": 76.28,
            "mean": 65.64,
            "mismatch": "training dataset/checkpoint differs; numerical context only, not reproduction",
            "differences_transfer_minus_paper": {
                "r10": r10 - 54.99,
                "r50": r50 - 76.28,
                "mean": (r10 + r50) / 2 - 65.64,
            },
        },
    }



def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", choices=CHECKPOINT_IDS, default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--capture-directory", type=Path,
                        help="save paired native/Adapter FashionIQ transfer arrays in this directory")
    parser.add_argument("--capture-stage", choices=("native", "adapter", "combined"), default="combined",
                        help="run native capture or Adapter comparison as separate processes")
    parser.add_argument("--v3-plan-directory", type=Path,
                        help="directory of immutable per-category PTHA V3 plans")
    parser.add_argument("--v3-role", choices=tuple(V3_ROLES),
                        help="pre-registered V3 capture role for this stage")
    parser.add_argument("--verify-adapter-runtime", action="store_true",
                        help="verify configured Adapter worker interpreter without model inference")
    parser.add_argument("--gallery-probe-batches", type=int, choices=(1, 2), default=0,
                        help="diagnostic only: run actual native gallery extractor on one or two G64 batches")
    parser.add_argument("--native-gallery-probe-child", action="store_true",
                        help="internal verified native worker")
    parser.add_argument("--native-capture-child", action="store_true",
                        help="internal pinned native worker")
    parser.add_argument("--preregister-v3-plans", type=Path,
                        help="derive and publish immutable V3 plans from actual inputs without inference")
    parser.add_argument(
        "--transfer-cirr-to-fashioniq",
        action="store_true",
        help="explicit exploratory transfer: CIRR-trained checkpoint evaluated on FashionIQ",
    )
    args = parser.parse_args(argv)
    if args.verify_adapter_runtime:
        try:
            print(json.dumps({"checkpoint_id": args.checkpoint_id,
                              "adapter_interpreter": _verify_adapter_runtime(args.checkpoint_id)}, indent=2))
        except Exception as error:
            print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0
    if args.preregister_v3_plans is not None:
        try:
            variant = next(item for item in MODEL["checkpoint_variants"]
                           if item["checkpoint_id"] == args.checkpoint_id)
            checkpoint = args.checkpoint_root / variant["filename"]
            identities = build_transfer_v3_identities(
                args.source_root, args.dataset_root, checkpoint, args.checkpoint_id)
            print(json.dumps(publish_transfer_v3_plans(
                args.preregister_v3_plans, identities, args.dataset_root), indent=2))
            return 0
        except Exception as error:
            import traceback
            print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
            traceback.print_exc()
            return 1

    if (args.v3_plan_directory is None) != (args.v3_role is None):
        print("[BLOCKED] --v3-plan-directory and --v3-role must be supplied together", file=sys.stderr)
        return 1
    if args.v3_plan_directory is not None and args.v3_role is not None:
        stage = "adapter" if args.capture_stage == "adapter" else "native"
        try:
            variant = next(item for item in MODEL["checkpoint_variants"]
                           if item["checkpoint_id"] == args.checkpoint_id)
            checkpoint = args.checkpoint_root / variant["filename"]
            expected = build_transfer_v3_identities(
                args.source_root, args.dataset_root, checkpoint, args.checkpoint_id)
            validate_transfer_v3_plans(args.v3_plan_directory, args.v3_role, stage, expected)
        except Exception as error:
            print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    blockers = preflight(
        args.source_root,
        args.dataset_root,
        args.checkpoint_root,
        checkpoint_id=args.checkpoint_id,
        transfer_cirr_to_fashioniq=args.transfer_cirr_to_fashioniq,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.capture_stage == "native" and not args.native_capture_child:
        try:
            interpreter = _verify_adapter_runtime(args.checkpoint_id)
            script = Path(__file__).resolve()
            command = [interpreter, str(script),
                       "--source-root", str(args.source_root), "--dataset-root", str(args.dataset_root),
                       "--checkpoint-root", str(args.checkpoint_root), "--checkpoint-id", args.checkpoint_id,
                       "--capture-directory", str(args.capture_directory), "--capture-stage", "native",
                       "--native-capture-child", "--transfer-cirr-to-fashioniq"]
            if args.v3_plan_directory is not None and args.v3_role is not None:
                command += ["--v3-plan-directory", str(args.v3_plan_directory),
                            "--v3-role", args.v3_role]
            if args.gallery_probe_batches:
                command += ["--gallery-probe-batches", str(args.gallery_probe_batches),
                            "--native-gallery-probe-child"]
            print(json.dumps({"native_worker": interpreter, "command": command}), flush=True)
            return subprocess.run(command, check=False).returncode
        except Exception as error:
            import traceback
            print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
            traceback.print_exc()
            return 1
    if args.capture_stage == "adapter":
        try:
            _verify_adapter_runtime(args.checkpoint_id)
        except Exception as error:
            print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    checkpoint_variant = next(
        item for item in MODEL["checkpoint_variants"] if item["checkpoint_id"] == args.checkpoint_id
    )
    checkpoint = args.checkpoint_root / checkpoint_variant["filename"]
    try:
        result = evaluate_transfer(
            args.source_root, args.dataset_root, checkpoint, args.checkpoint_id,
            args.capture_directory, args.capture_stage, args.gallery_probe_batches,
            args.v3_plan_directory, args.v3_role
        )
    except Exception as error:
        import traceback
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
