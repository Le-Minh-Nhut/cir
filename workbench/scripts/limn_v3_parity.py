"""LIMN FashionIQ native/adapter parity runner.

Plan phase is CPU-only and immutable. Native and adapter phases execute only after
Main leases GPU; each writes into its own run root and never publishes discovery state.
"""
from __future__ import annotations

import contextlib
import argparse
import uuid
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.operator_config import resolve_config
from workbench.replay import limn as replay

STANDARD = "parity-absolute-relative/v3"
T4_PATH = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
TOLERANCES = {"forward_max_abs": 1e-5, "forward_max_rel": 1e-4,
              "batch_max_abs": 1e-4, "batch_max_rel": 1e-3,
              "score_max_abs": 1e-6, "tie_epsilon": 1e-6}
ROLES = ("native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
         "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
         "adapter_batch_1", "adapter_batch_n")
NATIVE_ROLES = ("native_repeat_1", "native_repeat_2", "native_adapter_native", "native_batch_1", "native_batch_n")
ADAPTER_ROLES = ("adapter_repeat_1", "adapter_repeat_2", "native_adapter_adapter", "adapter_batch_1", "adapter_batch_n")


def _t4():
    spec = importlib.util.spec_from_file_location("limn_t4_parity", T4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source LIMN producer: {T4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sha(values: list[str]) -> str:
    return _sha(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _compare_g1_inputs(root: Path, category: str, native_inputs: dict) -> dict:
    """Compare independently captured native and Adapter inputs in one bounded order."""
    native_path = root / "native" / category / f"g2_{category}_native.npz"
    adapter_root = root / "adapter" / category / "g1"
    query_paths = sorted(adapter_root.glob("query-*.npz"))
    gallery_paths = sorted(adapter_root.glob("gallery-*.npz"))
    if not native_path.is_file() or not query_paths or not gallery_paths:
        raise FileNotFoundError("missing native/Adapter actual G1 input capture")
    native_keys = {"source_tensor", "target_tensor", "sample_token_ids", "sample_query_texts",
                   "sample_text_padding_mask", "causal_attn_mask_actual", "sample_query_ids",
                   "sample_target_ids", "sample_indices"}
    with np.load(native_path, allow_pickle=False) as blob:
        if not native_keys.issubset(blob.files):
            raise ValueError(f"native G1 capture missing required sample arrays: {native_path}")
        native = {key: blob[key].copy() for key in native_keys}
    query_keys = {"pixels", "paths", "texts", "token_ids", "token_padding_mask_derived",
                  "causal_attn_mask_actual"}
    queries, galleries = [], []
    for path in query_paths:
        with np.load(path, allow_pickle=False) as blob:
            if not query_keys.issubset(blob.files):
                raise ValueError(f"Adapter G1 query capture missing required arrays: {path}")
            queries.append({key: blob[key].copy() for key in query_keys})
    for path in gallery_paths:
        with np.load(path, allow_pickle=False) as blob:
            if not {"pixels", "paths"}.issubset(blob.files):
                raise ValueError(f"Adapter G1 gallery capture missing pixels/paths: {path}")
            galleries.append({key: blob[key].copy() for key in ("pixels", "paths")})
    def joined(records, key):
        return np.concatenate([record[key] for record in records], axis=0)
    adapter = {key: joined(queries, key) for key in query_keys}
    adapter.update({f"gallery_{key}": joined(galleries, key) for key in galleries[0]})
    inputs = native_inputs.get("queries")
    if not isinstance(inputs, list) or not inputs or len(inputs) > 32:
        raise ValueError("native G1 input paths must contain bounded selected queries")
    expected = {
        "sample_indices": native_inputs.get("sample_indices"),
        "sample_query_ids": native_inputs.get("reference_ids"),
        "sample_target_ids": native_inputs.get("target_ids"),
        "sample_query_texts": native_inputs.get("query_texts"),
        "sample_query_paths": [pair[0] for pair in inputs],
        "sample_target_paths": [pair[1] for pair in inputs],
    }
    for key, value in expected.items():
        if not isinstance(value, list) or len(value) != len(inputs):
            raise ValueError(f"native G1 inputs lack aligned metadata: {key}")
        observed = (native[key] if key in native else adapter["paths"] if key == "sample_query_paths"
                    else adapter["gallery_paths"])
        expected_array = np.asarray(value)
        if observed.shape != expected_array.shape or not np.array_equal(observed.astype(str), expected_array.astype(str)):
            raise ValueError(f"G1 source metadata differs from native_inputs: {key}")
    sample_count = len(inputs)
    pairs = (("query_pixels", native["source_tensor"], adapter["pixels"]),
             ("target_pixels", native["target_tensor"], adapter["gallery_pixels"]),
             ("token_ids", native["sample_token_ids"], adapter["token_ids"]),
             ("query_texts", native["sample_query_texts"], adapter["texts"]),
             ("padding_mask", native["sample_text_padding_mask"], adapter["token_padding_mask_derived"]),
             ("causal_mask", native["causal_attn_mask_actual"], adapter["causal_attn_mask_actual"]))
    comparisons = {}
    for name, left, right in pairs:
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"G1 {name} shape/dtype mismatch: native={left.shape}/{left.dtype}, Adapter={right.shape}/{right.dtype}")
        if name == "causal_mask":
            if np.isnan(left).any() or np.isposinf(left).any() or np.isnan(right).any() or np.isposinf(right).any():
                raise ValueError("G1 causal attention mask contains NaN or +Inf")
        elif left.dtype.kind in "biufc" and (not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError(f"G1 {name} contains nonfinite values")
        equal = np.array_equal(left, right)
        if name not in ("query_pixels", "target_pixels") and not equal:
            raise ValueError(f"G1 {name} differs between native and Adapter captures")
        comparisons[name] = {"sample_count": int(left.shape[0]) if left.ndim else 1,
                             "shape": list(left.shape), "dtype": str(left.dtype), "equal": equal}
        if name in ("query_pixels", "target_pixels"):
            comparisons[name].update({"max_abs": float(np.max(np.abs(left.astype(np.float64) -
                                                                                right.astype(np.float64)))),
                                      "bitwise_equal": bool(equal)})
    return {"sample_count": sample_count, "sample_indices": native["sample_indices"].astype(int).tolist(),
            "comparisons": comparisons}


def _protocol(config, category: str) -> dict:
    root = Path(config.FASHIONIQ_ROOT)
    try:
        split = json.loads((root / "image_splits" / f"split.{category}.val.json").read_text(encoding="utf-8"))
        captions = json.loads((root / "captions" / f"cap.{category}.val.json").read_text(encoding="utf-8"))
        correction = json.loads((root / "captions" / f"correction_dict_{category}.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read FashionIQ {category} protocol: {error}") from error
    trans = str.maketrans({char: " " for char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"})
    def clean(text: str) -> str:
        return " ".join(str(correction.get(token, token)) for token in text.lower().translate(trans).split())
    queries, gallery = [], []
    for row in captions:
        reference, target = str(row["candidate"]), str(row["target"])
        if reference not in split or target not in split:
            raise ValueError(f"FashionIQ {category} ID absent from val split: {reference}/{target}")
        queries.append({"reference_id": reference, "target_id": target,
                        "text": f"{clean(row['captions'][0])} and {clean(row['captions'][1])}"})
        for image_id in (reference, target):
            if image_id not in gallery:
                gallery.append(image_id)
    query_ids, exclusions = [], []
    for index, item in enumerate(queries):
        payload = {"index": index, "reference_id": item["reference_id"],
                   "target_id": item["target_id"], "text": item["text"]}
        query_id = "q:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        query_ids.append(query_id)
        exclusions.append(json.dumps({"query_id": query_id,
                                      "excluded_gallery_ids": [item["reference_id"]]},
                                     ensure_ascii=False, separators=(",", ":")))
    return {"queries": queries, "gallery_ids": gallery, "query_ids": query_ids,
            "target_ids": [item["target_id"] for item in queries],
            "reference_exclusions": exclusions}

def _environment_identity(config, controller_interpreter: str) -> str:
    interpreter = str(Path(_limn_interpreter(config)).absolute())
    controller_interpreter = str(Path(controller_interpreter).absolute())
    key = interpreter, controller_interpreter
    if key not in _ENVIRONMENT_CACHE:
        worker_runtime = _interpreter_runtime(interpreter)
        runtime = {"native": worker_runtime,
                   "adapter_controller": _worker_environment_identity(controller_interpreter),
                   "adapter_worker": worker_runtime}
        payload = json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode()
        _ENVIRONMENT_CACHE[key] = "runtime-" + _sha(payload)
    return _ENVIRONMENT_CACHE[key]


def _limn_interpreter(config) -> str:
    from workbench.backend.runtime import resolve_model_interpreter
    interpreter = resolve_model_interpreter(replay.MODEL, config)
    if not interpreter or not Path(interpreter).is_file():
        raise RuntimeError("LIMN configured model interpreter unavailable")
    return str(interpreter)


def _worker_environment_identity(interpreter: str) -> str:
    probe = ("import json,platform,sys; print(json.dumps({'python':sys.version,'prefix':sys.prefix,"
             "'implementation':platform.python_implementation(),'platform':platform.platform(),"
             "'machine':platform.machine()}, sort_keys=True))")
    try:
        completed = subprocess.run([interpreter, "-c", probe], capture_output=True, text=True,
                                   check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"LIMN interpreter probe failed: {error}") from error
    try:
        runtime = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LIMN interpreter probe returned invalid JSON: {error}") from error
    if not isinstance(runtime, dict):
        raise RuntimeError("LIMN interpreter probe must return an object")
    runtime["interpreter"] = interpreter
    runtime["source_commit"] = replay.MODEL["upstream_commit_sha"]
    return "runtime-" + _sha(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode())




def _cohort_id(query_ids: list[str], gallery_ids: list[str]) -> str:
    digest = _ordered_sha(query_ids + ["--gallery--"] + gallery_ids)
    return f"fashioniq-val-{digest[:24]}"
_ENVIRONMENT_CACHE: dict[tuple[str, str], str] = {}
_PROVENANCE_CACHE: dict[str, dict[str, dict[str, str | int]]] = {}
_CHECKPOINT_SHA_CACHE: dict[str, str] = {}


def _provenance(config, category: str) -> dict[str, dict[str, str | int]]:
    if category not in _PROVENANCE_CACHE:
        fashion = Path(config.FASHIONIQ_ROOT)
        source = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "LIMN" / "LIMN"
        checkpoint = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn" / replay.LIMN_CHECKPOINTS[category]["filename"]
        paths = [source / "test.py", source / "datasets.py", source / "model.py",
                 REPO / "workbench/composition/model_limn.py", REPO / "workbench/composition/adapters.py",
                 REPO / "workbench/replay/limn.py", REPO / "workbench/replay/common.py",
                 REPO / "workbench/scripts/limn_v3_parity.py", T4_PATH,
                 fashion / "image_splits" / f"split.{category}.val.json",
                 fashion / "captions" / f"cap.{category}.val.json",
                 fashion / "captions" / f"correction_dict_{category}.json", checkpoint]
        records = {}
        for path in paths:
            stat = path.stat()
            key = str(path)
            digest = replay.sha256_file(path) if path == checkpoint else _sha(path.read_bytes())
            if path == checkpoint:
                _CHECKPOINT_SHA_CACHE[key] = digest
            records[key] = {"bytes": stat.st_size, "sha256": digest}
        _PROVENANCE_CACHE[category] = records
    return _PROVENANCE_CACHE[category]


def _verify_provenance(config, category: str, expected: dict) -> None:
    global _PROVENANCE_CACHE
    _PROVENANCE_CACHE.pop(category, None)
    if _provenance(config, category) != expected:
        raise RuntimeError("LIMN source, checkpoint or FashionIQ input bytes changed after preregistration")


def _checkpoint_sha(config, category: str) -> str:
    path = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn" / replay.LIMN_CHECKPOINTS[category]["filename"]
    digest = _CHECKPOINT_SHA_CACHE.get(str(path))
    if digest is None:
        digest = replay.sha256_file(path)
        _CHECKPOINT_SHA_CACHE[str(path)] = digest
    return digest


def _identity(config, category: str, checkpoint: str, query_ids: list[str], gallery_ids: list[str],
              exclusions: list[str], targets: list[str], cohort: str, controller_interpreter: str) -> dict:
    model_config = {"model": replay.MODEL, "category": category,
                    "checkpoint": replay.LIMN_CHECKPOINTS[category], "source_inputs": _provenance(config, category),
                    "seed": 0, "batch_size": 32, "precision": "float32", "mode": "eval"}
    configuration = _sha(json.dumps(model_config, sort_keys=True, default=str,
                                    separators=(",", ":")).encode())
    labels = {"query_targets": [[query_id, target_id] for query_id, target_id in zip(query_ids, targets, strict=True)],
              "reference_exclusions": [[query_id, json.loads(exclusion)["excluded_gallery_ids"]]
                                       for query_id, exclusion in zip(query_ids, exclusions, strict=True)]}
    label_sha = _sha(json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {"checkpoint": checkpoint, "configuration": configuration,
            "environment": _environment_identity(config, controller_interpreter),
            "protocol": "fashioniq_val_split", "category": category, "cohort_id": cohort,
            "precision": "float32", "seed": 0, "query_order_sha256": _ordered_sha(query_ids),
            "gallery_order_sha256": _ordered_sha(gallery_ids),
            "reference_exclusions_sha256": label_sha,
            "query_count": len(query_ids), "gallery_count": len(gallery_ids)}


def _scope(coverage: str, population: int, ids: list[str], cohort: str, dimension: int = 768) -> dict:
    count = len(ids)
    if coverage == "sample" and not 0 < count < population:
        raise ValueError("sample scope must be non-empty and smaller than population")
    shape = [count, dimension]
    result = {"coverage": coverage, "population_count": population,
              "tensor_shape": shape, "component_shape": shape,
              "ordered_ids_sha256": _ordered_sha(ids), "cohort_id": cohort}
    if coverage == "sample":
        result["sample_count"] = count
    return result


def _execution(category: str, cohort: str, query_ids: list[str], gallery_ids: list[str], batch: int) -> dict:
    return {"partition_id": f"limn-{category}-{cohort}-v1",
            "query_partition_sha256": _ordered_sha(query_ids),
            "gallery_partition_sha256": _ordered_sha(gallery_ids),
            "query_batch_size": batch, "gallery_batch_size": batch}


def _plan(t4, config, root: Path, category: str) -> tuple[dict, str, Path]:
    allowed = {f"g1_g2.limn.{name}.v3-plan.json" for name in CATEGORIES}
    if root.exists():
        unexpected = {path.name for path in root.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"LIMN run root contains unexpected entries: {sorted(unexpected)}")
    checkpoint_root = config.WORKBENCH_CHECKPOINT_ROOT / "limn"
    problems = replay.validate_checkpoint_bundle(checkpoint_root, (category,))
    problems += replay.validate_dataset(config.FASHIONIQ_ROOT, (category,))
    problems += replay.validate_source(config.WORKBENCH_THIRD_PARTY_ROOT / "LIMN")
    if problems:
        raise RuntimeError("LIMN CPU preflight blocked: " + "; ".join(problems))
    checkpoint = replay.LIMN_CHECKPOINTS[category]
    checkpoint_sha = _checkpoint_sha(config, category)
    protocol = _protocol(config, category)
    qids, gids, exclusions = protocol["query_ids"], protocol["gallery_ids"], protocol["reference_exclusions"]
    sample_q, sample_g = qids[: min(33, len(qids))], gids[: min(64, len(gids))]
    sample_exclusions = exclusions[:len(sample_q)]
    controller_interpreter = str(Path(sys.executable).absolute())
    primary = _identity(config, category, checkpoint_sha, qids, gids, exclusions, protocol["target_ids"],
                        _cohort_id(qids, gids), controller_interpreter)
    sample_targets = protocol["target_ids"][:len(sample_q)]
    sample = _identity(config, category, checkpoint_sha, sample_q, sample_g, sample_exclusions,
                       sample_targets, _cohort_id(sample_q, sample_g), controller_interpreter)
    batch_n = 32
    defs = {"native_repeat_1": ("native", sample, sample_q, sample_g, replay.LIMN_BATCH_SIZE_DEFAULT),
            "native_repeat_2": ("native", sample, sample_q, sample_g, replay.LIMN_BATCH_SIZE_DEFAULT),
            "adapter_repeat_1": ("adapter", sample, sample_q, sample_g, replay.LIMN_BATCH_SIZE_DEFAULT),
            "adapter_repeat_2": ("adapter", sample, sample_q, sample_g, replay.LIMN_BATCH_SIZE_DEFAULT),
            "native_adapter_native": ("native", primary, qids, gids, replay.LIMN_BATCH_SIZE_DEFAULT),
            "native_adapter_adapter": ("adapter", primary, qids, gids, replay.LIMN_BATCH_SIZE_DEFAULT),
            "native_batch_1": ("native", sample, sample_q, sample_g, 1),
            "native_batch_n": ("native", sample, sample_q, sample_g, batch_n),
            "adapter_batch_1": ("adapter", sample, sample_q, sample_g, 1),
            "adapter_batch_n": ("adapter", sample, sample_q, sample_g, batch_n)}
    runs = {}
    for role, (side, identity, role_q, role_g, batch) in defs.items():
        runs[role] = {"run_id": f"limn:{category}:{role}:{uuid.uuid4().hex}",
                      "source_id": f"limn-{side}-source-{uuid.uuid4().hex}", "identity": identity,
                      "execution": _execution(category, identity["cohort_id"], role_q, role_g, batch)}
    controls = {}
    for name, ref, actual, side in (("native_repeat", "native_repeat_1", "native_repeat_2", "native"),
                                    ("adapter_repeat", "adapter_repeat_1", "adapter_repeat_2", "adapter"),
                                    ("native_adapter", "native_adapter_native", "native_adapter_adapter", "native"),
                                    ("native_batch_1_vs_n", "native_batch_1", "native_batch_n", "native"),
                                    ("adapter_batch_1_vs_n", "adapter_batch_1", "adapter_batch_n", "adapter")):
        batch = "batch" in name
        identity = runs[ref]["identity"]
        role_q = qids if identity is primary else sample_q
        role_g = gids if identity is primary else sample_g
        coverage = "full" if identity is primary else "sample"
        controls[name] = {"reference_run_id": runs[ref]["run_id"], "actual_run_id": runs[actual]["run_id"],
                          "reference_side": side, "dimension": "batch_parity" if batch else "forward_parity",
                          "dtype": "float32", "tolerance": {"atol": TOLERANCES["batch_max_abs"] if batch else TOLERANCES["forward_max_abs"],
                                                                "rtol": TOLERANCES["batch_max_rel"] if batch else TOLERANCES["forward_max_rel"]},
                          "scopes": {"query": _scope(coverage, len(qids), role_q, identity["cohort_id"]),
                                     "gallery": _scope(coverage, len(gids), role_g, identity["cohort_id"])}}
    provenance = _provenance(config, category)
    scoring_labels = {"query_targets": [[query_id, target_id] for query_id, target_id in
                                        zip(qids, protocol["target_ids"], strict=True)],
                      "reference_exclusions": [[query_id, [protocol["queries"][index]["reference_id"]]]
                                               for index, query_id in enumerate(qids)]}
    plan = {"standard": STANDARD, "model_id": "limn", "category": category, "identity": primary,
            "scoring_labels": scoring_labels,
            "metric_cutoffs": [1, 10, 50],
            "score_order_conventions": {"native": "fp32_descending_score",
                                        "adapter": "fp32_descending_score"},
            "scoring_proof_path": f"g1_g2.limn.{category}.native.score-ranks.jsonl",
            "supported_batch_policy": {"supported": True, "batch_sizes": [1, batch_n]},
            "population_scope": {"coverage": "full", "query_count": len(qids), "gallery_count": len(gids)},
            "controls": controls, "runs": runs, "tolerances": TOLERANCES, "provenance": provenance,
            "controller_interpreter": controller_interpreter,
            "resource_budget": {"model_resident_count": 1, "max_g1_images_per_capture": 32,
                                "full_gallery_pixel_capture": False, "cpu_threads": 10},
            "created_at": datetime.now(timezone.utc).isoformat()}
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"g1_g2.limn.{category}.v3-plan.json"
    return plan, t4.write_v3_plan(path, plan), path


def _plan_inputs(root: Path, category: str) -> tuple[dict, Path, str]:
    plan_path = root / f"g1_g2.limn.{category}.v3-plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"immutable plan missing: {plan_path}")
    plan = _read_json(plan_path)
    raw = plan_path.read_bytes()
    return plan, plan_path, _sha(raw)


def _interpreter_runtime(interpreter: str) -> dict:
    probe = ("import json,platform,sys,torch,open_clip; "
             "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'implementation':platform.python_implementation(),"
             "'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,"
             "'open_clip':getattr(open_clip,'__version__','unknown')}, sort_keys=True))")
    try:
        completed = subprocess.run([interpreter, "-c", probe], capture_output=True, text=True, check=True)
        runtime = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LIMN model runtime probe failed: {interpreter}: {error}") from error
    if not isinstance(runtime, dict) or not runtime.get("torch") or not runtime.get("open_clip"):
        raise RuntimeError(f"LIMN interpreter lacks observed torch/open_clip runtime: {interpreter}")
    return {"interpreter": interpreter, **runtime}


def execute_roles(root: Path, category: str, lease: str | None = None) -> None:
    if lease != "MAIN_GPU_LEASE":
        raise PermissionError("roles phase requires explicit Main GPU lease: --gpu-lease MAIN_GPU_LEASE")
    config = resolve_config()
    interpreter = str(Path(_limn_interpreter(config)).absolute())
    plan, _, _ = _plan_inputs(root, category)
    controller_interpreter = plan.get("controller_interpreter")
    if not isinstance(controller_interpreter, str) or not Path(controller_interpreter).is_absolute():
        raise RuntimeError("LIMN plan lacks absolute controller executable")
    if str(Path(sys.executable).absolute()) != controller_interpreter:
        raise RuntimeError("LIMN process executable differs from preregistered controller executable")
    _verify_provenance(config, category, plan["provenance"])
    for phase, child_python in (("roles-native", interpreter), ("roles-adapter", controller_interpreter)):
        completed = subprocess.run([child_python, str(Path(__file__).resolve()), "--run-root", str(root),
                                    "--category", category, "--gpu-lease", lease, "--phase", phase],
                                   cwd=REPO, check=False)
        if completed.returncode:
            raise RuntimeError(f"LIMN {phase} child failed with exit {completed.returncode}")


def _execute_roles_local(root: Path, category: str, lease: str | None = None, side: str = "both") -> None:
    if lease != "MAIN_GPU_LEASE":
        raise PermissionError("roles phase requires explicit Main GPU lease: --gpu-lease MAIN_GPU_LEASE")
    config, t4 = resolve_config(), _t4()
    plan, _, plan_sha = _plan_inputs(root, category)
    controller_interpreter = plan.get("controller_interpreter")
    if not isinstance(controller_interpreter, str) or not Path(controller_interpreter).is_absolute():
        raise RuntimeError("LIMN plan lacks absolute controller executable")
    expected_interpreter = (str(Path(_limn_interpreter(config)).absolute())
                            if side == "native" else controller_interpreter)
    if str(Path(sys.executable).absolute()) != expected_interpreter:
        raise RuntimeError(f"LIMN {side} process executable differs from preregistration: {sys.executable}")
    _verify_provenance(config, category, plan["provenance"])
    protocol = _protocol(config, category)
    role_root = root / "roles" / category
    role_root.mkdir(parents=True, exist_ok=True)
    full_q, full_g = list(range(len(protocol["queries"]))), list(range(len(protocol["gallery_ids"])))
    sample_q, sample_g = full_q[:33], full_g[:64]
    def selected(role):
        identity = plan["runs"][role]["identity"]
        return (full_q if identity["query_count"] == len(full_q) else sample_q,
                full_g if identity["gallery_count"] == len(full_g) else sample_g)
    def receipt(role, side, started, arrays, observed_batch, side_observation):
        run = plan["runs"][role]
        query_ids = arrays["query_features"][1]
        gallery_ids = arrays["gallery_features"][1]
        exclusion_by_query = {query_id: exclusion for query_id, exclusion in
                              zip(protocol["query_ids"], protocol["reference_exclusions"])}
        observed_cohort = _cohort_id(query_ids, gallery_ids)
        observed = _identity(config, category, _checkpoint_sha(config, category), query_ids, gallery_ids,
                             [exclusion_by_query[qid] for qid in query_ids],
                             [protocol["target_ids"][protocol["query_ids"].index(qid)] for qid in query_ids],
                             observed_cohort, controller_interpreter)
        if observed_cohort != run["identity"]["cohort_id"]:
            raise ValueError(f"observed cohort differs from preregistered run: {role}")
        qidx, gidx = selected(role)
        if query_ids != [protocol["query_ids"][i] for i in qidx] or gallery_ids != [protocol["gallery_ids"][i] for i in gidx]:
            raise ValueError(f"observed input order differs from preregistered run: {role}")
        if observed != run["identity"]:
            raise ValueError(f"observed identity differs from preregistered run: {role}")
        observed_gallery_batch = side_observation.get("gallery_batch_size", observed_batch)
        if (observed_batch != run["execution"]["query_batch_size"] or
                observed_gallery_batch != run["execution"]["gallery_batch_size"]):
            raise ValueError(f"observed query/gallery batches differ from preregistration: {role}")
        if side == "native":
            runtime = side_observation["native_identity"]["native_runtime"]
            if (runtime["batch_size"] != observed_batch or runtime["seed"] != 0 or
                    runtime["precision"] != "float32" or runtime["model_mode"] != "eval" or
                    runtime["model_training"]):
                raise ValueError(f"native runtime differs from registered execution: {role}")
        else:
            worker = side_observation["worker_observation"]
            model = worker["models"][category]
            if (worker["batch_size"] != observed_batch or worker["seed"] != 0 or
                    worker["precision"] != "float32" or worker["model_mode"] != "eval" or
                    model["checkpoint_sha256"] != observed["checkpoint"] or model["training"] or
                    model["parameter_dtype"] != "torch.float32"):
                raise ValueError(f"Adapter worker runtime differs from registered execution: {role}")
        execution = {**run["execution"], "query_batch_size": observed_batch,
                     "gallery_batch_size": observed_gallery_batch}
        for name, array, expected in (("query", arrays["query_features"][0], query_ids),
                                      ("gallery", arrays["gallery_features"][0], gallery_ids)):
            if array.dtype != np.float32 or array.ndim != 2 or array.shape != (len(expected), 768):
                raise ValueError(f"observed {name} features violate V3 tensor contract: {role}")
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite {name} features in role {role}")
        captured = {}
        for control_name, control in plan["controls"].items():
            if run["run_id"] in (control["reference_run_id"], control["actual_run_id"]):
                for scope_name, tensor_key in (("query", "query_features"), ("gallery", "gallery_features")):
                    captured[f"{control_name}.{scope_name}"] = arrays[tensor_key]
        payload = {**{key: value[0] for key, value in arrays.items()},
                   **{f"{key}_ids": np.asarray(value[1]) for key, value in arrays.items()}}
        if "image_ids" in side_observation:
            payload.update({f"{key}_image_ids": np.asarray(value)
                            for key, value in side_observation["image_ids"].items()})
        np.savez_compressed(role_root / f"{role}.npz", **payload)
        ranking_archive = (root / "native" / category / f"g2_{category}_native_ranking.npz"
                           if side == "native" and role == "native_adapter_native" else
                           root / "adapter" / category / f"g2_{category}_adapter_ranking.npz"
                           if side == "adapter" and role == "native_adapter_adapter" else None)
        ranking_sha256 = _file_sha256(ranking_archive) if ranking_archive is not None else None
        value = t4.v3_run_receipt(run_id=run["run_id"], side=side, source_id=run["source_id"],
                                  plan_sha256=plan_sha, started_at=started, arrays=captured,
                                  execution=execution, identity=observed)
        value["ranking_archive_sha256"] = ranking_sha256
        value["runtime_observation"] = side_observation
        (role_root / f"{role}.receipt.json").write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    native_results = {}
    native_context = t4.native_session(category, config) if side != "adapter" else contextlib.nullcontext(None)
    with native_context as loaded:
        for role in (NATIVE_ROLES if side != "adapter" else ()):
            qidx, gidx = selected(role)
            started = datetime.now(timezone.utc).isoformat()
            out = role_root / role
            out.mkdir(exist_ok=True)
            if role == "native_adapter_native":
                measured = t4.native_phase(category, config, artifact_root=out,
                                            batch_size=plan["runs"][role]["execution"]["query_batch_size"],
                                            capture_full_pixels=False, loaded=loaded)
                feature_path = out / f"g2_{category}_native.npz"
                query_key, gallery_key = "native_query", "native_gallery"
            else:
                measured = t4.native_selected_phase(category, config, loaded, artifact_root=out,
                                                    query_indices=qidx, gallery_indices=gidx,
                                                    batch_size=plan["runs"][role]["execution"]["query_batch_size"])
                feature_path = out / f"g2_{category}_selected.npz"
                query_key, gallery_key = "queries", "gallery"
            with np.load(feature_path, allow_pickle=False) as blob:
                observed_queries = blob["query_ids"].astype(str).tolist()
                observed_gallery = blob["gallery_ids"].astype(str).tolist()
                if role == "native_adapter_native":
                    if observed_queries != [protocol["queries"][i]["reference_id"] for i in qidx]:
                        raise ValueError("native source query reference order differs from preregistration")
                    query_ids = [protocol["query_ids"][i] for i in qidx]
                else:
                    if (blob["query_indices"].tolist() != qidx or
                            blob["gallery_indices"].tolist() != gidx):
                        raise ValueError("native selected query/gallery indices differ from preregistration")
                    query_ids = [protocol["query_ids"][i] for i in blob["query_indices"].tolist()]
                if observed_gallery != [protocol["gallery_ids"][i] for i in gidx]:
                    raise ValueError("native source gallery order differs from preregistration")
                arrays = {"query_features": (blob[query_key], query_ids),
                          "gallery_features": (blob[gallery_key], observed_gallery)}
            if role == "native_adapter_native":
                inputs_path = out / f"g2_{category}_inputs.json"
                inputs = _read_json(inputs_path)
                if inputs["full_reference_ids"] != [query["reference_id"] for query in protocol["queries"]]:
                    raise ValueError("native evaluator query references differ from preregistered order")
                inputs["full_query_ids"] = protocol["query_ids"]
                inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
            if role == "native_adapter_native":
                native_output = root / "native" / category
                native_output.mkdir(parents=True, exist_ok=True)
                for name in (f"g2_{category}_native.npz", f"g2_{category}_native_ranking.npz",
                             f"g2_{category}_inputs.json"):
                    shutil.copy2(out / name, native_output / name)
            if loaded is None:
                raise RuntimeError("native role lost resident source model")
            native_runtime = measured
            side_observation = {"python": sys.version, "executable": sys.executable,
                                "model_type": type(loaded["model"]).__qualname__,
                                "native_identity": measured}
            receipt(role, "native", started, arrays,
                    measured["native_runtime"]["batch_size"], side_observation)
            native_results[role] = measured
    adapter_root = root / "adapter" / category
    adapter_root.mkdir(parents=True, exist_ok=True)
    adapter_context = (t4.adapter_session(category, config, adapter_root,
                       plan["runs"]["adapter_repeat_1"]["execution"]["query_batch_size"])
                       if side != "native" else contextlib.nullcontext(None))
    with adapter_context as adapter:
        primary_result = None
        for role in (ADAPTER_ROLES if side != "native" else ()):
            if side == "native":
                break
            if adapter is None:
                raise RuntimeError("adapter child lacks resident production worker")
            started = datetime.now(timezone.utc).isoformat()
            adapter.batch_size = plan["runs"][role]["execution"]["query_batch_size"]
            qidx, gidx = selected(role)
            if role == "native_adapter_adapter":
                inputs = _read_json(role_root / "native_adapter_native" / f"g2_{category}_inputs.json")
                primary_result = t4.adapter_population_phase(adapter, artifact_root=adapter_root,
                                                             category=category, inputs=inputs, config=config,
                                                             capture_root=adapter_root / "g1",
                                                             native_ranking_path=root / "native" / category / f"g2_{category}_native_ranking.npz")
                with np.load(adapter_root / f"g2_{category}_population.npz", allow_pickle=False) as blob:
                    observed_query_images = blob["query_features_ids"].astype(str).tolist()
                    observed_gallery = blob["gallery_features_ids"].astype(str).tolist()
                    if observed_query_images != inputs["full_reference_ids"] or observed_gallery != inputs["gallery_ids"]:
                        raise ValueError("production Adapter feature row IDs differ from its native input rows")
                    arrays = {"query_features": (blob["queries"], protocol["query_ids"]),
                              "gallery_features": (blob["gallery"], protocol["gallery_ids"])}
                receipt(role, "adapter", started, arrays, adapter.batch_size,
                        {"worker_observation": primary_result["worker_observation"],
                         "image_ids": {"query_features": observed_query_images,
                                       "gallery_features": observed_gallery},
                         "runtime": _interpreter_runtime(_limn_interpreter(config))})
                (adapter_root / "adapter-observation.json").write_text(json.dumps({"status": "measured_phase_receipt", "adapter_result": primary_result}, sort_keys=True) + "\n", encoding="utf-8")
                continue
            image_root = Path(config.FASHIONIQ_ROOT) / "resized_image" / category
            query_paths = [str(image_root / f"{protocol['queries'][i]['reference_id']}.jpg") for i in qidx]
            query_texts = [protocol["queries"][i]["text"] for i in qidx]
            gallery_paths = [str(image_root / f"{protocol['gallery_ids'][i]}.jpg") for i in gidx]
            capture_root = role_root / role / "g1"
            queries, gallery = t4.adapter_selected_phase(adapter, query_paths=query_paths, query_texts=query_texts,
                                                         gallery_paths=gallery_paths, categories=[category] * len(qidx),
                                                         capture_root=capture_root)
            arrays = {"query_features": (queries, [protocol["query_ids"][i] for i in qidx]),
                      "gallery_features": (gallery, [protocol["gallery_ids"][i] for i in gidx])}
            receipt(role, "adapter", started, arrays, adapter.batch_size,
                    {"worker_observation": adapter._worker_observation,
                     "runtime": _interpreter_runtime(_limn_interpreter(config))})
    if side != "adapter":
        (root / "native" / category).mkdir(parents=True, exist_ok=True)
        if "native_adapter_native" not in native_results:
            raise RuntimeError("native child did not produce full source evaluator result")
        measured = native_results["native_adapter_native"]
        observation = {"status": "measured_phase_receipt", "native_result": measured,
                       "native_metrics": measured.get("native_metrics_recheck", {})}
        (root / "native" / category / "native-observation.json").write_text(
            json.dumps(observation, sort_keys=True) + "\n", encoding="utf-8")
def aggregate_category(root: Path, category: str) -> dict:
    plan, plan_path, plan_sha = _plan_inputs(root, category)
    role_root = root / "roles" / category
    receipts = {}
    for role in ROLES:
        receipt_path = role_root / f"{role}.receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing actual role receipt: {receipt_path}")
        receipts[role] = _read_json(receipt_path)
        if receipts[role].get("plan_sha256") != plan_sha:
            raise ValueError(f"role receipt plan mismatch: {role}")
        if role in ("native_adapter_native", "native_adapter_adapter"):
            archive = (root / "native" / category / f"g2_{category}_native_ranking.npz"
                       if role == "native_adapter_native" else
                       root / "adapter" / category / f"g2_{category}_adapter_ranking.npz")
            if receipts[role].get("ranking_archive_sha256") != _file_sha256(archive):
                raise ValueError(f"captured ranking archive digest mismatch: {role}")
    role_root = root / "roles" / category
    native_g1_path = root / "native" / category / f"g2_{category}_native.npz"
    adapter_g1_root = root / "adapter" / category / "g1"
    native_inputs = _read_json(root / "native" / category / f"g2_{category}_inputs.json")
    input_comparison = _compare_g1_inputs(root, category, native_inputs)
    adapter_result = _read_json(root / "adapter" / category / "adapter-observation.json")["adapter_result"]
    pixel_comparison = input_comparison["comparisons"]["query_pixels"]
    adapter_result["layer2_pixels"] = {"adapter_decode_vs_native_val_decode": pixel_comparison}
    adapter_result["layer2_input_identity"] = input_comparison
    if not pixel_comparison["bitwise_equal"]:
        raise ValueError("G1 pixel mismatch: Adapter pixels differ from native validation decode")
    target_pixel_comparison = input_comparison["comparisons"]["target_pixels"]
    if not target_pixel_comparison["bitwise_equal"]:
        raise ValueError("G1 target/gallery pixel mismatch: Adapter gallery pixels differ from native validation decode")
    g1 = {}
    g1_root = root / "raw" / category
    g1_root.mkdir(parents=True, exist_ok=True)
    for side, paths in (("native", [native_g1_path]),
                        ("adapter", sorted(adapter_g1_root.glob("query-*.npz")) +
                         sorted(adapter_g1_root.glob("gallery-*.npz")))):
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"missing actual bounded {side} G1 captures")
        records = []
        for index, source in enumerate(paths):
            with np.load(source, allow_pickle=False) as payload:
                if side == "native":
                    required = {"source_tensor", "target_tensor", "sample_token_ids", "sample_query_texts",
                                "sample_text_padding_mask", "causal_attn_mask_actual", "sample_query_ids",
                                "sample_target_ids", "sample_indices"}
                    if not required.issubset(payload.files):
                        raise ValueError(f"native G1 capture missing required sample arrays: {source}")
                elif "query-" in source.name:
                    required = {"pixels", "paths", "texts", "token_ids", "token_padding_mask_derived",
                                "causal_attn_mask_actual"}
                    if not required.issubset(payload.files):
                        raise ValueError(f"Adapter G1 query capture missing actual token/mask evidence: {source}")
                elif not {"pixels", "paths"}.issubset(payload.files):
                    raise ValueError(f"Adapter G1 gallery capture missing pixels/paths: {source}")
            suffix = source.name
            destination = g1_root / f"{side}-{index}-{suffix}"
            shutil.copy2(source, destination)
            records.append({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        g1[side] = {"capture_count": len(records), "captures": records,
                    "mask_semantics": "actual causal additive mask and separate boolean token padding mask"}
    g1["source_comparison"] = input_comparison
    t4 = _t4()
    scopes = {}
    for name, control in plan["controls"].items():
        reference_role = next(role for role in ROLES
                             if plan["runs"][role]["run_id"] == control["reference_run_id"])
        actual_role = next(role for role in ROLES
                           if plan["runs"][role]["run_id"] == control["actual_run_id"])
        records = {}
        for tensor, key in (("query", "query_features"), ("gallery", "gallery_features")):
            with (np.load(role_root / f"{reference_role}.npz", allow_pickle=False) as ref_blob,
                  np.load(role_root / f"{actual_role}.npz", allow_pickle=False) as actual_blob):
                reference, actual = ref_blob[key], actual_blob[key]
                ref_ids = ref_blob[f"{key}_ids"].astype(str).tolist()
                actual_ids = actual_blob[f"{key}_ids"].astype(str).tolist()
            if ref_ids != actual_ids:
                raise ValueError(f"V3 control {name}.{tensor} has different ordered IDs")
            component = t4.component_scope(name, control["dimension"], actual, reference,
                                           atol=control["tolerance"]["atol"],
                                           rtol=control["tolerance"]["rtol"])
            records[tensor] = {**control["scopes"][tensor], **component,
                               "coverage_scope": control["scopes"][tensor]["coverage"],
                               "reference_run_id": control["reference_run_id"],
                               "actual_run_id": control["actual_run_id"], "dtype": control["dtype"]}
        scopes[name] = records
    block = t4.v3_artifact_block(Path(plan_path.name), plan_sha, scopes=scopes, receipts=receipts)
    identity = plan["identity"]
    protocol = _protocol(resolve_config(), category)
    if (native_inputs["full_reference_ids"] != [query["reference_id"] for query in protocol["queries"]] or
            native_inputs["full_target_ids"] != protocol["target_ids"] or
            native_inputs["gallery_ids"] != protocol["gallery_ids"]):
        raise ValueError("native evaluator population differs from preregistered query/gallery order")
    if (len(native_inputs["full_query_paths"]) != identity["query_count"] or
            len(native_inputs["gallery_paths"]) != identity["gallery_count"]):
        raise ValueError("production Adapter population differs from native full population")
    with np.load(role_root / "native_adapter_adapter.npz", allow_pickle=False) as adapter_full:
        if (adapter_full["query_features_image_ids"].astype(str).tolist() !=
                [query["reference_id"] for query in protocol["queries"]] or
                adapter_full["gallery_features_image_ids"].astype(str).tolist() != protocol["gallery_ids"]):
            raise ValueError("production Adapter query/gallery IDs differ from native source evaluator")
    native_observation = _read_json(root / "native" / category / "native-observation.json")
    native_result = native_observation["native_result"]
    ranking = adapter_result["scoring_ranking"]
    native_ranking_path = root / "native" / category / f"g2_{category}_native_ranking.npz"
    with np.load(native_ranking_path, allow_pickle=False) as rank_blob:
        source_target_ranks = rank_blob["source_target_ranks"].astype(int).tolist()
    native_ranks = list(map(int, source_target_ranks))
    adapter_ranks = list(map(int, ranking["adapter_source_target_ranks"]))
    if len(native_ranks) != identity["query_count"] or len(adapter_ranks) != identity["query_count"]:
        raise ValueError("full metric rank vectors differ from preregistered query population")

    def side_metrics(values):
        return {"query_count": len(values), "hits": {f"r{k}": sum(rank <= k for rank in values)
                for k in (1, 10, 50)}}

    native_metrics = side_metrics(native_ranks)
    adapter_metrics = side_metrics(adapter_ranks)
    equality = {f"r{k}": native_metrics["hits"][f"r{k}"] == adapter_metrics["hits"][f"r{k}"]
                for k in (1, 10, 50)}
    scoring_labels = {"query_targets": [[query_id, target_id] for query_id, target_id in
                                        zip(protocol["query_ids"], protocol["target_ids"], strict=True)],
                      "reference_exclusions": [[query_id, [protocol["queries"][index]["reference_id"]]]
                                               for index, query_id in enumerate(protocol["query_ids"])]}
    proof_name = f"g1_g2.limn.{category}.native.score-ranks.jsonl"
    proof_path = root / proof_name
    with (np.load(native_ranking_path, allow_pickle=False) as native_blob,
          np.load(root / "adapter" / category / f"g2_{category}_adapter_ranking.npz", allow_pickle=False) as adapter_blob):
        native_scores = native_blob["source_score_rows"]
        adapter_scores = adapter_blob["adapter_masked_score_rows"]
        source_query_ids = native_blob["query_ids"].astype(str).tolist()
        source_gallery_ids = native_blob["gallery_ids"].astype(str).tolist()
        query_ids = list(protocol["query_ids"])
        gallery_ids = source_gallery_ids
        expected_source_query_ids = [query["reference_id"] for query in protocol["queries"]]
        native_orders = native_blob["source_orders"]
        adapter_orders = adapter_blob["adapter_source_orders"]
        if (adapter_blob["query_ids"].astype(str).tolist() != expected_source_query_ids or
                adapter_blob["gallery_ids"].astype(str).tolist() != gallery_ids or
                source_query_ids != expected_source_query_ids or gallery_ids != protocol["gallery_ids"]):
            raise ValueError("native/Adapter raw ranking IDs differ from preregistered order")
        if (native_scores.shape != (len(query_ids), len(gallery_ids))
                or adapter_scores.shape != native_scores.shape
                or native_orders.shape != native_scores.shape
                or adapter_orders.shape != native_scores.shape
                or not np.isfinite(native_scores).all()
                or not np.isfinite(adapter_scores).all()):
            raise ValueError("native/Adapter raw score and order arrays are malformed")
        with proof_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": "cir.score-ranks/v1", "identity": identity,
                "query_ids": query_ids, "gallery_ids": gallery_ids,
                "query_targets": scoring_labels["query_targets"],
                "reference_exclusions": scoring_labels["reference_exclusions"]},
                ensure_ascii=False, separators=(",", ":")) + "\n")
            for index, query_id in enumerate(query_ids):
                target_id = protocol["target_ids"][index]
                target_position = gallery_ids.index(target_id)
                excluded = scoring_labels["reference_exclusions"][index][1]
                stream.write(json.dumps({"query_id": query_id, "target_id": target_id,
                    "target_position": target_position, "excluded_gallery_ids": excluded,
                    "native_scores": native_scores[index].astype(float).tolist(),
                    "adapter_scores": adapter_scores[index].astype(float).tolist(),
                    "native_order": native_orders[index].astype(int).tolist(),
                    "adapter_order": adapter_orders[index].astype(int).tolist()},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
    block["scoring_proof_path"] = proof_name
    block["scoring_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    score_receipts = {}
    for role, side in (("native_adapter_native", "native"), ("native_adapter_adapter", "adapter")):
        rows = []
        for index, query_id in enumerate(protocol["query_ids"]):
            scores = native_scores[index].astype(float).tolist() if side == "native" else adapter_scores[index].astype(float).tolist()
            order = native_orders[index].astype(int).tolist() if side == "native" else adapter_orders[index].astype(int).tolist()
            rows.append(json.dumps({"query_id": query_id, "scores": scores, "order": order},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")
        run = plan["runs"][role]
        score_receipts[role] = {"run_id": run["run_id"], "source_id": run["source_id"],
            "side": side, "identity": run["identity"],
            "score_order_sha256": hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()}
    block["scoring_run_receipts"] = score_receipts
    metric_detail = {"declared_rule": "source native ranks and Adapter-feature source-policy ranks under same native reference mask",
                     "cutoffs": [1, 10, 50], "reference_mask": "native source -10e10 exclusion applied to both feature matrices",
                     "ranking_convention": "native source and Adapter features ranked with NumPy argsort(-scores) after the same native reference mask; Adapter unmasked tie-stable diagnostic retained separately",
                     "eligible_query_set_identical": (
                         native_inputs["full_reference_ids"] == [q["reference_id"] for q in protocol["queries"]] and
                         native_inputs["full_target_ids"] == protocol["target_ids"] and
                         native_inputs["gallery_ids"] == protocol["gallery_ids"]),
                     "eligible_query_set_evidence": "native references, targets, and gallery IDs checked against ordered protocol and Adapter feature rows",
                     "equality": equality,
                     "native": {**native_metrics, "query_count": identity["query_count"]},
                     "adapter": {**adapter_metrics, "query_count": identity["query_count"]}}
    native = {"model_id": "limn", "category": category, "standard": STANDARD,
              "identity": identity, "checkpoint_sha256": identity["checkpoint"],
              "protocol": identity["protocol"], "protocol_id": identity["protocol"],
              "checkpoint_id": identity["checkpoint"],
              "query_order_sha256": identity["query_order_sha256"],
              "gallery_order_sha256": identity["gallery_order_sha256"],
              "tolerances": {"score_max_abs": TOLERANCES["score_max_abs"],
                             "tie_epsilon": TOLERANCES["tie_epsilon"]},
              "native": {"layer1_identity": native_result},
              "native_metrics_parity_detail": metric_detail,
              "native_metrics_match_feature_rerank": all(equality.values()), "g1_capture": g1, "parity_v3": block}
    adapter = {"model_id": "limn", "category": category, "standard": STANDARD,
               "identity": identity, "checkpoint_sha256": identity["checkpoint"],
               "protocol": identity["protocol"], "protocol_id": identity["protocol"],
               "checkpoint_id": identity["checkpoint"],
               "query_order_sha256": identity["query_order_sha256"],
               "gallery_order_sha256": identity["gallery_order_sha256"],
              "tolerances": {"score_max_abs": TOLERANCES["score_max_abs"],
                             "tie_epsilon": TOLERANCES["tie_epsilon"]},
               "adapter": adapter_result, "g1_capture": g1, "scoring_ranking": ranking}
    staging = root / "staging" / category
    staging.mkdir(parents=True, exist_ok=True)
    for side, artifact in (("native", native), ("adapter", adapter)):
        (staging / f"g1_g2.limn.{category}.{side}.json").write_text(
            json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return {"native": native, "adapter": adapter, "plan_path": plan_path,
            "proof_paths": {proof_name: proof_path}}

def publish(root: Path, publication_root: Path) -> Path:
    gate = _gate_module()
    artifacts, plans, proofs = {}, {}, {}
    for category in CATEGORIES:
        result = aggregate_category(root, category)
        for side in ("native", "adapter"):
            artifacts[f"g1_g2.limn.{category}.{side}.json"] = result[side]
        plans[f"g1_g2.limn.{category}.v3-plan.json"] = result["plan_path"]
        proofs.update(result["proof_paths"])
    generation = "limn-v3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_generation = publication_root / "raw" / generation
    raw_generation.mkdir(parents=True, exist_ok=False)
    raw_hashes = {}
    for category in CATEGORIES:
        for side in ("native", "adapter"):
            record = artifacts[f"g1_g2.limn.{category}.{side}.json"]["g1_capture"][side]
            for capture in record["captures"]:
                source = Path(capture["path"])
                destination = raw_generation / category / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if digest != capture["sha256"]:
                    raise ValueError(f"raw G1 capture changed during generation staging: {source}")
                capture["path"] = str(destination)
                raw_hashes[str(destination.relative_to(publication_root))] = digest
    for category in CATEGORIES:
        for side, filename in (("native", f"g2_{category}_native_ranking.npz"),
                               ("adapter", f"g2_{category}_adapter_ranking.npz")):
            source = root / side / category / filename
            if not source.is_file():
                raise FileNotFoundError(f"missing actual {side} ranking evidence: {source}")
            destination = raw_generation / category / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            raw_hashes[str(destination.relative_to(publication_root))] = digest
            evidence = artifacts[f"g1_g2.limn.{category}.{side}.json"]
            evidence["ranking_evidence"] = {"path": str(destination.relative_to(publication_root)),
                                             "sha256": digest}
    staging = root / "publication-staging" / generation
    staging.mkdir(parents=True, exist_ok=False)
    for name, source in plans.items():
        shutil.copy2(source, staging / name)
    for name, source in proofs.items():
        shutil.copy2(source, staging / name)
    for name, payload in artifacts.items():
        (staging / name).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                                    encoding="utf-8")
    verdict = gate.evaluate_model("limn", root=staging, standard=STANDARD)
    verdict_data = verdict.to_dict()
    absent_mandatory = [check.name for check in verdict.checks if check.mandatory and check.absence]
    if (verdict.status in (gate.NOT_RUN, gate.EVIDENCE_MISSING, gate.INCOMPLETE, gate.MALFORMED)
            or verdict.missing or absent_mandatory):
        raise RuntimeError(f"refusing publication with incomplete or malformed LIMN evidence: status={verdict.status}, missing={verdict.missing}, absent_mandatory={absent_mandatory}")
    manifest = {"generation": generation, "model_id": "limn", "standard": STANDARD,
                "categories": list(CATEGORIES), "artifacts": sorted(artifacts),
                "plans": sorted(plans), "raw_assets": raw_hashes, "verdict": verdict_data,
                "created_at": datetime.now(timezone.utc).isoformat()}
    manifest_root = publication_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{generation}.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
    (staging / "limn-v3-verdict.json").write_text(
        json.dumps(verdict_data, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "limn-v3-generation-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return _t4().publish_v3_generation(publication_root, generation, artifacts,
                                       plan_paths=plans, proof_paths=proofs)


def _gate_module():
    import workbench.backend.parity_gate as gate
    return gate
def plan_only(root: Path, category: str) -> None:
    config, t4 = resolve_config(), _t4()
    plan, digest, path = _plan(t4, config, root, category)
    print(json.dumps({"plan_path": str(path), "plan_sha256": digest, "category": category,
                      "roles": list(plan["runs"]), "query_count": plan["identity"]["query_count"],
                      "gallery_count": plan["identity"]["gallery_count"], "cpu_only": True,
                      "cuda_execution": "blocked until Main lease"}, indent=2))




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "roles", "roles-native", "roles-adapter", "aggregate", "publish"), required=True)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--gpu-lease")
    parser.add_argument("--publication-root", type=Path)
    args = parser.parse_args(argv)
    root = args.run_root.expanduser().resolve()
    if args.phase == "publish":
        if args.publication_root is None:
            parser.error("--publication-root required for publish")
        print(str(publish(root, args.publication_root)))
        return 0
    if args.category is None:
        parser.error("--category required for this phase")
    if args.phase == "plan":
        plan_only(root, args.category)
    elif args.phase == "roles":
        execute_roles(root, args.category, args.gpu_lease)
    elif args.phase == "roles-native":
        _execute_roles_local(root, args.category, args.gpu_lease, "native")
    elif args.phase == "roles-adapter":
        _execute_roles_local(root, args.category, args.gpu_lease, "adapter")
    elif args.phase == "aggregate":
        print(json.dumps(aggregate_category(root, args.category), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
