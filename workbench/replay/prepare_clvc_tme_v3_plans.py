"""Pre-register native/adapter CLVC-Net and TME V3 roles without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN_ROOT = ROOT / "workbench/artifacts/parity/v3/clvc-tme/20261121-edge-g1"
CATEGORIES = ("dress", "shirt", "toptee")
ROLES = (
    "native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
    "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
    "adapter_batch_1", "adapter_batch_n",
)
CONTROLS = {
    "native_repeat": ("native_repeat_1", "native_repeat_2"),
    "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2"),
    "matched_native_adapter": ("native_adapter_native", "native_adapter_adapter"),
    "native_B1_vs_BN": ("native_batch_1", "native_batch_n"),
    "adapter_B1_vs_BN": ("adapter_batch_1", "adapter_batch_n"),
}
V3_CONTROL_ROLES = {
    "native_repeat": ("native_repeat_1", "native_repeat_2"),
    "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2"),
    "native_adapter": ("native_adapter_native", "native_adapter_adapter"),
    "native_batch_1_vs_n": ("native_batch_1", "native_batch_n"),
    "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n"),
}
IDENTITY_FIELDS = (
    "checkpoint", "configuration", "environment", "protocol", "category", "cohort_id",
    "precision", "seed", "query_order_sha256", "gallery_order_sha256",
    "reference_exclusions_sha256", "query_count", "gallery_count",
)



def _control_scope(coverage: str, population_count: int, ids: list[str], cohort_id: str,
                  component_shape: tuple[int, ...]) -> dict[str, Any]:
    count = len(ids)
    if (not ids or len(ids) != len(set(ids)) or not component_shape or
            any(dimension <= 0 for dimension in component_shape) or
            (coverage == "sample" and count >= population_count)):
        raise ValueError("control scope must contain ordered unique IDs and positive component dimensions")
    components = 1
    for dimension in component_shape:
        components *= dimension
    scope = {"coverage": coverage, "population_count": population_count,
             "tensor_shape": [count, *component_shape],
             "component_shape": [count, components],
             "ordered_ids_sha256": _digest(ids), "cohort_id": cohort_id}
    if coverage == "sample":
        scope["sample_count"] = count
    return scope

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error


def _load_checkpoint_hashes() -> dict[str, str]:
    checkpoint_root = ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints"
    paths = {
        name: checkpoint_root / "clvc" / name
        for name in (
            "dress_local_model.pt", "dress_global_model.pt", "shirt_local_model.pt",
            "shirt_global_model.pt", "toptee_local_model.pt", "toptee_global_model.pt",
        )
    }
    paths["fashioniq_0_best_model.pth"] = (
        checkpoint_root.parent / "cohort/downloads/tme/fashioniq_0_best_model.pth")
    found = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"checkpoint missing or empty: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        found[name] = digest.hexdigest()
    return found


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(model_id: str) -> dict[str, str]:
    adapter = ROOT / f"workbench/composition/model_{model_id}.py"
    replay = ROOT / ("workbench/replay/clvc_net.py" if model_id == "clvc_net" else "workbench/replay/tme.py")
    producer = ROOT / "workbench/scripts/clvc_tme_v3_parity.py"
    plan_builder = ROOT / "workbench/replay/prepare_clvc_tme_v3_plans.py"
    paths = ([ROOT / "workbench/third_party/CLVCNet/test.py",
              ROOT / "workbench/third_party/CLVCNet/datasets.py",
              ROOT / "workbench/third_party/CLVCNet/model.py",
              ROOT / "workbench/third_party/CLVCNet/train.py", replay, adapter, producer, plan_builder]
             if model_id == "clvc_net" else
             [ROOT / "workbench/third_party/TME/src/precompute_evaluation.py",
              ROOT / "workbench/third_party/TME/src/precompute_test.py",
              ROOT / "workbench/third_party/TME/src/data.py",
              ROOT / "workbench/third_party/TME/src/precompute.py", replay, adapter, producer, plan_builder])
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths}


def _environment_identity(model_id: str) -> dict[str, str]:
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    config = resolve_config()
    interpreter = resolve_model_interpreter(model_by_id(model_id), config)
    if not interpreter or not Path(interpreter).is_file():
        raise ValueError(f"registered model interpreter is unavailable: {model_id}")
    path = Path(interpreter).resolve()
    return {"python": str(path), "python_sha256": _sha256_file(path)}


def _clvc_rows(category: str) -> tuple[list[str], list[str]]:
    rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                      f"CLVC {category} captions")
    query_ids = ["clvc:" + hashlib.sha256(json.dumps(
        {"category": category, **row}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest() for row in rows]
    gallery_ids: list[str] = []
    for row in rows:
        for image_id in (row["candidate"], row["target"]):
            if image_id not in gallery_ids:
                gallery_ids.append(image_id)
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError(f"duplicate ordered native CLVC IDs: {category}")
    return query_ids, gallery_ids

def _clvc_reference_exclusions(category: str, query_ids: list[str]) -> list[dict[str, Any]]:
    rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                      f"CLVC {category} reference rows")
    if len(rows) != len(query_ids):
        raise ValueError(f"CLVC {category} reference/query counts differ")
    return [{"query_id": query_id, "reference_id": row["candidate"], "score": -10e10}
            for query_id, row in zip(query_ids, rows, strict=True)]


def _tme_rows(category: str) -> tuple[list[str], list[str]]:
    rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                      f"TME {category} captions")
    galleries = _read_json(ROOT / "data/FashionIQ/image_splits" / f"split.{category}.val.json",
                           f"TME {category} native gallery")
    query_ids = ["tme:" + category + ":" + _digest(
        {key: row[key] for key in ("candidate", "target", "captions")}
    ) for row in rows]
    if len(set(query_ids)) != len(query_ids) or len(set(galleries)) != len(galleries):
        raise ValueError(f"duplicate ordered native TME {category} IDs")
    return query_ids, galleries


def _g1_selection(model_id: str, category: str, query_ids: list[str], gallery_ids: list[str]) -> dict[str, list[str]]:
    if model_id == "clvc_net":
        rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                          f"CLVC {category} G1 boundary rows")
        references = [row["candidate"] for row in rows]
        targets = [row["target"] for row in rows]
    else:
        rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                          f"TME {category} G1 boundary rows")
        references = [row["candidate"] for row in rows]
        targets = [row["target"] for row in rows]
    if len(rows) != len(query_ids):
        raise ValueError(f"{model_id} {category} G1 query count differs from source rows")
    positions = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if any(image_id not in positions for image_id in references + targets):
        raise ValueError(f"{model_id} {category} G1 source pair outside gallery")
    boundary_positions = {0, len(gallery_ids) - 1}
    boundary_queries = [query_id for query_id, reference_id, target_id
                        in zip(query_ids, references, targets, strict=True)
                        if positions[reference_id] in boundary_positions or
                        positions[target_id] in boundary_positions]
    selected = list(dict.fromkeys([query_ids[0], *boundary_queries[:3], query_ids[-1]]))
    selected_set = set(selected)
    query_order = [query_id for query_id in query_ids if query_id in selected_set]
    selected_rows = [row for query_id, row in zip(query_ids, rows, strict=True)
                     if query_id in selected_set]
    sample_gallery = list(dict.fromkeys(
        [image_id for row in selected_rows for image_id in (row["candidate"], row["target"])] +
        gallery_ids[:1] + gallery_ids[-1:]))
    if len(sample_gallery) > 32:
        raise ValueError(f"{model_id} {category} G1 gallery exceeds native capture bound")
    return {"queries": query_order, "gallery": sample_gallery}


def _control_identity(identity: dict[str, Any], query_ids: list[str],
                      gallery_ids: list[str], cohort_id: str,
                      scoring_labels: dict[str, Any]) -> dict[str, Any]:
    selected = set(query_ids)
    labels = {key: [row for row in rows if row[0] in selected]
              for key, rows in scoring_labels.items()}
    return {**identity, "cohort_id": cohort_id,
            "query_order_sha256": _digest(query_ids),
            "gallery_order_sha256": _digest(gallery_ids),
            "reference_exclusions_sha256": _digest(labels),
            "query_count": len(query_ids), "gallery_count": len(gallery_ids)}


def _native_replay_command(argv: list[str], python: Path) -> str:
    invocation_id = f'{shlex.quote(str(python))} -c "import uuid; print(uuid.uuid4())"'
    return (f'invocation_id=$({invocation_id}); ' +
            " ".join(shlex.quote(value) for value in argv) +
            ' --invocation-id "$invocation_id"')

def _role_source(model_id: str, category: str, role: str, side: str,
                 identity: dict[str, Any], declared_upstream_ref: str) -> str:
    return f"{side}:{model_id}:{category}:{role}:{_digest({'identity': identity, 'declared_upstream_ref': declared_upstream_ref, 'role': role, 'schema': 'clvc-tme-native-v3'})}"


def _plan(model_id: str, category: str, hashes: dict[str, str],
          plan_root: Path = PLAN_ROOT) -> dict[str, Any]:
    if model_id == "clvc_net":
        query_ids, gallery_ids = _clvc_rows(category)
        checkpoint = [hashes[f"{category}_local_model.pt"], hashes[f"{category}_global_model.pt"]]
        protocol = "fashioniq_val_split; ordered pair-union gallery; per-query reference score -10e10"
        declared_upstream_ref = "bd9b6889489806cb60676597880f91d918b279cb"
        environment = "WORKBENCH_CLVC_PYTHON; adapted sm_120 runtime only when explicitly declared"
        batch, gallery_batch = 32, 32
    elif model_id == "tme":
        query_ids, gallery_ids = _tme_rows(category)
        checkpoint = [hashes["fashioniq_0_best_model.pth"]]
        protocol = f"fashioniq_val_split; source all-three-category API; {category} split gallery; references eligible"
        declared_upstream_ref = "394aad79df2c104feb32a2c96132582012765062"
        environment = "WORKBENCH_TME_PYTHON; mmap val image feature cache"
        batch, gallery_batch = 32, 256
    else:
        raise ValueError(f"unsupported model: {model_id}")
    plan_root = plan_root.resolve()
    selection_path = plan_root / f"{model_id}.{category}.capture-selection.json"
    plan_root.mkdir(parents=True, exist_ok=True)
    selection = _g1_selection(model_id, category, query_ids, gallery_ids)
    selection_raw = json.dumps(selection, sort_keys=True) + "\n"
    if selection_path.exists():
        if selection_path.read_text(encoding="utf-8") != selection_raw:
            raise FileExistsError(f"immutable capture selection differs: {selection_path}")
    else:
        selection_path.write_text(selection_raw, encoding="utf-8")
    identity = {
        "checkpoint": "sha256:" + _digest(checkpoint),
        "configuration": _digest({"model": model_id, "declared_upstream_ref": declared_upstream_ref,
                                   "source_files": _source_identity(model_id)}),
        "environment": _digest(_environment_identity(model_id)),
        "protocol": protocol, "category": category,
        "cohort_id": f"fashioniq-val-{model_id}-{category}", "precision": "float32", "seed": 0,
        "query_order_sha256": _digest(query_ids), "gallery_order_sha256": _digest(gallery_ids),
        "reference_exclusions_sha256": _digest(
            _clvc_reference_exclusions(category, query_ids) if model_id == "clvc_net" else []),
        "query_count": len(query_ids), "gallery_count": len(gallery_ids),
    }
    source_rows = _read_json(ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json",
                             f"{model_id} {category} scoring labels")
    if len(source_rows) != len(query_ids):
        raise ValueError(f"{model_id} {category} scoring label count differs from source queries")
    references = [row["candidate"] for row in source_rows]
    targets = [row["target"] for row in source_rows]
    scoring_labels = {
        "query_targets": [[query_id, target_id]
                          for query_id, target_id in zip(query_ids, targets, strict=True)],
        "reference_exclusions": [[query_id, [reference_id]]
                                  for query_id, reference_id in zip(query_ids, references, strict=True)
                                  if model_id == "clvc_net"],
    }
    identity["reference_exclusions_sha256"] = hashlib.sha256(
        json.dumps(scoring_labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    scoring_proof_path = f"g1_g2.{model_id}.{category}.native.score-ranks.jsonl"
    runs: dict[str, dict[str, Any]] = {}
    for role_name in ROLES:
        side = "adapter" if role_name.startswith("adapter") or role_name == "native_adapter_adapter" else "native"
        role_batch = 1 if role_name.endswith("batch_1") else batch
        role_gallery_batch = 1 if role_name.endswith("batch_1") else gallery_batch
        full_role = role_name in {"native_adapter_native", "native_adapter_adapter"}
        sample_query_ids = []
        if not full_role:
            selected = set(selection["queries"])
            selected.update(query_ids[round(i * (len(query_ids) - 1) / 26)] for i in range(27))
            sample_query_ids = [query_id for query_id in query_ids if query_id in selected]
        role_queries = query_ids if full_role or model_id == "clvc_net" else sample_query_ids
        role_identity = identity if model_id == "clvc_net" else _control_identity(
            identity, role_queries, gallery_ids,
            identity["cohort_id"] if full_role else
            f"{identity['cohort_id']}-control-{_digest([role_queries, gallery_ids])}",
            scoring_labels)
        partition_id = (f"{model_id}:{category}:full-population-v3" if full_role else
                        f"{model_id}:{category}:sample32-v1")
        execution = {"partition_id": partition_id,
                     "query_partition_sha256": role_identity["query_order_sha256"],
                     "gallery_partition_sha256": role_identity["gallery_order_sha256"],
                     "query_batch_size": role_batch, "gallery_batch_size": role_gallery_batch,
                     "memory_policy": "full native source population; G1 selection is capture-only"}
        if not full_role:
            execution.update(sample_query_ids=sample_query_ids,
                             sample_query_sha256=_digest(sample_query_ids),
                             sample_query_count=len(sample_query_ids))
        runs[role_name] = {"run_id": f"{model_id}:{category}:{role_name}", "side": side,
                           "source_id": _role_source(model_id, category, role_name, side, role_identity, declared_upstream_ref),
                           "identity": role_identity, "execution": execution}

    control_roles = V3_CONTROL_ROLES if model_id == "tme" else CONTROLS
    controls: dict[str, dict[str, Any]] = {}
    if model_id == "clvc_net":
        controls = {name: {"reference_role": pair[0], "actual_role": pair[1]}
                    for name, pair in control_roles.items()}
    for name, (reference_role, actual_role) in (V3_CONTROL_ROLES.items() if model_id == "tme" else ()):
        reference = runs[reference_role]
        actual = runs[actual_role]
        if reference["side"] != actual["side"] and name != "native_adapter":
            raise ValueError(f"control roles have different sides: {name}")
        coverage = "full" if name == "native_adapter" else "sample"
        query_ids_for_control = (query_ids if coverage == "full" else
                                 reference["execution"]["sample_query_ids"])
        if query_ids_for_control != (query_ids if coverage == "full" else
                                     actual["execution"]["sample_query_ids"]):
            raise ValueError(f"control roles have different selected query IDs: {name}")
        cohort_id = reference["identity"]["cohort_id"]
        batch = "batch" in name
        controls[name] = {
            "reference_run_id": reference["run_id"], "actual_run_id": actual["run_id"],
            "reference_side": reference["side"],
            "dimension": "batch_parity" if batch else "forward_parity",
            "dtype": "float32",
            "tolerance": {"atol": 1e-4 if batch else 1e-5,
                          "rtol": 1e-3 if batch else 1e-4},
            "scopes": {
                "query": _control_scope(coverage, len(query_ids), query_ids_for_control,
                    cohort_id, (256,)),
                "gallery": _control_scope("full", len(gallery_ids), gallery_ids,
                    cohort_id, (32, 256)),
            },
        }
    checkpoint_root = (ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc"
                       if model_id == "clvc_net" else
                       ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/tme")
    source_root = ROOT / ("workbench/third_party/CLVCNet" if model_id == "clvc_net" else "workbench/third_party/TME")
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    config = resolve_config()
    python_value = resolve_model_interpreter(model_by_id(model_id), config)
    if not python_value or not Path(python_value).is_file():
        raise ValueError(f"registered model interpreter is unavailable: {model_id}")
    python = Path(python_value).resolve()
    replay = ROOT / ("workbench/replay/clvc_net.py" if model_id == "clvc_net" else "workbench/replay/tme.py")
    plan_path = plan_root / f"g1_g2.{model_id}.{category}.v3-plan.json"
    run_commands: dict[str, dict[str, str]] = {}
    for role_name, role in runs.items():
        result_output = plan_root / "adapter" / model_id / category / f"{role_name}.npz"
        sample_selection_path = None
        if role["execution"].get("sample_query_count"):
            sample_selection_path = plan_root / "sample-queries" / model_id / category / f"{role_name}.json"
            sample_selection_path.parent.mkdir(parents=True, exist_ok=True)
            sample_raw = json.dumps(role["execution"]["sample_query_ids"], separators=(",", ":")) + "\n"
            if sample_selection_path.exists() and sample_selection_path.read_text(encoding="utf-8") != sample_raw:
                raise FileExistsError(f"immutable sample query selection differs: {sample_selection_path}")
            if not sample_selection_path.exists():
                sample_selection_path.write_text(sample_raw, encoding="utf-8")
        if role["side"] == "native":
            capture = plan_root / "native" / model_id / category / role_name
            native_capture = capture / f"{category}.npz"
            argv = [str(python), str(replay), "--checkpoint-root", str(checkpoint_root),
                    "--dataset-root", str(ROOT / "data/FashionIQ"), "--source-root", str(source_root),
                    "--output", str(capture / "result.json")]
            if model_id == "clvc_net":
                argv.extend(["--category", category, "--batch-size", str(role["execution"]["query_batch_size"])])
            else:
                argv.extend(["--checkpoint-id", "fashioniq_0"])
                argv = ["env", f"WORKBENCH_TME_QUERY_BS={role['execution']['query_batch_size']}",
                        f"WORKBENCH_TME_QFORMER_BS={role['execution']['gallery_batch_size']}", *argv]
            if sample_selection_path is not None:
                argv.extend(["--sample-queries", str(sample_selection_path)])
            command = (f'test ! -e {shlex.quote(str(capture))} && mkdir -p {shlex.quote(str(capture))} && ' +
                       f'invocation_id=$({shlex.quote(str(python))} -c ' +
                       '"import uuid; print(uuid.uuid4())") && export invocation_id && ' +
                       _native_replay_command(argv, python))
        else:
            argv = [str(python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"),
                    "--phase", "adapter", "--plan", str(plan_path), "--role", role_name,
                    "--output", str(result_output)]
            if sample_selection_path is not None:
                argv.extend(["--sample-queries", str(sample_selection_path)])
            command = " ".join(shlex.quote(value) for value in argv)
        unit = f"cir-{model_id}-{category}-{role_name}".replace("_", "-")
        guarded = ("set -euo pipefail; "
                   f"cg=$(systemctl --user show --property=ControlGroup --value {shlex.quote(unit + '.service')}); "
                   "test -n \"$cg\"; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.max\")\" = 12884901888; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.swap.max\")\" = 0; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.oom.group\")\" = 1; "
                   f"flock -x /tmp/cir-fix-all-20261120-cpu.lock -c {shlex.quote(command)}")
        run_commands[role_name] = {"command": (f"systemd-run --user --wait --pipe --collect --same-dir --service-type=exec --unit={unit} -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=kill env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH={ROOT} {ROOT / 'workbench/scripts/gpu_job.sh'} /bin/bash --noprofile --norc -c {shlex.quote(guarded)}"), "side": role["side"], "output": str(result_output)}
    native_role = run_commands["native_adapter_native"]
    native_role["convert_command"] = " ".join(shlex.quote(value) for value in [str(python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"), "--phase", "native", "--plan", str(plan_path), "--capture", str(plan_root / "native" / model_id / category / "native_adapter_native" / f"{category}.npz"), "--role", "native_adapter_native", "--output-root", str(plan_root)])
    compare_role = run_commands["native_adapter_adapter"]
    compare_role["compare_command"] = " ".join(shlex.quote(value) for value in [str(python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"), "--phase", "compare", "--plan", str(plan_path), "--native", str(plan_root / "native" / model_id / category / "native_adapter_native" / f"{category}.npz"), "--adapter", str(plan_root / "adapter" / model_id / category / "native_adapter_adapter.npz"), "--output", str(plan_root / "comparisons" / model_id / category / "native_adapter.json")])
    return {"standard": "parity-absolute-relative/v3", "model_id": model_id,
            "category": category, "declared_upstream_ref": declared_upstream_ref,
            "checkpoint_hashes": checkpoint, "identity": identity,
            "scoring_proof_path": scoring_proof_path, "scoring_labels": scoring_labels,
            "capture_selection": str(selection_path), "runs": runs,
            "controls": controls,
            "scopes": {"full": True, "G1": {"query_max": 32, "gallery_max": 32}, "scores": True, "ranks": True, "metrics": True},
            "resource_budget": {"score_row_ram_bytes": identity["gallery_count"] * 4,
                                "disk_policy": "versioned immutable generation; mmap full features; refuse overwrite",
                                "gpu_policy": "one model process; native exits before adapter; GPU lock required"},
            "commands": {"roles": run_commands}}


def _tme_family(plans: dict[str, dict[str, Any]], plan_paths: dict[str, Path]) -> dict[str, Any]:
    if set(plans) != set(CATEGORIES) or set(plan_paths) != set(CATEGORIES):
        raise ValueError("TME family requires exactly the three category plans")
    roles: dict[str, Any] = {}
    for role_name in ROLES:
        role_sides = {plans[category]["runs"][role_name]["side"] for category in CATEGORIES}
        if len(role_sides) != 1:
            raise ValueError(f"TME family role has category-dependent sides: {role_name}")
        execution = {category: plans[category]["runs"][role_name]["execution"]
                     for category in CATEGORIES}
        query_ids = {}
        for category in CATEGORIES:
            execution_for_category = execution[category]
            if "sample_query_ids" in execution_for_category:
                selected_ids = execution_for_category["sample_query_ids"]
            elif role_name in {"native_adapter_native", "native_adapter_adapter"}:
                selected_ids = _tme_rows(category)[0]
            else:
                raise ValueError(f"TME family sample query IDs are missing: {role_name}/{category}")
            query_ids[category] = selected_ids
        if any(not ids or len(ids) != len(set(ids)) for ids in query_ids.values()):
            raise ValueError(f"TME family role needs non-empty unique category queries: {role_name}")
        batch_pairs = {(item["query_batch_size"], item["gallery_batch_size"])
                       for item in execution.values()}
        if len(batch_pairs) != 1:
            raise ValueError(f"TME family role has category-dependent batch sizes: {role_name}")
        roles[role_name] = {"side": plans[CATEGORIES[0]]["runs"][role_name]["side"],
                            "execution": execution, "query_ids": query_ids,
                            "run_ids": {category: plans[category]["runs"][role_name]["run_id"]
                                        for category in CATEGORIES},
                            "source_ids": {category: plans[category]["runs"][role_name]["source_id"]
                                           for category in CATEGORIES},
                            "identities": {category: plans[category]["runs"][role_name]["identity"]
                                            for category in CATEGORIES}}
    family_root = next(iter(plan_paths.values())).parent
    selection = {category: _g1_selection("tme", category, _tme_rows(category)[0], _tme_rows(category)[1])
                 for category in CATEGORIES}
    selection_path = family_root / "tme-family.capture-selection.json"
    selection_raw = json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n"
    if selection_path.exists() and selection_path.read_text(encoding="utf-8") != selection_raw:
        raise FileExistsError(f"immutable TME family capture selection differs: {selection_path}")
    if not selection_path.exists():
        selection_path.write_text(selection_raw, encoding="utf-8")
    family_sample_paths: dict[str, Path] = {}
    family_capture_root = family_root / "native" / "tme-family"
    for role_name in ROLES:
        sample_map = roles[role_name]["query_ids"]
        sample_map_path = family_root / "sample-queries" / "tme-family" / f"{role_name}.json"
        sample_raw = json.dumps(sample_map, sort_keys=True, separators=(",", ":")) + "\n"
        if sample_map_path.exists() and sample_map_path.read_text(encoding="utf-8") != sample_raw:
            raise FileExistsError(f"immutable TME family sample map differs: {sample_map_path}")
        if not sample_map_path.exists():
            sample_map_path.parent.mkdir(parents=True, exist_ok=True)
            sample_map_path.write_text(sample_raw, encoding="utf-8")
        family_sample_paths[role_name] = sample_map_path
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    runtime_python_value = resolve_model_interpreter(model_by_id("tme"), resolve_config())
    if not runtime_python_value or not Path(runtime_python_value).is_file():
        raise ValueError("registered TME interpreter is unavailable")
    native_python = Path(runtime_python_value).resolve()
    adapter_python = ROOT / "workbench/envs/ctl312h/bin/python"
    if not adapter_python.is_file():
        raise ValueError(f"ctl312h Adapter controller is unavailable: {adapter_python}")
    commands: dict[str, str] = {}
    for role_name in ROLES:
        role = roles[role_name]
        if role["side"] == "native":
            native_dir = family_capture_root / role_name
            argv = [str(native_python), str(ROOT / "workbench/replay/tme.py"),
                    "--checkpoint-root", str(ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/tme"),
                    "--dataset-root", str(ROOT / "data/FashionIQ"),
                    "--source-root", str(ROOT / "workbench/third_party/TME"),
                    "--checkpoint-id", "fashioniq_0", "--capture-root", str(native_dir),
                    "--capture-selection", str(selection_path),
                    "--output", str(native_dir / "result.json")]
            execution = role["execution"]["dress"]
            argv = ["env", f"WORKBENCH_TME_QUERY_BS={execution['query_batch_size']}",
                    f"WORKBENCH_TME_QFORMER_BS={execution['gallery_batch_size']}", *argv]
            replay_command = (f'test ! -e {shlex.quote(str(native_dir))} && mkdir -p {shlex.quote(str(native_dir))} && ' +
                             _native_replay_command(argv, native_python))
            binder = [str(native_python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"),
                      "--phase", "bind-tme-family", "--family-plan",
                      str(family_root / "tme-family.v3-plan.json"), "--capture",
                      str(native_dir), "--role", role_name]
            replay_command += " && " + " ".join(shlex.quote(value) for value in binder)
            if role_name == "native_adapter_native":
                for category in CATEGORIES:
                    native_capture = native_dir / f"{category}.npz"
                    convert = [str(native_python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"),
                               "--phase", "native", "--plan", str(plan_paths[category]),
                               "--capture", str(native_capture), "--role", role_name,
                               "--output-root", str(plan_paths[category].parent)]
                    replay_command += " && " + " ".join(shlex.quote(value) for value in convert)
        else:
            category_commands = []
            for category in CATEGORIES:
                plan = plans[category]
                role_plan = plan["runs"][role_name]
                output = plan["commands"]["roles"][role_name]["output"]
                argv = [str(adapter_python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"),
                        "--phase", "adapter", "--plan", str(plan_paths[category]), "--role", role_name,
                        "--output", output]
                if role_plan["execution"].get("sample_query_count"):
                    sample_path = family_root / "sample-queries" / "tme-family" / role_name / f"{category}.json"
                    sample_ids = role_plan["execution"]["sample_query_ids"]
                    sample_raw = json.dumps(sample_ids, separators=(",", ":")) + "\n"
                    if sample_path.exists() and sample_path.read_text(encoding="utf-8") != sample_raw:
                        raise FileExistsError(f"immutable category sample differs: {sample_path}")
                    if not sample_path.exists():
                        sample_path.parent.mkdir(parents=True, exist_ok=True)
                        sample_path.write_text(sample_raw, encoding="utf-8")
                    argv.extend(["--sample-queries", str(sample_path)])
                command = (f"test ! -e {shlex.quote(output)} && mkdir -p {shlex.quote(str(Path(output).parent))} && " +
                           " ".join(shlex.quote(value) for value in argv))
                if role_name == "native_adapter_adapter":
                    compare = [str(adapter_python), str(ROOT / "workbench/scripts/clvc_tme_v3_parity.py"),
                               "--phase", "compare", "--plan", str(plan_paths[category]),
                               "--native", str(family_capture_root / "native_adapter_native" / f"{category}.npz"),
                               "--adapter", output, "--output",
                               str(plan_paths[category].parent / "comparisons" / "tme" / category / "native_adapter.json")]
                    command += " && " + " ".join(shlex.quote(value) for value in compare)
                category_commands.append(command)
            replay_command = " && ".join(f"({command})" for command in category_commands)
        unit = f"cir-tme-family-{role_name}".replace("_", "-")
        guarded = ("set -euo pipefail; "
                   "cg=$(awk -F: '$1==\"0\" {print $3}' /proc/self/cgroup); "
                   "test -n \"$cg\"; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.max\")\" = 12884901888; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.swap.max\")\" = 0; "
                   "test \"$(cat \"/sys/fs/cgroup$cg/memory.oom.group\")\" = 1; "
                   f"flock -x /tmp/cir-fix-all-20261120-cpu.lock -c {shlex.quote(replay_command)}")
        commands[role_name] = (f"{ROOT / 'workbench/scripts/gpu_job.sh'} systemd-run --user --wait --pipe --collect --same-dir --service-type=exec "
                              f"--unit={unit} -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=kill "
                              f"env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                              f"MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH={ROOT} "
                              f"/bin/bash --noprofile --norc -c {shlex.quote(guarded)}")
    return {"schema": "cir-tme-family-plan-v1",
            "categories": {category: {"plan_path": str(plan_paths[category].resolve()),
                                      "plan_sha256": hashlib.sha256(plan_paths[category].read_bytes()).hexdigest(),
                                      "identity": plans[category]["identity"],
                                      "query_ids": _tme_rows(category)[0]}
                           for category in CATEGORIES},
            "roles": roles, "commands": commands,
            "capture_selection": str(selection_path),
            "gallery_scope": "full native per-category gallery; category sample query IDs only"}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("clvc_net", "tme"))
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family-output", type=Path)
    parser.add_argument("--family-only", action="store_true")
    args = parser.parse_args(argv)
    if args.family_only:
        if args.model != "tme" or args.category is not None or args.output is not None or args.family_output is None:
            parser.error("--family-only needs --model tme and --family-output only")
        destination = args.family_output
        category_plans = {}
        category_paths = {}
        for category in CATEGORIES:
            path = destination.parent / f"g1_g2.tme.{category}.v3-plan.json"
            category_plans[category] = json.loads(path.read_text(encoding="utf-8"))
            category_paths[category] = path
        if any(not isinstance(plan.get("controls"), dict) or
               any(not isinstance(control.get("scopes"), dict) or
                   set(control["scopes"]) != {"query", "gallery"}
                   for control in plan["controls"].values())
               for plan in category_plans.values()):
            raise ValueError("TME family requires complete preregistered V3 control scopes")
        family = _tme_family(category_plans, category_paths)
        raw = json.dumps(family, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"family_plan": str(destination),
                          "family_plan_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
        return 0
    if args.model is None or args.category is None or args.family_output is not None:
        parser.error("plan generation needs --model and --category; family is emitted with --family-only")
    hashes = _load_checkpoint_hashes()
    destination = args.output or (PLAN_ROOT / f"g1_g2.{args.model}.{args.category}.v3-plan.json")
    plan = _plan(args.model, args.category, hashes, destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable V3 plan exists: {destination}")
    raw = json.dumps(plan, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"plan": str(destination), "plan_sha256": hashlib.sha256(raw).hexdigest(),
                      "model_id": args.model, "category": args.category}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
