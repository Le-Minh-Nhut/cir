#!/usr/bin/env python3
"""Orchestrate existing workbench operator scripts without reimplementing them."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import auxiliary_model_ids, load_registry
SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parents[1]
STAGE_STATE_PATH = REPOSITORY / "workbench" / "artifacts" / "pipeline_state.json"


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[list[str], ...] = ()
    skipped: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    protocol: str | None = None
    dependencies: tuple[str, ...] = ()
    input_paths: tuple[Path, ...] = ()
    output_paths: tuple[Path, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    receipt: bool = False
    run_id: str | None = None
    evidence: Callable[[dict | None], tuple[bool, str | None, dict | None]] | None = None


def stage_key(stage: Stage) -> str:
    parts = [stage.name]
    if stage.model_id:
        parts.append(stage.model_id)
    if stage.checkpoint_id:
        parts.append(stage.checkpoint_id)
    if stage.protocol:
        parts.append(stage.protocol)
    return ":".join(parts)


def dependency_key(stage: Stage) -> str:
    """Checkpoint/protocol-independent key used by stage dependency edges."""
    return f"{stage.name}:{stage.model_id}" if stage.model_id else stage.name



_CONTENT_HASH_LIMIT = 64 * 1024 * 1024
_CONTENT_HASH_SUFFIXES = (".json", ".yaml", ".yml", ".txt", ".pkl", ".pt", ".pth", ".cfg", ".toml")


def compute_stage_input_fingerprint(stage: Stage, config: WorkbenchConfig | None = None, state: dict | None = None) -> str:
    import hashlib
    from workbench.backend.registry import sha256_file

    hasher = hashlib.sha256()
    hasher.update(stage.name.encode("utf-8"))
    if stage.model_id:
        hasher.update(f"MODEL:{stage.model_id}".encode("utf-8"))
    if stage.checkpoint_id:
        hasher.update(f"CKPT:{stage.checkpoint_id}".encode("utf-8"))
    if stage.protocol:
        hasher.update(f"PROTO:{stage.protocol}".encode("utf-8"))
    for cmd in stage.commands:
        hasher.update(shlex.join(cmd).encode("utf-8"))

    # Hashing declared input paths
    for p in stage.input_paths:
        hasher.update(str(p).encode("utf-8"))
        if not p.exists():
            hasher.update(b":MISSING\0")
        elif p.is_file():
            hasher.update(b":FILE\0")
            hasher.update(str(p.stat().st_size).encode("utf-8"))
            if p.stat().st_size <= 10 * 1024 * 1024 or p.suffix in (".json", ".yaml", ".yml", ".txt", ".pkl", ".pt", ".pth"):
                try:
                    hasher.update(sha256_file(p).encode("utf-8"))
                except Exception:
                    hasher.update(str(p.stat().st_mtime_ns).encode("utf-8"))
            else:
                hasher.update(str(p.stat().st_mtime_ns).encode("utf-8"))
        elif p.is_dir():
            hasher.update(b":DIR\0")
            try:
                hasher.update(sha256_file(p).encode("utf-8"))
            except Exception:
                hasher.update(str(p.stat().st_mtime_ns).encode("utf-8"))

    # Source identity: commit, cleanliness, and provenance digest.
    if stage.model_id and config:
        from workbench.backend.registry import model_by_id
        from workbench.backend.runtime import source_provenance
        try:
            m = model_by_id(stage.model_id)
            sdir = m.get("source_dir") or stage.model_id
            source_dir = config.WORKBENCH_THIRD_PARTY_ROOT / sdir
            if source_dir.is_dir() and (source_dir / ".git").exists():
                provenance = source_provenance(m, source_dir)
                hasher.update(f"GIT:{provenance.get('actual_commit')}".encode("utf-8"))
                hasher.update(f"DIRTY:{provenance.get('dirty')}".encode("utf-8"))
                hasher.update(f"PROV:{provenance.get('provenance_digest')}".encode("utf-8"))
        except Exception:
            pass
    # Dependency state
    if state and stage.dependencies:
        stages_rec = state.get("stages", {})
        for dep in stage.dependencies:
            dep_data = stages_rec.get(dep, {})
            hasher.update(f"DEP:{dep}:{dep_data.get('status')}:{dep_data.get('fingerprint')}".encode("utf-8"))

    return hasher.hexdigest()


def stage_input_fingerprint_info(stage: Stage, config: WorkbenchConfig | None = None, state: dict | None = None) -> dict:
    """Return {"digest": str, "weak": bool}.

    ``weak`` is True when any declared input could only be identified by metadata
    (size/mtime) rather than content. Weak identities must never silently grant a
    verified resume of a side-effecting stage.
    """
    digest = compute_stage_input_fingerprint(stage, config, state)
    weak = False
    for p in stage.input_paths:
        if not p.exists() or p.is_dir():
            continue
        if p.is_file() and p.stat().st_size > _CONTENT_HASH_LIMIT and p.suffix not in _CONTENT_HASH_SUFFIXES:
            weak = True
    return {"digest": digest, "weak": weak}


def stage_receipt_path(config: WorkbenchConfig, stage: Stage) -> Path:
    repo_root = getattr(config, "CIR_REPO_ROOT", REPOSITORY)
    key = stage_key(stage).replace(":", "_")
    return repo_root / "workbench" / "artifacts" / "receipts" / f"{key}.json"


def write_stage_receipt(config: WorkbenchConfig, stage: Stage, *, return_code: int) -> Path:
    """Durable completion proof for a provisioning stage that produces no artifact file."""
    path = stage_receipt_path(config, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_id": stage_key(stage),
        "model_id": stage.model_id,
        "commands": [list(c) for c in stage.commands],
        "return_code": return_code,
        "completed_at": datetime.now(UTC).isoformat(),
        "kind": "stage_completion_receipt",
    }
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def compute_stage_output_fingerprints(stage: Stage) -> dict[str, str]:
    from workbench.backend.registry import sha256_file

    res = {}
    for p in stage.output_paths:
        if not p.exists():
            res[str(p)] = "MISSING"
        elif p.is_file():
            if p.stat().st_size == 0:
                res[str(p)] = "EMPTY"
            else:
                try:
                    res[str(p)] = sha256_file(p)
                except Exception:
                    res[str(p)] = "ERROR"
        elif p.is_dir():
            if not any(p.iterdir()):
                res[str(p)] = "EMPTY"
            else:
                # Hash the resolved target so a symlinked dataset link is content-verified.
                target = p.resolve()
                try:
                    res[str(p)] = sha256_file(target)
                except Exception:
                    res[str(p)] = "DIR"
    return res


def validate_stage_outputs(stage: Stage, recorded: dict[str, str] | None = None,
                           config: WorkbenchConfig | None = None,
                           record: dict | None = None) -> bool:
    from workbench.backend.registry import sha256_file

    if stage.evidence is not None:
        # The stage's completion proof is supplied by the stage itself: for an
        # evaluation this is the run-bound completion artifact of THIS invocation,
        # so a leftover report from an earlier run can never satisfy it.
        ok, _, _ = stage.evidence(record)
        return bool(ok)

    if stage.receipt and not stage.output_paths:
        cfg = config or resolve_config()
        path = stage_receipt_path(cfg, stage)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("stage_id") == stage_key(stage) and payload.get("return_code") == 0

    if not stage.output_paths:
        # Read-only stages may be skipped on fingerprint alone. A side-effecting stage
        # must carry a durable completion receipt; without one it is not trusted.
        return not stage.required_capabilities
    for p in stage.output_paths:
        if not p.exists():
            return False
        if p.is_dir():
            # A declared directory output must resolve to a populated tree. A symlinked
            # directory is acceptable only when its target is itself populated (the
            # CSMCIR dataset link is exactly this case); dangling/empty links fail.
            if not any(p.iterdir()):
                return False
        if p.is_file():
            if p.stat().st_size == 0:
                return False
            if p.suffix == ".json":
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return False
                # A run manifest (anything named *_latest.json) must reference an
                # existing non-empty run artifact; an empty/garbage manifest is not proof.
                if p.name.endswith("_latest.json"):
                    if not isinstance(payload, dict) or not isinstance(payload.get("report"), str):
                        return False
                    referenced = Path(payload["report"])
                    if not referenced.is_file() or referenced.stat().st_size == 0:
                        return False
                    try:
                        json.loads(referenced.read_text(encoding="utf-8"))
                    except Exception:
                        return False
        if recorded and str(p) in recorded:
            expected = recorded[str(p)]
            if expected in ("MISSING", "EMPTY"):
                return False
            try:
                if p.is_dir():
                    actual = sha256_file(p.resolve())
                elif p.is_file():
                    actual = sha256_file(p)
                else:
                    actual = None
                if actual != expected:
                    return False
            except Exception:
                return False
    return True


_UNUSABLE_DEP_STATUSES = {"FAILED", "BLOCKED_AUTHORIZATION_REQUIRED", "SKIPPED_DEPENDENCY", "INTERRUPTED"}


class WorkflowLockError(RuntimeError):
    """Raised when another live workflow already owns this state file."""


def _lock_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + ".lock")


_LOCK_HANDLES: dict[str, object] = {}


def acquire_workflow_lock(state_path: Path, *, timeout: float = 0.0):
    """Acquire an exclusive advisory lock on the workflow state.

    Uses ``flock`` so the lock is released automatically if the process dies.
    Returns the live file object; the caller must keep it referenced and pass it
    to ``release_workflow_lock``. Raises WorkflowLockError if held by a live owner
    and ``timeout`` elapses.
    """
    import fcntl
    import time

    path = _lock_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            _LOCK_HANDLES[str(path)] = handle
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise WorkflowLockError(
                    f"another workflow owns {state_path} (lock held); retry later"
                )
            time.sleep(0.05)


def release_workflow_lock(handle) -> None:
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass
    for key, value in list(_LOCK_HANDLES.items()):
        if value is handle:
            del _LOCK_HANDLES[key]


def load_pipeline_state(path: Path = STAGE_STATE_PATH) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_pipeline_state(state: dict, path: Path = STAGE_STATE_PATH) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def new_run_id() -> str:
    """Invocation identity shared by the master and the evaluator it launches."""
    import uuid

    return f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}"


def command(script: str, *arguments: str | Path) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *(str(argument) for argument in arguments)]


def run_stage(
    name: str,
    *commands: list[str],
    skipped: str | None = None,
    model_id: str | None = None,
    checkpoint_id: str | None = None,
    protocol: str | None = None,
    dependencies: tuple[str, ...] = (),
    input_paths: tuple[Path, ...] = (),
    output_paths: tuple[Path, ...] = (),
    required_capabilities: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
    required_capability: str | None = None,
    receipt: bool = False,
    run_id: str | None = None,
    evidence: Callable[[dict | None], tuple[bool, str | None, dict | None]] | None = None,
) -> Stage:
    caps = frozenset(required_capabilities)
    if required_capability:
        caps = caps | {required_capability}
    return Stage(
        name=name,
        commands=commands,
        skipped=skipped,
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        protocol=protocol,
        dependencies=dependencies,
        input_paths=input_paths,
        output_paths=output_paths,
        required_capabilities=caps,
        receipt=receipt,
        run_id=run_id,
        evidence=evidence,
    )


def selection(args: argparse.Namespace, *, runnable: bool = False) -> list[str]:
    return ["--model", args.model] if args.model else (["--all-runnable"] if runnable else ["--all"])


def dataset_root(args: argparse.Namespace, config: WorkbenchConfig) -> Path:
    return args.dataset_root or config.FASHIONIQ_ROOT


def dataset_check(args: argparse.Namespace, config: WorkbenchConfig) -> Stage:
    root = dataset_root(args, config)
    val_splits = tuple(root / "image_splits" / f"split.{c}.val.json" for c in ("dress", "shirt", "toptee"))
    return run_stage("dataset", command("prepare_dataset.py", "--dataset-root", root, "--check-only"), input_paths=val_splits)


def layout_prepare(args: argparse.Namespace, config: WorkbenchConfig) -> Stage:
    active = args.model == "csmcir" or (args.model is None and any((args.sync_sources, args.download_checkpoints, args.download_auxiliary_assets, args.evaluate)))
    if not active:
        return Stage("dataset-link", skipped="CSMCIR preparation not selected")
    return run_stage(
        "dataset-link",
        command("prepare_dataset.py", "--dataset-root", dataset_root(args, config), "--model", "csmcir"),
        model_id="csmcir",
        output_paths=(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset",),
        required_capability="allow_preparation",
    )


def model_eval_inputs(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig) -> tuple[Path, ...]:
    from workbench.backend.registry import checkpoint_path
    inputs = []
    mid = model_dict["model_id"]
    ckpts = model_dict.get("checkpoint_variants", [])
    if args.checkpoint:
        ckpts = [c for c in ckpts if c["checkpoint_id"] == args.checkpoint]
    for c in ckpts:
        inputs.append(checkpoint_path(mid, c, config.WORKBENCH_CHECKPOINT_ROOT))
    if mid == "limn":
        for cat in ("dress", "shirt", "toptee"):
            inputs.append(config.WORKBENCH_CHECKPOINT_ROOT / "limn" / f"0_{cat}_best_model.pt")
    ds_root = dataset_root(args, config)
    for cat in ("dress", "shirt", "toptee"):
        inputs.append(ds_root / "image_splits" / f"split.{cat}.val.json")
        inputs.append(ds_root / "captions" / f"cap.{cat}.val.json")
    if mid == "csmcir":
        for cat in ("dress", "shirt", "toptee"):
            inputs.append(ds_root / "qwen_captions" / f"{cat}_cot_val.json")
            inputs.append(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "COT_ours2" / "fashioniq" / f"{cat}_cot_val.json")
    return tuple(inputs)


def resolved_checkpoint_id(model_dict: dict, args: argparse.Namespace) -> str:
    """The checkpoint id evaluate_models.py will actually use for this model.

    LIMN evaluates as one bundle candidate keyed by its bundle id, so the proof
    filename must use the bundle id, not the first category variant.
    """
    explicit = getattr(args, "checkpoint", None)
    if explicit:
        return explicit
    bundles = model_dict.get("checkpoint_bundles") or []
    if model_dict["model_id"] == "limn" and bundles:
        return bundles[0]["bundle_id"]
    variants = model_dict.get("checkpoint_variants") or []
    return variants[0]["checkpoint_id"] if variants else "default"


def resolved_checkpoint_file(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig) -> Path:
    """The primary checkpoint file the evaluator will be handed for this selection."""
    from workbench.backend.registry import checkpoint_path

    mid = model_dict["model_id"]
    variants = {variant["checkpoint_id"]: variant for variant in model_dict.get("checkpoint_variants") or []}
    ckpt_id = resolved_checkpoint_id(model_dict, args)
    variant = variants.get(ckpt_id)
    if variant is None:
        for bundle in model_dict.get("checkpoint_bundles") or []:
            if bundle["bundle_id"] == ckpt_id:
                variant = variants.get(bundle["required_checkpoint_ids"][0])
                break
    if variant is None and variants:
        variant = next(iter(variants.values()))
    return checkpoint_path(mid, variant, config.WORKBENCH_CHECKPOINT_ROOT) if variant else Path(".")


def evaluation_evidence(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig):
    """Run-bound completion evidence for one evaluation stage."""
    from workbench.scripts.evaluate_models import completion_validator

    mid = model_dict["model_id"]
    ckpt_id = resolved_checkpoint_id(model_dict, args)
    protocol = args.protocol or model_dict.get("native_protocol")
    return completion_validator(mid, ckpt_id, protocol, model_dict.get("upstream_commit_sha"), config,
                                resolved_checkpoint_file(model_dict, args, config))


def model_eval_outputs(model_dict: dict, args: argparse.Namespace, config: WorkbenchConfig) -> tuple[Path, ...]:
    """Declared outputs are informational only.

    A run-scoped report cannot be named before the run id exists, so the real
    completion proof is the run-bound completion artifact validated by
    :func:`evaluation_evidence`. The pointer returned here is the *successful*
    pointer, which only ever advances after a proof-backed success.
    """
    mid = model_dict["model_id"]
    ckpt_id = resolved_checkpoint_id(model_dict, args)
    repo_root = getattr(config, "CIR_REPO_ROOT", REPOSITORY)
    return (repo_root / "workbench" / "artifacts" / "reports" / f"{mid}_{ckpt_id}_latest_successful.json",)
def real_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    from workbench.backend.registry import checkpoint_path, model_by_id
    model_obj = model_by_id(args.model) if args.model else None
    sdir = model_obj.get("source_dir") if model_obj else None
    sync_outputs = (config.WORKBENCH_THIRD_PARTY_ROOT / sdir,) if sdir else ()

    sync = (
        run_stage(
            "sync",
            command("sync_upstreams.py", *selection(args), "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT),
            model_id=args.model,
            output_paths=sync_outputs,
            required_capability="allow_network",
        )
        if args.sync_sources
        else Stage("sync", skipped="pass --sync-sources")
    )
    link = layout_prepare(args, config)
    download_args = [*selection(args), *(["--checkpoint", args.checkpoint] if args.checkpoint else []), "--output-root", config.WORKBENCH_CHECKPOINT_ROOT]
    ckpt_outputs = ()
    if model_obj:
        ckpts = model_obj.get("checkpoint_variants", [])
        if args.checkpoint:
            ckpts = [c for c in ckpts if c["checkpoint_id"] == args.checkpoint]
        ckpt_outputs = tuple(checkpoint_path(args.model, c, config.WORKBENCH_CHECKPOINT_ROOT) for c in ckpts)

    if not args.download_checkpoints:
        checkpoints = Stage("checkpoint", skipped="pass --download-checkpoints")
    elif not ckpt_outputs:
        checkpoints = Stage("checkpoint", skipped="no registered checkpoint output for this selection")
    else:
        checkpoints = run_stage(
            "checkpoint",
            command("download_checkpoints.py", *download_args),
            model_id=args.model,
            checkpoint_id=args.checkpoint,
            output_paths=ckpt_outputs,
            required_capabilities={"allow_large_downloads", "allow_network"},
        )
    selected_auxiliary_models = [args.model] if args.model in auxiliary_model_ids() else ([] if args.model else sorted(auxiliary_model_ids()))
    if args.download_auxiliary_assets and selected_auxiliary_models:
        auxiliary_args = ["--model", selected_auxiliary_models[0]] if args.model else ["--all"]
        auxiliary = run_stage("auxiliary-assets", command("download_auxiliary_assets.py", *auxiliary_args), model_id=args.model, receipt=True, required_capabilities={"allow_network", "allow_large_downloads"})
    elif args.download_auxiliary_assets:
        auxiliary = Stage("auxiliary-assets", skipped="selected model has no registered auxiliary assets")
    else:
        auxiliary = Stage("auxiliary-assets", skipped="pass --download-auxiliary-assets")
    final_doctor_args: list[str] = ["--scope", "real", "--dataset-root", str(dataset_root(args, config))]
    if args.model:
        final_doctor_args.extend(["--model", args.model])
        strict_runtime = run_stage("runtime-preflight", command("doctor.py", *final_doctor_args), model_id=args.model, dependencies=("sync", "checkpoint")) if args.evaluate else Stage("runtime-preflight", skipped="pass --evaluate")
    else:
        strict_runtime = Stage("runtime-preflight", skipped="per-model strict guards run inside --all-runnable evaluation")
    if args.evaluate:
        evaluation_args = [*selection(args, runnable=True), "--dataset-root", str(dataset_root(args, config))]
        if args.model == "csmcir":
            evaluation_args[-1] = str(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset")
            evaluation_args.extend(["--canonical-dataset-root", str(dataset_root(args, config))])
        if args.checkpoint:
            evaluation_args.extend(["--checkpoint", args.checkpoint])
        if args.protocol:
            evaluation_args.extend(["--protocol", args.protocol])
        evaluation_args.extend(["--top-k", str(args.top_k)])
        if args.continue_on_error:
            evaluation_args.append("--continue-on-error")
        eval_inputs = model_eval_inputs(model_obj, args, config) if model_obj else ()
        eval_outputs = model_eval_outputs(model_obj, args, config) if model_obj else ()
        evaluation = run_stage(
            "evaluation",
            command("evaluate_models.py", *evaluation_args),
            model_id=args.model,
            checkpoint_id=args.checkpoint,
            protocol=args.protocol,
            dependencies=("runtime-preflight",),
            input_paths=eval_inputs,
            output_paths=eval_outputs,
            required_capabilities={"allow_gpu_eval", "allow_preparation"},
            run_id=new_run_id() if model_obj else None,
            evidence=evaluation_evidence(model_obj, args, config) if model_obj else None,
        )
    else:
        evaluation = Stage("evaluation", skipped="pass --evaluate")
    validation_index = (
        run_stage(
            "validation-index",
            command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"),
            command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
            dependencies=("evaluation",),
            input_paths=(config.WORKBENCH_RESULTS_ROOT,),
            output_paths=(getattr(config, "CIR_REPO_ROOT", REPOSITORY) / "workbench" / "artifacts" / "workbench.duckdb",),
            required_capabilities={"allow_index_write"},
        )
        if args.rebuild_index
        else Stage("validation-index", skipped="pass --rebuild-index after schema-v2 result JSON exists")
    )
    serve = run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")
    early_doctor_args: list[str] = ["--scope", "workbench"]
    if args.serve:
        early_doctor_args.append("--serve")
    return [
        run_stage("doctor", command("doctor.py", *early_doctor_args)),
        dataset_check(args, config),
        sync,
        link,
        checkpoints,
        auxiliary,
        strict_runtime,
        evaluation,
        validation_index,
        serve,
    ]


def all_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    from workbench.backend.registry import checkpoint_path, auxiliary_destination, auxiliary_assets_for_model
    registry = load_registry()
    models = registry["models"]
    if args.model:
        selected_models = [m for m in models if m["model_id"] == args.model]
    else:
        selected_models = models

    early_doctor = run_stage("doctor", command("doctor.py", "--scope", "workbench"))
    ds_check = dataset_check(args, config)
    stages = [early_doctor, ds_check]

    synced_sources = set()
    for m in selected_models:
        mid = m["model_id"]
        sdir = m.get("source_dir")
        sync_dep = f"sync:{mid}"
        sync_out = (config.WORKBENCH_THIRD_PARTY_ROOT / sdir,) if sdir else ()
        if sdir and sdir in synced_sources:
            first_m = next(m2["model_id"] for m2 in selected_models if m2.get("source_dir") == sdir)
            sync_dep = f"sync:{first_m}"
        else:
            if sdir:
                synced_sources.add(sdir)
            stages.append(run_stage(
                "sync",
                command("sync_upstreams.py", "--model", mid, "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT),
                model_id=mid,
                output_paths=sync_out,
                receipt=True,
                required_capabilities={"allow_network", "allow_preparation"},
            ))

        # Only an actual provisioning action is side-effecting; plain verification is read-only.
        will_create_env = bool(m.get("environment_required")) and getattr(args, "allow_env_install", False) and getattr(args, "apply", False)
        env_cmd = ["--model", mid]
        if will_create_env:
            env_cmd.extend(["--create", "--allow-env-install"])
        stages.append(run_stage(
            "environment",
            command("manage_environment.py", *env_cmd),
            model_id=mid,
            dependencies=(sync_dep,),
            receipt=True,
            required_capabilities=({"allow_env_install", "allow_network"} if will_create_env else frozenset()),
        ))

        prep_cmd = ["--model", mid, "--dataset-root", str(dataset_root(args, config))]
        prep_out = (config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset",) if mid == "csmcir" else ()
        will_prepare = mid == "csmcir" and getattr(args, "allow_preparation", False) and getattr(args, "apply", False)
        if will_prepare:
            prep_cmd.extend(["--execute", "--allow-preparation"])
        stages.append(run_stage(
            "preparation",
            command("prepare_model.py", *prep_cmd),
            model_id=mid,
            dependencies=("dataset", sync_dep),
            output_paths=prep_out,
            required_capabilities=({"allow_preparation", "allow_network"} if will_prepare else frozenset()),
        ))

        ckpt_args = ["--model", mid, "--output-root", str(config.WORKBENCH_CHECKPOINT_ROOT)]
        if args.checkpoint:
            ckpt_args.extend(["--checkpoint", args.checkpoint])
        ckpts = m.get("checkpoint_variants", [])
        if args.checkpoint:
            ckpts = [c for c in ckpts if c["checkpoint_id"] == args.checkpoint]
        ckpt_outs = tuple(checkpoint_path(mid, c, config.WORKBENCH_CHECKPOINT_ROOT) for c in ckpts)
        if not ckpt_outs:
            # No registry checkpoint exists for this model: nothing to acquire.
            stages.append(Stage("checkpoint", skipped="model has no registered checkpoint variant"))
        else:
            stages.append(run_stage(
                "checkpoint",
                command("download_checkpoints.py", *ckpt_args),
                model_id=mid,
                checkpoint_id=args.checkpoint,
                output_paths=ckpt_outs,
                required_capabilities={"allow_large_downloads", "allow_network"},
            ))

        if mid in auxiliary_model_ids():
            aux_outs = tuple(auxiliary_destination(a, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT) for a in auxiliary_assets_for_model(mid))
            stages.append(run_stage(
                "auxiliary-assets",
                command("download_auxiliary_assets.py", "--model", mid),
                model_id=mid,
                output_paths=aux_outs,
                receipt=True,
                required_capabilities={"allow_network", "allow_large_downloads"},
            ))

        preflight_deps = (sync_dep, f"checkpoint:{mid}", f"preparation:{mid}", f"environment:{mid}")
        stages.append(run_stage(
            "runtime-preflight",
            command("doctor.py", "--scope", "real", "--dataset-root", str(dataset_root(args, config)), "--model", mid),
            model_id=mid,
            dependencies=preflight_deps,
        ))

        eval_args = ["--model", mid, "--dataset-root", str(dataset_root(args, config))]
        if mid == "csmcir":
            eval_args[2] = str(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset")
            eval_args.extend(["--canonical-dataset-root", str(dataset_root(args, config))])
        if args.checkpoint:
            eval_args.extend(["--checkpoint", args.checkpoint])
        if args.protocol:
            eval_args.extend(["--protocol", args.protocol])
        eval_args.extend(["--top-k", str(args.top_k)])
        if args.continue_on_error:
            eval_args.append("--continue-on-error")

        eval_deps = (f"runtime-preflight:{mid}",)
        eval_ins = model_eval_inputs(m, args, config)
        eval_outs = model_eval_outputs(m, args, config)
        stages.append(run_stage(
            "evaluation",
            command("evaluate_models.py", *eval_args),
            model_id=mid,
            checkpoint_id=args.checkpoint,
            protocol=args.protocol,
            dependencies=eval_deps,
            input_paths=eval_ins,
            output_paths=eval_outs,
            required_capabilities={"allow_gpu_eval", "allow_preparation"},
            run_id=new_run_id(),
            evidence=evaluation_evidence(m, args, config),
        ))

    all_eval_deps = tuple(f"evaluation:{m['model_id']}" for m in selected_models)
    stages.append(run_stage(
        "validation-index",
        command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"),
        command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
        dependencies=all_eval_deps,
        input_paths=(config.WORKBENCH_RESULTS_ROOT,),
        output_paths=(getattr(config, "CIR_REPO_ROOT", REPOSITORY) / "workbench" / "artifacts" / "workbench.duckdb",),
        required_capabilities={"allow_index_write"},
    ))
    return stages
def setup_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    all_s = all_stages(args, config)
    return [s for s in all_s if s.name in ("doctor", "dataset", "sync", "environment", "preparation", "checkpoint", "auxiliary-assets")]


def reproduce_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    all_s = all_stages(args, config)
    return [s for s in all_s if s.name in ("runtime-preflight", "evaluation")]


def analyze_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    """Analysis mode is the explicit opt-in to rebuild the derived index."""
    all_s = all_stages(args, config)
    selected = []
    for stage in all_s:
        if stage.name == "validation-index":
            selected.append(replace(stage, required_capabilities=frozenset({"allow_index_write"})))
    return selected

def stages_for(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    if args.mode == "mock":
        doctor_args = ["--scope", "workbench", *( ["--serve"] if args.serve else [])]
        stages = [
            run_stage("doctor", command("doctor.py", *doctor_args)),
            run_stage("load mock", command("load_mock_results.py", "--output-root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_mock_write"}),
            run_stage("validate", command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_mock_write"}),
            run_stage("rebuild", command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
                      input_paths=(config.WORKBENCH_RESULTS_ROOT,),
                      receipt=True, required_capabilities={"allow_index_write"}),
        ]
        return [*stages, run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")]
    if args.mode == "prepare":
        return [dataset_check(args, config), layout_prepare(args, config)]
    if args.mode == "real":
        return real_stages(args, config)
    if args.mode == "serve":
        return [run_stage("serve", command("serve_workbench.py"))]
    if args.mode == "setup":
        return setup_stages(args, config)
    if args.mode == "reproduce":
        return reproduce_stages(args, config)
    if args.mode == "analyze":
        return analyze_stages(args, config)
    if args.mode == "all":
        return all_stages(args, config)
    if args.mode == "report":
        report_args = ["--scope", "real", "--json"]
        if args.model:
            report_args.extend(["--model", args.model])
        else:
            report_args.append("--all")
        return [run_stage("report", command("doctor.py", *report_args))]
    return [run_stage("doctor", command("doctor.py", *( ["--model", args.model] if args.model else [])))]


def run_command(argv: list[str]) -> int:
    try:
        return subprocess.run(argv, cwd=REPOSITORY, check=False).returncode
    except OSError as error:
        print(f"[BLOCKED] cannot start {argv[1]}: {error}", file=sys.stderr)
        return 127


def run_stages(
    stages: list[Stage],
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
    resume: bool = False,
    force_stages: set[str] | None = None,
    state_path: Path | None = None,
    apply: bool = False,
    authorized_capabilities: set[str] | None = None,
    config: WorkbenchConfig | None = None,
    mode: str = "real",
    report_path: Path | None = None,
) -> int:
    cfg = config or resolve_config()
    if state_path is None:
        repo_root = getattr(cfg, "CIR_REPO_ROOT", REPOSITORY)
        state_path = repo_root / "workbench" / "artifacts" / "pipeline_state.json"
    failed = False
    force_set = set(force_stages or ())
    if force_set:
        known = {s.name for s in stages} | {stage_key(s) for s in stages} | {dependency_key(s) for s in stages}
        unknown = sorted(force_set - known)
        if unknown:
            raise WorkflowLockError(f"unknown --force-stage value(s): {', '.join(unknown)}")
    authorized = set(authorized_capabilities or ())
    lock_fd = None
    if not dry_run:
        lock_fd = acquire_workflow_lock(state_path)
    state = load_pipeline_state(state_path) if resume else {}
    stage_records = state.setdefault("stages", {})
    rerun_stages = set()
    failed_stages = set()
    completed_stages = set()
    planning_mode = dry_run or (mode in ("all", "setup", "reproduce") and not apply)
    if mode in ("all", "setup", "reproduce") and not apply and not dry_run:
        print("  [PLAN] planning mode by default; pass --apply with capability authorization flags to execute")

    def write_report() -> None:
        if report_path and not dry_run:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_report = {
                "mode": mode,
                "timestamp": datetime.now(UTC).isoformat(),
                "success": not failed,
                "status": "SUCCESS" if not failed else ("PARTIAL" if any(s.get("status") == "COMPLETE" for s in stage_records.values()) else "FAILED"),
                "stages": stage_records,
            }
            temp_rep = report_path.with_suffix(f".tmp.{os.getpid()}")
            temp_rep.write_text(json.dumps(workflow_report, indent=2) + "\n", encoding="utf-8")
            temp_rep.replace(report_path)
    try:
        for number, stage in enumerate(stages, 1):
            skey = stage_key(stage)
            print(f"[{number}/{len(stages)}] {stage.name}")
            if not stage.commands:
                print(f"  [SKIP] {stage.skipped}")
                continue

            is_side_effecting = bool(stage.required_capabilities)
            stage_planning = planning_mode or (is_side_effecting and not apply)

            # Dependency execution prerequisite check. A dependency that is failing in
            # this sweep OR persisted as failing/blocked in the state file must block.
            def _dep_unusable(dep: str) -> bool:
                if dep in failed_stages:
                    return True
                # A dependency string may omit the checkpoint/protocol qualifier that the
                # recorded key carries. Qualify it with THIS stage's own model so the lookup
                # never matches a different model's stage.
                own = stage.model_id

                def _bad(key: str) -> bool:
                    if key in failed_stages:
                        return True
                    data = stage_records.get(key)
                    return isinstance(data, dict) and data.get("status") in _UNUSABLE_DEP_STATUSES

                # Exact match only: a bare edge name ("sync") must never match another
                # model's qualified key ("sync:othermodel").
                if _bad(dep):
                    return True
                # Same edge qualified by THIS stage's model (dep -> dep:<own>), including
                # any checkpoint/protocol variants of that same edge.
                if own and ":" not in dep:
                    qualified = f"{dep}:{own}"
                    if _bad(qualified):
                        return True
                    for recorded, data in stage_records.items():
                        if recorded.startswith(qualified + ":") and isinstance(data, dict) \
                                and data.get("status") in _UNUSABLE_DEP_STATUSES:
                            return True
                # A dependency that already names a model: allow its checkpoint variants.
                if ":" in dep:
                    for recorded, data in stage_records.items():
                        if recorded.startswith(dep + ":") and isinstance(data, dict) \
                                and data.get("status") in _UNUSABLE_DEP_STATUSES:
                            return True
                return False

            unmet_deps = [dep for dep in stage.dependencies if _dep_unusable(dep)]
            if unmet_deps:
                print(f"  [SKIP] {stage.name}: SKIPPED_DEPENDENCY: dependency {unmet_deps[0]} failed", file=sys.stderr)
                failed = True
                failed_stages.add(stage.name)
                failed_stages.add(dependency_key(stage))
                if not stage_planning:
                    stage_records[skey] = {
                        "stage_id": skey,
                        "name": stage.name,
                        "model_id": stage.model_id,
                        "status": "SKIPPED_DEPENDENCY",
                        "fingerprint": None,
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": 125,
                    }
                    save_pipeline_state(state, state_path)
                continue

            # Explicit authorization check: every required capability must be granted.
            if is_side_effecting:
                missing_caps = sorted(stage.required_capabilities - authorized)
                if not apply and not dry_run:
                    flags = " ".join(f"--{c.replace('_', '-')}" for c in sorted(stage.required_capabilities))
                    print(f"  [PLAN] {stage.name}: side-effecting stage requires --apply and {flags} to execute")
                elif apply and missing_caps:
                    flags = " ".join(f"--{c.replace('_', '-')}" for c in missing_caps)
                    print(f"  [BLOCKED] {stage.name}: BLOCKED_AUTHORIZATION_REQUIRED: pass {flags}", file=sys.stderr)
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    stage_records[skey] = {
                        "stage_id": skey,
                        "name": stage.name,
                        "model_id": stage.model_id,
                        "status": "BLOCKED_AUTHORIZATION_REQUIRED",
                        "fingerprint": None,
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": 126,
                    }
                    if not dry_run:
                        save_pipeline_state(state, state_path)
                    if not continue_on_error:
                        return 1
                    continue

            fingerprint_info = stage_input_fingerprint_info(stage, cfg, state)
            fingerprint = fingerprint_info["digest"]

            # Check dependency invalidation
            dep_invalidated = any(dep in rerun_stages for dep in stage.dependencies)

            # Resume check
            can_skip = False
            if resume and not dep_invalidated and (not force_set or (stage.name not in force_set and skey not in force_set)):
                prev = stage_records.get(skey) or stage_records.get(stage.name)
                if prev and prev.get("status") == "COMPLETE" and not fingerprint_info["weak"]:
                    if prev.get("fingerprint") == fingerprint:
                        if validate_stage_outputs(stage, prev.get("output_fingerprints"), config=cfg, record=prev):
                            can_skip = True

            if can_skip:
                print(f"  [SKIP] resume: already completed with matching fingerprint and verified outputs")
                completed_stages.add(stage.name)
                completed_stages.add(skey)
                continue

            rerun_stages.add(stage.name)
            rerun_stages.add(skey)

            if not stage_planning:
                stage_records[skey] = {
                    "stage_id": skey,
                    "name": stage.name,
                    "model_id": stage.model_id,
                    "status": "RUNNING",
                    "fingerprint": fingerprint,
                    "run_id": stage.run_id,
                    "started_at": datetime.now(UTC).isoformat(),
                }
                save_pipeline_state(state, state_path)

            stage_failed = False
            for argv in stage.commands:
                if stage.run_id and stage.name == "evaluation" and "--run-id" not in argv:
                    argv = [*argv, "--run-id", stage.run_id]
                print(f"  [{'PLAN' if stage_planning else 'RUN'}] {shlex.join(argv)}")
                if stage.name == "evaluation":
                    print("  [INFO] official reproduction only; no instrumented ranking export")
                if stage_planning:
                    continue
                status = run_command(argv)
                if status != 0:
                    stage_failed = True
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    print(f"  [BLOCKED] {stage.name} exited with status {status}", file=sys.stderr)
                    stage_records[skey].update({
                        "status": "FAILED",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": status,
                    })
                    save_pipeline_state(state, state_path)
                    break
                # Single-command stages can be output-validated per command; multi-command
                # stages validate once after all commands succeed (below).
                if stage.receipt and not stage.output_paths:
                    continue  # proof is written after all commands succeed (below)
                if len(stage.commands) == 1 and not validate_stage_outputs(stage, config=cfg, record=stage_records.get(skey)):
                    stage_failed = True
                    failed = True
                    failed_stages.add(stage.name)
                    failed_stages.add(dependency_key(stage))
                    print(f"  [BLOCKED] {stage.name}: declared outputs missing or invalid after exit 0", file=sys.stderr)
                    stage_records[skey].update({
                        "status": "FAILED",
                        "finished_at": datetime.now(UTC).isoformat(),
                        "return_code": 0,
                        "failure_reason": "output_contract_not_satisfied",
                    })
                    save_pipeline_state(state, state_path)
                    break

            if stage_failed:
                if not continue_on_error:
                    return 1
                continue

            # Provisioning stages that produce no artifact file get a durable receipt
            # as their completion proof (validated on later resumes).
            if not stage_planning and stage.receipt and not stage.output_paths:
                write_stage_receipt(cfg, stage, return_code=0)

            # Output proof for multi-command stages.
            if not stage_planning and not validate_stage_outputs(stage, config=cfg, record=stage_records.get(skey)):
                failed = True
                failed_stages.add(stage.name)
                failed_stages.add(dependency_key(stage))
                stage_records[skey].update({
                    "status": "FAILED",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "return_code": 0,
                    "failure_reason": "output_contract_not_satisfied",
                })
                save_pipeline_state(state, state_path)
                _, evidence_reason, _ = stage.evidence(stage_records.get(skey)) if stage.evidence else (False, None, None)
                detail = f": {evidence_reason}" if evidence_reason else ""
                print(f"  [BLOCKED] {stage.name}: declared outputs missing or invalid after exit 0{detail}", file=sys.stderr)
                if not continue_on_error:
                    return 1
                continue

            if not stage_planning:
                completed_stages.add(stage.name)
                completed_stages.add(skey)
                evidence_detail = None
                if stage.evidence is not None:
                    _, _, evidence_detail = stage.evidence(stage_records.get(skey))
                stage_records[skey].update({
                    "status": "COMPLETE",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "return_code": 0,
                    "run_id": stage.run_id,
                    "evidence": evidence_detail,
                    "output_fingerprints": compute_stage_output_fingerprints(stage),
                })
                save_pipeline_state(state, state_path)
    finally:
        write_report()
        if lock_fd is not None:
            release_workflow_lock(lock_fd)

    return 1 if failed else 0


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mock", "prepare", "real", "serve", "status", "setup", "reproduce", "analyze", "all", "report"))
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ dataset root")
    parser.add_argument("--sync-sources", action="store_true", help="sync selected official source repositories")
    parser.add_argument("--download-checkpoints", action="store_true", help="download selected registry checkpoints")
    parser.add_argument("--download-auxiliary-assets", action="store_true", help="obtain selected model auxiliary evaluation assets")
    parser.add_argument("--evaluate", action="store_true", help="run guarded official reproduction commands")
    parser.add_argument("--rebuild-index", action="store_true", help="validate and rebuild the derived result index")
    parser.add_argument("--serve", action="store_true", help="serve after pipeline stages")
    parser.add_argument("--dry-run", action="store_true", help="print ordered stage plan without subprocesses or mutations")
    parser.add_argument("--continue-on-error", action="store_true", help="continue pipeline after a failed stage")
    parser.add_argument("--resume", action="store_true", help="skip stages already completed with matching input fingerprints")
    parser.add_argument("--force-stage", action="append", default=[], help="force re-execution of a stage even when --resume is set")
    parser.add_argument("--apply", action="store_true", help="authorize execution of expensive operations")
    parser.add_argument("--allow-network", action="store_true", help="authorize network access (git clone/fetch, downloads)")
    parser.add_argument("--allow-large-downloads", action="store_true", help="authorize large checkpoint downloads")
    parser.add_argument("--allow-env-install", action="store_true", help="authorize environment creation and package installation")
    parser.add_argument("--allow-preparation", action="store_true", help="authorize model-specific preparation execution")
    parser.add_argument("--allow-gpu-eval", action="store_true", help="authorize model evaluation subprocess execution")
    parser.add_argument("--allow-index-write", action="store_true", help="authorize rebuilding the derived DuckDB index")
    parser.add_argument("--model", help="registry model selection")
    parser.add_argument("--all-models", action="store_true", help="select all registry models")
    parser.add_argument("--protocol", help="official protocol selection for evaluation")
    parser.add_argument("--checkpoint", help="checkpoint selection")
    parser.add_argument("--top-k", type=int, default=200, help="requested saved retrieval depth")
    parser.add_argument("--report", type=Path, help="write machine-readable execution report JSON to path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    if args.top_k < 1:
        parser_for().error("--top-k must be positive")
    if args.model and args.model not in {model["model_id"] for model in load_registry()["models"]}:
        parser_for().error(f"unknown model: {args.model}")
    config = resolve_config()
    if args.mode == "prepare" and args.dataset_root is None and not config.FASHIONIQ_ROOT:
        parser_for().error("prepare requires --dataset-root or FASHIONIQ_ROOT")
    force_stages = set(args.force_stage) if args.force_stage else None
    if args.all_models and args.model:
        parser_for().error("--all-models cannot be combined with --model")
    if args.mode in ("all", "setup", "reproduce") and not args.model and not args.all_models:
        parser_for().error(f"{args.mode} requires --model MODEL_ID or --all-models (explicit selection)")
    if force_stages:
        known = {s.name for s in stages_for(args, config)}
        unknown = sorted(force_stages - known)
        if unknown:
            parser_for().error(f"unknown --force-stage value(s): {', '.join(unknown)}")
    if args.dry_run and args.apply:
        parser_for().error("--dry-run cannot be combined with --apply")
    authorized_caps = set()
    if args.allow_network:
        authorized_caps.add("allow_network")
    if args.allow_large_downloads:
        authorized_caps.add("allow_large_downloads")
    if args.allow_env_install:
        authorized_caps.add("allow_env_install")
    if args.allow_preparation:
        authorized_caps.add("allow_preparation")
    if args.allow_gpu_eval:
        authorized_caps.add("allow_gpu_eval")
    if args.allow_index_write:
        authorized_caps.add("allow_index_write")
    if args.mode == "mock":
        # `mock` is an explicit, self-contained development workflow; it carries its own
        # authorization for writing mock results and rebuilding the derived index only.
        authorized_caps.update({"allow_mock_write", "allow_index_write"})
    if args.mode == "analyze":
        # `analyze` exists solely to rebuild the derived index from validated results.
        authorized_caps.add("allow_index_write")
    # `mock` and `analyze` are explicit, self-contained operator workflows: they carry
    # their own authorization for the exact writes they perform. `--dry-run` still plans only.
    effective_apply = args.apply or args.mode in ("mock", "analyze")
    return run_stages(
        stages_for(args, config),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        resume=args.resume,
        force_stages=force_stages,
        apply=effective_apply,
        authorized_capabilities=authorized_caps,
        config=config,
        mode=args.mode,
        report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
