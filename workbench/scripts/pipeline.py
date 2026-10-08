#!/usr/bin/env python3
"""Orchestrate existing workbench operator scripts without reimplementing them."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    required_capability: str | None = None


def stage_key(stage: Stage) -> str:
    parts = [stage.name]
    if stage.model_id:
        parts.append(stage.model_id)
    if stage.checkpoint_id:
        parts.append(stage.checkpoint_id)
    if stage.protocol:
        parts.append(stage.protocol)
    return ":".join(parts)


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

    # Source git commit
    if stage.model_id and config:
        source_dir = config.WORKBENCH_THIRD_PARTY_ROOT / stage.model_id
        if source_dir.is_dir() and (source_dir / ".git").exists():
            try:
                head = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
                hasher.update(f"GIT:{head}".encode("utf-8"))
            except Exception:
                pass

    # Dependency state
    if state and stage.dependencies:
        stages_rec = state.get("stages", {})
        for dep in stage.dependencies:
            dep_data = stages_rec.get(dep, {})
            hasher.update(f"DEP:{dep}:{dep_data.get('status')}:{dep_data.get('fingerprint')}".encode("utf-8"))

    return hasher.hexdigest()


def stage_fingerprint(stage: Stage) -> str:
    if stage.input_paths or stage.model_id:
        try:
            return compute_stage_input_fingerprint(stage, resolve_config())
        except Exception:
            pass
    import hashlib
    payload = json.dumps([stage.name, [list(c) for c in stage.commands]], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            try:
                res[str(p)] = sha256_file(p)
            except Exception:
                res[str(p)] = "DIR"
    return res


def validate_stage_outputs(stage: Stage, recorded: dict[str, str] | None = None) -> bool:
    from workbench.backend.registry import sha256_file

    if not stage.output_paths:
        return True
    for p in stage.output_paths:
        if not p.exists():
            return False
        if p.is_file():
            if p.stat().st_size == 0:
                return False
            if p.suffix == ".json":
                try:
                    json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return False
        if recorded and str(p) in recorded:
            expected = recorded[str(p)]
            if expected in ("MISSING", "EMPTY"):
                return False
            try:
                actual = sha256_file(p) if (p.is_file() or p.is_dir()) else None
                if actual != expected:
                    return False
            except Exception:
                return False
    return True


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
    required_capability: str | None = None,
) -> Stage:
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
        required_capability=required_capability,
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
    return run_stage("dataset-link", command("prepare_dataset.py", "--dataset-root", dataset_root(args, config), "--model", "csmcir"), model_id="csmcir", required_capability="allow_preparation")


def real_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    sync = (
        run_stage("sync", command("sync_upstreams.py", *selection(args), "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT), model_id=args.model, required_capability="allow_network")
        if args.sync_sources
        else Stage("sync", skipped="pass --sync-sources")
    )
    link = layout_prepare(args, config)
    download_args = [*selection(args), *(["--checkpoint", args.checkpoint] if args.checkpoint else []), "--output-root", config.WORKBENCH_CHECKPOINT_ROOT]
    checkpoints = run_stage("checkpoint", command("download_checkpoints.py", *download_args), model_id=args.model, checkpoint_id=args.checkpoint, required_capability="allow_large_downloads") if args.download_checkpoints else Stage("checkpoint", skipped="pass --download-checkpoints")
    selected_auxiliary_models = [args.model] if args.model in auxiliary_model_ids() else ([] if args.model else sorted(auxiliary_model_ids()))
    if args.download_auxiliary_assets and selected_auxiliary_models:
        auxiliary_args = ["--model", selected_auxiliary_models[0]] if args.model else ["--all"]
        auxiliary = run_stage("auxiliary-assets", command("download_auxiliary_assets.py", *auxiliary_args), model_id=args.model, required_capability="allow_network")
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
        evaluation = run_stage("evaluation", command("evaluate_models.py", *evaluation_args), model_id=args.model, checkpoint_id=args.checkpoint, protocol=args.protocol, dependencies=("runtime-preflight",), required_capability="allow_gpu_eval")
    else:
        evaluation = Stage("evaluation", skipped="pass --evaluate")
    validation_index = (
        run_stage(
            "validation-index",
            command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"),
            command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
            dependencies=("evaluation",),
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
    model = args.model
    early_doctor = run_stage("doctor", command("doctor.py", "--scope", "workbench"))
    ds_check = dataset_check(args, config)
    sync = run_stage("sync", command("sync_upstreams.py", *selection(args), "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT), model_id=model, required_capability="allow_network")
    env_cmd = ["--model", model] if model else ["--list"]
    if getattr(args, "allow_env_install", False) and getattr(args, "apply", False):
        env_cmd.extend(["--create", "--allow-env-install"])
    env_stage = run_stage("environment", command("manage_environment.py", *env_cmd), model_id=model, dependencies=("sync",), required_capability="allow_env_install" if getattr(args, "allow_env_install", False) else None)
    prep_cmd = ["--model", model, "--dataset-root", str(dataset_root(args, config))] if model else ["--list"]
    if getattr(args, "allow_preparation", False) and getattr(args, "apply", False):
        prep_cmd.extend(["--execute", "--allow-preparation"])
    prep_stage = run_stage("preparation", command("prepare_model.py", *prep_cmd), model_id=model, dependencies=("dataset", "sync"), required_capability="allow_preparation" if getattr(args, "allow_preparation", False) else None)
    download_args = [*selection(args), *(["--checkpoint", args.checkpoint] if args.checkpoint else []), "--output-root", config.WORKBENCH_CHECKPOINT_ROOT]
    ckpt_stage = run_stage("checkpoint", command("download_checkpoints.py", *download_args), model_id=model, checkpoint_id=args.checkpoint, required_capability="allow_large_downloads")
    selected_auxiliary_models = [args.model] if args.model in auxiliary_model_ids() else ([] if args.model else sorted(auxiliary_model_ids()))
    if selected_auxiliary_models:
        aux_args = ["--model", selected_auxiliary_models[0]] if args.model else ["--all"]
        aux_stage = run_stage("auxiliary-assets", command("download_auxiliary_assets.py", *aux_args), model_id=model, required_capability="allow_network")
    else:
        aux_stage = Stage("auxiliary-assets", skipped="selected model has no registered auxiliary assets")
    final_doctor_args = ["--scope", "real", "--dataset-root", str(dataset_root(args, config))]
    if model:
        final_doctor_args.extend(["--model", model])
    runtime_stage = run_stage("runtime-preflight", command("doctor.py", *final_doctor_args), model_id=model, dependencies=("sync", "checkpoint", "preparation", "environment"))
    eval_args = [*selection(args, runnable=True), "--dataset-root", str(dataset_root(args, config))]
    if model == "csmcir":
        eval_args[-1] = str(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset")
        eval_args.extend(["--canonical-dataset-root", str(dataset_root(args, config))])
    if args.checkpoint:
        eval_args.extend(["--checkpoint", args.checkpoint])
    if args.protocol:
        eval_args.extend(["--protocol", args.protocol])
    eval_args.extend(["--top-k", str(args.top_k)])
    if args.continue_on_error:
        eval_args.append("--continue-on-error")
    eval_stage = run_stage("evaluation", command("evaluate_models.py", *eval_args), model_id=model, checkpoint_id=args.checkpoint, protocol=args.protocol, dependencies=("runtime-preflight",), required_capability="allow_gpu_eval")
    val_idx_stage = run_stage("validation-index", command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"), command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT), dependencies=("evaluation",))
    return [early_doctor, ds_check, sync, env_stage, prep_stage, ckpt_stage, aux_stage, runtime_stage, eval_stage, val_idx_stage]


def setup_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    return all_stages(args, config)[:7]


def reproduce_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    return all_stages(args, config)[7:9]


def analyze_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    return [all_stages(args, config)[9]]


def stages_for(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    if args.mode == "mock":
        doctor_args = ["--scope", "workbench", *( ["--serve"] if args.serve else [])]
        stages = [
            run_stage("doctor", command("doctor.py", *doctor_args)),
            run_stage("load mock", command("load_mock_results.py", "--output-root", config.WORKBENCH_RESULTS_ROOT)),
            run_stage("validate", command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT)),
            run_stage("rebuild", command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT)),
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
    state_path: Path = STAGE_STATE_PATH,
    apply: bool = False,
    authorized_capabilities: set[str] | None = None,
    config: WorkbenchConfig | None = None,
    mode: str = "real",
    report_path: Path | None = None,
) -> int:
    cfg = config or resolve_config()
    failed = False
    force_set = set(force_stages or ())
    authorized = set(authorized_capabilities or ())
    state = load_pipeline_state(state_path) if resume else {}
    stage_records = state.setdefault("stages", {})
    rerun_stages = set()

    planning_mode = dry_run or (mode in ("all", "setup", "reproduce") and not apply)
    if mode in ("all", "setup", "reproduce") and not apply and not dry_run:
        print("  [PLAN] planning mode by default; pass --apply with capability authorization flags to execute")

    for number, stage in enumerate(stages, 1):
        skey = stage_key(stage)
        print(f"[{number}/{len(stages)}] {stage.name}")
        if not stage.commands:
            print(f"  [SKIP] {stage.skipped}")
            continue

        # Authorization check
        if apply and stage.required_capability and stage.required_capability not in authorized:
            print(f"  [BLOCKED] {stage.name}: BLOCKED_AUTHORIZATION_REQUIRED: pass --{stage.required_capability.replace('_', '-')}", file=sys.stderr)
            failed = True
            if not continue_on_error:
                return 1
            continue

        fingerprint = compute_stage_input_fingerprint(stage, cfg, state)

        # Check dependency invalidation
        dep_invalidated = any(dep in rerun_stages for dep in stage.dependencies)

        # Resume check
        can_skip = False
        if resume and not dep_invalidated and (not force_set or (stage.name not in force_set and skey not in force_set)):
            prev = stage_records.get(skey) or stage_records.get(stage.name)
            if prev and prev.get("status") == "COMPLETE":
                if prev.get("fingerprint") == fingerprint:
                    if validate_stage_outputs(stage, prev.get("output_fingerprints")):
                        can_skip = True

        if can_skip:
            print(f"  [SKIP] resume: already completed with matching fingerprint and verified outputs")
            continue

        rerun_stages.add(stage.name)
        rerun_stages.add(skey)

        if not planning_mode:
            stage_records[skey] = {
                "stage_id": skey,
                "name": stage.name,
                "model_id": stage.model_id,
                "status": "RUNNING",
                "fingerprint": fingerprint,
                "started_at": datetime.now(UTC).isoformat(),
            }
            save_pipeline_state(state, state_path)

        stage_failed = False
        for argv in stage.commands:
            print(f"  [{'PLAN' if planning_mode else 'RUN'}] {shlex.join(argv)}")
            if stage.name == "evaluation":
                print("  [INFO] official reproduction only; no instrumented ranking export")
            if planning_mode:
                continue
            status = run_command(argv)
            if status == 0:
                continue
            stage_failed = True
            failed = True
            print(f"  [BLOCKED] {stage.name} exited with status {status}", file=sys.stderr)
            if not planning_mode:
                stage_records[skey].update({
                    "status": "FAILED",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "return_code": status,
                })
                save_pipeline_state(state, state_path)
            if stage.name == "runtime-preflight" or not continue_on_error:
                return 1
            break

        if not planning_mode and not stage_failed:
            stage_records[skey].update({
                "status": "COMPLETE",
                "finished_at": datetime.now(UTC).isoformat(),
                "return_code": 0,
                "output_fingerprints": compute_stage_output_fingerprints(stage),
            })
            save_pipeline_state(state, state_path)

    if report_path and not planning_mode:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_report = {
            "mode": mode,
            "timestamp": datetime.now(UTC).isoformat(),
            "success": not failed,
            "stages": stage_records,
        }
        report_path.write_text(json.dumps(workflow_report, indent=2) + "\n", encoding="utf-8")

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
    return run_stages(
        stages_for(args, config),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        resume=args.resume,
        force_stages=force_stages,
        apply=args.apply,
        authorized_capabilities=authorized_caps,
        config=config,
        mode=args.mode,
        report_path=args.report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
