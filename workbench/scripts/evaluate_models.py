#!/usr/bin/env python3
"""Guard and run audited official evaluation commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import os
import platform
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.metrics_extraction import extract_aggregate_metrics
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import checkpoint_path, load_registry, sha256_file
from workbench.backend.runtime import runtime_blockers


@dataclass(frozen=True)
class EvaluationPlan:
    model_id: str
    checkpoint_id: str
    protocol: str
    dataset_root: Path
    source: Path
    checkpoint: Path
    cwd: Path
    command: list[str]
    pin: str | None
    actual_source_commit: str | None
    paper_metrics: dict[str, float | None] | None = None
    checkpoint_bundle: dict[str, dict[str, Any]] | None = None
    bundle_manifest_digest: str | None = None
    artifact_type: str = "single_file"
    directory_members: list[dict[str, Any]] | None = None
    checkpoint_sha256: str | None = None
def command_is_audited(adapter_type: type[OfficialScriptAdapter]) -> bool:
    return getattr(adapter_type, "command", None) is not OfficialScriptAdapter.command


def pinned_revision(source: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_cwd(model: dict, source: Path) -> Path:
    return source / "src" if model["model_id"] == "csmcir" else source


def candidates(args: argparse.Namespace, models: list[dict]) -> list[tuple[dict, dict]]:
    if args.model:
        models = [model for model in models if model["model_id"] == args.model]
        if not models:
            raise ValueError(f"unknown model: {args.model}")
    selected: list[tuple[dict, dict]] = []
    for model in models:
        checkpoints = model["checkpoint_variants"]
        if model["model_id"] == "limn":
            bundle = model["checkpoint_bundles"][0]
            if args.checkpoint == bundle["bundle_id"] or (not args.checkpoint):
                anchor = next(checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == bundle["required_checkpoint_ids"][0])
                checkpoints = [{**anchor, "checkpoint_id": bundle["bundle_id"], "bundle_id": bundle["bundle_id"]}]
            elif args.checkpoint:
                checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == args.checkpoint]
        elif args.checkpoint:
            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == args.checkpoint]
        if args.checkpoint and not checkpoints and args.model:
            raise ValueError(f"unknown checkpoint for {args.model}: {args.checkpoint}")
        selected.extend((model, checkpoint) for checkpoint in checkpoints)
    if args.checkpoint and not selected:
        raise ValueError(f"unknown checkpoint: {args.checkpoint}")
    return selected


def list_candidates(items: Iterable[tuple[dict, dict]]) -> None:
    for model, checkpoint in items:
        adapter_type = ADAPTERS.get(model["model_id"])
        readiness = "audited" if adapter_type and command_is_audited(adapter_type) else "not audited"
        print(f"[OK] {model['model_id']} / {checkpoint['checkpoint_id']}: protocol={model['native_protocol'] or 'unavailable'} command={readiness}")


def blockers(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, config: WorkbenchConfig) -> tuple[Path, Path, str | None, list[str]]:
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else config.WORKBENCH_THIRD_PARTY_ROOT
    actual_pin = pinned_revision(source) if source.is_dir() else None
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    reasons = runtime_blockers(model, checkpoint, config, protocol, dataset_root)
    if model["model_id"] == "limn" and checkpoint["checkpoint_id"] != checkpoint.get("bundle_id"):
        reasons = [reason for reason in reasons if not reason.startswith("checkpoint bundle incomplete:")]
    if model.get("source_available") and actual_pin is None and source.is_dir():
        reasons.append(f"source pin unavailable: {source}")
    return source, checkpoint_file, actual_pin, reasons


def guarded_plan(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, top_k: int, config: WorkbenchConfig | None = None) -> tuple[EvaluationPlan | None, list[str]]:
    config = config or resolve_config()
    adapter_type = ADAPTERS.get(model["model_id"])
    if adapter_type is None or not command_is_audited(adapter_type):
        return None, ["SKIPPED: adapter command not audited"]
    source, checkpoint_file, actual_pin, blocked = blockers(model, checkpoint, protocol, dataset_root, config)
    script = getattr(adapter_type, "script", None)
    if source.is_dir() and script and not (source / script).is_file():
        blocked.append(f"official evaluator missing: {source / script}")
    if blocked:
        return None, blocked
    assert dataset_root is not None
    request = EvalRequest(model_id=model["model_id"], checkpoint_id=checkpoint["checkpoint_id"], protocol_id=protocol, dataset_root=dataset_root, output_path=config.WORKBENCH_RESULTS_ROOT / protocol / model["model_id"] / f"{checkpoint['checkpoint_id']}.json", top_k=top_k)
    try:
        command = adapter_type().command(source, checkpoint_file, request)
    except Exception as error:
        return None, [f"official command construction failed: {error}"]
    paper_metrics = {key.removeprefix("reported_"): model.get(key) for key in ("reported_r10", "reported_r50", "reported_mean")}
    checkpoint_bundle = None
    bundle_manifest_digest = None
    artifact_type = checkpoint.get("artifact_type", "single_file")
    directory_members = None

    if model["model_id"] == "limn" and checkpoint["checkpoint_id"] == "base_iter0_all_categories":
        artifact_type = "category_specific_whole_models"
        checkpoint_root = checkpoint_file.parent
        checkpoint_bundle = {}
        for cat in ("dress", "shirt", "toptee"):
            fn = f"0_{cat}_best_model.pt"
            fp = checkpoint_root / fn
            c_sha = sha256_file(fp) if fp.is_file() else None
            checkpoint_bundle[cat] = {
                "checkpoint_id": f"base_iter0_{cat}",
                "filename": fn,
                "path": str(fp),
                "sha256": c_sha,
            }
        bundle_manifest_digest = hashlib.sha256(
            json.dumps(checkpoint_bundle, sort_keys=True).encode("utf-8")
        ).hexdigest()
    elif artifact_type in ("run_directory", "bundle_directory"):
        req_files = checkpoint.get("required_files") or [m["filename"] for m in checkpoint.get("bundle_members", [])]
        directory_members = [
            {
                "filename": name,
                "path": str(checkpoint_file / name),
                "sha256": sha256_file(checkpoint_file / name) if (checkpoint_file / name).is_file() else None,
            }
            for name in req_files
        ]
        bundle_manifest_digest = hashlib.sha256(
            json.dumps(directory_members, sort_keys=True).encode("utf-8")
        ).hexdigest()

    checkpoint_digest = None
    if checkpoint_file.is_file():
        checkpoint_digest = sha256_file(checkpoint_file)
    elif checkpoint_file.is_dir():
        required = [member["filename"] for member in (directory_members or [])] or None
        try:
            checkpoint_digest = sha256_file(checkpoint_file, required_files=required)
        except (OSError, ValueError):
            checkpoint_digest = None

    return EvaluationPlan(
        model["model_id"],
        checkpoint["checkpoint_id"],
        protocol,
        dataset_root,
        source,
        checkpoint_file,
        source_cwd(model, source),
        command,
        model.get("upstream_commit_sha"),
        actual_pin,
        paper_metrics,
        checkpoint_bundle=checkpoint_bundle,
        bundle_manifest_digest=bundle_manifest_digest,
        artifact_type=artifact_type,
        directory_members=directory_members,
        checkpoint_sha256=checkpoint_digest,
    ), []


def parser_for(registry: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list registry candidates and command-audit state")
    selector.add_argument("--model", help="select one registry model")
    selector.add_argument("--all-runnable", action="store_true", help="guard every registry candidate for official execution")
    parser.add_argument("--checkpoint", help="select one checkpoint ID")
    parser.add_argument("--protocol", choices=sorted(registry["protocols"]), help="protocol; defaults to each model native protocol")
    parser.add_argument("--dataset-root", type=Path, help="FashionIQ root; CSMCIR requires its fixed upstream layout")
    parser.add_argument("--canonical-dataset-root", type=Path, help="canonical FashionIQ root behind a CSMCIR fixed link")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--run-id", help="invocation run identity supplied by the orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="print guarded command and log plan without writing")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def make_run_id(plan: EvaluationPlan, timestamp: datetime) -> str:
    """Collision-resistant run identity: time + model + checkpoint + random suffix."""
    return f"{timestamp:%Y%m%dT%H%M%S%fZ}_{plan.model_id}_{plan.checkpoint_id}_{uuid.uuid4().hex[:8]}"

def invoke_evaluator(plan: EvaluationPlan, config: WorkbenchConfig, run_id: str,
                     output_root: Path) -> tuple[int, Path, dict]:
    """Run the official command under this invocation's identity and collect evidence.

    Returns ``(return_code, log_directory, environment)``. The environment carries
    the run identity the evaluator is expected to stamp onto any completion proof.
    """
    environment = {
        "orchestrator_python": sys.version,
        "orchestrator_executable": sys.executable,
        "platform": sys.platform,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "evaluation_run_id": run_id,
        "evaluation_output_root": str(output_root),
    }
    directory = log_dir(config, plan, datetime.now(UTC), run_id)
    child_env = dict(os.environ)
    child_env.update({key: str(value) for key, value in environment.items()})

    # Start the process first: if it cannot be started at all, nothing is written and
    # no orphaned log directory or burnt run id is left behind.
    try:
        process = subprocess.Popen(plan.command, cwd=plan.cwd, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env)
    except OSError as error:
        raise RuntimeError(f"cannot start the official command {plan.command[0]!r}: {error}") from error

    try:
        directory.mkdir(parents=True, exist_ok=False)
    except BaseException:
        # The process already started; a directory collision must not orphan it.
        process.kill()
        process.wait()
        raise
    try:
        with (directory / "stdout.log").open("w", encoding="utf-8", newline="") as stdout, \
                (directory / "stderr.log").open("w", encoding="utf-8", newline="") as stderr:

            def tee(stream, destination, terminal):
                for line in stream:
                    destination.write(line)
                    destination.flush()
                    terminal.write(line)
                    terminal.flush()

            threads = [
                threading.Thread(target=tee, args=(process.stdout, stdout, sys.stdout)),
                threading.Thread(target=tee, args=(process.stderr, stderr, sys.stderr)),
            ]
            for thread in threads:
                thread.start()
            code = process.wait()
            for thread in threads:
                thread.join()
    except BaseException:
        process.kill()
        process.wait()
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return code, directory, environment


# Descriptor keys a completion proof may use to spell the run identity it belongs to.
_RUN_ID_KEYS = ("run_id", "evaluation_run_id", "invocation_run_id")
_MODEL_ID_KEYS = ("model_id", "model")
_CHECKPOINT_ID_KEYS = ("checkpoint_id", "checkpoint")
_PROTOCOL_ID_KEYS = ("protocol_id", "protocol")
_SOURCE_COMMIT_KEYS = ("source_actual_commit", "source_commit", "actual_source_commit")
_CHECKPOINT_SHA_KEYS = ("checkpoint_sha256", "checkpoint_digest")


class EvaluationOutputError(RuntimeError):
    """The evaluator's declared output does not prove *this* invocation succeeded."""


def _first(payload: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def validate_run_manifest(plan: EvaluationPlan, proof_path: Path, *, run_id: str,
                          output_root: Path, config: WorkbenchConfig | None = None,
                          require_checksum: bool = False,
                          require_environment_identity: bool = False) -> dict:
    """Validate that ``proof_path`` is a completion proof for *this* invocation.

    A pre-existing report, a pre-existing latest pointer, or a merely-newer
    timestamp is never sufficient: the proof must name the current run and agree
    with this plan's model, checkpoint, protocol, and source provenance.
    """
    if not proof_path.exists():
        raise EvaluationOutputError(f"declared output missing: {proof_path}")
    if not proof_path.is_file():
        raise EvaluationOutputError(f"declared output is not a file: {proof_path}")
    if proof_path.stat().st_size == 0:
        raise EvaluationOutputError(f"declared output is empty: {proof_path}")
    try:
        resolved_proof = proof_path.resolve()
        resolved_root = Path(output_root).resolve()
    except OSError as error:
        raise EvaluationOutputError(f"declared output is unreadable: {error}") from error
    if not (resolved_proof == resolved_root or resolved_proof.is_relative_to(resolved_root)):
        raise EvaluationOutputError(
            f"declared output {resolved_proof} is outside the run output root {resolved_root}"
        )
    try:
        payload = json.loads(resolved_proof.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationOutputError(f"declared output is malformed JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationOutputError("declared output is not a JSON object")

    proof_run = _first(payload, _RUN_ID_KEYS)
    if proof_run != run_id:
        raise EvaluationOutputError(
            f"declared output belongs to run {proof_run!r}, not the current run {run_id!r}"
        )
    for label, keys, expected in (
        ("model", _MODEL_ID_KEYS, plan.model_id),
        ("checkpoint", _CHECKPOINT_ID_KEYS, plan.checkpoint_id),
        ("protocol", _PROTOCOL_ID_KEYS, plan.protocol),
    ):
        declared = _first(payload, keys)
        if declared is None:
            raise EvaluationOutputError(f"declared output does not state its {label}")
        if str(declared) != str(expected):
            raise EvaluationOutputError(
                f"declared output {label} is {declared!r}, expected {expected!r}"
            )
    if plan.pin is not None:
        source_commit = _first(payload, _SOURCE_COMMIT_KEYS)
        if source_commit is None:
            raise EvaluationOutputError("declared output does not state its source commit")
        if str(source_commit) != str(plan.pin):
            raise EvaluationOutputError(
                f"declared output source commit is {source_commit!r}, expected {plan.pin!r}"
            )
    declared_checkpoint = _first(payload, _CHECKPOINT_SHA_KEYS)
    if declared_checkpoint is None:
        raise EvaluationOutputError("declared output does not state its checkpoint digest")
    # The checkpoint identity is part of the completion contract for *every* artifact
    # type, including run/bundle directories.
    from workbench.backend.registry import sha256_file as _sha

    def _hash_checkpoint() -> str | None:
        checkpoint = plan.checkpoint
        # A placeholder path (unresolved registry entry) is not a checkpoint.
        if str(checkpoint) in (".", "") or not checkpoint.exists():
            return None
        try:
            if checkpoint.is_file():
                return _sha(checkpoint)
            if checkpoint.is_dir():
                required = [m["filename"] for m in (plan.directory_members or [])] or None
                return _sha(checkpoint, required_files=required)
        except (OSError, ValueError):
            return None
        return None

    live_digest = _hash_checkpoint() or plan.checkpoint_sha256
    if live_digest is not None and str(declared_checkpoint) != live_digest:
        raise EvaluationOutputError(
            "declared output checkpoint digest does not match the selected checkpoint"
        )
    if live_digest is None:
        # No resolvable checkpoint to re-hash. The proof must at least agree with the
        # checkpoint digest recorded by its own run-scoped report, which is written by
        # the run and cannot be transplanted to another run.
        recorded = _recorded_report_checkpoint_digest(proof_path)
        if recorded is None:
            raise EvaluationOutputError("declared output does not reference a run report")
        if str(declared_checkpoint) != recorded:
            raise EvaluationOutputError(
                "declared output checkpoint digest disagrees with its run report"
            )

    if require_checksum:
        digest = payload.get("report_digest")
        if not isinstance(digest, str) or not digest:
            raise EvaluationOutputError("declared output does not carry a report digest")
        inner = payload.get("report")
        if not isinstance(inner, str):
            raise EvaluationOutputError("declared output does not reference its report")
        report_path = Path(inner)
        if not report_path.is_absolute():
            report_path = resolved_proof.parent / report_path
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise EvaluationOutputError(f"declared output references a missing report: {report_path}")
        try:
            resolved_report = report_path.resolve()
        except OSError as error:
            raise EvaluationOutputError(f"declared output report is unreadable: {error}") from error
        if not (resolved_report == resolved_root or resolved_report.is_relative_to(resolved_root)):
            raise EvaluationOutputError(
                f"declared output report {resolved_report} is outside the artifact root {resolved_root}"
            )
        if sha256_file(report_path) != digest:
            raise EvaluationOutputError("declared output report digest does not match its report")
    if require_environment_identity:
        env_digest = payload.get("environment_digest")
        if not isinstance(env_digest, str) or not env_digest:
            raise EvaluationOutputError("declared output does not carry an environment digest")
        evidence_digest = payload.get("evidence_digest")
        if not isinstance(evidence_digest, str) or not evidence_digest:
            raise EvaluationOutputError("declared output does not carry invocation evidence")
        run_directory = payload.get("run_directory")
        if not isinstance(run_directory, str) or not run_directory:
            raise EvaluationOutputError("declared output does not state its run directory")
        run_dir = Path(run_directory)
        # The proof must belong to THIS invocation: the run directory is derived from
        # the run id, so a copied or renamed proof can never satisfy a new run.
        # The run directory embeds both the checkpoint that was evaluated and the
        # invocation identity, so a copied or renamed proof cannot be re-pointed at a
        # different run or a different checkpoint.
        expected_directory = log_dir(config, plan, datetime.now(UTC), run_id) if config is not None else None
        expected_name = expected_directory.name if expected_directory is not None else f"{plan.checkpoint_id}__{run_id}"
        if run_dir.name != expected_name:
            raise EvaluationOutputError(
                f"declared output belongs to run directory {run_dir.name!r}, "
                f"expected this invocation's {expected_name!r}")
        if not run_dir.is_dir():
            raise EvaluationOutputError(f"declared output run directory is missing: {run_dir}")
        if not run_dir.resolve().is_relative_to(resolved_root):
            raise EvaluationOutputError(f"declared output run directory escapes the artifact root: {run_dir}")
        logs = {}
        for name in ("stdout.log", "stderr.log"):
            log_path = run_dir / name
            if not log_path.is_file():
                raise EvaluationOutputError(f"invocation log missing: {log_path}")
            logs[name[:-4]] = log_path.read_text(encoding="utf-8")
        observed = hashlib.sha256(json.dumps(logs, sort_keys=True).encode("utf-8")).hexdigest()
        if observed != evidence_digest:
            raise EvaluationOutputError("invocation evidence does not match the recorded digest")
        if not (logs["stdout"].strip() or logs["stderr"].strip()):
            raise EvaluationOutputError("invocation produced no observable evaluation output")
    return payload


def validate_run_id(run_id: str) -> str:
    """Reject a run identity that could name a path outside the artifact root.

    Run ids come from the orchestrator (or, in tests, from a caller), so they are
    validated at the entry point rather than trusted: a separator or a parent
    reference would let a run write outside ``workbench/artifacts``.
    """
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")) or ".." in run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    return run_id


def log_dir(config: WorkbenchConfig, plan: EvaluationPlan, timestamp: datetime,
            run_id: str | None = None) -> Path:
    """Where a run's own logs live: one directory per (invocation, checkpoint).

    A single orchestrator invocation may evaluate several checkpoint variants of one
    model, so the directory is qualified by the checkpoint the run actually used.
    """
    repo_root = getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])
    identifier = validate_run_id(run_id) if run_id else make_run_id(plan, timestamp)
    return (repo_root / "workbench" / "artifacts" / "logs"
            / f"{plan.model_id}__{plan.checkpoint_id}__{identifier}")


def expected_run_identity(config: WorkbenchConfig, plan: EvaluationPlan, run_id: str) -> dict[str, str]:
    """The identity every completion proof must carry for this invocation."""
    return {
        "run_id": run_id,
        "run_directory": str(log_dir(config, plan, datetime.now(UTC), run_id)),
        "report": str(run_report_path(config, plan, run_id)),
        "proof": str(completion_proof_path(config, plan, run_id)),
    }


def artifact_root(config: WorkbenchConfig) -> Path:
    repo_root = getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])
    return repo_root / "workbench" / "artifacts"


def reports_root(config: WorkbenchConfig) -> Path:
    return artifact_root(config) / "reports"


def run_report_path(config: WorkbenchConfig, plan: EvaluationPlan, run_id: str) -> Path:
    return reports_root(config) / f"{plan.model_id}_{plan.checkpoint_id}_{run_id}_aggregate.json"


def completion_proof_path(config: WorkbenchConfig, plan: EvaluationPlan, run_id: str) -> Path:
    return reports_root(config) / f"{plan.model_id}_{plan.checkpoint_id}_{run_id}_completion.json"


def latest_pointer(config: WorkbenchConfig, plan: EvaluationPlan, kind: str) -> Path:
    """`kind` is ``attempt`` or ``successful``. Attempts and successes never mix."""
    return reports_root(config) / f"{plan.model_id}_{plan.checkpoint_id}_latest_{kind}.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        temp.replace(path)
    except BaseException:
        # A failed commit must leave the previous pointer intact and no debris behind.
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def execute(plan: EvaluationPlan, config: WorkbenchConfig, run_id: str | None = None) -> int:
    started = datetime.now(UTC)
    run_id = validate_run_id(run_id) if run_id else make_run_id(plan, started)
    root = artifact_root(config)
    reports = reports_root(config)

    checkpoint_local_sha = None
    if plan.checkpoint.is_file():
        checkpoint_local_sha = sha256_file(plan.checkpoint)
    elif plan.checkpoint.is_dir():
        req_files = [m["filename"] for m in plan.directory_members] if plan.directory_members else None
        checkpoint_local_sha = sha256_file(plan.checkpoint, required_files=req_files)

    run_report = run_report_path(config, plan, run_id)
    proof_path = completion_proof_path(config, plan, run_id)
    if run_report.exists() or proof_path.exists():
        raise RuntimeError(f"refusing to overwrite existing run artifacts for run id {run_id}")

    # The evaluator is launched under this invocation's identity; the identity is
    # exported so a repeating evaluator can stamp it onto its own output.
    code, directory, environment = invoke_evaluator(plan, config, run_id, root)
    command_json = json.dumps(plan.command, separators=(",", ":"))
    environment_json = json.dumps(environment, sort_keys=True, separators=(",", ":"))

    metadata = {
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "run_id": run_id,
        "argv": plan.command,
        "cwd": str(plan.cwd),
        "upstream_expected_pin": plan.pin,
        "actual_source_commit": plan.actual_source_commit,
        "checkpoint_local_sha256": checkpoint_local_sha,
        "checkpoint_artifact_type": plan.artifact_type,
        "checkpoint_bundle": plan.checkpoint_bundle,
        "bundle_manifest_digest": plan.bundle_manifest_digest,
        "directory_members": plan.directory_members,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "command_digest": hashlib.sha256(command_json.encode()).hexdigest(),
        "environment": environment,
        "environment_digest": hashlib.sha256(environment_json.encode()).hexdigest(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "return_code": code,
        "dataset_root": str(plan.dataset_root),
    }
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    stdout_content = (directory / "stdout.log").read_text(encoding="utf-8")
    stderr_content = (directory / "stderr.log").read_text(encoding="utf-8")
    extraction = extract_aggregate_metrics(
        plan.model_id,
        stdout_content,
        stderr_content,
        code,
        paper_metrics=plan.paper_metrics,
    )

    paper = plan.paper_metrics or {}
    report = {
        "artifact_type": "official_aggregate_evaluation_report",
        "schema_version": 1,
        "run_id": run_id,
        "run_directory": str(directory),
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        # Execution completion and metric verification are separate states: a run that
        # exited 0 with unusable aggregate output is still a completed execution.
        "execution_status": "EXECUTION_COMPLETED" if code == 0 else "EXECUTION_FAILED",
        "reproduction_status": extraction.extraction_status,
        "per_query_export_available": False,
        "extraction_status": extraction.extraction_status,
        "metric_extraction_available": extraction.observed_metrics is not None,
        "parser_id": extraction.parser_id,
        "parser_version": extraction.parser_version,
        "observed_metrics": extraction.observed_metrics,
        "category_metrics": extraction.category_metrics,
        "paper_sanity_metrics": paper,
        "parity_status": extraction.parity_status,
        "metric_source": extraction.metric_source,
        "command_json": "command.json",
        "stdout_log": "stdout.log",
        "stderr_log": "stderr.log",
        "command_digest": metadata["command_digest"],
        "source_expected_commit": plan.pin,
        "source_actual_commit": plan.actual_source_commit,
        "checkpoint_artifact_type": plan.artifact_type,
        "checkpoint_sha256": checkpoint_local_sha,
        "checkpoint_bundle": plan.checkpoint_bundle,
        "bundle_manifest_digest": plan.bundle_manifest_digest,
        "directory_members": plan.directory_members,
        "environment": environment,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "environment_digest": metadata["environment_digest"],
        "return_code": code,
    }
    report_json = json.dumps(report, indent=2) + "\n"

    # Immutable, run-scoped artifact. Exclusive create: a colliding run id must never
    # overwrite a previous experiment.
    reports.mkdir(parents=True, exist_ok=True)
    try:
        with run_report.open("x", encoding="utf-8") as handle:
            handle.write(report_json)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing run report: {run_report}") from error
    (directory / "aggregate_report.json").write_text(report_json, encoding="utf-8")

    # A completion proof must be backed by evidence *this* invocation produced. If the
    # official command exits 0 but emits nothing at all, there is no run-specific
    # evaluation evidence, so no proof may be minted — and the stage cannot COMPLETE.
    produced_output = bool(stdout_content.strip() or stderr_content.strip())
    evidence_digest = hashlib.sha256(
        json.dumps({"stdout": stdout_content, "stderr": stderr_content},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()
    if code == 0 and not produced_output:
        print("[BLOCKED] official command exited 0 but produced no observable evaluation "
              "output; no completion proof can be minted for this invocation", file=sys.stderr)
        _atomic_write_json(latest_pointer(config, plan, "attempt"), {
            "run_id": run_id,
            "run_directory": str(directory),
            "report": str(run_report),
            "return_code": code,
            "completion_status": "NO_OBSERVABLE_OUTPUT",
            "updated_at": datetime.now(UTC).isoformat(),
        })
        return code

    proof = {
        "artifact_type": "official_evaluation_completion_proof",
        "schema_version": 1,
        "completion_status": "COMPLETED" if code == 0 else "FAILED",
        "run_id": run_id,
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "source_expected_commit": plan.pin,
        "source_actual_commit": plan.actual_source_commit,
        "checkpoint_artifact_type": plan.artifact_type,
        "checkpoint_sha256": checkpoint_local_sha,
        "return_code": code,
        "report": str(run_report),
        "report_digest": sha256_file(run_report),
        "report_relative": run_report.name,
        "metric_extraction_available": extraction.observed_metrics is not None,
        "extraction_status": extraction.extraction_status,
        "environment_digest": metadata["environment_digest"],
        "command_digest": metadata["command_digest"],
        "evidence_digest": evidence_digest,
        "stdout_log": "stdout.log",
        "stderr_log": "stderr.log",
        "run_directory": str(directory),
        "run_id_expected_directory": str(log_dir(config, plan, datetime.now(UTC), run_id)),
        "invocation_identity": expected_run_identity(config, plan, run_id),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    try:
        with proof_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(proof, indent=2) + "\n")
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing completion proof: {proof_path}") from error

    # Attempts always advance; successes only advance on a genuinely successful,
    # proof-backed execution. A failed rerun can never replace latest_successful.
    _atomic_write_json(latest_pointer(config, plan, "attempt"), {
        "run_id": run_id,
        "run_directory": str(directory),
        "report": str(run_report),
        "proof": str(proof_path),
        "return_code": code,
        "completion_status": proof["completion_status"],
        "updated_at": proof["completed_at"],
    })
    if code == 0:
        try:
            validate_run_manifest(plan, proof_path, run_id=run_id, output_root=root,
                                  config=config, require_checksum=True,
                                  require_environment_identity=True)
        except EvaluationOutputError as error:
            print(f"[BLOCKED] completion proof rejected: {error}", file=sys.stderr)
        else:
            _atomic_write_json(latest_pointer(config, plan, "successful"), {
                "run_id": run_id,
                "run_directory": str(directory),
                "report": str(run_report),
                "proof": str(proof_path),
                "return_code": code,
                "completion_status": "COMPLETED",
                "report_digest": proof["report_digest"],
                "updated_at": proof["completed_at"],
            })
    return code


def _recorded_report_checkpoint_digest(proof_path: Path) -> str | None:
    try:
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
        report = Path(str(payload.get("report") or ""))
        if not report.is_file():
            return None
        digest = json.loads(report.read_text(encoding="utf-8")).get("checkpoint_sha256")
        return str(digest) if digest else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def completion_validator(model_id: str, checkpoint_id: str, protocol: str, pin: str | None,
                         config: WorkbenchConfig, checkpoint_file: Path | None = None):
    """Build the pipeline-side validator for one evaluation stage.

    It answers a single question: *did THIS invocation produce a valid completion
    proof?* A leftover report or an old latest pointer is never accepted.
    """
    def _resolve_checkpoint() -> tuple[Path, list[dict[str, Any]] | None]:
        """The checkpoint this (model, checkpoint id) pair evaluates, with its members.

        Directory/bundle checkpoints are identified by their *declared* members, so the
        validator must use exactly the member list the runner used. Hashing every file
        in a run directory would reject a perfectly good proof as soon as the directory
        contains an undeclared extra file (a config, a log, a README).
        """
        if checkpoint_file is not None:
            return Path(checkpoint_file), None
        try:
            from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id

            model = model_by_id(model_id)
            variant = checkpoint_by_id(model, checkpoint_id)
            path = checkpoint_path(model_id, variant, config.WORKBENCH_CHECKPOINT_ROOT)
            required = variant.get("required_files") or [
                member["filename"] for member in (variant.get("bundle_members") or [])
            ]
            members = None
            if required:
                members = [{"filename": name, "path": str(path / name), "sha256": None}
                           for name in required]
            return path, members
        except Exception:
            return Path("."), None

    def _validate(record: dict | None) -> tuple[bool, str | None, dict | None]:
        if not isinstance(record, dict):
            return False, "no evaluation invocation recorded", None
        run_id = record.get("run_id")
        if not run_id:
            return False, "recorded evaluation has no invocation run id", None
        reports = reports_root(config)
        proof_path = reports / f"{model_id}_{checkpoint_id}_{run_id}_completion.json"
        resolved_checkpoint, resolved_members = _resolve_checkpoint()
        plan = EvaluationPlan(model_id, checkpoint_id, protocol, Path("."), Path("."),
                              resolved_checkpoint, Path("."), [], pin, None,
                              directory_members=resolved_members)
        if not (resolved_checkpoint.is_file() or resolved_checkpoint.is_dir()):
            # No checkpoint file to re-hash (e.g. a non-file artifact): bind the proof to
            # the digest recorded by the run itself, which is run-scoped and unforgeable
            # across runs.
            recorded = _recorded_report_checkpoint_digest(proof_path)
            if recorded is None:
                return False, "cannot verify the checkpoint identity of this proof", None
            plan = replace(plan, checkpoint_sha256=recorded)
        try:
            payload = validate_run_manifest(plan, proof_path, run_id=run_id,
                                            output_root=artifact_root(config), config=config,
                                            require_checksum=True,
                                            require_environment_identity=True)
        except EvaluationOutputError as error:
            return False, str(error), None
        if payload.get("return_code") != 0 or payload.get("completion_status") != "COMPLETED":
            return False, f"recorded evaluation did not complete (return code {payload.get('return_code')})", None
        return True, None, {
            "completion_proof": str(proof_path),
            "run_report": str(payload.get("report")),
            "report_digest": payload.get("report_digest"),
            "run_id": run_id,
        }
    return _validate


def main(argv: list[str] | None = None) -> int:
    registry = load_registry()
    args = parser_for(registry).parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    try:
        items = candidates(args, registry["models"])
    except ValueError as error:
        raise SystemExit(str(error))
    if args.list:
        list_candidates(items)
        return 0
    config = resolve_config()
    if args.canonical_dataset_root:
        config = replace(config, FASHIONIQ_ROOT=args.canonical_dataset_root.resolve())
    dataset_root = args.dataset_root or config.FASHIONIQ_ROOT
    failed = False
    for model, checkpoint in items:
        protocol = args.protocol or model["native_protocol"]
        print(f"[RUN] {model['model_id']} / {checkpoint['checkpoint_id']}")
        if protocol is None:
            print("[BLOCKED] protocol unavailable"); failed = True; continue
        plan, reasons = guarded_plan(model, checkpoint, protocol, dataset_root, args.top_k, config)
        if plan is None:
            for reason in reasons: print(f"{'[SKIP]' if reason.startswith('SKIPPED:') else '[BLOCKED]'} {reason}")
            failed = failed or any(not reason.startswith("SKIPPED:") for reason in reasons)
            continue
        planned_logs = log_dir(config, plan, datetime.now(UTC))
        print(f"[OK] protocol: {protocol}")
        print(f"[RUN] cwd: {plan.cwd}")
        print(f"[RUN] command: {shlex.join(plan.command)}")
        print(f"[RUN] log directory: {planned_logs}")
        print(f"[RUN] provenance: {planned_logs / 'command.json'}")
        if args.dry_run:
            print("[OK] dry-run: official command not executed; no logs created")
            continue
        if execute(plan, config, run_id=args.run_id):
            print("[BLOCKED] official command failed"); failed = True
            if not args.continue_on_error: break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
