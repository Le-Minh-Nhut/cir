#!/usr/bin/env python3
"""Guard and run audited official evaluation commands."""
from __future__ import annotations

import fcntl
import stat
import tempfile
import argparse
import hashlib
import json
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
import platform
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.metrics_extraction import extract_aggregate_metrics
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import (
    checkpoint_bundle_missing_paths, checkpoint_is_present, checkpoint_missing_paths,
    checkpoint_path, load_registry, sha256_file,
)
from workbench.backend import runtime
from workbench.backend.preflight import preflight_blockers
from workbench.backend.runtime import runtime_plan_blockers


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
    #: True only when the registry declares the model's source as synthetic. A synthetic evaluator
    #: has no published output format, so its own declared aggregate is the evidence and the report
    #: records that this was the case.
    synthetic_source: bool = False
def _synthetic_aggregate(stdout_text: str) -> dict[str, dict[str, float]] | None:
    """Read the aggregate a registry-declared synthetic evaluator prints.

    The declared schema is one JSON object:

        {"aggregate": {"macro_r10": .., "macro_r50": .., "macro_mean": ..},
         "categories": [{"category": "dress", "r10": .., "r50": ..}, ...]}

    It is the contract for models the registry marks `synthetic_source: true`, which have no
    published output format to parse. Returns None when the object is absent or incomplete, so a
    synthetic run that printed nothing usable is still reported as having no metrics.
    """
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate")
        categories = payload.get("categories")
        if not isinstance(aggregate, dict) or not isinstance(categories, list):
            continue
        emitted = {}
        for item in categories:
            if not isinstance(item, dict):
                continue
            name = item.get("category")
            if name not in ("dress", "shirt", "toptee"):
                continue
            if item.get("r10") is None or item.get("r50") is None:
                continue
            emitted[name] = {"r10": float(item["r10"]), "r50": float(item["r50"])}
        if set(emitted) != {"dress", "shirt", "toptee"}:
            continue
        if aggregate.get("macro_r10") is None or aggregate.get("macro_r50") is None:
            continue
        return {
            "observed": {
                "r10": float(aggregate["macro_r10"]),
                "r50": float(aggregate["macro_r50"]),
                "mean": float(aggregate.get("macro_mean",
                                            (aggregate["macro_r10"] + aggregate["macro_r50"]) / 2)),
            },
            "categories": emitted,
        }
    return None


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
        requested = list(args.checkpoint or [])
        if model["model_id"] == "limn":
            # LIMN evaluates its three per-category checkpoints as one declared bundle. A registry
            # that does not declare one is not a reason to raise from inside candidate resolution:
            # fall back to the plain variants so the model is either evaluated directly or refused
            # by the ordinary guard, with a message that names the real problem.
            bundles = model.get("checkpoint_bundles") or []
            if not bundles:
                if requested:
                    checkpoints = [checkpoint for checkpoint in checkpoints
                                   if checkpoint["checkpoint_id"] in set(requested)]
                if requested and not checkpoints and args.model:
                    raise ValueError(f"unknown checkpoint for {args.model}: {', '.join(requested)}")
                selected.extend((model, checkpoint) for checkpoint in checkpoints)
                continue
            bundle = bundles[0]
            if args.checkpoint == [bundle["bundle_id"]] or not requested:
                anchor = next(checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == bundle["required_checkpoint_ids"][0])
                checkpoints = [{**anchor, "checkpoint_id": bundle["bundle_id"], "bundle_id": bundle["bundle_id"]}]
            elif requested:
                # Naming every member of a declared bundle is the same experiment as naming the
                # bundle, so resolve it to the bundle anchor rather than evaluating the categories
                # as three unrelated runs.
                required = set(bundle["required_checkpoint_ids"])
                if required and set(requested) == required:
                    anchor = next(checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == bundle["required_checkpoint_ids"][0])
                    checkpoints = [{**anchor, "checkpoint_id": bundle["bundle_id"], "bundle_id": bundle["bundle_id"]}]
                else:
                    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] in set(requested)]
        elif requested:
            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] in set(requested)]
        if requested and not checkpoints and args.model:
            raise ValueError(f"unknown checkpoint for {args.model}: {', '.join(requested)}")
        selected.extend((model, checkpoint) for checkpoint in checkpoints)
    if args.checkpoint and not selected:
        raise ValueError(f"unknown checkpoint: {', '.join(args.checkpoint)}")
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
    reasons = runtime_plan_blockers(model, checkpoint, config, protocol, dataset_root)
    if model["model_id"] == "limn" and checkpoint["checkpoint_id"] != checkpoint.get("bundle_id"):
        reasons = [reason for reason in reasons if not reason.startswith("checkpoint bundle incomplete:")]
    shared = preflight_blockers(model, config, dataset_root=dataset_root)
    reasons.extend(reason for reason in shared if not reason.startswith("execution environment not verified:") and reason not in reasons)
    return source, checkpoint_file, actual_pin, list(dict.fromkeys(reasons))

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
        names = [f"0_{cat}_best_model.pt" for cat in ("dress", "shirt", "toptee")]
        checkpoint_bundle = build_artifact_manifest(checkpoint_file.parent, names)
        directory_members = list(checkpoint_bundle.values())
        bundle_manifest_digest = manifest_digest(checkpoint_bundle)
    elif artifact_type in ("run_directory", "bundle_directory"):
        req_files = (checkpoint.get("required_files")
                     or [m["filename"] for m in checkpoint.get("bundle_members", [])])
        manifest = build_artifact_manifest(checkpoint_file, req_files)
        directory_members = list(manifest.values())
        bundle_manifest_digest = manifest_digest(manifest)

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
        synthetic_source=bool(model.get("synthetic_source")),
    ), []


def parser_for(registry: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list registry candidates and command-audit state")
    selector.add_argument("--model", help="select one registry model")
    selector.add_argument("--all-runnable", action="store_true", help="guard every registry candidate for official execution")
    parser.add_argument("--checkpoint", action="append",
                        help="select a checkpoint ID; repeat to select a bundle's members")
    parser.add_argument("--protocol", choices=sorted(registry["protocols"]), help="protocol; defaults to each model native protocol")
    parser.add_argument("--dataset-root", type=Path, help="FashionIQ root; CSMCIR requires its fixed upstream layout")
    parser.add_argument("--canonical-dataset-root", type=Path, help="canonical FashionIQ root behind a CSMCIR fixed link")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--run-id", help="invocation run identity supplied by the orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="print guarded command and log plan without writing")
    parser.add_argument("--verify-gpu", action="store_true",
                        help="authorize or refresh GPU capability verification before/instead of evaluation")
    parser.add_argument("--verify-gpu-only", action="store_true",
                        help="verify GPU readiness only: record the capability and exit, never "
                             "loading a checkpoint or running an evaluator")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def make_run_id(plan: EvaluationPlan, timestamp: datetime) -> str:
    """Collision-resistant run identity: time + model + checkpoint + random suffix."""
    return f"{timestamp:%Y%m%dT%H%M%S%fZ}_{plan.model_id}_{plan.checkpoint_id}_{uuid.uuid4().hex[:8]}"

#: Names an official evaluator legitimately needs. Everything else in the operator's environment
#: is withheld: it is model code, and a research script has no business reading the operator's
#: tokens, cloud credentials or ledger key. An authenticated download belongs in an explicit
#: acquisition step, not in the evaluation.
_EVALUATOR_ENV_ALLOWLIST = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
    # CUDA/GPU visibility: which devices the run may use is a deliberate operator decision.
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    # Python and library discovery for the model's own interpreter.
    "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    # Where models and datasets live, so the evaluator can find its inputs.
    "CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT",
    "WORKBENCH_CHECKPOINT_ROOT", "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT",
    "TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
)

#: Prefixes that are never forwarded, whatever else matches: a prefix rule catches names this
#: list has not seen yet, which is the failure mode a fixed allowlist alone would miss.
_EVALUATOR_ENV_DENY_PREFIXES = (
    "WORKBENCH_LEDGER_KEY", "WORKBENCH_SECRET_",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AZURE_", "GOOGLE_APPLICATION_CREDENTIALS", "GCP_",
    "DOCKER_", "KUBECONFIG", "SLACK_TOKEN", "STRIPE_", "NPM_TOKEN",
)


def _evaluator_environment(identity: dict[str, object]) -> dict[str, str]:
    """Build the environment for the untrusted official evaluator, by allowlist.

    Copying `os.environ` and removing a few names propagated every unrelated credential the
    operator happened to hold — tokens, cloud keys, CI secrets — into model code. Here only names
    the evaluator genuinely needs are forwarded, and anything matching a sensitive prefix is
    refused even if it would otherwise be allowed, so an unfamiliar secret name cannot slip
    through by not being on a list.
    """
    child_env: dict[str, str] = {}
    # An operator may declare extra names a specific model legitimately needs (a library prefix
    # path, a cache location). The declaration is explicit and reviewable, rather than implied by
    # whatever happened to be in the caller's shell.
    extra = os.environ.get("WORKBENCH_EVALUATOR_ENV_PASSTHROUGH", "")
    declared = [name.strip() for name in extra.split(":") if name.strip()]
    for name in (*_EVALUATOR_ENV_ALLOWLIST, *declared):
        value = os.environ.get(name)
        if value is not None:
            child_env[name] = value
    for name in list(child_env):
        if any(name == prefix or name.startswith(prefix)
               for prefix in _EVALUATOR_ENV_DENY_PREFIXES):
            child_env.pop(name, None)
    # The run identity is the operator's own contract with the evaluator, so it is set last and
    # cannot be shadowed by an inherited name.
    child_env.update({key: str(value) for key, value in identity.items()})
    return child_env


def invoke_evaluator(plan: EvaluationPlan, config: WorkbenchConfig, run_id: str,
                     output_root: Path) -> tuple[int, Path, dict]:
    """Run the official command under this invocation's identity and collect evidence.

    Returns ``(return_code, log_directory, environment)``. The environment carries
    the run identity the evaluator is expected to stamp onto any completion proof.
    """
    directory = log_dir(config, plan, datetime.now(timezone.utc), run_id)
    environment = {
        "orchestrator_python": sys.version,
        "orchestrator_executable": sys.executable,
        "platform": sys.platform,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "evaluation_run_id": run_id,
        "evaluation_output_root": str(output_root),
    }
    child_env = _evaluator_environment(environment)

    # Claim the run directory EXCLUSIVELY, before anything is launched. `exist_ok=False` is an
    # atomic filesystem claim, so of two concurrent invocations sharing a run id exactly one
    # wins. An existing directory belongs to an earlier invocation: it is refused as-is, never
    # reused, never deleted, and never silently replaced by a different run id — its evidence
    # must survive untouched.
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to overwrite existing run directory for run id {run_id}: {directory}"
        ) from error

    # Private temp space per run, created before the process so the child's TMPDIR is fixed at
    # launch. The host temp dir is world-readable, so anything the evaluator writes there
    # outlives the run and is exposed to every local user; a run-scoped 0700 directory keeps
    # leaked material inside a tree the operator already controls.
    temp_root = directory / "tmp"
    temp_root.mkdir(mode=0o700)
    os.chmod(temp_root, 0o700)
    child_env["TMPDIR"] = str(temp_root)
    environment["TMPDIR"] = str(temp_root)

    try:
        process = subprocess.Popen(plan.command, cwd=plan.cwd, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env)
    except OSError as error:
        # Only this invocation's own freshly claimed directory is removed; nothing else can be
        # in it, because the claim above refuses a directory that already existed. The run id
        # is released with it, so no burnt identity is left behind.
        shutil.rmtree(directory, ignore_errors=True)
        raise RuntimeError(f"cannot start the official command {plan.command[0]!r}: {error}") from error
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
    if payload.get("completion_status") != "COMPLETED":
        raise EvaluationOutputError("declared output is not a successful completion proof")
    if payload.get("return_code") != 0:
        raise EvaluationOutputError("declared output has a nonzero native return code")

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
        members = [m["filename"] for m in (plan.directory_members or [])]
        try:
            if checkpoint.is_dir() and members:
                # Directory/bundle artifacts are identified by their declared members:
                # an undeclared extra file in a run directory must not change identity.
                return _sha(checkpoint, required_files=members)
            if checkpoint.is_file():
                return _sha(checkpoint)
            if checkpoint.is_dir():
                return _sha(checkpoint)
        except (OSError, ValueError):
            return None
        return None

    def _members_live() -> bool:
        """Compare the live artifact members with the manifest recorded by the *run*.

        Two independent checks, so neither a rewritten proof nor an absent manifest can
        disable change detection:

        * the live member files must reproduce the manifest digest the proof records;
        * that digest must equal the one the run's own report records.

        The report is corroborated by the out-of-band ledger, so rewriting the proof alone
        cannot re-bless changed checkpoint members.
        """
        multi_member = (plan.artifact_type in ("run_directory", "bundle_directory",
                                               "category_specific_whole_models")
                        or bool(plan.checkpoint_bundle)
                        or bool(plan.directory_members))
        if not multi_member:
            return True

        members: dict[str, dict[str, Any]] = {}
        for name, member in (plan.checkpoint_bundle or {}).items():
            members[name] = member
        for member in plan.directory_members or []:
            name = str(member.get("filename") or "")
            if name:
                members.setdefault(name, member)

        recorded_manifest = payload.get("bundle_manifest_digest")
        if not recorded_manifest:
            # An absent manifest may not disable change detection: the per-member digests
            # the run recorded are the fallback, and they must actually be present.
            recorded_by_name = {str(m.get("filename")): m.get("sha256")
                                for m in (plan.directory_members or []) if m.get("filename")}
            checked = False
            for name, member in members.items():
                path = Path(str(member.get("path") or ""))
                if not path.is_file():
                    raise EvaluationOutputError(f"checkpoint member {name} is missing: {path}")
                expected = member.get("sha256") or recorded_by_name.get(str(name))
                if expected:
                    checked = True
                    if _sha(path) != expected:
                        raise EvaluationOutputError(
                            f"checkpoint member {name} no longer matches its recorded digest")
            if not checked:
                raise EvaluationOutputError(
                    "no recorded member digest is available for this multi-member checkpoint")
            recorded_report = _recorded_report_manifest(proof_path)
            if recorded_report:
                raise EvaluationOutputError(
                    "declared output does not record the artifact manifest its run produced")
            return True

        if not members:
            raise EvaluationOutputError(
                "declared output records an artifact manifest but no members could be resolved")
        live: dict[str, dict[str, Any]] = {}
        for name, member in members.items():
            path = Path(str(member.get("path") or ""))
            if not path.is_file():
                raise EvaluationOutputError(f"checkpoint member {name} is missing: {path}")
            live[name] = {
                "filename": member.get("filename") or path.name,
                "path": str(path),
                "sha256": _sha(path),
            }
        if manifest_digest(live) != recorded_manifest:
            raise EvaluationOutputError(
                "checkpoint members no longer match the recorded artifact manifest")
        recorded_report = _recorded_report_manifest(proof_path)
        if recorded_report is not None and recorded_report != recorded_manifest:
            raise EvaluationOutputError(
                "declared output artifact manifest disagrees with its run report")
        return True

    def _hash_checkpoint() -> str | None:
        checkpoint = plan.checkpoint
        # A placeholder path (unresolved registry entry) is not a checkpoint.
        if str(checkpoint) in (".", "") or not checkpoint.exists():
            return None
        members = [m["filename"] for m in (plan.directory_members or [])]
        try:
            if checkpoint.is_dir() and members:
                # Directory/bundle artifacts are identified by their declared members:
                # an undeclared extra file in a run directory must not change identity.
                return _sha(checkpoint, required_files=members)
            if checkpoint.is_file():
                return _sha(checkpoint)
            if checkpoint.is_dir():
                return _sha(checkpoint)
        except (OSError, ValueError):
            return None
        return None

    def _bundle_members_live() -> bool:
        """Every bundle member must still match the manifest digest the proof recorded.

        The proof carries ``bundle_manifest_digest`` (computed by the run). Recomputing the
        members here and comparing them to *fresh* hashes would always agree, so the
        manifest recorded by the proof is what is compared against the live member files.
        """
        recorded_manifest = payload.get("bundle_manifest_digest")
        members = plan.checkpoint_bundle or {}
        if not recorded_manifest:
            # Without a manifest digest the bundle is still identified by the per-member
            # digests the proof recorded: a changed member must invalidate the proof.
            if not members:
                return True
            recorded_members = plan.directory_members or []
            recorded_by_name = {
                str(member.get("filename")): member.get("sha256")
                for member in recorded_members if member.get("filename")
            }
            for category, member in members.items():
                path = Path(str(member.get("path") or ""))
                if not path.is_file():
                    raise EvaluationOutputError(
                        f"checkpoint bundle member {category} is missing: {path}")
                expected = member.get("sha256") or recorded_by_name.get(str(member.get("filename")))
                if expected and _sha(path) != expected:
                    raise EvaluationOutputError(
                        f"checkpoint bundle member {category} no longer matches its recorded digest")
            return True
        if not members:
            raise EvaluationOutputError(
                "proof records a bundle manifest but no bundle members could be resolved")
        live = {}
        for category, member in members.items():
            path = Path(str(member.get("path") or ""))
            if not path.is_file():
                raise EvaluationOutputError(
                    f"checkpoint bundle member {category} is missing: {path}")
            live[category] = {
                "checkpoint_id": member.get("checkpoint_id"),
                "filename": member.get("filename") or path.name,
                "path": str(path),
                "sha256": _sha(path),
            }
        recomputed = hashlib.sha256(
            json.dumps(live, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if recomputed != recorded_manifest:
            raise EvaluationOutputError(
                "checkpoint bundle members no longer match the recorded bundle manifest")
        return True

    # A bundle is one experiment: every recorded member is verified, not just the anchor.
    _members_live()
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
        # The report records the run's own outcome; the proof may not disagree with it.
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationOutputError(f"declared output report is unreadable: {error}") from error
        if not isinstance(report_payload, dict):
            raise EvaluationOutputError("declared output report is not a JSON object")
        for label, report_key, proof_value in (
            ("run", "run_id", payload.get("run_id")),
            ("model", "model_id", payload.get("model_id")),
            ("checkpoint", "checkpoint_id", payload.get("checkpoint_id")),
            ("protocol", "protocol_id", payload.get("protocol_id")),
            ("return code", "return_code", payload.get("return_code")),
            ("invocation status", "invocation_status", payload.get("invocation_status")),
        ):
            if str(report_payload.get(report_key)) != str(proof_value):
                raise EvaluationOutputError(
                    f"declared output {label} ({proof_value!r}) disagrees with its report "
                    f"({report_payload.get(report_key)!r})")
        report_status = str(report_payload.get("execution_status") or "")
        if report_status and report_status != "EXECUTION_COMPLETED":
            raise EvaluationOutputError(
                f"declared output reports execution status {report_status!r}")
        # Cross-check the out-of-band ledger: an actor who rewrites every artifact inside
        # the run directory still has to rewrite the whole chained ledger.
        claimed_entry = payload.get("ledger_entry_digest")
        if config is not None:
            # Every completion is ledger-backed: the claim is required, not optional, so a
            # rewritten proof cannot switch corroboration off by dropping the field.
            if not claimed_entry:
                raise EvaluationOutputError(
                    "declared output does not name its evidence-ledger entry")
            ledger = read_evidence_ledger(config)
            if ledger is None:
                raise EvaluationOutputError(
                    "the evidence ledger is missing or its chain is broken, so the recorded "
                    "invocation cannot be corroborated")
        else:
            ledger = None
        if ledger is not None:
            matching = [item for item in ledger if item.get("run_id") == payload.get("run_id")]
            if not matching:
                raise EvaluationOutputError(
                    "no evidence-ledger entry for this run: the invocation was never recorded")
            witness = matching[-1]
            if claimed_entry and witness.get("entry_digest") != claimed_entry:
                raise EvaluationOutputError(
                    "declared output names a different evidence-ledger entry than the ledger holds")
            for label, key, proof_value in (
                ("model", "model_id", payload.get("model_id")),
                ("checkpoint", "checkpoint_id", payload.get("checkpoint_id")),
                ("protocol", "protocol_id", payload.get("protocol_id")),
                ("invocation status", "invocation_status", payload.get("invocation_status")),
                ("return code", "return_code", payload.get("return_code")),
            ):
                if str(witness.get(key)) != str(proof_value):
                    raise EvaluationOutputError(
                        f"declared output {label} ({proof_value!r}) disagrees with the "
                        f"evidence ledger ({witness.get(key)!r})")
            if witness.get("return_code") not in (0, None):
                raise EvaluationOutputError(
                    f"the evidence ledger records a failed execution "
                    f"(return code {witness.get('return_code')})")
            if str(witness.get("report_digest") or "") != str(payload.get("report_digest") or ""):
                raise EvaluationOutputError(
                    "declared output report digest disagrees with the evidence ledger")
        report_evidence = report_payload.get("evidence_digest")
        if isinstance(report_evidence, str) and report_evidence and report_evidence != payload.get("evidence_digest"):
            raise EvaluationOutputError("declared output evidence digest disagrees with its report")
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
        expected_directory = log_dir(config, plan, datetime.now(timezone.utc), run_id) if config is not None else None
        expected_name = expected_directory.name if expected_directory is not None else f"{plan.checkpoint_id}__{run_id}"
        if run_dir.name != expected_name:
            raise EvaluationOutputError(
                f"declared output belongs to run directory {run_dir.name!r}, "
                f"expected this invocation's {expected_name!r}")
        if not run_dir.is_dir():
            raise EvaluationOutputError(f"declared output run directory is missing: {run_dir}")
        if not run_dir.resolve().is_relative_to(resolved_root):
            raise EvaluationOutputError(f"declared output run directory escapes the artifact root: {run_dir}")
        # ``command.json`` is written by the runner around the subprocess itself. It is
        # not derived from the report or the proof, so it is the run's independent witness
        # that the execution really succeeded.
        command_path = run_dir / "command.json"
        if not command_path.is_file():
            raise EvaluationOutputError(f"invocation record missing: {command_path}")
        try:
            command_payload = json.loads(command_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationOutputError(f"invocation record is unreadable: {error}") from error
        for label, key, proof_value in (
            ("run", "run_id", payload.get("run_id")),
            ("model", "model_id", payload.get("model_id")),
            ("checkpoint", "checkpoint_id", payload.get("checkpoint_id")),
            ("protocol", "protocol_id", payload.get("protocol_id")),
            ("return code", "return_code", payload.get("return_code")),
        ):
            if str(command_payload.get(key)) != str(proof_value):
                raise EvaluationOutputError(
                    f"declared output {label} ({proof_value!r}) disagrees with the "
                    f"invocation record ({command_payload.get(key)!r})")
        if command_payload.get("invocation_status") != "INVOCATION_COMPLETED":
            raise EvaluationOutputError("the invocation record is not a completed run")
        if command_payload.get("return_code") not in (0, None):
            raise EvaluationOutputError(
                f"the invocation record shows a failed execution "
                f"(return code {command_payload.get('return_code')})")

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
    artifacts_root = getattr(config, "WORKBENCH_ARTIFACTS_ROOT", None)
    if artifacts_root is None:
        artifacts_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])) / "workbench" / "artifacts"
    identifier = validate_run_id(run_id) if run_id else make_run_id(plan, timestamp)
    return (Path(artifacts_root) / "logs" / f"{plan.model_id}__{plan.checkpoint_id}__{identifier}")


def expected_run_identity(config: WorkbenchConfig, plan: EvaluationPlan, run_id: str) -> dict[str, str]:
    """The identity every completion proof must carry for this invocation."""
    return {
        "run_id": run_id,
        "run_directory": str(log_dir(config, plan, datetime.now(timezone.utc), run_id)),
        "report": str(run_report_path(config, plan, run_id)),
        "proof": str(completion_proof_path(config, plan, run_id)),
    }


def _plan_members(plan: "EvaluationPlan") -> dict[str, dict[str, Any]] | None:
    """The plan's multi-member manifest, with each member's digest taken from disk.

    A plan may carry recorded members whose digests are placeholders (the pipeline builds
    them before the run). The manifest that identity is computed over must describe the
    files as they are, so the digests are read here rather than copied from the plan.
    """
    names: list[str] = []
    root: Path | None = None
    for name in plan.directory_members or []:
        filename = str(name.get("filename") or "")
        if filename:
            names.append(filename)
            root = root or Path(str(name.get("path") or "")).parent
    for name in (plan.checkpoint_bundle or {}):
        names.append(str(name))
    if not names:
        return None
    if root is None:
        first = next(iter((plan.checkpoint_bundle or {}).values()), None)
        root = Path(str(first.get("path") or "")).parent if first else None
    if root is None:
        return None
    return build_artifact_manifest(root, sorted(set(names)))


def build_artifact_manifest(checkpoint_path_value: Path, members: list[str]) -> dict[str, dict[str, Any]]:
    """The canonical member manifest for a multi-file checkpoint artifact.

    Bundle checkpoints (LIMN's three category models) and run/bundle directories are
    described by the *same* shape: a filename-keyed record of filename, path, and digest.
    One builder means the planner, the evaluator, and the validator cannot disagree about
    what was consumed.
    """
    manifest: dict[str, dict[str, Any]] = {}
    for name in members:
        member_path = Path(checkpoint_path_value) / name
        manifest[name] = {
            "filename": name,
            "path": str(member_path),
            "sha256": sha256_file(member_path) if member_path.is_file() else None,
        }
    return manifest


def manifest_digest(manifest: dict[str, dict[str, Any]] | None) -> str | None:
    if not manifest:
        return None
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def evidence_ledger_path(config: WorkbenchConfig) -> Path:
    """Append-only record of every invocation, kept *outside* the run directory."""
    artifacts_root = getattr(config, "WORKBENCH_ARTIFACTS_ROOT", None)
    if artifacts_root is None:
        artifacts_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])) / "workbench" / "artifacts"
    return Path(artifacts_root) / "evidence_ledger.jsonl"



class LedgerKeyError(RuntimeError):
    """The evidence ledger cannot be authenticated without its operator key."""


class LedgerCorruptError(RuntimeError):
    """The existing evidence ledger is malformed or does not verify."""


class LedgerDuplicateRunError(RuntimeError):
    """An invocation run id already exists in the evidence ledger."""
def ledger_hmac_key(config: WorkbenchConfig) -> bytes:
    """Load the operator key, creating a private random key only for a new ledger."""
    configured = os.environ.get("WORKBENCH_LEDGER_KEY")
    if configured is not None:
        key = configured.strip().encode("utf-8")
        if key:
            return key
        raise LedgerKeyError("WORKBENCH_LEDGER_KEY is set but empty")

    # The fallback key lives in the operator's private config directory, not in the repository: the
    # repository root is forwarded to the official evaluator child, so a key kept there is readable
    # by the very code the ledger is meant to be independent of. A key already present at the old
    # repository-root location is still honoured, so an existing ledger stays verifiable.
    repo_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2]))
    legacy_key_path = repo_root / ".workbench_ledger_key"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    private_key_path = config_home / "cir" / "workbench_ledger_key"
    if legacy_key_path.is_file() and private_key_path.is_file():
        # Two key files is a state that cannot be resolved by preference: whichever this function
        # picked would be the wrong one for a ledger signed with the other, and the operator would
        # see only an unverifiable ledger. Fail closed and say exactly what to do.
        raise LedgerKeyError(
            "two ledger keys exist "
            f"({legacy_key_path} and {private_key_path}); keep one and remove the other, or set "
            "WORKBENCH_LEDGER_KEY explicitly to choose")
    if legacy_key_path.is_file():
        key_path = legacy_key_path
        key_lock_path = repo_root / ".workbench_ledger_key.lock"
    else:
        private_dir = config_home / "cir"
        key_path = private_key_path
        key_lock_path = private_dir / "workbench_ledger_key.lock"
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(key_path.parent, 0o700)
        except OSError:
            pass
        lock_fd = os.open(key_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(lock_fd, "r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                try:
                    mode = key_path.lstat().st_mode
                except FileNotFoundError:
                    mode = None
                if mode is not None:
                    if not stat.S_ISREG(mode):
                        raise LedgerKeyError("ledger key file is not a regular file")
                    key = key_path.read_bytes()
                    if not key.strip():
                        raise LedgerKeyError("ledger key file is empty")
                    return key
                if evidence_ledger_path(config).exists():
                    raise LedgerKeyError("ledger key is missing for an existing evidence ledger")
                key = secrets.token_bytes(32)
                fd, temporary = tempfile.mkstemp(prefix=".workbench_ledger_key.", dir=key_path.parent)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(key)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, key_path)
                except BaseException:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    Path(temporary).unlink(missing_ok=True)
                    raise
                return key
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    except LedgerKeyError:
        raise
    except OSError as error:
        raise LedgerKeyError("cannot read or create the evidence ledger key") from error


def _ledger_authentication(config: WorkbenchConfig, record_without_mac: dict) -> str:
    import hmac as _hmac

    key = ledger_hmac_key(config)
    canonical = json.dumps(record_without_mac, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _ledger_entries_unlocked(config: WorkbenchConfig, path: Path) -> list[dict]:
    """Read and verify a ledger while caller holds its lock."""
    ledger_hmac_key(config)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise LedgerCorruptError("evidence ledger cannot be read") from error
    if raw and not raw.endswith(b"\n"):
        raise LedgerCorruptError("evidence ledger ends with an incomplete line")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise LedgerCorruptError("evidence ledger is not valid UTF-8") from error

    entries: list[dict] = []
    previous = ""
    seen_runs: set[str] = set()
    try:
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise LedgerCorruptError("evidence ledger entry is not an object")
            claimed = record.pop("entry_digest", None)
            if not isinstance(claimed, str) or record.get("previous_digest") != previous:
                raise LedgerCorruptError("evidence ledger chain is broken")
            if _ledger_authentication(config, record) != claimed:
                raise LedgerCorruptError("evidence ledger authentication failed")
            run_id = str(record.get("run_id") or "")
            if run_id and run_id in seen_runs:
                raise LedgerCorruptError("evidence ledger contains a duplicate run id")
            if run_id:
                seen_runs.add(run_id)
            record["entry_digest"] = claimed
            entries.append(record)
            previous = claimed
    except LedgerKeyError:
        raise
    except LedgerCorruptError:
        raise
    except LedgerDuplicateRunError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise LedgerCorruptError("evidence ledger entry is malformed") from error
    return entries


def _ledger_lock(path: Path, operation: int):
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    lock = os.fdopen(fd, "r+b")
    try:
        fcntl.flock(lock, operation)
    except BaseException:
        lock.close()
        raise
    return lock


def append_evidence_entry(config: WorkbenchConfig, entry: dict) -> dict:
    """Append one invocation record, chained and authenticated under an exclusive lock."""
    path = evidence_ledger_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _ledger_lock(path, fcntl.LOCK_EX)
    try:
        entries = _ledger_entries_unlocked(config, path)
        run_id = str(entry.get("run_id") or "")
        if run_id and any(item.get("run_id") == run_id for item in entries):
            raise LedgerDuplicateRunError("evidence ledger already records this run id")
        previous = entries[-1]["entry_digest"] if entries else ""
        record = {**entry, "previous_digest": previous}
        record["entry_digest"] = _ledger_authentication(config, record)
        with path.open("ab") as handle:
            handle.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return record
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def read_evidence_ledger(config: WorkbenchConfig) -> list[dict] | None:
    """Every authenticated ledger entry, or None when the chain does not verify."""
    path = evidence_ledger_path(config)
    try:
        lock = _ledger_lock(path, fcntl.LOCK_SH)
        try:
            if not path.is_file():
                return None
            return _ledger_entries_unlocked(config, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    except (LedgerKeyError, LedgerCorruptError, OSError):
        return None

def artifact_root(config: WorkbenchConfig) -> Path:
    artifacts_root = getattr(config, "WORKBENCH_ARTIFACTS_ROOT", None)
    if artifacts_root is None:
        artifacts_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2])) / "workbench" / "artifacts"
    return Path(artifacts_root)


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
    started = datetime.now(timezone.utc)
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
    # The ledger key and chain must be usable *before* the evaluator starts: an invocation
    # that cannot be audited must not be launched at all, otherwise a started run would leave
    # no audit entry.
    try:
        ledger_hmac_key(config)
        if evidence_ledger_path(config).exists() and read_evidence_ledger(config) is None:
            raise LedgerCorruptError("evidence ledger does not verify")
        # The ledger itself refuses a duplicate run id, but only when the entry is appended —
        # which is after the evaluator has already run. Two plans sharing an explicit run id
        # would therefore both start, and the second would be unauditable. Check first: an
        # invocation that cannot be recorded must not be launched.
        existing = [item.get("run_id") for item in read_evidence_ledger(config) or []]
        if run_id in existing:
            raise LedgerDuplicateRunError(f"evidence ledger already records run id {run_id}")
    except LedgerKeyError as error:
        raise RuntimeError(f"evaluation could not be audited: {error}") from error
    except LedgerDuplicateRunError as error:
        raise RuntimeError(f"evaluation could not be audited: {error}") from error

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
        "bundle_manifest_digest": (plan.bundle_manifest_digest
                                   or manifest_digest(_plan_members(plan))),
        "directory_members": plan.directory_members,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "command_digest": hashlib.sha256(command_json.encode()).hexdigest(),
        "environment": environment,
        "environment_digest": hashlib.sha256(environment_json.encode()).hexdigest(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": code,
        "dataset_root": str(plan.dataset_root),
    }
    stdout_content = (directory / "stdout.log").read_text(encoding="utf-8")
    stderr_content = (directory / "stderr.log").read_text(encoding="utf-8")
    produced_output = bool(stdout_content.strip() or stderr_content.strip())
    invocation_status = ("INVOCATION_FAILED" if code != 0 else
                         "INVOCATION_COMPLETED" if produced_output else
                         "INVOCATION_NO_OBSERVABLE_OUTPUT")
    metadata["invocation_status"] = invocation_status
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Bound before the report is serialised: the report itself carries the digest of the
    # invocation output, so a log edit invalidates the report as well as the proof.
    evidence_digest = hashlib.sha256(
        json.dumps({"stdout": stdout_content, "stderr": stderr_content},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()
    extraction = extract_aggregate_metrics(
        plan.model_id,
        stdout_content,
        stderr_content,
        code,
        paper_metrics=plan.paper_metrics,
    )
    # A model the registry marks as synthetic has no published output format, so its evaluator's own
    # JSON aggregate is the evidence. This is an explicit registry declaration, not a guess from the
    # model id, and it is recorded in the report so a reader can tell the two apart.
    synthetic = bool(getattr(plan, "synthetic_source", False))
    if synthetic and code == 0 and extraction.category_metrics is None and extraction.observed_metrics is None:
        declared = _synthetic_aggregate(stdout_content)
        if declared is not None:
            extraction = replace(
                extraction,
                extraction_status="EXTRACTED_SYNTHETIC",
                parser_id=extraction.parser_id or "synthetic_declared_aggregate",
                observed_metrics=declared["observed"],
                category_metrics=declared["categories"],
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
        "invocation_status": invocation_status,
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
        "is_synthetic": synthetic,
        "command_json": "command.json",
        "stdout_log": "stdout.log",
        "stderr_log": "stderr.log",
        "command_digest": metadata["command_digest"],
        "source_expected_commit": plan.pin,
        "source_actual_commit": plan.actual_source_commit,
        "checkpoint_artifact_type": plan.artifact_type,
        "checkpoint_sha256": checkpoint_local_sha,
        "checkpoint_bundle": plan.checkpoint_bundle,
        "bundle_manifest_digest": (plan.bundle_manifest_digest
                                   or manifest_digest(_plan_members(plan))),
        "directory_members": plan.directory_members,
        "environment": environment,
        "model_interpreter": plan.command[0] if plan.command else sys.executable,
        "environment_digest": metadata["environment_digest"],
        "evidence_digest": evidence_digest,
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


    # Append the out-of-band witness before any pointer moves; failed and silent invocations
    # are recorded exactly like successful ones, but never earn a completion proof.
    ledger_entry = append_evidence_entry(config, {
        "run_id": run_id,
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "return_code": code,
        "invocation_status": invocation_status,
        "execution_status": report["execution_status"],
        "report_digest": sha256_file(run_report),
        "evidence_digest": evidence_digest,
        "run_directory": str(directory),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })
    if invocation_status == "INVOCATION_NO_OBSERVABLE_OUTPUT":
        print("[BLOCKED] official command exited 0 but produced no observable evaluation "
              "output; no completion proof can be minted for this invocation", file=sys.stderr)
        _atomic_write_json(latest_pointer(config, plan, "attempt"), {
            "run_id": run_id,
            "run_directory": str(directory),
            "report": str(run_report),
            "return_code": code,
            "completion_status": "NO_OBSERVABLE_OUTPUT",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return code

    proof = {
        "artifact_type": "official_evaluation_completion_proof",
        "schema_version": 1,
        "completion_status": "COMPLETED" if invocation_status == "INVOCATION_COMPLETED" and code == 0 else "FAILED",
        "run_id": run_id,
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "source_expected_commit": plan.pin,
        "source_actual_commit": plan.actual_source_commit,
        "checkpoint_artifact_type": plan.artifact_type,
        "checkpoint_sha256": checkpoint_local_sha,
        # A bundle is one experiment: the manifest recorded here is what the validator
        # compares against the live member files, so a changed member invalidates the proof.
        "bundle_manifest_digest": (plan.bundle_manifest_digest
                                   or manifest_digest(_plan_members(plan))),
        "checkpoint_bundle": plan.checkpoint_bundle,
        "directory_members": plan.directory_members,
        "invocation_status": invocation_status,
        "return_code": code,
        "ledger_entry_digest": ledger_entry["entry_digest"],
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
        "run_id_expected_directory": str(log_dir(config, plan, datetime.now(timezone.utc), run_id)),
        "invocation_identity": expected_run_identity(config, plan, run_id),
        "completed_at": datetime.now(timezone.utc).isoformat(),
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
    if invocation_status == "INVOCATION_COMPLETED" and code == 0:
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


def _recorded_report_manifest(proof_path: Path) -> str | None:
    """The artifact manifest digest recorded by the proof's own run report."""
    try:
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
        report = Path(str(payload.get("report") or ""))
        if not report.is_file():
            return None
        value = json.loads(report.read_text(encoding="utf-8")).get("bundle_manifest_digest")
        return str(value) if value else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def completion_validator(model_id: str, checkpoint_id: str, protocol: str, pin: str | None,
                         config: WorkbenchConfig, checkpoint_file: Path | None = None):
    """Build the pipeline-side validator for one evaluation stage.

    It answers a single question: *did THIS invocation produce a valid completion
    proof?* A leftover report or an old latest pointer is never accepted.
    """
    def _resolve_bundle() -> dict[str, dict[str, Any]] | None:
        """Recorded bundle members for a bundled checkpoint, in the canonical shape.

        ``checkpoint_id`` may be a *bundle* id, which is not itself a checkpoint variant:
        the bundle's required member checkpoints are the ones the evaluator consumes. The
        shape must be exactly what ``build_artifact_manifest`` produces — filename-keyed
        with filename/path/digest — or the live manifest cannot reproduce the recorded
        digest and an unchanged bundle would be rejected.
        """
        try:
            from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id

            model = model_by_id(model_id)
            bundle = next((b for b in model.get("checkpoint_bundles") or []
                           if b["bundle_id"] == checkpoint_id), None)
            if bundle is None:
                return None
            names: list[str] = []
            for member_id in bundle["required_checkpoint_ids"]:
                variant = checkpoint_by_id(model, member_id)
                names.append(str(checkpoint_path(model_id, variant,
                                                 config.WORKBENCH_CHECKPOINT_ROOT).name))
            root = config.WORKBENCH_CHECKPOINT_ROOT / model_id
            return build_artifact_manifest(root, names)
        except Exception:
            return None

    def _resolve_directory_manifest(path: Path, members: list[dict[str, Any]] | None
                                    ) -> dict[str, dict[str, Any]] | None:
        names = [str(m.get("filename")) for m in (members or []) if m.get("filename")]
        if not names:
            return None
        return build_artifact_manifest(path, names)

    def _resolve_checkpoint() -> tuple[Path, list[dict[str, Any]] | None]:
        """The checkpoint this (model, checkpoint id) pair evaluates, with its members.

        Directory/bundle checkpoints are identified by their *declared* members, so the
        validator must use exactly the member list the runner used. Hashing every file
        in a run directory would reject a perfectly good proof as soon as the directory
        contains an undeclared extra file (a config, a log, a README).
        """
        if checkpoint_file is not None:
            # The caller knows the path but not necessarily the member contract: resolve
            # the declared members from the registry so a directory artifact is never
            # hashed over undeclared extras.
            try:
                from workbench.backend.registry import checkpoint_by_id, model_by_id

                model = model_by_id(model_id)
                variant = checkpoint_by_id(model, checkpoint_id)
                required = variant.get("required_files") or [
                    member["filename"] for member in (variant.get("bundle_members") or [])
                ]
                members = ([{"filename": name, "path": str(Path(checkpoint_file) / name),
                             "sha256": None} for name in required] if required else None)
            except Exception:
                members = None
            return Path(checkpoint_file), members
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
        resolved_bundle = _resolve_bundle()
        resolved_directory = _resolve_directory_manifest(resolved_checkpoint, resolved_members)
        if resolved_directory is not None and resolved_bundle is None:
            resolved_bundle = resolved_directory
        plan = EvaluationPlan(model_id, checkpoint_id, protocol, Path("."), Path("."),
                              resolved_checkpoint, Path("."), [], pin, None,
                              directory_members=resolved_members,
                              checkpoint_bundle=resolved_bundle,
                              bundle_manifest_digest=manifest_digest(resolved_bundle))
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


def verify_gpu_only(items: list[tuple[dict, dict]], config: WorkbenchConfig) -> int:
    """Probe GPU readiness for each selected model, record the capability, and stop.

    This is the documented hardware preflight. It builds no evaluation plan, so it cannot load
    a checkpoint, start an evaluator, compute Recall, write a completion proof, or move
    ``latest_successful``. A model that declares no CUDA requirement is reported as such rather
    than probed.
    """
    failed = False
    for model, checkpoint in items:
        gpu = (model.get("environment") or {}).get("cuda")
        gpu_required = gpu is not None and str(gpu).upper() != "UNKNOWN"
        print(f"[RUN] {model['model_id']} / {checkpoint['checkpoint_id']}")
        if not gpu_required:
            print(f"[SKIP] {model['model_id']} declares no CUDA requirement")
            continue
        try:
            capability = runtime.verify_gpu_runtime(model, config, authorized=True,
                                                    require_cuda=True, force=True)
        except (runtime.GpuVerificationError, RuntimeError) as error:
            print(f"[BLOCKED] GPU verification failed: {error}", file=sys.stderr)
            failed = True
            continue
        print(json.dumps(capability, sort_keys=True))
        capability_file = runtime.gpu_capability_path(config, capability.get("capability_key"))
        print(f"[OK] capability recorded: {capability_file}")
    return 1 if failed else 0


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
    if args.verify_gpu_only:
        # An explicit hardware preflight: probe and record, then stop. It must not build a plan
        # at all, because planning is what loads checkpoint and dataset contracts.
        return verify_gpu_only(items, config)
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
        planned_logs = log_dir(config, plan, datetime.now(timezone.utc))
        print(f"[OK] protocol: {protocol}")
        print(f"[RUN] cwd: {plan.cwd}")
        print(f"[RUN] command: {shlex.join(plan.command)}")
        print(f"[RUN] log directory: {planned_logs}")
        print(f"[RUN] provenance: {planned_logs / 'command.json'}")
        gpu = (model.get("environment") or {}).get("cuda")
        gpu_required = gpu is not None and str(gpu).upper() != "UNKNOWN"
        if args.dry_run:
            print("[OK] dry-run: official command not executed; no logs created")
            continue
        if gpu_required and args.verify_gpu:
            # `--verify-gpu` means "establish or refresh the capability", whether or not one is
            # already recorded. `force` makes that explicit instead of leaving it to the cache.
            try:
                capability = runtime.verify_gpu_runtime(
                    model, config, authorized=True, require_cuda=True, force=True)
                print(json.dumps(capability, sort_keys=True))
            except (runtime.GpuVerificationError, RuntimeError) as error:
                print(f"[BLOCKED] GPU verification failed: {error}", file=sys.stderr)
                failed = True
                if not args.continue_on_error: break
                continue
        # There is deliberately no "you forgot --verify-gpu" refusal here. A CUDA-declaring
        # model needs a *valid recorded capability*, and --verify-gpu is how one is created or
        # refreshed; requiring the flag on every invocation both forced a re-probe of an
        # already-verified environment and made the API/queue contract contradictory, because
        # the queue validates readiness and then does not pass the flag. The launch gate below
        # is the single authority: no valid capability, no launch.
        try:
            runtime.require_launch_ready(model)
        except (runtime.GpuVerificationError, RuntimeError) as error:
            suffix = ("; run `--verify-gpu-only` on the GPU host to record a capability"
                      if gpu_required else "")
            print(f"[BLOCKED] evaluation launch refused: {error}{suffix}", file=sys.stderr)
            failed = True
            if not args.continue_on_error: break
            continue
        run_id = args.run_id
        if run_id and len(items) > 1:
            run_id = f"{run_id}_{model['model_id']}_{checkpoint['checkpoint_id']}"
        if not run_id:
            run_id = make_run_id(plan, datetime.now(timezone.utc))
        try:
            code = execute(plan, config, run_id=run_id)
        except RuntimeError as error:
            print(f"[BLOCKED] evaluation could not be audited: {error}", file=sys.stderr)
            failed = True
            if not args.continue_on_error: break
            continue
        if code:
            print("[BLOCKED] official command failed"); failed = True
            if not args.continue_on_error: break
            continue
        evidence = completion_validator(plan.model_id, plan.checkpoint_id, plan.protocol,
                                        plan.pin, config, plan.checkpoint)
        ok, reason, _ = evidence({"run_id": run_id})
        if not ok:
            print(f"[BLOCKED] evaluation produced no valid completion proof: {reason}",
                  file=sys.stderr)
            failed = True
            if not args.continue_on_error: break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
