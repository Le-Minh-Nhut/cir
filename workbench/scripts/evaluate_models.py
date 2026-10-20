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
    checkpoint_bundle_missing_paths, checkpoint_is_present, checkpoint_member_root,
    checkpoint_missing_paths, checkpoint_path, load_registry, sha256_file,
)
from workbench.backend import runtime
from workbench.backend.preflight import approved_patch_state, model_requirements, preflight_blockers
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
    #: The directory this artifact's MEMBERS occupy, resolved once from the declared artifact type.
    #: Consumers (the adapter's command, the observer's command, the manifest) must read this rather
    #: than re-deriving it: three separate `is_dir()`-based expressions is how the observer ended up
    #: one directory too high for LIMN's bundle, and `is_dir()` also answers differently before the
    #: weights are downloaded than after.
    member_root: Path | None = None
    checkpoint_sha256: str | None = None
    #: The training-noise condition of the checkpoint this plan will evaluate, from the registry. The
    #: benchmark reference is selected by condition, so a 0%-noise row cannot judge a noisy weight.
    checkpoint_training_noise_pct: int | None = None
    #: True only when the registry declares the model's source as synthetic. A synthetic evaluator
    #: has no published output format, so its own declared aggregate is the evidence and the report
    #: records that this was the case.
    synthetic_source: bool = False
    #: Declared-variant provenance for this run, from the shared preflight: the staged caption alias
    #: evidence and the approved source-patch note. Carried into the report and into the gate's
    #: `run_meta`, so `SUBSTITUTED_CAPTION` is derived from what the run actually used rather than
    #: inferred from the model id. `None`-free dict is always present for a real plan.
    variant_provenance: dict[str, Any] | None = None
    #: Whether the registry verified WHICH upstream asset this checkpoint is. Numerical comparability
    #: and this identity fact are separate claims; the gate records the distinction.
    checkpoint_mapping_verified: bool = False
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

def _benchmark_gate_verdict(plan, extraction, return_code: int) -> dict[str, Any] | None:
    """Judge the run against the committed paper benchmark reference.

    The reference is selected by *condition* -- the checkpoint's training noise, the protocol the run
    used -- so a 0%-noise paper row can never be the reference for a noise-trained checkpoint. When no
    committed row describes this run, the verdict says so rather than falling back on the registry's
    transcribed headline numbers, which are a sanity reference and not a benchmark.
    """
    from workbench.backend import benchmark_references as benchmarks
    from workbench.backend import paper_gate

    observed = extraction.observed_metrics if isinstance(extraction.observed_metrics, dict) else None
    variant = getattr(plan, "variant_provenance", None) or {}
    # Comparability fields are declared facts, not assumptions. `evaluation_noise_pct` is read from
    # the protocol the plan runs; `split` is the annotation cohort (RESULT_SCHEMA.md v2 uses
    # `split: "val"` for every FashionIQ run and carries the gallery vocabulary separately as
    # `literature_split_label`/`protocol_id`), which is also the vocabulary the committed paper
    # rows use — so the cohort, not the gallery label, is what the gate compares.
    _registry = load_registry()
    protocols = _registry.get("protocols", {}) if isinstance(_registry, dict) else {}
    protocol_def = (protocols.get(plan.protocol) if isinstance(protocols, dict) else None) or {}
    run_meta = {
        "protocol_id": plan.protocol,
        "checkpoint_id": plan.checkpoint_id,
        "checkpoint_training_noise_pct": plan.checkpoint_training_noise_pct,
        "evaluation_noise_pct": protocol_def.get("evaluation_noise_pct"),
        "split": "val",
        "dataset": "fashioniq",
        # Declared-variant facts, from the run's own preflight evidence. The gate reads these to
        # label a passing comparison as a variant, never to wave one through.
        "caption_alias_staged": bool(variant.get("caption_alias_staged")),
        # An adapted runtime is a declared variant of the historical environment: the run's
        # numbers come from a different software stack than the authors', so the gate must
        # label the verdict as a variant (PASS_WITHIN_5_PERCENT_VARIANT) rather than an
        # official reproduction.
        "declared_variant": bool(variant.get("approved_source_patch")) or bool(variant.get("adapted_runtime")),
        # Identity evidence, distinct from numerical comparability: an unverified mapping means the
        # gate does not know WHICH upstream asset produced these numbers, so a metric match cannot
        # certify an official reproduction of the paper's experiment.
        "checkpoint_mapping_verified": plan.checkpoint_mapping_verified,
    }
    try:
        reference, reason = benchmarks.select_reference(
            plan.model_id,
            checkpoint_training_noise_pct=plan.checkpoint_training_noise_pct,
            protocol_id=plan.protocol,
        )
    except (FileNotFoundError, ValueError):
        reference, reason = None, "the benchmark registry could not be read"

    verdict = paper_gate.evaluate_gate(
        reference,
        observed,
        run_meta=run_meta,
        # A registry-declared synthetic source has no published paper row to reproduce: feeding
        # the gate `data_kind="experiment"` would let its numbers mint a PASS against whichever
        # reference happened to select, so a synthetic run is judged as what it is — not a real
        # experiment — and never certifies reproduction.
        data_kind="synthetic" if bool(getattr(plan, "synthetic_source", False)) else "experiment",
        # The report is only reached after an invocation was actually made, so the run is real by
        # construction; whether it *completed* is what the exit code says. Collapsing the two would
        # report a crashed evaluation as `NOT_RUN`, which hides a failure behind a state that reads
        # like "not attempted yet".
        real_execution=True,
        evaluation_completed=return_code == 0,
        reproduced_category_metrics=getattr(extraction, "category_metrics", None),
    )
    payload = verdict.to_dict()
    payload["selection_reason"] = reason
    if reference is None:
        payload["selection_reason"] = reason
    return payload


def pinned_revision(source: Path, model: dict | None = None) -> str | None:
    """The revision actually on disk for *source*, for either checkout layout.

    A bare `git -C <source> rev-parse HEAD` is wrong for a vendored checkout. `third_party/<MODEL>`
    is delivered with its nested `.git` removed and a `provenance.json` recording the upstream
    commit, so Git walks UP to the enclosing workbench repository and answers with the WORKBENCH's
    HEAD. The completion proof then compared the workbench's own commit against the model's
    `upstream_commit_sha`, refused with

        declared output source commit is '<workbench sha>', expected '<model sha>'

    and no evaluation could ever produce a valid proof -- the two values are different projects and
    can never agree. Passing `model` delegates to `runtime.source_provenance`, which already answers
    this correctly for both layouts (recorded manifest for a vendored checkout, nested Git for one
    that carries it) and which the replay path has always used. Without `model` the git answer is
    still returned for callers that hold only a path.
    """
    if model is not None:
        from workbench.backend.runtime import source_provenance as runtime_source_provenance

        return runtime_source_provenance(model, source).get("actual_commit")
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
                # Through the registry resolver, which is the one place a bundle record is built:
                # a hand-rebuilt fold drops the member facts, and `checkpoint_path` then resolves
                # to a directory that the presence check reports as missing.
                from workbench.backend.registry import checkpoint_by_id

                checkpoints = [checkpoint_by_id(model, bundle["bundle_id"])]
            elif requested:
                # Naming every member of a declared bundle is the same experiment as naming the
                # bundle, so resolve it to the bundle anchor rather than evaluating the categories
                # as three unrelated runs.
                required = set(bundle["required_checkpoint_ids"])
                if required and set(requested) == required:
                    from workbench.backend.registry import checkpoint_by_id

                    checkpoints = [checkpoint_by_id(model, bundle["bundle_id"])]
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


def _variant_provenance(model: dict, config: WorkbenchConfig) -> dict[str, Any]:
    """What this run substitutes or repairs, from the preflight's own evidence.

    Only declared facts count: a caption alias recorded in the download manifest whose bytes match
    the published stand-in, and an applied approved source patch. Inferring "CSMCIR therefore uses
    the alias" from the model id is exactly what this refuses to do — a run that never staged the
    alias gets no `SUBSTITUTED_CAPTION` label.
    """
    requirements = model_requirements(model, config)
    warnings = list(requirements.get("warnings") or [])
    caption_aliases: list[dict[str, Any]] = []
    for asset in requirements["evidence"].get("auxiliary_assets") or []:
        alias = asset.get("caption_alias")
        if alias:
            caption_aliases.append({
                "asset_family": asset.get("name") or asset.get("family"),
                "path": asset.get("path"),
                **alias,
            })
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    patch = approved_patch_state(model, source)
    approved_patch = None
    if patch is not None and patch.get("applied"):
        approved_patch = {key: patch[key] for key in
                          ("patch_file", "patch_sha256", "applied", "patches") if key in patch}
    from workbench.backend.runtime import runtime_adaptation

    try:
        adaptation = runtime_adaptation(model)
    except Exception:
        adaptation = None
    adapted_profile = None
    if adaptation:
        adapted_profile = {
            "runtime_adaptation": str(adaptation.get("target_compute_capability")),
            # The declared constraints this run actually verified against. A run under an
            # adapted runtime is recorded as adapted evidence and must never be presented as
            # an exact reproduction of the authors' historical environment.
            "pytorch": adaptation.get("pytorch"),
            "torchvision": adaptation.get("torchvision"),
            "cuda": adaptation.get("cuda"),
            "packages": list(adaptation.get("packages") or []),
            "note": "ADAPTED_RUNTIME: modern package stack for the target GPU; not the "
                    "authors' historical environment, not an exact reproduction claim",
        }
    return {
        "caption_aliases": caption_aliases,
        "caption_alias_staged": bool(caption_aliases),
        "approved_source_patch": approved_patch,
        "adapted_runtime": adapted_profile,
        "warnings": warnings,
    }


#: BLOCKER TEXT THAT DESCRIBES A *DIFFERENT* SELECTION, not this one. The message names the
#: bundle the blocker belongs to, so a caller can tell "my selection is incomplete" from "some
#: other checkpoint group on this model still has a member missing". A bundle selection whose
#: own members are all present must not be refused by a blocker about itself, and a selection
#: that is not a bundle must not inherit a bundle's blocker.
def _bundle_blocker_names(reason: str) -> str | None:
    prefix = "checkpoint bundle incomplete: "
    if not reason.startswith(prefix):
        return None
    return reason[len(prefix):].split(":", 1)[0].strip()


def blockers(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, config: WorkbenchConfig) -> tuple[Path, Path, str | None, list[str]]:
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else config.WORKBENCH_THIRD_PARTY_ROOT
    actual_pin = pinned_revision(source, model) if source.is_dir() else None
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    reasons = runtime_plan_blockers(model, checkpoint, config, protocol, dataset_root)
    # A bundle-incomplete blocker is advisory for a *non-bundle* selection: it states that some
    # other group of checkpoints is incomplete, which is not a reason to refuse a run of one
    # category. It is authoritative for the bundle itself, and the identity that decides which is
    # `bundle_id` on the selection, never the checkpoint id — a bundle selection keeps its own
    # bundle id as `checkpoint_id` (`registry.checkpoint_by_id`), so comparing the two fields
    # would ask the question with the answer already folded in and always conclude "not a bundle".
    selected_bundle = checkpoint.get("bundle_id")
    reasons = [reason for reason in reasons
               if (named := _bundle_blocker_names(reason)) is None or named == selected_bundle]
    shared = preflight_blockers(model, config, dataset_root=dataset_root)
    reasons.extend(reason for reason in shared if not reason.startswith("execution environment not verified:") and reason not in reasons)
    return source, checkpoint_file, actual_pin, list(dict.fromkeys(reasons))

def guarded_plan(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, top_k: int, config: WorkbenchConfig | None = None, *, dry_run: bool = False) -> tuple[EvaluationPlan | None, list[str]]:
    config = config or resolve_config()
    adapter_type = ADAPTERS.get(model["model_id"])
    if adapter_type is None or not command_is_audited(adapter_type):
        return None, ["SKIPPED: adapter command not audited"]
    source, checkpoint_file, actual_pin, blocked = blockers(model, checkpoint, protocol, dataset_root, config)
    if dry_run and model["model_id"] == "csmcir":
        # Pinned LAVIS downloads these caches via download_cached_file/from_pretrained; absence is
        # expected before first use, but only execution can prove acquisition works.
        planned = ("LAVIS BLIP-2 pretrain weights", "EVA-ViT-g backbone weights", "bert-base-uncased tokenizer")
        blocked = [reason for reason in blocked if not any(name in reason for name in planned)]
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
        # The members' own directory. `checkpoint_path` resolves a declared bundle to the
        # directory its members share, so that resolution is what the manifest is built over --
        # `.parent` named the whole checkpoint root for a bundle and produced a manifest over
        # paths that do not exist.
        # Declared, not filesystem-derived: a bundle is a member directory regardless of whether the
        # weights have been downloaded, and `is_dir()` answers differently before and after.
        bundle_root = checkpoint_file if checkpoint_file.is_dir() else checkpoint_file.parent
        checkpoint_bundle = build_artifact_manifest(bundle_root, names)
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

    # One resolution of the member directory, from the DECLARATION. A multi-member artifact keeps its
    # members as siblings inside its resolved directory; a single artifact is itself the member.
    multi_member = (artifact_type in ("category_specific_whole_models", "run_directory",
                                      "bundle_directory", "bundle_placeholder")
                    or bool(checkpoint.get("bundle_id")))
    member_root = checkpoint_member_root(checkpoint_file, artifact_type=artifact_type,
                                         is_multi_member=multi_member)

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
        member_root=member_root,
        checkpoint_bundle=checkpoint_bundle,
        bundle_manifest_digest=bundle_manifest_digest,
        artifact_type=artifact_type,
        directory_members=directory_members,
        checkpoint_sha256=checkpoint_digest,
        checkpoint_training_noise_pct=checkpoint.get("checkpoint_training_noise_pct"),
        checkpoint_mapping_verified=checkpoint.get("checkpoint_mapping_status") == "VERIFIED_METADATA",
        synthetic_source=bool(model.get("synthetic_source")),
        variant_provenance=_variant_provenance(model, config),
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


def read_executed_config(plan: EvaluationPlan) -> dict | None:
    """The config the evaluator actually read, when its artifact is a run directory.

    DCNet prints six anonymous `comb:` cells in the order of
    `config['data_loader']['args']['categories']`, so the parser cannot attribute them without this
    file. It is the run directory's own `config.json` -- the same file the native `parse_config.py:28`
    reads -- and never a re-derived or edited copy. A missing or malformed file returns None, which
    makes the parser refuse rather than assume an order.
    """
    if plan.model_id != "dcnet":
        return None
    config_file = plan.checkpoint if plan.checkpoint.is_file() else plan.checkpoint / "config.json"
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def make_run_id(plan: EvaluationPlan, timestamp: datetime) -> str:
    """Collision-resistant run identity: time + model + checkpoint + random suffix."""
    return f"{timestamp:%Y%m%dT%H%M%S%fZ}_{plan.model_id}_{plan.checkpoint_id}_{uuid.uuid4().hex[:8]}"

#: Names an official evaluator legitimately needs. Everything else in the operator's environment
#: is withheld: it is model code, and a research script has no business reading the operator's
#: tokens, cloud credentials or ledger key. An authenticated download belongs in an explicit
#: acquisition step, not in the evaluation.
# HOME is deliberately NOT forwarded: the ledger HMAC key lives under the operator's
# config home (derived from HOME), and the untrusted evaluator child must never be able to
# read it. Everything the child needs from HOME it gets explicitly: TORCH_HOME, HF_HOME,
# TRANSFORMERS_CACHE, XDG_CACHE_HOME are on this list.
_EVALUATOR_ENV_ALLOWLIST = (
    "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ",
    # CUDA/GPU visibility: which devices the run may use is a deliberate operator decision.
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    # Python and library discovery for the model's own interpreter.
    "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    # Where models and datasets live, so the evaluator can find its inputs.
    "CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT",
    "WORKBENCH_CHECKPOINT_ROOT", "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT",
    # The operator-selected runtime profile (e.g. sm_120): the model's own official command
    # validates the same dependency contract the orchestrator planned. A profile name, not a
    # secret; secrets stay excluded by the deny prefixes below.
    "WORKBENCH_RUNTIME_ADAPTATION",
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
    # The ledger key is resolved from these, and the parent uses `Path.home()` for the same
    # directory when XDG_CONFIG_HOME is unset. Withholding only XDG_CONFIG_HOME left the key
    # reachable through HOME: an operator passthrough of `HOME` (or a shell that has it set)
    # let the untrusted evaluator child read the HMAC key, which is the one secret it could
    # use to forge a ledger entry. The prefix rule is what makes an explicit passthrough of one
    # of these names a no-op instead of an override.
    "HOME", "USERPROFILE", "XDG_",
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
    # Evidence that the corroborator has to be able to re-read, not a convenience copy: DCNet's six
    # anonymous `comb:` cells can only be attributed through the category order in the run directory's
    # own `config.json` (the file `parse_config.py:28` read), and `verify_run` re-parses the recorded
    # stdout with the same parser. A run whose config is missing makes that re-extraction refuse,
    # which is how a transposed category would be caught rather than trusted.
    if plan.model_id == "dcnet":
        source_config = (plan.checkpoint if plan.checkpoint.is_file()
                         else plan.checkpoint / "config.json")
        if source_config.is_file():
            shutil.copyfile(source_config, directory / "config.json")

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
    """Reject a run identity that could name a path outside the artifact root, or that claims to be
    a development run.

    Run ids come from the orchestrator (or, in tests, from a caller), so they are validated at the
    entry point rather than trusted: a separator or a parent reference would let a run write outside
    ``workbench/artifacts``.

    The ``dev-`` prefix is reserved for a run that deliberately has NO orchestrator identity, and it
    is what makes an artifact development data instead of an experiment. An operator-supplied
    ``dev-...`` id used to be accepted here and then silently RELABELLED as a development run further
    down, which turns a naming mistake into a data-classification decision. A caller that wants a
    development artifact simply omits ``run_id``; passing one that looks like a development id is
    refused by name.
    """
    if not run_id or any(char in run_id for char in ("/", "\\", "\0")) or ".." in run_id:
        raise ValueError(f"unsafe run id: {run_id!r}")
    if run_id.startswith("dev-"):
        raise ValueError(
            f"a run id may not use the reserved development prefix: {run_id!r}; omit the run id to "
            "record a development artifact, or pass the identity the orchestrator generated")
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
    """Digest of a member manifest, independent of where the artifact happens to be mounted.

    The manifest is compared across contexts: the container records members under
    ``/checkpoints/...`` while a host-side validator resolves the same files under
    ``<artifacts>/checkpoints/...``. Hashing the absolute ``path`` made an unchanged artifact
    fail its own manifest purely because the mount prefix differed, so a legitimate evaluation
    was rejected as tampered. Identity is the *member set and its bytes*: filename plus content
    digest. The path stays in the manifest for readers, but is not part of the identity.
    """
    if not manifest:
        return None
    # Keyed by the MEMBER'S OWN filename, not by whatever key the caller's map happens to use. Two
    # callers describe the same bundle with different map keys -- the evaluator keys by filename, the
    # publisher's member map keys by category -- and using the map key as the identity made the two
    # halves of one run record two unequal digests for one artifact (found by an independent audit).
    identity = sorted(
        ({"filename": member.get("filename") or name, "sha256": member.get("sha256")}
         for name, member in manifest.items()),
        key=lambda entry: str(entry["filename"]),
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


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
def _config_home_is_usable(config_home: Path) -> bool:
    """Whether a key can actually be created and kept under ``config_home``.

    The deployment points ``XDG_CONFIG_HOME`` at a bind-mounted, operator-writable volume. A
    container whose home resolves somewhere read-only (or unset) must not be chosen for a key
    that then cannot be written — that is how a capability record ends up unsigned and a run
    unverifiable — so the fallback to the repository-root location stays available, but only
    when the durable location genuinely cannot be used.
    """
    candidate = config_home / "cir"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(candidate, os.W_OK | os.X_OK)

def ledger_hmac_key(config: WorkbenchConfig) -> bytes:
    """Load the operator key, creating a private random key only for a new ledger."""
    configured = os.environ.get("WORKBENCH_LEDGER_KEY")
    if configured is not None:
        key = configured.strip().encode("utf-8")
        if key:
            return key
        raise LedgerKeyError("WORKBENCH_LEDGER_KEY is set but empty")

    # Two candidate locations, and the choice matters for durability. The private config
    # directory is on a writable, bind-mounted volume in the deployment (`XDG_CONFIG_HOME`), so a
    # key there survives container recreation. The repository root is a mounted-or-baked checkout
    # the runtime user usually cannot write, and a key there is lost when the container is
    # replaced — which invalidates every ledger entry signed with it. The config-home key is
    # therefore preferred; the repository-root key is still honoured when it is the only one, so
    # an existing ledger stays verifiable.
    repo_root = Path(getattr(config, "CIR_REPO_ROOT", Path(__file__).resolve().parents[2]))
    legacy_key_path = repo_root / ".workbench_ledger_key"
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    private_key_path = config_home / "cir" / "workbench_ledger_key"
    if legacy_key_path.is_file() and private_key_path.is_file():
        # Two key files disagreeing is a state that cannot be resolved by preference: whichever this
        # function picked would be the wrong one for a ledger signed with the other, and the operator
        # would see only an unverifiable ledger. Identical bytes are a completed migration, not a
        # conflict, so that case proceeds.
        try:
            same = legacy_key_path.read_bytes() == private_key_path.read_bytes()
        except OSError as error:
            raise LedgerKeyError(
                "cannot compare the two ledger key files "
                f"({legacy_key_path} and {private_key_path}); check that both are readable, or set "
                "WORKBENCH_LEDGER_KEY explicitly") from error
        if not same:
            raise LedgerKeyError(
                "two different ledger keys exist "
                f"({legacy_key_path} and {private_key_path}); keep one file and remove the other, or "
                "set WORKBENCH_LEDGER_KEY explicitly to choose one")
    if legacy_key_path.is_file() and not private_key_path.is_file():
        # An existing repository-root key already signed the ledger on disk: honouring it is the
        # only way that ledger stays verifiable. Creating a fresh key elsewhere would silently
        # invalidate every entry the operator already has. The durable location is used when it
        # already holds a key, or when no key exists anywhere yet.
        key_path = legacy_key_path
        key_lock_path = repo_root / ".workbench_ledger_key.lock"
    elif private_key_path.is_file() or _config_home_is_usable(config_home):
        # Preferred location: durable across container recreation, and writable by the runtime
        # user. Falls back to the legacy path only when there is nowhere durable to put a key.
        private_dir = config_home / "cir"
        key_path = private_key_path
        key_lock_path = private_dir / "workbench_ledger_key.lock"
    else:
        key_path = legacy_key_path
        key_lock_path = repo_root / ".workbench_ledger_key.lock"
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
    """Every authenticated ledger entry, or None when the chain does not verify.

    None means exactly one thing to a caller: *nothing is corroborated here*. An unusable key is
    deliberately not folded into it any more: a caller that catches LedgerKeyError can state the
    actionable cause by name -- the conflicting key paths and the remedy -- while a caller that
    does not still gets the fail-closed None it always got.
    """
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
    except LedgerKeyError:
        raise
    except (LedgerCorruptError, OSError):
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


def invocation_outcome_path(config: WorkbenchConfig, model_id: str, checkpoint_id: str,
                            run_id: str) -> Path:
    """The machine-readable verdict of one invocation, beside its report and proof.

    A caller that dispatched an evaluation needs to know one thing the exit code cannot tell it:
    did this run produce a *valid completion proof*? ``execute()`` returns the native evaluator
    code (that is evidence, and it is never overwritten with a workbench status), so a run whose
    proof failed validation still exits 0. This file is where the operator answers the real
    question, and it is named by the run identity so a leftover file can never be read as this
    run's verdict.
    """
    return reports_root(config) / f"{model_id}_{checkpoint_id}_{run_id}_invocation_outcome.json"

def report_identity(report: dict | None) -> dict[str, Any]:
    """The comparability identity a report was produced under.

    A run's numbers are only meaningful together with the commit they came from and the runtime
    they ran on; this is the shape every consumer of a report (the launcher's summary, a verifier)
    reads, so it is derived once, here, from the report itself.
    """
    report = report or {}
    return {
        "run_id": report.get("run_id"),
        "model_id": report.get("model_id"),
        "checkpoint_id": report.get("checkpoint_id"),
        "protocol_id": report.get("protocol_id"),
        "git_commit": report.get("source_actual_commit"),
        "runtime_adaptation": report.get("runtime_adaptation"),
    }


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


def record_invocation_outcome(directory: Path, config: WorkbenchConfig, *, run_id: str,
                              model_id: str, checkpoint_id: str, protocol_id: str,
                              return_code: int, invocation_status: str, verdict: str,
                              reason: str | None) -> dict:
    """Record what this invocation actually achieved, in one machine-readable file.

    ``verdict`` is one of ``PROOF_VALIDATED`` (the completion proof passed the evaluator's own
    ``validate_run_manifest``), ``NO_OBSERVABLE_OUTPUT`` (exited 0 and printed nothing, so no
    proof was minted), ``PROOF_REJECTED`` (a proof was minted but failed validation) or
    ``EXECUTION_FAILED`` (nonzero exit, so no proof could be minted). ``completed`` states in one
    boolean what a consumer must not infer from the exit code.

    Written both beside the report (for an auditor) and into the run directory (for a caller
    that only knows the run directory): the two are the same payload, computed once.
    """
    payload = {
        "artifact_type": "invocation_outcome",
        "schema_version": 1,
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "protocol_id": protocol_id,
        "return_code": return_code,
        "invocation_status": invocation_status,
        "verdict": verdict,
        "completed": verdict == "PROOF_VALIDATED",
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(invocation_outcome_path(config, model_id, checkpoint_id, run_id), payload)
    _atomic_write_json(directory / "invocation_outcome.json", payload)
    return payload


def _atomic_exclusive_write(destination: Path, content: str) -> None:
    """Create `destination` with `content`, never exposing a partial file at that name.

    Raises ``FileExistsError`` when the name already exists, preserving the exclusive-create
    contract the run-scoped report relies on (a colliding run id must never overwrite an earlier
    experiment). The content is written and fsynced in a private temporary first, then linked into
    place: the link fails atomically if the name appeared meanwhile, so two writers cannot collide.
    """
    import tempfile as _tempfile

    handle, temporary = _tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_replace(destination: Path, content: str) -> None:
    """Write `content` to `destination` atomically, replacing any existing file."""
    import tempfile as _tempfile

    handle, temporary = _tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


#: The registry field that declares a model has a per-query observer. A model without it is not a
#: failed export -- it is a model whose result is aggregate-only by design, and the launcher must not
#: treat the absence as a defect.
PER_QUERY_CAPABILITY = "per_query_observer"

#: The per-query outcome vocabulary, used by the report and consumed by the launcher. Distinct from
#: the EVIDENCE states (schemas/results.py): these describe what the RUN achieved, those describe what
#: a stored artifact may be called.
PER_QUERY_STATES = (
    "CAPABILITY_ABSENT",       # the model declares no observer: aggregate-only by design
    "PER_QUERY_EXPORT_FAILED",  # an observer exists and did not produce an artifact
    "PER_QUERY_EXPORTED_UNVERIFIED",  # an artifact exists; the verifier has not spoken
    "PER_QUERY_VERIFIED",      # exported AND the verifier corroborated it
)


def per_query_state(capability: str | None, artifact: Path | None, note: str) -> str:
    """What this run achieved on the per-query axis. One function, so the report and any consumer
    cannot disagree about the same facts."""
    if not capability:
        return "CAPABILITY_ABSENT"
    if artifact is None:
        return "PER_QUERY_EXPORT_FAILED"
    # An artifact exists. Whether it may be called VERIFIED is the verifier's answer, never this
    # function's: the run can only report that it produced one.
    return "PER_QUERY_EXPORTED_UNVERIFIED"


def per_query_capability(model_id: str) -> str | None:
    """The declared per-query capability of a model, or None when it has none.

    Read from the registry rather than hardcoded here, so adding an observer to another model is a
    registry change and the launcher follows automatically.
    """
    from workbench.backend.registry import model_by_id

    try:
        return model_by_id(model_id).get(PER_QUERY_CAPABILITY)
    except Exception:
        return None


def _status_object(stdout: str, field: str) -> str:
    """The named field of the LAST JSON object on a child's stdout.

    Both children emit exactly one compact status line, but a compiled dependency can still write to
    fd 1 directly, and `json.loads` on the whole stream would then fail and the parent would discard
    a valid payload. Scanning for the status object makes the contract robust without relaxing it:
    the object must still carry the field being asked for.
    """
    for line in reversed([line for line in (stdout or "").splitlines() if line.strip()]):
        if not line.lstrip().startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and field in candidate:
            return candidate[field]
    raise ValueError(f"no status object carrying {field!r} on stdout")


def _run_observation(plan: EvaluationPlan, config: WorkbenchConfig, run_id: str,
                     directory: Path, command_digest: str | None,
                     environment_digest: str | None) -> tuple[Path | None, str, str | None, str | None]:
    """Run the per-query observer as a child of the model's interpreter, then verify its artifact.

    Returns ``(artifact_path, note, artifact_digest, parity)``. The artifact digest is computed HERE,
    by the parent, from the bytes the child wrote — the child never reports the digest of its own
    output, because a child evaluator must not self-certify.

    Observation is deliberately non-fatal: the aggregate evaluation already succeeded and is the
    authoritative result, so a missing per-query artifact is recorded as a reason, not an error. What
    it must never do is make an unverified artifact look verified, which is why the artifact is
    re-validated and its admission state recomputed by the parent.
    """
    results_root = getattr(config, "WORKBENCH_RESULTS_ROOT",
                           Path(config.WORKBENCH_ARTIFACTS_ROOT) / "results")
    interpreter = plan.command[0] if plan.command else sys.executable
    # The SAME allowlisted environment the official evaluator gets. `os.environ` would hand model
    # code the operator's tokens, cloud credentials and -- worst of all -- the ledger signing key,
    # which is the one secret that must never reach a child that writes evidence.
    child_env = _evaluator_environment({
        "orchestrator_python": sys.version,
        "orchestrator_executable": sys.executable,
        "platform": sys.platform,
        "model_interpreter": interpreter,
        "evaluation_run_id": run_id,
        "evaluation_output_root": str(artifact_root(config)),
    })
    # The observer is workbench code, so the repository must be importable in its child. This is the
    # one addition to the evaluator's allowlist, and it is deliberate: the evaluator gets the same
    # value through its own PYTHONPATH handling when the operator sets it, and without it the child
    # cannot import `workbench.replay.limn_export` at all.
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    argv = [interpreter, "-m", "workbench.replay.limn_export",
            # The plan's own recorded member root: resolved once from the DECLARED artifact type,
            # so it is the same directory whether or not the weights are on disk yet. `plan.
            # checkpoint.parent` named the whole checkpoint root for LIMN's three-category bundle
            # and every member lookup came up one level too high, so observation failed while the
            # aggregate succeeded (P1 -- the exact regression the 1309-test release did not catch).
            "--checkpoint-root", str(
                plan.member_root
                or checkpoint_member_root(plan.checkpoint, artifact_type=plan.artifact_type)),
            "--dataset-root", str(plan.dataset_root),
            "--source-root", str(plan.source),
            "--results-root", str(results_root),
            "--run-id", run_id]
    # The observer runs a DIFFERENT command than the aggregate, so its own digests would describe
    # the wrong invocation. The parent passes the aggregate invocation's digests, which it owns.
    if command_digest:
        argv += ["--command-digest", command_digest]
    if environment_digest:
        argv += ["--environment-digest", environment_digest]
    adaptation = (getattr(plan, "variant_provenance", None) or {}).get("adapted_runtime")
    if isinstance(adaptation, dict) and adaptation.get("runtime_adaptation"):
        argv += ["--runtime-adaptation", str(adaptation["runtime_adaptation"])]
    # The digest over the members THIS run will evaluate, recomputed exactly as the report records
    # it. The publisher compares the observer's staged members against it, so the per-query artifact
    # and the aggregate report cannot disagree about which bundle was evaluated.
    planned_digest = plan.bundle_manifest_digest or manifest_digest(_plan_members(plan))
    if planned_digest:
        argv += ["--planned-bundle-manifest-digest", str(planned_digest)]
    try:
        completed = subprocess.run(argv, cwd=str(plan.cwd), env=child_env, text=True,
                                   capture_output=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"observer could not run: {type(error).__name__}: {error}", None, None
    (directory / "observation.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (directory / "observation.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        return None, f"observer exited {completed.returncode}: {' | '.join(tail)}", None, None

    # Two-stage: the child STAGED a plain payload (model interpreter), and the parent PUBLISHES it
    # here, in the workbench interpreter, because only this side has the canonical schema. The child
    # cannot do it: pydantic is not in the model's provisioned environment, and importing it there
    # made every export fail while the aggregate succeeded.
    try:
        staged = Path(_status_object(completed.stdout, "payload"))
    except (ValueError, KeyError, TypeError) as error:
        return None, f"observer did not report a staged payload: {error}", None, None
    # The publisher is invoked as a CHILD under the WORKBENCH interpreter, never imported here.
    # Importing it would put the canonical schema (pydantic) on this module's import graph, and this
    # module is reachable from `workbench/replay/limn.py` -- which the environment planner traces to
    # decide LIMN's interpreter. That is the regression that broke cold start twice.
    publish_argv = [sys.executable, "-m", "workbench.replay.limn_publish",
                    "--payload", str(staged), "--results-root", str(results_root)]
    try:
        published = subprocess.run(publish_argv, cwd=str(plan.cwd), env=child_env, text=True,
                                   capture_output=True, timeout=900)
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"publisher could not run: {type(error).__name__}: {error}", None, None
    (directory / "publish.stdout.log").write_text(published.stdout or "", encoding="utf-8")
    (directory / "publish.stderr.log").write_text(published.stderr or "", encoding="utf-8")
    if published.returncode != 0:
        tail = (published.stderr or published.stdout or "").strip().splitlines()[-3:]
        return None, f"publisher exited {published.returncode}: {' | '.join(tail)}", None, None
    try:
        artifact = Path(_status_object(published.stdout, "published"))
    except (ValueError, KeyError, TypeError) as error:
        return None, f"publisher did not report a published artifact: {error}", None, None
    try:
        payload = json.loads(artifact.read_text())
        claimed_run = payload["run"]["run_id"]
        parity = payload["run"].get("instrumentation_parity")
        data_kind = payload["run"].get("data_kind")
    except (ValueError, KeyError, TypeError) as error:
        return None, f"published artifact is not a readable canonical result: {error}", None, None
    if claimed_run != run_id:
        return None, (f"observer artifact claims run {claimed_run!r}, not this invocation "
                      f"{run_id!r}"), None, None
    # The artifact must describe THIS plan. The observer always observes the full three-category
    # bundle, so a single-category plan would otherwise have a bundle artifact -- with its own
    # checkpoint/protocol identity -- accepted and bound to the diagnostic run's report.
    for field, expected in (("model_id", plan.model_id), ("protocol_id", plan.protocol)):
        claimed = payload["run"].get(field)
        if claimed is not None and expected is not None and claimed != expected:
            return None, (f"observer artifact describes {field}={claimed!r}, not this run's "
                          f"{expected!r}"), None, None
    # THE PARENT BINDS THE PUBLISHED BUNDLE IDENTITY TO THE PLAN, and this check is the one that
    # holds, because the parent is the only participant that derives the digest from OUTSIDE the
    # payload.
    #
    # The publisher compares the payload's staged members against the payload's own
    # `planned_bundle_manifest_digest`, and it does refuse a disagreement -- but the observer writes
    # BOTH fields, so an observer that staged the wrong checkpoint could declare a matching planned
    # digest and the two halves would agree while describing a bundle this run never evaluated.
    # Payload-versus-payload is not a binding. The plan is: `_plan_members` hashes the members on
    # DISK, before the observer runs, and `plan.bundle_manifest_digest` is derived from that. When
    # the plan resolves members, the artifact must record exactly that digest.
    #
    # The expression below is deliberately the same one the observer is handed and the same one the
    # completion proof records (an earlier line in this file, and the `checkpoint_local_sha256`
    # block's neighbour). Three sites, one expression: the artifact, the proof and the run's own
    # record of the plan must agree, and a difference between them is the defect this check exists
    # to catch. `_plan_members` re-reads each member from disk rather than copying the plan's
    # recorded digest, because a plan built before the run may carry placeholders.
    planned_digest = plan.bundle_manifest_digest or manifest_digest(_plan_members(plan))
    if planned_digest:
        recorded_digest = payload["run"].get("bundle_manifest_digest")
        if recorded_digest != planned_digest:
            return None, (
                f"observer artifact records bundle_manifest_digest={recorded_digest!r}, but this "
                f"run planned {planned_digest!r}: the per-query artifact must describe the members "
                "the aggregate run actually evaluated, and neither the observer nor the publisher "
                "can be the authority for that"), None, None
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # The staging file is an interchange detail, not evidence: leaving it beside the canonical
    # artifacts would make the results directory hold two descriptions of one run.
    try:
        staged.unlink(missing_ok=True)
    except OSError:
        pass
    note = f"artifact exported (data_kind={data_kind!r}); admission is the verifier's to decide"
    return artifact, note, digest, parity


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
        executed_config=read_executed_config(plan),
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

    # Per-query observation, bound to THIS invocation (F-08/F-09/F-11/F-12). It runs as a separate
    # process under the model's own interpreter because in production the parent cannot import the
    # model environment; the parent then VERIFIES what the child wrote rather than trusting it. A
    # failure here never changes the aggregate result, and never promotes anything.
    # The per-query capability is DECLARED, not inferred from a model id, so a capability-absent
    # model (one the registry gives no observer) is distinguishable from a capability that FAILED.
    # The launcher needs that distinction: only the second is a pilot blocker.
    capability = per_query_capability(plan.model_id)
    if capability and plan.checkpoint_id != "base_iter0_all_categories":
        # The observer exports the FULL three-category bundle. A single-category diagnostic is a
        # legitimate run, but its per-query artifact would describe a different experiment, so it is
        # not attempted -- and saying so is better than binding a bundle artifact to a diagnostic.
        capability = None
        observation_note = ("CAPABILITY_ABSENT_FOR_DIAGNOSTIC: the observer covers the full bundle, "
                            "and this run selected a single category")
    elif not capability:
        observation_note = "CAPABILITY_ABSENT"
    else:
        observation_note = f"{capability}: not attempted"
    per_query_artifact = None
    per_query_artifact_digest = None
    instrumentation_parity = None
    # The field NAME is the constant; its VALUE is the observer's identity (e.g.
    # "LIMN_QUERY_OBSERVER_V1"). The gate is "the registry declares one", not equality with the key.
    if code == 0 and capability:
        per_query_artifact, observation_note, per_query_artifact_digest, instrumentation_parity = \
            _run_observation(plan, config, run_id, directory,
                             metadata.get("command_digest"), metadata.get("environment_digest"))

    paper = plan.paper_metrics or {}
    gate_verdict = _benchmark_gate_verdict(plan, extraction, code)
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
        "per_query_export_available": per_query_artifact is not None,
        "per_query_export_reason": observation_note,
        # What this RUN achieved on the per-query axis, distinct from the artifact's evidence state.
        # `CAPABILITY_ABSENT` is not a failure; `PER_QUERY_EXPORT_FAILED` is a pilot blocker for a
        # model that declares an observer.
        "per_query_capability": capability,
        "per_query_state": per_query_state(capability, per_query_artifact, observation_note),
        "per_query_artifact": str(per_query_artifact) if per_query_artifact else None,
        "per_query_artifact_digest": per_query_artifact_digest,
        "instrumentation_parity": instrumentation_parity,
        "extraction_status": extraction.extraction_status,
        "metric_extraction_available": extraction.observed_metrics is not None,
        "parser_id": extraction.parser_id,
        "parser_version": extraction.parser_version,
        "observed_metrics": extraction.observed_metrics,
        "category_metrics": extraction.category_metrics,
        "paper_sanity_metrics": paper,
        "parity_status": extraction.parity_status,
        # The canonical acceptance verdict. `paper_sanity_metrics` above is the registry's transcribed
        # row, kept only as a sanity reference; the gate judges the run against the committed paper
        # benchmark reference, which records what the paper actually printed and under which
        # checkpoint condition.
        "benchmark_gate": gate_verdict,
        "benchmark_reference": (gate_verdict or {}).get("reference_evidence"),
        "variant_provenance": getattr(plan, "variant_provenance", None) or {},
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
        # The adaptation the run executed under, named on the report itself (not only in
        # variant_provenance): a reader of the environment block must not mistake an
        # adapted stack for the authors' historical one.
        "runtime_adaptation": (getattr(plan, "variant_provenance", None) or {}).get(
            "adapted_runtime", {}).get("runtime_adaptation") if isinstance(
            (getattr(plan, "variant_provenance", None) or {}).get("adapted_runtime"), dict) else None,
        "evidence_digest": evidence_digest,
        "return_code": code,
    }
    report_json = json.dumps(report, indent=2) + "\n"

    # Immutable, run-scoped artifact. Exclusive create: a colliding run id must never
    # overwrite a previous experiment.
    reports.mkdir(parents=True, exist_ok=True)
    try:
        # `open("x")` is atomic for the NAME but not for the BYTES: an interruption mid-write leaves
        # a partial file at the final path, which then fails its own digest on every later run and can
        # never be replaced. Written to a private temporary and linked, so the name appears only once
        # the content is complete.
        _atomic_exclusive_write(run_report, report_json)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing run report: {run_report}") from error
    # The run-directory copy is replaced wholesale on a rerun, so it is written atomically rather
    # than exclusively: a reader must never see a truncated report either way.
    _atomic_replace(directory / "aggregate_report.json", report_json)


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
        record_invocation_outcome(
            directory, config, run_id=run_id, model_id=plan.model_id,
            checkpoint_id=plan.checkpoint_id, protocol_id=plan.protocol, return_code=code,
            invocation_status=invocation_status, verdict="NO_OBSERVABLE_OUTPUT",
            reason="the official command exited 0 and produced no observable evaluation output")
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
            # The native code stays 0 — it is the evaluator's own evidence — but the run is not
            # a completion, and the outcome file says so instead of leaving a caller to guess
            # from an exit code that cannot distinguish the two.
            record_invocation_outcome(
                directory, config, run_id=run_id, model_id=plan.model_id,
                checkpoint_id=plan.checkpoint_id, protocol_id=plan.protocol, return_code=code,
                invocation_status=invocation_status, verdict="PROOF_REJECTED",
                reason=str(error))
        else:
            record_invocation_outcome(
                directory, config, run_id=run_id, model_id=plan.model_id,
                checkpoint_id=plan.checkpoint_id, protocol_id=plan.protocol, return_code=code,
                invocation_status=invocation_status, verdict="PROOF_VALIDATED", reason=None)
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
    else:
        record_invocation_outcome(
            directory, config, run_id=run_id, model_id=plan.model_id,
            checkpoint_id=plan.checkpoint_id, protocol_id=plan.protocol, return_code=code,
            invocation_status=invocation_status, verdict="EXECUTION_FAILED",
            reason=f"the official command exited with code {code}")
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
            # Path(".") is not a sentinel a caller can distinguish from a real path, and it made
            # `checkpoint_sha256` a hash of a DIRECTORY the run never used. The caller tests for it.
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
            # No checkpoint to RE-HASH. Two different situations reach here and they must not be
            # treated alike.
            #
            # (a) The registry does not describe this (model, checkpoint id) pair, so the resolution
            #     above caught an exception and the path is not one. Falling back to the digest the
            #     REPORT claims about itself makes the proof self-referential: a proof repointed to
            #     another run, or a tampered weight file, is then checked against the claim it makes
            #     rather than against bytes on disk (found by an independent audit). Refuse instead --
            #     an unresolvable identity is an unverifiable one.
            #
            # (b) The artifact is genuinely not a single file (a directory or bundle already handled
            #     above), where the run's own recorded digest is the only available binding.
            if not (resolved_members or resolved_bundle):
                return False, (f"the registry does not resolve checkpoint {checkpoint_id!r} for "
                               f"{model_id!r}, so this proof's checkpoint identity cannot be "
                               f"re-derived from disk"), None
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
        plan, reasons = guarded_plan(model, checkpoint, protocol, dataset_root, args.top_k, config, dry_run=args.dry_run)
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
            if model["model_id"] == "csmcir":
                print("[PLAN] first-use downloads: LAVIS BLIP-2, EVA-ViT-g, bert-base-uncased; dry-run cannot verify acquisition or evaluator success")
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
