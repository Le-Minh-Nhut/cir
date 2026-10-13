#!/usr/bin/env python3
"""Inspect local workbench readiness without changing files or using network."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.runtime import environment_blockers, environment_status, source_status
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination, checkpoint_bundle_missing_paths, checkpoint_missing_paths, checkpoint_path, external_asset_blockers, fashioniq_required_paths, load_registry, preparation_contract_for_model, preparation_paths, required_runtime_assets, sha256_file

from workbench.backend.fashioniq_layout import (
    CATEGORIES, image_extension_problems, missing_paths, standard_paths,
)


from workbench.backend.preflight import model_requirements, preflight_blockers
def record(name: str, status: str, detail: str, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "evidence": evidence}


def git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_state(model: dict[str, Any], root: Path) -> dict[str, Any]:
    source_dir = model.get("source_dir")
    if not source_dir:
        return {"path": None, "state": "unavailable", "head": None, "dirty": None, "pin": model.get("upstream_commit_sha")}
    path = root / source_dir
    if not path.is_dir():
        return {"path": str(path), "state": "missing", "head": None, "dirty": None, "pin": model.get("upstream_commit_sha")}
    head = git(path, "rev-parse", "HEAD")
    if head is None:
        return {"path": str(path), "state": "not_git", "head": None, "dirty": None, "pin": model.get("upstream_commit_sha")}
    # Classify the checkout exactly like the execution gate: tracked edits and
    # unauthorized untracked/ignored files are dirty, while declared runtime
    # artifacts (the CSMCIR canonical dataset link, recorded auxiliary assets) are not.
    entries = source_status(path)
    if entries is None:
        return {"path": str(path), "state": "not_git", "head": head, "dirty": None, "pin": model.get("upstream_commit_sha")}
    dirty = bool(entries["tracked"] or entries["unauthorized"] or entries["invalid"])
    reason = None
    if dirty:
        offenders = entries["tracked"] or entries["invalid"] or entries["unauthorized"]
        reason = offenders[0] if offenders else None
    return {"path": str(path), "state": "dirty" if dirty else "clean", "head": head, "dirty": dirty,
            "pin": model.get("upstream_commit_sha"), "dirty_reason": reason,
            "tracked_modifications": entries["tracked"],
            "unauthorized_untracked": entries["unauthorized"],
            "approved_runtime_artifacts": entries["approved"],
            "invalid_runtime_artifacts": entries["invalid"]}


def fashioniq_base_paths(root: Path) -> tuple[Path, ...]:
    return standard_paths(root)


def csmcir_auxiliary_paths(config: WorkbenchConfig, family: str) -> tuple[Path, ...]:
    return tuple(
        auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
        for asset in auxiliary_assets_for_model("csmcir")
        if asset["family"] == family
    )


def missing_path_strings(paths: tuple[Path, ...]) -> list[str]:
    return [str(path) for path in missing_paths(paths)]


def preparation_checks(model: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    config = replace(resolve_config(), FASHIONIQ_ROOT=root)
    requirements = model_requirements(model, config, canonical_root=root)
    evidence = requirements["evidence"]
    contract = preparation_contract_for_model(model)
    raw_missing = evidence["raw_inputs_missing"]
    generated_missing = evidence["generated_artifacts_missing"]
    extension_problems = evidence["image_extension_problems"]
    if contract is None:
        blockers = [f"FashionIQ {model.get('fashioniq_layout') or 'FashionIQ'} requirement missing: {path}"
                    for path in raw_missing]
        blockers.extend(f"FashionIQ image extension requirement: {problem}"
                        for problem in extension_problems)
        return [], blockers
    raw_status = "OK" if not raw_missing and not extension_problems else "BLOCKED"
    preparation_status = "BLOCKED" if generated_missing else "MANUAL"
    checks = [
        record(f"raw-dataset:{model['model_id']}", raw_status,
               "raw benchmark inputs available" if raw_status == "OK" else "raw benchmark inputs missing",
               preparation_id=contract["preparation_id"], missing=raw_missing,
               image_extension_problems=extension_problems),
        record(f"preparation:{model['model_id']}", preparation_status,
               "manual source-specific preparation required" if not generated_missing else "required generated artifacts missing",
               preparation_id=contract["preparation_id"],
               required_generated_artifacts=contract["generated_artifacts"],
               external_assets=contract["external_assets"],
               deterministic_status=contract["deterministic_status"],
               automation_policy=contract["automation_policy"], notes=contract["notes"],
               missing=generated_missing),
    ]
    blockers = [f"raw benchmark input missing: {path}" for path in raw_missing]
    blockers.extend(f"required generated artifact missing: {path}" for path in generated_missing)
    blockers.extend(f"FashionIQ image extension requirement: {problem}" for problem in extension_problems)
    return checks, blockers


def external_asset_checks(model: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    assets = required_runtime_assets(model)
    blockers = external_asset_blockers(model)
    status = "OK" if not blockers else "BLOCKED"
    return [record(f"external-assets:{model['model_id']}", status, "required evaluation assets resolved" if not blockers else "required evaluation assets unresolved", assets=assets, blockers=blockers)], blockers


def fashioniq_check(root: Path) -> dict[str, Any]:
    missing = missing_path_strings(fashioniq_base_paths(root))
    status = "OK" if not missing else "BLOCKED"
    return record("fashioniq", status, "FashionIQ base layout available" if not missing else "FashionIQ base layout missing", root=str(root), missing=missing, categories=list(CATEGORIES))


def writable_path_check(name: str, path: Path) -> dict[str, Any]:
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
    return record(f"generated:{name}", "OK" if writable else "BLOCKED", "generated path writable" if writable else "generated path not writable", path=str(path), checked_parent=str(ancestor))


def model_checks(model: dict[str, Any], config: WorkbenchConfig,
                 allow_gpu_pending: bool = False) -> list[dict[str, Any]]:
    model_id = model["model_id"]
    state = source_state(model, config.WORKBENCH_THIRD_PARTY_ROOT)
    source_ok = state["state"] == "clean" and (not state["pin"] or state["head"] == state["pin"])
    checks = [record(f"source:{model_id}", "OK" if source_ok else "BLOCKED",
                     "source pin ready" if source_ok else "source unavailable, dirty, or not pinned", **state)]
    blockers = preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)
    env_state = environment_status(model, probe=True)
    if allow_gpu_pending and env_state["status"] == "GPU_VERIFICATION_PENDING":
        blockers = [reason for reason in blockers if "GPU_VERIFICATION_PENDING" not in reason]
    env_blockers = environment_blockers(model)
    if allow_gpu_pending:
        env_blockers = [reason for reason in env_blockers if "GPU_VERIFICATION_PENDING" not in reason]
    environment_check = record(
        f"environment:{model_id}", "OK" if not env_blockers else "BLOCKED",
        "authorized GPU probe runs in the next stage" if allow_gpu_pending and env_state["status"] == "GPU_VERIFICATION_PENDING" else env_state["status"],
        environment=model.get("environment"), interpreter=env_state["interpreter"], tier=env_state["tier"],
        runtime_verified=env_state["verified"], gpu_deferred=env_state["gpu_deferred"],
        reason=env_state["reason"], dependencies=env_state.get("packages"))
    adapter = ADAPTERS.get(model_id)
    command_status = model.get("command_status")
    command_implemented = adapter is not None and getattr(adapter, "command", None) is not OfficialScriptAdapter.command
    command_audited = command_implemented and command_status in {None, "COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"}
    command_check = record(f"command:{model_id}", "OK" if command_audited else "BLOCKED",
                           command_status or "COMMAND_AUDITED", script=getattr(adapter, "script", None))

    deferred_checks: list[dict[str, Any]] = []
    csmcir_assets: list[dict[str, Any]] = []
    if model_id == "csmcir":
        groups = (
            ("csmcir:base-dataset", "FashionIQ base dataset", fashioniq_base_paths(config.FASHIONIQ_ROOT)),
            ("csmcir:qwen-captions", "CSMCIR Qwen captions", csmcir_auxiliary_paths(config, "Qwen captions")),
            ("csmcir:cot-captions", "CSMCIR COT_ours2 captions", csmcir_auxiliary_paths(config, "COT_ours2 captions")),
        )
        for name, label, paths in groups:
            missing = missing_path_strings(paths)
            root = config.FASHIONIQ_ROOT if name != "csmcir:cot-captions" else Path(state["path"] or config.WORKBENCH_THIRD_PARTY_ROOT)
            csmcir_assets.append(record(name, "OK" if not missing else "BLOCKED",
                                        f"{label} available" if not missing else f"{label} missing",
                                        root=str(root), missing=missing))
    else:
        preparation, _ = preparation_checks(model, config.FASHIONIQ_ROOT)
        deferred_checks.extend(preparation)
    external_checks, _ = external_asset_checks(model)
    deferred_checks.extend(external_checks)

    env_evidence = {"status": env_state["status"], "tier": env_state["tier"], "variable": env_state["variable"],
                    "interpreter": env_state["interpreter"], "runtime_verified": env_state["verified"],
                    "gpu_deferred": env_state["gpu_deferred"], "reason": env_state["reason"]}
    incomplete_bundles = checkpoint_bundle_missing_paths(model, config.WORKBENCH_CHECKPOINT_ROOT)
    bundle_reports = [{
        "bundle_id": bundle["bundle_id"], "required_checkpoint_ids": bundle["required_checkpoint_ids"],
        "complete": not incomplete_bundles.get(bundle["bundle_id"], []),
        "status": "READY" if not incomplete_bundles.get(bundle["bundle_id"], []) else "BLOCKED",
        "blockers": [f"checkpoint bundle member missing or invalid: {path}"
                     for path in incomplete_bundles.get(bundle["bundle_id"], [])],
    } for bundle in model.get("checkpoint_bundles", [])]
    checkpoints = []
    for checkpoint in model["checkpoint_variants"]:
        path = checkpoint_path(model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        missing = checkpoint_missing_paths(path, checkpoint)
        installed = not missing
        local_sha = sha256_file(path, required_files=checkpoint.get("required_files")) if installed else None
        official_sha = checkpoint.get("expected_sha256")
        mapping = checkpoint["checkpoint_mapping_status"] == "VERIFIED_METADATA"
        reasons = list(blockers)
        reasons.extend(f"checkpoint missing or invalid: {item}" for item in missing)
        registry_status = checkpoint.get("status")
        unavailable = registry_status in {"NO_CLEAN_CHECKPOINT", "BLOCKED"} or (registry_status and registry_status.startswith("BLOCKED_"))
        if unavailable:
            reasons.append(f"registry checkpoint unavailable: {registry_status}")
        if not mapping:
            reasons.append("checkpoint mapping unresolved")
        if official_sha and local_sha != official_sha:
            reasons.append("official hash mismatch")
        for bundle in bundle_reports:
            if checkpoint["checkpoint_id"] in bundle["required_checkpoint_ids"]:
                reasons.extend(f"{bundle['bundle_id']} incomplete: {item}" for item in bundle["blockers"])
        checkpoints.append({"checkpoint_id": checkpoint["checkpoint_id"], "path": str(path),
                            "artifact_type": checkpoint.get("artifact_type", "file"), "installed": installed,
                            "local_sha256": local_sha, "official_sha256": official_sha,
                            "official_sha256_available": official_sha is not None, "mapping_verified": mapping,
                            "provenance_verified": mapping and official_sha is not None and local_sha == official_sha,
                            "registry_status": registry_status,
                            "readiness": "UNAVAILABLE" if unavailable else "BLOCKED" if reasons else "RUNNABLE",
                            "runtime_blockers": reasons, "command_ready": not reasons})
    runnable = bool(checkpoints) and all(item["command_ready"] for item in checkpoints) and all(item["complete"] for item in bundle_reports)
    checks.append(record(f"runtime:{model_id}", "OK" if runnable else "BLOCKED",
                         "prerequisites present; evaluation not run" if runnable else "command unavailable",
                         checkpoints=checkpoints, bundles=bundle_reports, environment=env_evidence,
                         runnable=runnable, reproduction_status=model.get("reproduction_status", "NOT_REPORTED")))
    checks.extend(deferred_checks)
    checks.append(command_check)
    checks.append(environment_check)
    checks.extend(csmcir_assets)
    return checks


def workbench_checks(config: WorkbenchConfig, serving: bool = False) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    branch = git(config.CIR_REPO_ROOT, "branch", "--show-current")
    dirty = git(config.CIR_REPO_ROOT, "status", "--porcelain")
    checks.append(record("repository", "OK", "repository information", root=str(config.CIR_REPO_ROOT), branch=branch, dirty=bool(dirty)))
    checks.append(record("python", "OK", platform.python_version(), executable=sys.executable))
    for name in ("fastapi", "uvicorn", "duckdb", "pydantic", "yaml"):
        present = importlib.util.find_spec(name) is not None
        checks.append(record(f"import:{name}", "OK" if present else "BLOCKED", "available" if present else "missing", module=name))
    checks.append(writable_path_check("results", config.WORKBENCH_RESULTS_ROOT))
    if serving:
        for command in ("node", "npm"):
            present = shutil.which(command) is not None
            checks.append(record(f"command:{command}", "OK" if present else "BLOCKED", "available" if present else "missing", command=command))
        node_modules = config.CIR_REPO_ROOT / "workbench" / "frontend" / "node_modules"
        checks.append(record("frontend:node_modules", "OK" if node_modules.is_dir() else "BLOCKED", "available" if node_modules.is_dir() else "missing", path=str(node_modules)))
    return checks


def collect(config: WorkbenchConfig | None = None, model_id: str | None = None, all_models: bool = False, scope: str = "workbench", serving: bool = False, allow_gpu_pending: bool = False) -> dict[str, Any]:
    config = config or resolve_config()
    scope = "real" if scope == "workbench" and (model_id or all_models) else scope
    checks = workbench_checks(config, serving)
    if scope == "workbench":
        overall = "BLOCKED" if any(item["status"] == "BLOCKED" for item in checks) else "READY"
        return {"overall": overall, "checks": checks}
    checks.append(fashioniq_check(config.FASHIONIQ_ROOT))
    try:
        registry = load_registry()
        checks.append(record("registry", "OK", "registry parsed", models=len(registry["models"])))
    except Exception as error:
        return {"overall": "BLOCKED", "checks": checks + [record("registry", "BLOCKED", str(error))]}
    models = registry["models"]
    if model_id:
        models = [model for model in models if model["model_id"] == model_id]
        if not models:
            return {"overall": "BLOCKED", "checks": checks + [record("selection", "BLOCKED", f"unknown model: {model_id}")]}
    for model in models if (all_models or model_id or scope == "real") else []:
        checks.extend(model_checks(model, config, allow_gpu_pending=allow_gpu_pending))
    overall = "BLOCKED" if any(item["status"] == "BLOCKED" for item in checks) else "WARNING" if any(item["status"] == "WARNING" for item in checks) else "READY"
    return {"overall": overall, "checks": checks}


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", help="inspect one registry model in real scope")
    group.add_argument("--all", action="store_true", help="inspect every registry model in real scope")
    parser.add_argument("--scope", choices=("workbench", "real"), default="workbench")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-gpu-pending", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = parser_for()
    args = parser.parse_args(argv)
    if args.dataset_root:
        config = replace(resolve_config(), FASHIONIQ_ROOT=args.dataset_root.resolve())
        report = collect(config, model_id=args.model, all_models=args.all, scope=args.scope,
                         serving=args.serve, allow_gpu_pending=args.allow_gpu_pending)
    else:
        report = collect(model_id=args.model, all_models=args.all, scope=args.scope,
                         serving=args.serve, allow_gpu_pending=args.allow_gpu_pending)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for item in report["checks"]:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        print(f"[{report['overall']}] doctor")
    return 0 if report["overall"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
