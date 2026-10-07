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

from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination, checkpoint_path, fashioniq_required_paths, load_registry, sha256_file

from workbench.backend.fashioniq_layout import CATEGORIES, missing_paths, standard_paths


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
    dirty = bool(git(path, "status", "--porcelain"))
    return {"path": str(path), "state": "dirty" if dirty else "clean", "head": head, "dirty": dirty, "pin": model.get("upstream_commit_sha")}


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


def model_checks(model: dict[str, Any], config: WorkbenchConfig) -> list[dict[str, Any]]:
    model_id = model["model_id"]
    state = source_state(model, config.WORKBENCH_THIRD_PARTY_ROOT)
    source_ok = state["state"] == "clean" and (not state["pin"] or state["head"] == state["pin"])
    checks = [record(f"source:{model_id}", "OK" if source_ok else "BLOCKED", "source pin ready" if source_ok else "source unavailable, dirty, or not pinned", **state)]
    runtime_blockers: list[str] = []
    csmcir_assets: list[dict[str, Any]] = []
    if not model.get("source_available"):
        runtime_blockers.append("upstream source unavailable")
    elif state["state"] != "clean":
        runtime_blockers.append(f"source {state['state']}: {state['path']}")
    elif state["pin"] and state["head"] != state["pin"]:
        runtime_blockers.append(f"source pin mismatch: expected {state['pin']}, found {state['head']}")
    source = Path(state["path"]) if state["path"] else config.WORKBENCH_THIRD_PARTY_ROOT
    adapter = ADAPTERS.get(model_id)
    if adapter is None or getattr(adapter, "command", None) is OfficialScriptAdapter.command:
        runtime_blockers.append("adapter command not audited")
    elif getattr(adapter, "script", None) and not (source / adapter.script).is_file():
        runtime_blockers.append(f"official evaluator missing: {source / adapter.script}")
    if model_id == "encoder" and not (source / "open_clip_pytorch_model.bin").is_file():
        runtime_blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    if model_id == "csmcir":
        layout = source / "fashionIQ_dataset"
        if not layout.is_symlink():
            runtime_blockers.append(f"CSMCIR fixed dataset layout missing: {layout}")
        elif layout.resolve() != config.FASHIONIQ_ROOT.resolve():
            runtime_blockers.append(f"CSMCIR dataset link resolves to {layout.resolve()}, expected {config.FASHIONIQ_ROOT.resolve()}")
        asset_groups = (
            ("csmcir:base-dataset", "FashionIQ base dataset", fashioniq_base_paths(config.FASHIONIQ_ROOT)),
            ("csmcir:qwen-captions", "CSMCIR Qwen captions", csmcir_auxiliary_paths(config, "Qwen captions")),
            ("csmcir:cot-captions", "CSMCIR COT_ours2 captions", csmcir_auxiliary_paths(config, "COT_ours2 captions")),
        )
        for name, label, paths in asset_groups:
            missing = missing_path_strings(paths)
            root = config.FASHIONIQ_ROOT if name != "csmcir:cot-captions" else source
            csmcir_assets.append(record(name, "OK" if not missing else "BLOCKED", f"{label} available" if not missing else f"{label} missing", root=str(root), missing=missing))
            runtime_blockers.extend(f"{label} missing: {path}" for path in missing)
            if name == "csmcir:cot-captions" and missing:
                runtime_blockers.append("required COT_ours2 captions have no verified automatic acquisition source")
    if model_id != "csmcir":
        missing = missing_path_strings(fashioniq_required_paths(model, config.FASHIONIQ_ROOT))
        runtime_blockers.extend(f"FashionIQ {model['fashioniq_layout']} requirement missing: {path}" for path in missing)
    checkpoints = []
    for checkpoint in model["checkpoint_variants"]:
        path = checkpoint_path(model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        installed = path.is_file()
        local_sha = sha256_file(path) if installed else None
        official_sha = checkpoint.get("expected_sha256")
        mapping = checkpoint["checkpoint_mapping_status"] != "UNVERIFIED"
        blockers = list(runtime_blockers)
        if not installed:
            blockers.append(f"checkpoint missing: {path}")
        if not mapping:
            blockers.append("checkpoint mapping unresolved")
        if official_sha and local_sha != official_sha:
            blockers.append("official hash mismatch")
        checkpoints.append({"checkpoint_id": checkpoint["checkpoint_id"], "path": str(path), "installed": installed, "local_sha256": local_sha, "official_sha256": official_sha, "official_sha256_available": official_sha is not None, "mapping_verified": mapping, "runtime_blockers": blockers, "command_ready": not blockers})
    status = "OK" if checkpoints and all(item["command_ready"] for item in checkpoints) else "BLOCKED"
    checks.append(record(f"runtime:{model_id}", status, "official command ready" if status == "OK" else "official command blocked", checkpoints=checkpoints))
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


def collect(config: WorkbenchConfig | None = None, model_id: str | None = None, all_models: bool = False, scope: str = "workbench", serving: bool = False) -> dict[str, Any]:
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
        checks.extend(model_checks(model, config))
    overall = "BLOCKED" if any(item["status"] == "BLOCKED" for item in checks) else "WARNING" if any(item["status"] == "WARNING" for item in checks) else "READY"
    return {"overall": overall, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", help="inspect one registry model in real scope")
    group.add_argument("--all", action="store_true", help="inspect every registry model in real scope")
    parser.add_argument("--scope", choices=("workbench", "real"), default="workbench", help="workbench checks local service prerequisites; real checks model and dataset prerequisites")
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ root for real preflight")
    parser.add_argument("--serve", action="store_true", help="also require frontend service dependencies")
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args(argv)
    if args.dataset_root:
        config = replace(resolve_config(), FASHIONIQ_ROOT=args.dataset_root.resolve())
        report = collect(config, model_id=args.model, all_models=args.all, scope=args.scope, serving=args.serve)
    else:
        report = collect(model_id=args.model, all_models=args.all, scope=args.scope, serving=args.serve)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for item in report["checks"]:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        print(f"[{report['overall']}] doctor")
    return 0 if report["overall"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
