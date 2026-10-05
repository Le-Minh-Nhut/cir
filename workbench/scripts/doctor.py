#!/usr/bin/env python3
"""Inspect local workbench readiness without changing files or using network."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import checkpoint_path, load_registry, sha256_file

CATEGORIES = ("dress", "shirt", "toptee")


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


def fashioniq_check(root: Path) -> dict[str, Any]:
    required = [root / name for name in ("captions", "image_splits", "images")]
    missing = [str(path) for path in required if not path.is_dir()]
    status = "OK" if not missing else "BLOCKED"
    return record("fashioniq", status, "FashionIQ layout available" if not missing else "FashionIQ layout missing", root=str(root), missing=missing, categories=list(CATEGORIES))


def result_counts(root: Path) -> tuple[int, int]:
    mock = experiment = 0
    for path in root.rglob("*.json") if root.is_dir() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            kind = payload.get("run", {}).get("data_kind", "experiment")
        except (OSError, json.JSONDecodeError):
            continue
        if kind == "mock":
            mock += 1
        else:
            experiment += 1
    return mock, experiment


def model_checks(model: dict[str, Any], config: WorkbenchConfig) -> list[dict[str, Any]]:
    model_id = model["model_id"]
    state = source_state(model, config.WORKBENCH_THIRD_PARTY_ROOT)
    source_ok = state["state"] == "clean" and (not state["pin"] or state["head"] == state["pin"])
    checks = [record(f"source:{model_id}", "OK" if source_ok else "BLOCKED", "source pin ready" if source_ok else "source unavailable, dirty, or not pinned", **state)]
    runtime_blockers: list[str] = []
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
        paths = [layout / part for part in ("captions", "image_splits", "images")]
        paths += [source / "qwen_captions" / f"{category}_cot_val.json" for category in CATEGORIES]
        paths += [source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json" for category in CATEGORIES]
        missing = next((path for path in paths if not path.exists()), None)
        if missing:
            runtime_blockers.append(f"CSMCIR fixed dataset layout missing: {missing}")
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
    return checks


def collect(config: WorkbenchConfig | None = None, model_id: str | None = None, all_models: bool = False) -> dict[str, Any]:
    config = config or resolve_config()
    checks: list[dict[str, Any]] = []
    branch = git(config.CIR_REPO_ROOT, "branch", "--show-current")
    dirty = git(config.CIR_REPO_ROOT, "status", "--porcelain")
    checks.append(record("repository", "OK", "repository information", root=str(config.CIR_REPO_ROOT), branch=branch, dirty=bool(dirty)))
    checks.append(record("python", "OK", platform.python_version(), executable=sys.executable))
    for name in ("fastapi", "uvicorn", "duckdb", "pydantic", "yaml"):
        present = importlib.util.find_spec(name) is not None
        checks.append(record(f"import:{name}", "OK" if present else "WARNING", "available" if present else "missing", module=name))
    torch = importlib.util.find_spec("torch")
    checks.append(record("torch", "OK" if torch else "WARNING", "available" if torch else "not installed", cuda_available=None))
    if torch:
        try:
            import torch as torch_module
            checks[-1]["evidence"]["cuda_available"] = torch_module.cuda.is_available()
        except Exception as error:
            checks[-1] = record("torch", "WARNING", "import failed", error=str(error), cuda_available=None)
    frontend = config.CIR_REPO_ROOT / "workbench" / "frontend"
    checks.append(record("frontend", "OK" if (frontend / "package.json").is_file() and (frontend / "node_modules").is_dir() else "WARNING", "frontend dependencies available" if (frontend / "node_modules").is_dir() else "package.json or node_modules missing", node=bool(shutil_which("node")), npm=bool(shutil_which("npm")), package_json=str(frontend / "package.json"), node_modules=str(frontend / "node_modules")))
    checks.append(fashioniq_check(config.FASHIONIQ_ROOT))
    try:
        registry = load_registry()
        checks.append(record("registry", "OK", "registry parsed", models=len(registry["models"])))
    except Exception as error:
        return {"overall": "BLOCKED", "checks": checks + [record("registry", "BLOCKED", str(error))]}
    results_root = config.WORKBENCH_RESULTS_ROOT
    mock, experiment = result_counts(results_root)
    checks.append(record("results", "OK", "canonical result files counted", root=str(results_root), mock=mock, experiment=experiment))
    database = config.CIR_REPO_ROOT / "workbench" / "artifacts" / "workbench.duckdb"
    if database.is_file() and importlib.util.find_spec("duckdb"):
        try:
            import duckdb
            with duckdb.connect(str(database), read_only=True) as connection:
                count = connection.execute("select count(*) from runs").fetchone()[0]
            checks.append(record("duckdb", "OK", "index readable", path=str(database), indexed_runs=count))
        except Exception as error:
            checks.append(record("duckdb", "WARNING", "index unreadable", path=str(database), error=str(error)))
    else:
        checks.append(record("duckdb", "WARNING", "index unavailable", path=str(database), indexed_runs=None))
    models = registry["models"]
    if model_id:
        models = [model for model in models if model["model_id"] == model_id]
        if not models:
            return {"overall": "BLOCKED", "checks": checks + [record("selection", "BLOCKED", f"unknown model: {model_id}")]}
    for model in models if (all_models or model_id) else []:
        checks.extend(model_checks(model, config))
    overall = "BLOCKED" if any(item["status"] == "BLOCKED" for item in checks) else "WARNING" if any(item["status"] == "WARNING" for item in checks) else "READY"
    return {"overall": overall, "checks": checks}


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", help="inspect one registry model")
    group.add_argument("--all", action="store_true", help="inspect every registry model")
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args(argv)
    report = collect(model_id=args.model, all_models=args.all)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for item in report["checks"]:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        print(f"[{report['overall']}] doctor")
    return 0 if report["overall"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
