from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def environment_status(model: dict[str, Any]) -> dict[str, Any]:
    environment = model.get("environment") or {}
    variable = environment.get("python_env_var")
    if not variable:
        return {"status": "UNCONFIGURED", "variable": None, "interpreter": None}
    configured = os.environ.get(variable)
    if not configured:
        return {"status": "UNCONFIGURED", "variable": variable, "interpreter": None}
    interpreter = Path(configured).expanduser().resolve()
    if not interpreter.is_file():
        return {"status": "MISSING", "variable": variable, "interpreter": str(interpreter)}
    if not os.access(interpreter, os.X_OK):
        return {"status": "NOT_EXECUTABLE", "variable": variable, "interpreter": str(interpreter)}
    return {"status": "READY", "variable": variable, "interpreter": str(interpreter)}


def environment_blockers(model: dict[str, Any]) -> list[str]:
    if not model.get("environment_required"):
        return []
    state = environment_status(model)
    if state["status"] == "READY":
        return []
    variable = state["variable"] or "model-specific Python"
    if state["status"] == "MISSING":
        return [f"execution environment interpreter missing: {variable}={state['interpreter']}"]
    if state["status"] == "NOT_EXECUTABLE":
        return [f"execution environment interpreter not executable: {variable}={state['interpreter']}"]
    return [f"execution environment not configured: set {variable}"]


def python_executable(model: dict[str, Any]) -> str:
    state = environment_status(model)
    if state["status"] == "READY":
        return state["interpreter"]
    if model.get("environment_required"):
        raise RuntimeError(f"execution environment not ready: set {state['variable']}")
    return "python"


def source_clean_and_pinned(model: dict[str, Any], source: Path) -> tuple[bool, str | None, str | None]:
    try:
        top_level = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=source,
                                   check=True, capture_output=True, text=True).stdout.strip()
        if Path(top_level).resolve() != source.resolve():
            return False, None, "source path is not a Git checkout root"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, None, "source is not a readable Git checkout"
    expected = model.get("upstream_commit_sha")
    if dirty:
        return False, head, "source checkout is dirty"
    if expected and head != expected:
        return False, head, f"source pin mismatch: expected {expected}, found {head}"
    return True, head, None


def runtime_blockers(model: dict[str, Any], checkpoint: dict[str, Any], config: Any,
                     protocol: str | None = None, dataset_root: Path | None = None) -> list[str]:
    """Shared final preflight used by UI availability and guarded execution."""
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
    from workbench.backend.fashioniq_layout import missing_paths
    from workbench.backend.registry import (
        checkpoint_bundle_missing_paths, checkpoint_is_present, checkpoint_missing_paths,
        checkpoint_path, external_asset_blockers, fashioniq_required_paths,
        preparation_contract_for_model, preparation_paths, sha256_file,
    )

    blockers: list[str] = []
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    if source is None or not source.is_dir():
        blockers.append(f"source missing: {source}")
    else:
        clean, _, reason = source_clean_and_pinned(model, source)
        if not clean:
            blockers.append(reason or "source unavailable")
    if protocol is not None and protocol not in model.get("supported_protocols", []):
        blockers.append(f"protocol incompatible: {protocol}")
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    if not checkpoint_is_present(checkpoint_file, checkpoint):
        blockers.append(f"checkpoint missing: {checkpoint_missing_paths(checkpoint_file, checkpoint)[0]}")
    if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append("checkpoint mapping unresolved")
    expected_sha = checkpoint.get("expected_sha256")
    if expected_sha and checkpoint_file.is_file() and sha256_file(checkpoint_file) != expected_sha:
        blockers.append("official checkpoint hash mismatch")
    for bundle_id, missing in checkpoint_bundle_missing_paths(model, config.WORKBENCH_CHECKPOINT_ROOT).items():
        blockers.append(f"checkpoint bundle incomplete: {bundle_id}: {missing[0]}")
    adapter_type = ADAPTERS.get(model["model_id"])
    command_implemented = (adapter_type is not None and
                           getattr(adapter_type, "command", None) is not OfficialScriptAdapter.command)
    command_audited = (model.get("command_status") in {"COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"} or
                       (model.get("command_status") is None and command_implemented)) and command_implemented
    if not command_audited:
        blockers.append("adapter command not audited")
    elif source is not None and getattr(adapter_type, "script", None) and not (source / adapter_type.script).is_file():
        blockers.append(f"official evaluator missing: {source / adapter_type.script}")
    blockers.extend(environment_blockers(model))
    dataset_root = dataset_root or config.FASHIONIQ_ROOT
    contract = preparation_contract_for_model(model)
    if contract is None:
        missing = missing_paths(fashioniq_required_paths(model, dataset_root))
    else:
        missing = missing_paths(preparation_paths(contract, dataset_root, "raw_inputs") +
                                preparation_paths(contract, dataset_root, "generated_artifacts"))
    if missing:
        layout = model.get("fashioniq_layout") or "FashionIQ"
        blockers.append(f"FashionIQ {layout} requirement missing: {missing[0]}")
    blockers.extend(external_asset_blockers(model))
    if model["model_id"] == "encoder" and source is not None:
        if not (source / "datasets1.py").is_file():
            blockers.append(f"ENCODER evaluator import missing: {source / 'datasets1.py'}")
        if not (source / "open_clip_pytorch_model.bin").is_file():
            blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    if model["model_id"] == "csmcir" and source is not None:
        layout = source / "fashionIQ_dataset"
        if not layout.is_symlink() or layout.resolve() != config.FASHIONIQ_ROOT.resolve():
            blockers.append(f"CSMCIR requires canonical dataset link: {layout} -> {config.FASHIONIQ_ROOT}")
        for group, paths in (
            ("Qwen captions", tuple(config.FASHIONIQ_ROOT / "qwen_captions" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))),
            ("COT_ours2 captions", tuple(source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))),
        ):
            group_missing = missing_paths(paths)
            if group_missing:
                blockers.append(f"CSMCIR {group} missing: {group_missing[0]}")
    return blockers
