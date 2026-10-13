"""Shared model and dataset readiness requirements for operator entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def model_requirements(
    model: dict,
    config: Any,
    *,
    dataset_root: Path | None = None,
    canonical_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve data, preparation, external asset, and fixed-link requirements."""
    from workbench.backend.fashioniq_layout import image_extension_problems, missing_paths
    from workbench.backend.registry import (
        auxiliary_assets_for_model, auxiliary_destination, external_asset_statuses,
        fashioniq_required_paths, preparation_contract_for_model, preparation_paths,
        validate_auxiliary_asset,
    )

    root = Path(dataset_root) if dataset_root else config.FASHIONIQ_ROOT
    canonical_root = Path(canonical_root) if canonical_root is not None else config.FASHIONIQ_ROOT
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    contract = preparation_contract_for_model(model)
    raw = preparation_paths(contract, root, "raw_inputs") if contract else fashioniq_required_paths(model, root)
    generated = preparation_paths(contract, root, "generated_artifacts") if contract else ()
    paths = list(dict.fromkeys((*raw, *generated)))
    raw_missing, generated_missing = missing_paths(raw), missing_paths(generated)
    blocks: list[str] = []
    layout = model.get("fashioniq_layout") or "FashionIQ"
    if contract:
        blocks.extend(f"raw benchmark input missing: {path}" for path in raw_missing)
        blocks.extend(f"required generated artifact missing: {path}" for path in generated_missing)
    else:
        blocks.extend(f"FashionIQ {layout} requirement missing: {path}" for path in raw_missing)
    image_problems = [problem for extension in model.get("fashioniq_image_extensions") or ()
                      for problem in image_extension_problems(root, extension, model.get("fashioniq_layout"))]
    blocks.extend(f"FashionIQ image extension requirement: {problem}" for problem in image_problems)

    auxiliary = []
    for asset in auxiliary_assets_for_model(model["model_id"]):
        path = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, canonical_root)
        paths.append(path)
        res = validate_auxiliary_asset(asset, path, fashioniq_root=canonical_root)
        missing = res.status == "MISSING"
        auxiliary.append({"name": asset["family"], "path": str(path), "missing": missing, "status": res.status, "issues": res.issues})
        prefix = "CSMCIR " if model["model_id"] == "csmcir" else ""
        if missing:
            blocks.append(f"{prefix}{asset['family']} missing: {path}")
            if asset["family"] == "COT_ours2 captions":
                blocks.append("required COT_ours2 captions have no verified automatic acquisition source")
        elif not res.ok:
            if res.status == "OPERATOR_SUPPLIED_UNVERIFIED":
                blocks.append(f"{prefix}{asset['family']} unverified: {path} (operator-supplied unverified provenance)")
            else:
                for issue in res.issues:
                    blocks.append(f"{prefix}{asset['family']} invalid: {issue}")
    link = model.get("canonical_dataset_link")
    link_evidence = None
    if link and source is not None:
        link_path = source / link
        actual = link_path.resolve() if link_path.is_symlink() else None
        expected = canonical_root.resolve()
        paths.append(link_path)
        link_evidence = {"path": str(link_path), "expected": str(expected),
                         "actual": str(actual) if actual is not None else None}
        if not link_path.is_symlink():
            blocks.append(f"canonical dataset link missing: {link_path} -> {expected}")
        elif actual != expected:
            blocks.append(f"canonical dataset link resolves to {actual}, expected {expected}")
        elif not actual.is_dir():
            blocks.append(f"canonical dataset link target missing: {actual}")
    if model["model_id"] == "encoder" and source is not None:
        paths.extend((source / "datasets1.py", source / "open_clip_pytorch_model.bin"))

    asset_statuses = external_asset_statuses(model)
    paths.extend(Path(item["path"]) for item in asset_statuses if item["path"])
    messages = {"UNRESOLVED_PROVENANCE": "provenance unresolved", "UNCONFIGURED_LOCATION": "local location unconfigured",
                "CONFIGURED_MISSING": "file missing", "INVALID_FILE_TYPE": "expected regular file, found non-file",
                "EMPTY_PLACEHOLDER": "empty placeholder file", "CHECKSUM_MISMATCH": "checksum mismatch",
                "EXISTS_UNVERIFIED_CHECKSUM": "file exists but no verified checksum is available"}
    blocks.extend(f"external runtime asset {messages[item['status']]}: {item['name']}" +
                  (f" ({item['path']})" if item["path"] else "")
                  for item in asset_statuses if item["status"] != "VERIFIED_LOCAL_ASSET")
    return {"paths": list(dict.fromkeys(paths)), "blocks": list(dict.fromkeys(blocks)), "warnings": [],
            "evidence": {"raw_inputs_missing": [str(path) for path in raw_missing],
                         "generated_artifacts_missing": [str(path) for path in generated_missing],
                         "image_extension_problems": image_problems, "auxiliary_assets": auxiliary,
                         "dataset_link": link_evidence}}


def preflight_blockers(
    model: dict,
    config: Any,
    *,
    dataset_root: Path | None = None,
    canonical_root: Path | None = None,
) -> list[str]:
    """Single shared blocker list for model data and runtime prerequisites."""
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
    from workbench.backend.runtime import environment_blockers, source_clean_and_pinned

    requirements = model_requirements(model, config, dataset_root=dataset_root, canonical_root=canonical_root)
    blockers = list(requirements["blocks"])
    model_id = model["model_id"]
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    if not model.get("source_available") or source is None or not source.is_dir():
        blockers.append("upstream source unavailable")
    else:
        clean, _, reason = source_clean_and_pinned(model, source)
        if not clean:
            blockers.append(reason or "source unavailable")

    adapter = ADAPTERS.get(model_id)
    command = getattr(adapter, "command", None) if adapter is not None else None
    audited = (command is not None and command is not OfficialScriptAdapter.command and
               model.get("command_status") in {None, "COMMAND_AUDITED", "REPLAY_COMMAND_IMPLEMENTED"})
    if not audited:
        blockers.append(f"command not runnable: {model.get('command_status') or 'COMMAND_UNAUDITED'}")
    elif source is not None and getattr(adapter, "script", None) and not (source / adapter.script).is_file():
        blockers.append(f"official evaluator missing: {source / adapter.script}")

    blockers.extend(environment_blockers(model))
    if model_id == "encoder" and source is not None:
        if not (source / "datasets1.py").is_file():
            blockers.append(f"ENCODER evaluator import missing: {source / 'datasets1.py'}")
        if not (source / "open_clip_pytorch_model.bin").is_file():
            blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    return list(dict.fromkeys(blockers))
