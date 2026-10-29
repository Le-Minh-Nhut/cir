#!/usr/bin/env python3
"""Opt-in CIRCUS composition CLI; WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE=1 is TEST-ONLY."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import autopilot
from workbench.backend.checkpoint_selection import select_checkpoints
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_path, load_registry
from workbench.composition import data as composition_data
from workbench.composition.adapters import adapter_class
from workbench.composition.runner import (
    ModelSpec,
    assert_composition_destination,
    inspect_image_inventory,
    model_interpreter_identity,
    run_composition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="autopilot YAML config")
    parser.add_argument("--models", default="selected",
                        help="comma-separated model IDs or 'selected' authorized cohort")
    parser.add_argument("--subsets", default="full,sf,validated",
                        help="comma-separated subsets: full,sf,validated")
    parser.add_argument("--resume", action="store_true", help="reuse matching composition results")
    parser.add_argument("--smoke", nargs="?", const=1, type=int,
                        help="limit query count (default 1); always preserve full gallery")
    parser.add_argument("--plan", action="store_true", help="show work and paths without writing state")
    parser.add_argument("--prepare-data", action="store_true",
                        help="verify release and inventory missing gallery/reference/positive images")
    parser.add_argument("--seed", type=int, default=0, help="seed recorded for deterministic summaries")
    return parser


def _load_configuration(path: Path | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    registry_override = os.environ.get("WORKBENCH_REGISTRY_PATH")
    if path is None:
        config = resolve_config()
        registry = load_registry(Path(registry_override) if registry_override else None)
        return None, None, config, registry
    profile = autopilot.load_config(path)
    config = autopilot._workbench_config(profile)
    registry = load_registry(Path(registry_override)) if registry_override else profile.registry
    return profile, profile, config, registry


def _available_models(registry: dict[str, Any]) -> list[str]:
    return [str(model["model_id"]) for model in registry.get("models", [])
            if model.get("composition_adapter")]


def _path_ready(model: dict[str, Any], checkpoint_id: str, config: Any) -> tuple[bool, str | None]:
    from workbench.backend.registry import checkpoint_by_id

    try:
        checkpoint = checkpoint_by_id(model, checkpoint_id)
        path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    except Exception as error:
        return False, f"checkpoint selection unavailable: {type(error).__name__}: {error}"
    missing = ([name for name in checkpoint.get("required_files", []) if not (path / name).is_file()]
               if checkpoint.get("bundle_id") else [])
    if missing:
        return False, f"checkpoint bundle missing member(s): {', '.join(missing)}"
    if not path.exists():
        return False, f"checkpoint not staged: {path}"
    return True, None


def _composition_checkpoint(model: dict[str, Any], checkpoint_ids: list[str], bundle_id: str | None,
                            unavailable: list[dict[str, Any]]) -> list[str] | None:
    """Resolve bundles to one job and independent variants to one job each."""
    if bundle_id:
        return [bundle_id]
    selected = set(checkpoint_ids)
    for bundle in model.get("checkpoint_bundles", []):
        members = set(bundle.get("required_checkpoint_ids", []))
        if selected & members:
            if selected.issuperset(members):
                return [str(bundle["bundle_id"])]
            unavailable.append({"model_id": model["model_id"], "status": "UNSUPPORTED",
                                "reason": f"composition supports full checkpoint bundle {bundle['bundle_id']!r}; "
                                          "individual or partial bundle members are unsupported"})
            return None
    return list(dict.fromkeys(checkpoint_ids)) or None


def _select_models(raw: str, registry: dict[str, Any], profile: Any,
                   config: Any) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    models = registry.get("models", [])
    by_id = {model["model_id"]: model for model in models}
    pairs: list[tuple[dict[str, Any], str]] = []
    unavailable: list[dict[str, Any]] = []
    if raw == "selected":
        if profile is None:
            raise ValueError("--models selected requires --config to reuse authorized autopilot cohort")
        plan = autopilot.resolved_plan(profile, dry_run=True)
        rows = {row["model_id"]: row for row in plan["models"]}
        for model_id in profile.selected_model_ids:
            model = by_id[model_id]
            row = rows.get(model_id, {})
            checkpoint_ids = row.get("checkpoint_ids") or []
            if not checkpoint_ids:
                unavailable.append({"model_id": model_id, "status": row.get("status", "NOT_READY"),
                                    "reason": "; ".join(row.get("blockers", [])) or
                                    "authorized cohort selection has no checkpoint"})
                continue
            selection = row.get("checkpoint_selection") or {}
            checkpoint_ids = _composition_checkpoint(model, checkpoint_ids,
                                                     selection.get("bundle_id"), unavailable)
            if checkpoint_ids is None:
                continue
            for checkpoint_id in checkpoint_ids:
                pairs.append((model, checkpoint_id))
                ready, reason = _path_ready(model, checkpoint_id, config)
                if not ready:
                    unavailable.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                                        "status": "NOT_READY", "reason": reason})
        return pairs, unavailable
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("--models needs model IDs or literal 'selected'")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown model ID(s): {', '.join(unknown)}; available: {', '.join(_available_models(registry))}")
    for model_id in requested:
        model = by_id[model_id]
        if not model.get("composition_adapter"):
            unavailable.append({"model_id": model_id, "status": "UNSUPPORTED",
                                "reason": "registry has no composition_adapter module path"})
            continue
        selection = select_checkpoints(
            model, policy=getattr(profile, "checkpoint_policy", "strict_clean"),
            requested=getattr(profile, "checkpoint_overrides", {}).get(model_id),
            checkpoint_root=config.WORKBENCH_CHECKPOINT_ROOT)
        if selection.blocked:
            unavailable.append({"model_id": model_id, "status": "NOT_READY", "reason": selection.reason})
            continue
        checkpoint_ids = _composition_checkpoint(model, list(selection.checkpoint_ids),
                                                 selection.bundle_id, unavailable)
        if checkpoint_ids is None:
            continue
        for checkpoint_id in checkpoint_ids:
            pairs.append((model, checkpoint_id))
            ready, reason = _path_ready(model, checkpoint_id, config)
            if not ready:
                unavailable.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                                    "status": "NOT_READY", "reason": reason})
    return pairs, unavailable


def _checkpoint_identity(model: dict[str, Any], checkpoint_id: str, config: Any) -> str:
    from workbench.backend.registry import checkpoint_by_id, sha256_file

    checkpoint = checkpoint_by_id(model, checkpoint_id)
    path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(child for child in path.rglob("*") if child.is_file()):
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(sha256_file(item).encode("ascii"))
        return digest.hexdigest()
    return "missing"


def _load_data(config: Any, *, synthetic: bool):
    root = getattr(config, "WORKBENCH_CIRCUS_RELEASE_ROOT", None)
    if root is None:
        root = Path(config.CIR_DATA_ROOT) / "circus_release"
    return composition_data.load_release(Path(root), verify_hashes=not synthetic,
                                         expect_released_counts=not synthetic)


def _plan(pairs: list[tuple[dict[str, Any], str]], unavailable: list[dict[str, Any]],
          data: Any, config: Any, subsets: list[str]) -> dict[str, Any]:
    artifacts = Path(config.WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()
    inventory = inspect_image_inventory(data, [model for model, _checkpoint in pairs],
                                        Path(config.FASHIONIQ_ROOT))
    outputs = []
    readiness = []
    for model, checkpoint_id in pairs:
        try:
            adapter = adapter_class(model["model_id"], {"models": [model]})
            status = adapter.status(config)
            runtime_identity = model_interpreter_identity(model, config, verify=False)
            readiness.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                              "ready": status.ready,
                              "status": "READY" if status.ready else "SELECTED_BUT_BLOCKED",
                              "reason": status.reason,
                              "adapter": model.get("composition_adapter"),
                              "runtime": {"model_interpreter": runtime_identity}})
        except Exception as error:
            readiness.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                              "ready": False, "status": "SELECTED_BUT_BLOCKED",
                              "reason": f"{type(error).__name__}: {error}",
                              "adapter": model.get("composition_adapter"),
                              "runtime": {"model_interpreter": model_interpreter_identity(model, config, verify=False)}})
        for subset in subsets:
            for condition in composition_data.CONDITIONS:
                outputs.append(str(assert_composition_destination(
                    artifacts, artifacts / "composition" / "results" / data.gallery.protocol_id /
                    model["model_id"] / checkpoint_id / "<attempt>" / subset / f"{condition}.jsonl")))
            outputs.append(str(assert_composition_destination(
                artifacts, artifacts / "composition" / "summary" /
                f"{model['model_id']}_{checkpoint_id}_<attempt>" / f"{subset}.json")))
    gallery_identity = hashlib.sha256(json.dumps(
        {"protocol_id": data.gallery.protocol_id, "image_ids": data.gallery.image_ids},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"selected_models": [{"model_id": model["model_id"], "checkpoint_id": checkpoint}
                                 for model, checkpoint in pairs],
            "unavailable": unavailable, "readiness": readiness,
            "protocol_id": data.gallery.protocol_id, "gallery_count": len(data.gallery.image_ids),
            "gallery_identity_sha256": gallery_identity, "missing_assets": inventory,
            "expected_output_paths": outputs,
            "category_resolution": data.provenance.get("category_resolution", {}),
            "is_synthetic": bool(data.provenance.get("is_synthetic")),
            "failed_expectations": data.provenance.get("failed_expectations", ())}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile, _, config, registry = _load_configuration(args.config)
        subsets = [item.strip() for item in args.subsets.split(",") if item.strip()]
        allowed = {"full", "sf", "validated"}
        if not subsets or len(set(subsets)) != len(subsets) or set(subsets) - allowed:
            raise ValueError("--subsets must be comma-separated values from full,sf,validated")
        fixture_value = os.environ.get("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE")
        if fixture_value not in (None, "0", "1"):
            raise ValueError("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE must be exactly '1' for tests or unset/'0'")
        synthetic = fixture_value == "1"
        requested_order = (list(profile.selected_model_ids) if args.models == "selected" and profile is not None
                           else [item.strip() for item in args.models.split(",") if item.strip()])
        pairs, unavailable = _select_models(args.models, registry, profile, config)
        data = _load_data(config, synthetic=synthetic)
        data = composition_data.assign_garment_categories(data, config.FASHIONIQ_ROOT)
        if args.plan or args.prepare_data:
            report = _plan(pairs, unavailable, data, config, subsets)
            if args.prepare_data:
                from workbench.composition.data import resolve_release_paths, verify_release_hashes

                reports = verify_release_hashes(resolve_release_paths(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
                report["release_hashes"] = [{"path": str(item.path), "expected": item.expected,
                                             "actual": item.actual, "matches": item.matches}
                                            for item in reports]
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            return 0
        blocked = [{"model_id": item["model_id"], "checkpoint_id": item.get("checkpoint_id"),
                    "status": item["status"] if item["status"] == "UNSUPPORTED" else "SELECTED_BUT_BLOCKED",
                    "reason": item["reason"]}
                   for item in unavailable]
        specs = []
        blocked_pairs = {(item["model_id"], item.get("checkpoint_id")) for item in unavailable}
        for model, checkpoint_id in pairs:
            if (model["model_id"], checkpoint_id) in blocked_pairs:
                continue
            try:
                adapter_type = adapter_class(model["model_id"], registry)
                adapter_status = adapter_type.status(config)
            except Exception as error:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED",
                                "reason": f"adapter readiness failed: {type(error).__name__}: {error}"})
                continue
            if not adapter_status.ready:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED",
                                "reason": adapter_status.reason or "adapter is not ready"})
                continue
            ready, reason = _path_ready(model, checkpoint_id, config)
            if not ready:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED", "reason": reason})
                continue
            specs.append(ModelSpec(model["model_id"], checkpoint_id,
                                   lambda cfg, cid, cls=adapter_type: cls(cfg, cid), model,
                                   _checkpoint_identity(model, checkpoint_id, config)))
        report = run_composition(data, specs, config, subsets=subsets,
                                 resume=args.resume, smoke=args.smoke, seed=args.seed,
                                 synthetic=bool(data.provenance.get("is_synthetic"))) if specs else {
                                     "run_count": 0, "models": [], "gallery_count": len(data.gallery.image_ids),
                                     "gallery_identity_sha256": hashlib.sha256(json.dumps(
                                         {"protocol_id": data.gallery.protocol_id,
                                          "image_ids": data.gallery.image_ids}, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest(),
                                     "protocol_id": data.gallery.protocol_id,
                                     "synthetic": bool(data.provenance.get("is_synthetic"))}
        report["models"].extend(blocked)
        report["run_count"] = len(report["models"])
        order = {model_id: index for index, model_id in enumerate(requested_order)}
        report["models"].sort(key=lambda row: order.get(row["model_id"], len(order)))
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        failures = {"FAILED", "BLOCKED_MISSING_GALLERY_IMAGES", "MODEL_UNAVAILABLE", "PARTIAL"}
        if args.models != "selected":
            failures.update({"SELECTED_BUT_BLOCKED", "UNSUPPORTED"})
        # Fail runtime errors, explicit unavailable requests, or selected cohorts with zero runnable results.
        selected_empty = args.models == "selected" and not any(
            row["status"] in {"COMPLETED", "PARTIAL", "REUSED"} for row in report["models"])
        return 1 if selected_empty or any(row["status"] in failures for row in report["models"]) else 0
    except Exception as error:
        print(f"[COMPOSITION-EVAL-FAILED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
