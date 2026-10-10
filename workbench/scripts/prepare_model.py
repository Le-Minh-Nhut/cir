#!/usr/bin/env python3
"""Inspect model preparation status. Never executes preparation scripts without explicit authorization.

Default: read-only status from preparation_contracts.yaml. Use --execute --allow-preparation
to run only the one verified automated CSMCIR dataset link step. All other preparation is manual.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.fashioniq_layout import missing_paths
from workbench.backend.registry import (
    load_registry,
    preparation_contract_for_model,
    preparation_paths,
)


def _check_paths(paths: tuple[Path, ...]) -> list[str]:
    return [str(p) for p in missing_paths(paths)]


def status(model: dict, config) -> dict:
    model_id = model["model_id"]
    root = config.FASHIONIQ_ROOT
    contract = preparation_contract_for_model(model)
    if contract is None:
        from workbench.backend.fashioniq_layout import standard_paths
        missing = _check_paths(standard_paths(root))
        return {
            "model_id": model_id,
            "preparation_id": None,
            "contract_type": "standard_fashioniq",
            "automation_policy": "NONE_REQUIRED",
            "deterministic": "N/A",
            "raw_inputs_missing": missing,
            "generated_artifacts_missing": [],
            "external_assets": [],
            "fully_prepared": not missing,
            "notes": "Standard FashionIQ layout required; no custom generated artifacts.",
        }
    raw_missing = _check_paths(preparation_paths(contract, root, "raw_inputs"))
    gen_missing = _check_paths(preparation_paths(contract, root, "generated_artifacts"))
    return {
        "model_id": model_id,
        "preparation_id": contract["preparation_id"],
        "contract_type": contract.get("source_family"),
        "automation_policy": contract["automation_policy"],
        "deterministic": contract["deterministic_status"],
        "raw_inputs_missing": raw_missing,
        "generated_artifacts_missing": gen_missing,
        "external_assets": contract["external_assets"],
        "fully_prepared": not raw_missing and not gen_missing,
        "notes": contract["notes"],
    }


def print_status(s: dict) -> None:
    label = "OK" if s["fully_prepared"] else ("BLOCKED" if s["raw_inputs_missing"] else "MANUAL")
    print(f"[{label}] {s['model_id']}: policy={s['automation_policy']} deterministic={s['deterministic']}")
    if s["raw_inputs_missing"]:
        print(f"  raw inputs missing ({len(s['raw_inputs_missing'])}): {s['raw_inputs_missing'][0]}")
    if s["generated_artifacts_missing"]:
        print(f"  generated artifacts missing ({len(s['generated_artifacts_missing'])}): {s['generated_artifacts_missing'][0]}")
    if s["external_assets"]:
        blocked_ext = [a for a in s["external_assets"] if a.get("required_for_evaluation") and not a.get("optional")]
        if blocked_ext:
            print(f"  required external assets: {', '.join(a['name'] for a in blocked_ext)}")
    if s["notes"]:
        print(f"  notes: {s['notes']}")


def _csmcir_link(config, dry_run: bool, check_only: bool) -> bool:
    """Delegate CSMCIR dataset symlink to prepare_dataset.py."""
    scripts = Path(__file__).resolve().parent
    argv = [sys.executable, str(scripts / "prepare_dataset.py"),
            "--dataset-root", str(config.FASHIONIQ_ROOT), "--model", "csmcir"]
    if dry_run:
        argv.append("--dry-run")
    elif check_only:
        argv.append("--check-only")
    print(f"[RUN] CSMCIR preparation: {' '.join(argv)}")
    result = subprocess.run(argv)
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list all model preparation statuses")
    group.add_argument("--model", help="show or execute preparation for one model")
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ root (overrides config)")
    parser.add_argument("--execute", action="store_true",
                        help="execute preparation for the one verified automated step (CSMCIR symlink only)")
    parser.add_argument("--allow-preparation", action="store_true",
                        help="authorize dataset-link creation; required with --execute")
    parser.add_argument("--dry-run", action="store_true", help="show planned operations without changes")
    parser.add_argument("--json", action="store_true", help="emit machine-readable status")
    args = parser.parse_args(argv)

    if args.execute and not args.allow_preparation:
        parser.error("--execute requires --allow-preparation (explicit authorization)")

    config = resolve_config()
    if args.dataset_root:
        from dataclasses import replace
        config = replace(config, FASHIONIQ_ROOT=args.dataset_root.resolve())

    registry = load_registry()
    models = registry["models"]
    if args.model:
        models = [m for m in models if m["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")

    statuses = [status(m, config) for m in models]

    if args.json:
        import json
        print(json.dumps(statuses if args.list else statuses[0], indent=2))
        return 0 if all(s["fully_prepared"] for s in statuses) else 1

    if args.list:
        for s in statuses:
            print_status(s)
        unprepared = [s for s in statuses if not s["fully_prepared"]]
        print(f"\n{len(unprepared)} model(s) not fully prepared.")
        return 0

    s = statuses[0]
    model = models[0]
    print_status(s)

    if args.execute:
        if model["model_id"] == "csmcir":
            ok = _csmcir_link(config, args.dry_run, check_only=False)
            return 0 if ok else 1
        print(f"[BLOCKED] {model['model_id']}: no verified automated preparation step; manual preparation required.",
              file=sys.stderr)
        print(f"  policy={s['automation_policy']} notes={s['notes']}", file=sys.stderr)
        return 2

    return 0 if s["fully_prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
