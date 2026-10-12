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
from workbench.backend.preflight import model_requirements, preflight_blockers
from workbench.backend.registry import load_registry, preparation_contract_for_model

def status(model: dict, config) -> dict:
    blockers = preflight_blockers(model, config)
    model_id = model["model_id"]
    contract = preparation_contract_for_model(model)
    requirements = model_requirements(model, config)
    evidence = requirements["evidence"]
    raw_missing = evidence["raw_inputs_missing"]
    generated_missing = evidence["generated_artifacts_missing"]
    result = {
        "model_id": model_id,
        "preparation_id": contract["preparation_id"] if contract else None,
        "contract_type": contract.get("source_family") if contract else "standard_fashioniq",
        "automation_policy": contract["automation_policy"] if contract else "NONE_REQUIRED",
        "deterministic": contract["deterministic_status"] if contract else "N/A",
        "raw_inputs_missing": raw_missing,
        "generated_artifacts_missing": generated_missing,
        "external_assets": contract["external_assets"] if contract else [],
        "fully_prepared": not requirements["blocks"],
        "blockers": blockers,
        "notes": contract["notes"] if contract else "Standard FashionIQ layout required; no custom generated artifacts.",
    }
    if evidence["image_extension_problems"]:
        result["image_extension_problems"] = evidence["image_extension_problems"]
    if evidence["dataset_link"]:
        result["dataset_link"] = evidence["dataset_link"]
    return result


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


def _fixed_link(config, model_id: str, dry_run: bool, check_only: bool) -> bool:
    """Stage the model's declared fixed dataset link through the shared helper.

    Any model whose pinned evaluator hard-codes its dataset root declares
    `canonical_dataset_link`, and `prepare_dataset.py` implements the staging. Delegating keeps
    one implementation of link creation and verification instead of a second one here.
    """
    scripts = Path(__file__).resolve().parent
    argv = [sys.executable, str(scripts / "prepare_dataset.py"),
            "--dataset-root", str(config.FASHIONIQ_ROOT), "--model", model_id]
    if dry_run:
        argv.append("--dry-run")
    elif check_only:
        argv.append("--check-only")
    print(f"[RUN] {model_id} dataset-link preparation: {' '.join(argv)}")
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
        if model.get("canonical_dataset_link"):
            ok = _fixed_link(config, model["model_id"], args.dry_run, check_only=False)
            return 0 if ok else 1
        print(f"[BLOCKED] {model['model_id']}: no verified automated preparation step; manual preparation required.",
              file=sys.stderr)
        print(f"  policy={s['automation_policy']} notes={s['notes']}", file=sys.stderr)
        return 2

    if model.get("canonical_dataset_link") and not args.dry_run:
        # A read-only status run still reports whether the declared link is correct, without
        # creating it: the check is part of readiness, the creation is the authorized step.
        _fixed_link(config, model["model_id"], dry_run=False, check_only=True)

    return 0 if s["fully_prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
