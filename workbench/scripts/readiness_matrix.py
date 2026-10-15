#!/usr/bin/env python3
"""Registry-wide readiness matrix.

Composes what already exists -- the registry, the doctor's per-model checks, the environment plan and
the checkpoint policy -- into one table rather than inventing a second readiness model. Every cell is
a fact one of those components already verified; nothing here claims a state that was not observed.

The distinction that matters: an environment that only *solved* is not installed, an installed
environment is not runtime-verified, and none of those is scientific reproduction. The matrix keeps
those columns separate on purpose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import load_registry, model_by_id


def _environment_rows(config: WorkbenchConfig, models: list[dict]) -> dict[str, dict]:
    from workbench.backend.environment_planner import plan_environments

    try:
        plan = plan_environments(models, artifacts_root=config.WORKBENCH_ARTIFACTS_ROOT,
                                 third_party_root=config.WORKBENCH_THIRD_PARTY_ROOT)
    except Exception as error:
        return {model["model_id"]: {"blocked": f"environment plan failed: {error}"} for model in models}
    rows: dict[str, dict] = {}
    for group in plan.get("groups", []):
        for member in group["members"]:
            rows[member] = {
                "environment_id": group["environment_id"],
                "members": list(group["members"]),
                "state": group.get("state"),
                "solver_verification_status": group.get("solver_verification_status"),
                "installation_status": group.get("installation_status"),
                "runtime_verification_status": group.get("runtime_verification_status"),
                "conda_env_path": group.get("conda_env_path"),
                "interpreter": group.get("interpreter"),
                "runtime_adaptation": group.get("runtime_adaptation"),
                "split_reasons": list(group.get("split_reasons") or []),
            }
    for item in plan.get("blocked", []):
        rows[item["model_id"]] = {"blocked": item["reason"]}
    return rows


def _checkpoint_rows(config: WorkbenchConfig, models: list[dict], policy: str,
                     overrides: dict[str, list[str]]) -> dict[str, dict]:
    from workbench.backend.checkpoint_selection import select_checkpoints

    rows: dict[str, dict] = {}
    for model in models:
        selection = select_checkpoints(model, policy=policy,
                                       requested=overrides.get(model["model_id"]),
                                       checkpoint_root=config.WORKBENCH_CHECKPOINT_ROOT)
        rows[model["model_id"]] = selection.as_dict()
    return rows


def matrix(config: WorkbenchConfig, *, policy: str = "strict_clean",
           overrides: dict[str, list[str]] | None = None) -> dict:
    from workbench.scripts.doctor import collect

    registry = load_registry()
    models = registry["models"]
    environment = _environment_rows(config, models)
    checkpoints = _checkpoint_rows(config, models, policy, overrides or {})
    doctor = collect(config, all_models=True, scope="real")

    checks_by_model: dict[str, list[dict]] = {}
    for check in doctor.get("checks", []):
        name = check.get("name", "")
        if ":" in name:
            checks_by_model.setdefault(name.split(":", 1)[1], []).append(check)

    rows = []
    for model in models:
        model_id = model["model_id"]
        checks = checks_by_model.get(model_id, [])
        source = next((item for item in checks if item["name"].startswith("source:")), {})
        command = next((item for item in checks if item["name"].startswith("command:")), {})
        runtime = next((item for item in checks if item["name"].startswith("runtime:")), {})
        evidence = runtime.get("evidence") or {}
        checkpoint_evidence = evidence.get("checkpoints") or []
        env = environment.get(model_id) or {}
        selection = checkpoints.get(model_id) or {}
        readiness = {
            "SOURCE_VERIFIED": source.get("status") == "OK",
            "ENVIRONMENT_PLANNED": bool(env.get("environment_id")),
            "ENVIRONMENT_INSTALLED": env.get("installation_status") == "INSTALLED",
            "RUNTIME_VERIFIED": env.get("runtime_verification_status") == "READY",
            "GPU_VERIFIED": any("gpu" in item["name"] and item.get("status") == "OK" for item in checks),
            "CHECKPOINT_READY": bool(selection.get("checkpoint_ids"))
                                and bool(checkpoint_evidence)
                                and all(item.get("installed") for item in checkpoint_evidence),
            "EVALUATOR_READY": bool(command.get("status") == "OK" and evidence.get("runnable")),
            "SCIENTIFICALLY_REPRODUCED": model.get("reproduction_status") == "REPRODUCED",
        }
        blockers = []
        if not readiness["SOURCE_VERIFIED"]:
            blockers.append(source.get("detail") or "source not verified")
        if env.get("blocked"):
            blockers.append(f"environment: {env['blocked']}")
        if selection.get("blocked"):
            blockers.append(f"checkpoint policy: {selection.get('reason')}")
        for item in checkpoint_evidence:
            blockers.extend(item.get("runtime_blockers") or [])
        if command.get("status") != "OK":
            blockers.append(f"evaluator command: {command.get('detail')}")
        rows.append({
            "model_id": model_id,
            "method_name": model.get("method_name"),
            "native_protocol": model.get("native_protocol"),
            "source_dir": model.get("source_dir"),
            "upstream_commit_sha": model.get("upstream_commit_sha"),
            "environment_id": env.get("environment_id"),
            "environment_state": env.get("state"),
            # A planned ADAPTED runtime means RUNTIME_VERIFIED refers to the adapted
            # contract, not the authors' historical environment. Recorded here so the
            # readiness label cannot be read as an exact-reproduction claim.
            "environment_adapted": bool(env.get("runtime_adaptation")),
            "interpreter": env.get("interpreter"),
            "checkpoint_ids": selection.get("checkpoint_ids") or [],
            "checkpoint_policy_status": selection.get("status"),
            "checkpoint_conditions": selection.get("conditions") or {},
            "strict_clean_checkpoint_available": selection.get("strict_clean_checkpoint_available"),
            "readiness": readiness,
            "blockers": list(dict.fromkeys(blockers)),
            "reported_r10": model.get("reported_r10"),
            "reported_r50": model.get("reported_r50"),
        })

    return {
        "schema_version": 1,
        "checkpoint_policy": policy,
        "dataset_root": str(config.FASHIONIQ_ROOT),
        "third_party_root": str(config.WORKBENCH_THIRD_PARTY_ROOT),
        "checkpoint_root": str(config.WORKBENCH_CHECKPOINT_ROOT),
        "models": rows,
        "counts": {
            key: sum(1 for row in rows if row["readiness"][key])
            for key in ("SOURCE_VERIFIED", "ENVIRONMENT_PLANNED", "ENVIRONMENT_INSTALLED",
                        "RUNTIME_VERIFIED", "GPU_VERIFIED", "CHECKPOINT_READY", "EVALUATOR_READY",
                        "SCIENTIFICALLY_REPRODUCED")
        },
        "total_models": len(rows),
    }


def render(document: dict) -> str:
    lines = [f"# Registry readiness matrix (checkpoint policy: {document['checkpoint_policy']})", "",
             f"- dataset root: `{document['dataset_root']}`",
             f"- upstream root: `{document['third_party_root']}`",
             f"- checkpoint root: `{document['checkpoint_root']}`",
             f"- models: {document['total_models']}", "",
             "## States verified", "",
             "| State | Models |", "|---|---|"]
    for key, count in document["counts"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "## Per model", "",
              "| Model | Protocol | Source pin | Env group | Env state | Selected checkpoints | Evaluator | Blocker |",
              "|---|---|---|---|---|---|---|---|"]
    for row in document["models"]:
        pin = (row["upstream_commit_sha"] or "none")[:12]
        selected = ", ".join(row["checkpoint_ids"]) or "—"
        # No truncation: a model with several blockers used to hide all but the first, so the most
        # fundamental reason a model cannot run (for CSMCIR, that its pinned source cannot import)
        # could be invisible behind an incidental one further down the list.
        blocked = "; ".join(row["blockers"]) or "—"
        lines.append(f"| `{row['model_id']}` | {row['native_protocol'] or '—'} | {pin} | "
                     f"{row['environment_id'] or '—'} | {row['environment_state'] or '—'} | {selected} | "
                     f"{'READY' if row['readiness']['EVALUATOR_READY'] else 'BLOCKED'} | {blocked} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the machine-readable matrix")
    parser.add_argument("--output", type=Path, help="also write the rendered markdown here")
    parser.add_argument("--checkpoint-policy", choices=("strict_clean", "include_flagged_conditions"),
                        default="strict_clean")
    parser.add_argument("--model", action="append", dest="models",
                        help="restrict the matrix to these model ids; repeatable")
    args = parser.parse_args(argv)

    config = resolve_config()
    if args.models:
        selected = []
        for model_id in dict.fromkeys(args.models):
            selected.append(model_by_id(model_id))
        # The matrix answers questions about the registry, so keep it registry-wide but note the
        # restriction rather than silently narrowing what the operator sees.
        document = matrix(config, policy=args.checkpoint_policy)
        wanted = {model["model_id"] for model in selected}
        document["models"] = [row for row in document["models"] if row["model_id"] in wanted]
        document["total_models"] = len(document["models"])
    else:
        document = matrix(config, policy=args.checkpoint_policy)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(document), encoding="utf-8")
    if args.json:
        print(json.dumps(document, indent=2))
    elif not args.output:
        print(render(document))
    else:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
