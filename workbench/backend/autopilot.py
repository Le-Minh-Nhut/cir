from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import load_registry

REPOSITORY = Path(__file__).resolve().parents[2]
PERMISSIONS = {"allow_network", "allow_large_downloads", "allow_env_install", "allow_preparation", "allow_gpu_eval", "allow_index_write"}
PATH_KEYS = {"registry", "dataset_root", "artifacts", "checkpoints", "results", "third_party"}
BUDGET_KEYS = {"max_download_bytes", "max_models", "top_k"}
STATUSES = {"BLOCKED_SOURCE", "BLOCKED_DATASET", "BLOCKED_PREPARATION", "BLOCKED_CHECKPOINT", "BLOCKED_ENVIRONMENT", "BLOCKED_GPU", "BLOCKED_EVALUATOR", "BLOCKED_AUTHORIZATION", "FAILED_EVALUATION", "FAILED_OUTPUT_VALIDATION", "COMPLETED", "COMPLETED_WITH_LIMITATIONS", "NOT_RUN"}


@dataclass(frozen=True)
class AutopilotConfig:
    path: Path
    selected_model_ids: tuple[str, ...]
    paths: dict[str, Path]
    budgets: dict[str, int]
    permissions: dict[str, bool]
    registry: dict[str, Any]


def _mapping(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown, missing = set(value) - keys, keys - set(value)
    if unknown:
        raise ValueError(f"unknown {name} key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {name} key(s): {', '.join(sorted(missing))}")


def load_config(path: Path, *, registry: dict[str, Any] | None = None) -> AutopilotConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _mapping(raw, {"schema_version", "selected_models", "paths", "budgets", "permissions"}, "config")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    _mapping(raw["paths"], PATH_KEYS, "paths")
    paths: dict[str, Path] = {}
    for name, value in raw["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} must be a non-empty path")
        item = Path(value).expanduser()
        item = (path.parent / item if not item.is_absolute() else item).resolve()
        if item.exists() and (item.is_file() != (name == "registry")):
            raise ValueError(f"paths.{name} must name a {'file' if name == 'registry' else 'directory'}")
        if name == "registry" and not item.is_file():
            raise ValueError(f"paths.registry does not exist: {item}")
        if name == "dataset_root" and not item.is_dir():
            raise ValueError(f"paths.dataset_root must name an existing directory: {item}")
        parent = item if item.exists() and item.is_dir() else item.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError(f"paths.{name} has no existing directory parent: {item}")
        paths[name] = item
    budgets = raw["budgets"]
    _mapping(budgets, BUDGET_KEYS, "budgets")
    for name, value in budgets.items():
        if type(value) is not int or value < 0 or (name != "max_download_bytes" and value < 1):
            raise ValueError(f"budgets.{name} must be a {'non-negative' if name == 'max_download_bytes' else 'positive'} integer")
    permissions = raw["permissions"]
    _mapping(permissions, PERMISSIONS, "permissions")
    if any(type(value) is not bool for value in permissions.values()):
        raise ValueError("every permissions value must be boolean")
    registry = registry if registry is not None else load_registry(paths["registry"])
    models = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(models, list):
        raise ValueError("registry must contain a models list")
    ids = [model.get("model_id") for model in models if isinstance(model, dict)]
    if len(ids) != len(models) or any(not isinstance(mid, str) or not mid for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("registry model IDs must be unique, non-empty strings")
    selected_models = raw["selected_models"]
    if selected_models == "all_registered":
        selected = tuple(ids)
    elif isinstance(selected_models, list) and selected_models and all(isinstance(mid, str) and mid for mid in selected_models):
        if len(selected_models) != len(set(selected_models)):
            raise ValueError("selected_models must not contain duplicates")
        unknown = sorted(set(selected_models) - set(ids))
        if unknown:
            raise ValueError(f"unknown model ID(s): {', '.join(unknown)}")
        selected = tuple(selected_models)
    else:
        raise ValueError("selected_models must be 'all_registered' or a non-empty list of model IDs")
    if len(selected) > budgets["max_models"]:
        raise ValueError(f"selected model count {len(selected)} exceeds budgets.max_models={budgets['max_models']}")
    return AutopilotConfig(path, selected, paths, dict(budgets), dict(permissions), registry)


def _configuration_digest(config: AutopilotConfig) -> str:
    value = {"selected_models": config.selected_model_ids, "paths": {key: str(path) for key, path in config.paths.items()},
             "budgets": config.budgets, "permissions": config.permissions}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _environment_plan(models: list[dict[str, Any]], *, dry_run: bool,
                      artifacts_root: Path | None = None) -> dict[str, Any]:
    if not dry_run:
        try:
            from workbench.backend.environment_planner import plan_environments
        except ImportError:
            pass
        else:
            result = plan_environments(models, artifacts_root=artifacts_root)
            # Read the statuses the planner actually writes: VERIFIED and CACHED_VERIFIED come from
            # a real solve, NOT_RUN_REUSED means a previously verified interpreter was reused, and
            # NOT_RUN / UNAVAILABLE / PARTIAL mean no successful solve backs the plan. `COMPLETED`
            # requires every group to carry real evidence -- a solve, or a runtime-verified
            # interpreter -- and `solver_verified` names only the solved case.
            status = result.get("solver_verification_status") or (
                "UNAVAILABLE" if result.get("solver") in (None, "unavailable") else "NOT_RUN")
            groups = result.get("groups", [])
            solved = [group for group in groups
                      if group.get("solver_verification_status") in ("VERIFIED", "CACHED_VERIFIED")]
            reused_ready = [group for group in groups
                            if group.get("solver_verification_status") == "NOT_RUN_REUSED"
                            and group.get("runtime_verification_status") == "READY"]
            covered = len(solved) + len(reused_ready)
            verified = bool(groups) and covered == len(groups)
            source = ("environment_planner" if solved
                      else "runtime_verified_reuse" if reused_ready
                      else "solver_unavailable")
            return {"solver_verified": verified, "solver_solved": bool(solved),
                    "source": source, "solver_verification_status": status, **result}
    return {"solver_verified": False, "source": "registry_only_fallback", "blocked": [],
            "groups": [{"environment_id": model["model_id"], "members": [model["model_id"]],
                        "python": (model.get("environment") or {}).get("python"), "packages": {},
                        "cuda_runtime": (model.get("environment") or {}).get("cuda"),
                        "conda_env_path": None, "interpreter": None,
                        "lock_digest": "registry-only-unverified", "evidence": [], "split_reasons": [],
                        "installation_status": "NOT_VERIFIED", "runtime_verification_status": "NOT_VERIFIED",
                        "unresolved": []} for model in models]}


def _artifact_paths(config: AutopilotConfig) -> dict[str, Path]:
    artifacts, reports = config.paths["artifacts"], config.paths["artifacts"] / "reports"
    return {"plan": artifacts / "autopilot_execution_plan.json", "pipeline_state": artifacts / "autopilot_pipeline_state.json",
            "summary_json": reports / "autopilot_summary.json", "summary_md": reports / "autopilot_summary.md",
            "csv": reports / "autopilot_results.csv", "blockers": reports / "blockers.json",
            "manifest": reports / "run_manifest.json"}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _workbench_config(config: AutopilotConfig) -> WorkbenchConfig:
    return replace(resolve_config(), CIR_REPO_ROOT=REPOSITORY, # the code checkout: scripts and pinned sources live here
                   FASHIONIQ_ROOT=config.paths["dataset_root"],
                   WORKBENCH_ARTIFACTS_ROOT=config.paths["artifacts"],
                   WORKBENCH_CHECKPOINT_ROOT=config.paths["checkpoints"], WORKBENCH_RESULTS_ROOT=config.paths["results"],
                   WORKBENCH_THIRD_PARTY_ROOT=config.paths["third_party"])


def _enforce_download_budget(config: AutopilotConfig, rows: list[dict[str, Any]]) -> None:
    from workbench.backend.registry import checkpoint_path

    models = {model["model_id"]: model for model in config.registry["models"]}
    remaining = config.budgets["max_download_bytes"]
    for row in rows:
        if row["status"] != "NOT_RUN":
            continue
        required, unknown = 0, False
        for checkpoint in models[row["model_id"]].get("checkpoint_variants", []):
            if not checkpoint.get("download_url"):
                continue
            destination = checkpoint_path(row["model_id"], checkpoint, config.paths["checkpoints"])
            if destination.is_file():
                continue
            size = checkpoint.get("expected_size_bytes")
            if type(size) is not int or size < 0:
                unknown = True
                break
            required += size
        if unknown:
            row.update(status="BLOCKED_CHECKPOINT", blockers=["download budget cannot be verified: checkpoint size unknown"])
        elif required > remaining:
            row.update(status="BLOCKED_CHECKPOINT", blockers=[f"download budget exceeded: requires {required} bytes, {remaining} remain"])
        else:
            remaining -= required



def resolved_plan(config: AutopilotConfig, *, dry_run: bool = False) -> dict[str, Any]:
    by_id = {model["model_id"]: model for model in config.registry["models"]}
    models = [by_id[mid] for mid in config.selected_model_ids]
    environment_plan = _environment_plan(models, dry_run=dry_run,
                                          artifacts_root=config.paths["artifacts"])
    blocked = {item["model_id"]: item["reason"] for item in environment_plan.get("blocked", [])}
    plan = {"schema_version": 1, "generated_at": datetime.now(UTC).isoformat(),
            "configuration_digest": _configuration_digest(config), "selected_models": list(config.selected_model_ids),
            "execution_started": False, "solver_verified": environment_plan["solver_verified"],
            "environment_plan": environment_plan,
            "models": [{"model_id": model["model_id"],
                        "status": "BLOCKED_ENVIRONMENT" if model["model_id"] in blocked else "NOT_RUN",
                        "blockers": [blocked[model["model_id"]]] if model["model_id"] in blocked else [], "metrics": None}
                       for model in models]}
    if not dry_run and config.permissions["allow_network"] and config.permissions["allow_large_downloads"]:
        _enforce_download_budget(config, plan["models"])
    return plan


def _execute(config: AutopilotConfig, plan: dict[str, Any], *, resume: bool, force_models: list[str]) -> dict[str, Any]:
    from argparse import Namespace
    from unittest.mock import patch
    from workbench.scripts import pipeline

    by_id = {model["model_id"]: model for model in config.registry["models"]}
    env_blocked = {item["model_id"] for item in plan["environment_plan"].get("blocked", [])}
    plan_blocked = {row["model_id"] for row in plan["models"] if row["status"] != "NOT_RUN"}
    models = [by_id[mid] for mid in config.selected_model_ids if mid not in env_blocked | plan_blocked]
    if not models:
        return {"exit_code": 0, "stages": {}, "missing_permissions": {}}
    try:
        from workbench.backend.environment_planner import apply_interpreter_mapping
        mapping = apply_interpreter_mapping(models, plan["environment_plan"], environ=os.environ)
    except ImportError:
        mapping = {}
    child_environment = {
        "WORKBENCH_REGISTRY_PATH": str(config.paths["registry"]),
        "FASHIONIQ_ROOT": str(config.paths["dataset_root"]),
        "WORKBENCH_ARTIFACTS_ROOT": str(config.paths["artifacts"]),
        "WORKBENCH_CHECKPOINT_ROOT": str(config.paths["checkpoints"]),
        "WORKBENCH_RESULTS_ROOT": str(config.paths["results"]),
        "WORKBENCH_THIRD_PARTY_ROOT": str(config.paths["third_party"]),
    }
    child_environment.update({
        (by_id[mid].get("environment") or {}).get("python_env_var"): str(interpreter)
        for mid, interpreter in mapping.items()
        if interpreter and (by_id[mid].get("environment") or {}).get("python_env_var")
    })
    args = Namespace(
        mode="all", model=None, selected_model_ids=list(config.selected_model_ids),
        dataset_root=config.paths["dataset_root"], checkpoint=None,
        protocol=None, top_k=config.budgets["top_k"], continue_on_error=True,
        allow_env_install=config.permissions["allow_env_install"],
        allow_preparation=config.permissions["allow_preparation"], apply=True,
        verify_gpu=False, sync_sources=True, download_checkpoints=True,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=True, serve=False,
    )
    with patch.dict(os.environ, child_environment):
        stages = pipeline.all_stages(args, _workbench_config(config))
        environment_plan_path = config.paths["artifacts"] / "environment_plan.json"
        stages.insert(0, pipeline.run_stage(
            "environment-plan", pipeline.command("plan_environments.py", "--output", environment_plan_path),
            output_paths=(environment_plan_path,),
        ))
        allowed = {name for name, value in config.permissions.items() if value}
        force = {pipeline.stage_key(stage) for stage in stages if stage.model_id in set(force_models)}
        state_path = _artifact_paths(config)["pipeline_state"]
        exit_code = pipeline.run_stages(
            stages, continue_on_error=True, resume=resume, force_stages=force,
            state_path=state_path, apply=True, authorized_capabilities=allowed,
            config=_workbench_config(config), mode="all",
        )
        records = pipeline.load_pipeline_state(state_path).get("stages", {})
    missing: dict[str, list[str]] = {}
    for stage in stages:
        record = records.get(pipeline.stage_key(stage), {})
        missing_caps = sorted(stage.required_capabilities - allowed) if record.get("status") == "BLOCKED_AUTHORIZATION_REQUIRED" else []
        if missing_caps:
            recipients = [stage.model_id] if stage.model_id else config.selected_model_ids
            for model_id in recipients:
                missing.setdefault(model_id, []).extend(missing_caps)
    return {"exit_code": exit_code, "stages": records,
            "missing_permissions": {mid: sorted(set(items)) for mid, items in missing.items()}}


def _preflight(model: dict[str, Any], config: AutopilotConfig) -> list[str]:
    try:
        from workbench.backend.preflight import preflight_blockers
    except ImportError:
        return []
    try:
        return list(preflight_blockers(model, _workbench_config(config)))
    except Exception as error:
        return [f"preflight failed: {error}"]


def _reason_status(reason: str) -> str:
    text = reason.lower()
    for terms, status in ((("gpu", "cuda"), "BLOCKED_GPU"), (("dataset", "fashioniq", "path missing"), "BLOCKED_DATASET"),
                          (("preparation", "prepare"), "BLOCKED_PREPARATION"), (("command", "evaluator", "adapter"), "BLOCKED_EVALUATOR"),
                          (("source", "upstream"), "BLOCKED_SOURCE"), (("checkpoint",), "BLOCKED_CHECKPOINT"),
                          (("environment", "interpreter", "dependency"), "BLOCKED_ENVIRONMENT")):
        if any(term in text for term in terms):
            return status
    return "BLOCKED_EVALUATOR"


def _classify(plan: dict[str, Any], execution: dict[str, Any], config: AutopilotConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = {model["model_id"]: model for model in config.registry["models"]}
    plan_rows = {row["model_id"]: row for row in plan["models"]}
    records = execution.get("stages", {})
    global_failures = [item for item in records.values() if isinstance(item, dict) and item.get("model_id") is None
                       and item.get("status") not in {"COMPLETE", "RUNNING"}]
    stage_status = {"doctor": "BLOCKED_EVALUATOR", "environment-plan": "BLOCKED_ENVIRONMENT",
                    "sync": "BLOCKED_SOURCE", "dataset": "BLOCKED_DATASET", "preparation": "BLOCKED_PREPARATION",
                    "checkpoint": "BLOCKED_CHECKPOINT", "auxiliary-assets": "BLOCKED_CHECKPOINT",
                    "environment": "BLOCKED_ENVIRONMENT", "evaluation": "FAILED_EVALUATION",
                    "validation-index": "FAILED_OUTPUT_VALIDATION"}
    rows, blockers = [], []
    for model_id in config.selected_model_ids:
        prior = plan_rows[model_id]
        status, reasons = prior["status"], list(prior.get("blockers", []))
        related = [item for item in records.values() if isinstance(item, dict) and item.get("model_id") == model_id]
        if status == "NOT_RUN" and execution.get("exit_code") is not None:
            failed = [item for item in related if item.get("status") not in {"COMPLETE", "RUNNING"}]
            successful_eval = any(item.get("name") == "evaluation" and item.get("status") == "COMPLETE" for item in related)
            if not failed and successful_eval and not global_failures:
                status = "COMPLETED_WITH_LIMITATIONS" if not plan["solver_verified"] else "COMPLETED"
            else:
                failed = failed or global_failures
            for failure in failed:
                name, state = failure.get("name"), failure.get("status")
                if state == "BLOCKED_AUTHORIZATION_REQUIRED":
                    status = "BLOCKED_AUTHORIZATION"
                elif state == "SKIPPED_DEPENDENCY":
                    continue
                elif failure.get("failure_reason") == "output_contract_not_satisfied":
                    status = "FAILED_OUTPUT_VALIDATION"
                elif name == "runtime-preflight":
                    reasons.extend(_preflight(models[model_id], config) or ["runtime preflight failed"])
                    status = _reason_status(reasons[-1])
                else:
                    status = stage_status.get(name, "FAILED_EVALUATION")
                missing = execution.get("missing_permissions", {}).get(model_id, [])
                reasons.append("missing permission: " + ", ".join(missing) if state == "BLOCKED_AUTHORIZATION_REQUIRED" and missing
                               else (failure.get("failure_reason") or f"{name} stage {str(state).lower()}"))
                break
            if status == "NOT_RUN":
                evals = [item for item in related if item.get("name") == "evaluation"]
                if execution.get("exit_code") == 0 or evals and all(item.get("status") == "COMPLETE" for item in evals):
                    status = "COMPLETED_WITH_LIMITATIONS" if not plan["solver_verified"] else "COMPLETED"
                else:
                    reasons.extend(_preflight(models[model_id], config))
                    status = _reason_status(reasons[0]) if reasons else "FAILED_EVALUATION"
                    if not reasons:
                        reasons = ["pipeline failed without a model-specific stage record"]
        missing = execution.get("missing_permissions", {}).get(model_id, [])
        if status not in STATUSES:
            status = "FAILED_EVALUATION"
        row = {"model_id": model_id, "status": status, "blockers": reasons,
               "missing_permissions": missing, "run_id": None, "metrics": None}
        rows.append(row)
        if reasons:
            blockers.append({"model_id": model_id, "status": status, "reasons": reasons, "missing_permissions": missing})
    return rows, blockers


def _write_reports(config: AutopilotConfig, plan: dict[str, Any], rows: list[dict[str, Any]], blockers: list[dict[str, Any]],
                   execution: dict[str, Any], *, planned: bool) -> dict[str, Any]:
    paths = _artifact_paths(config)
    completed = sum(row["status"] == "COMPLETED" for row in rows)
    limited = sum(row["status"] == "COMPLETED_WITH_LIMITATIONS" for row in rows)
    failed = len(rows) - completed - limited
    if planned:
        outcome, code = "PLANNED", 0
    elif failed == 0 and limited == 0:
        outcome, code = "FULL_SUCCESS", 0
    elif completed + limited:
        outcome, code = "PARTIAL", 2
    else:
        outcome, code = "WORKFLOW_FAILURE", 1
    generated = datetime.now(UTC).isoformat()
    summary = {"schema_version": 1, "status": outcome, "exit_code": code, "generated_at": generated,
               "selected_models": list(config.selected_model_ids), "solver_verified": plan["solver_verified"],
               "counts": {"selected": len(rows), "completed": completed, "completed_with_limitations": limited,
                          "blocked_or_failed": failed}, "models": rows}
    manifest = {"schema_version": 1, "configuration_digest": plan["configuration_digest"],
                "selected_models": list(config.selected_model_ids), "execution_started": not planned,
                "finished_at": generated, "pipeline_exit_code": execution.get("exit_code"), "outcome": outcome}
    buffer = io.StringIO(newline="")
    fields = ("model_id", "status", "blockers", "missing_permissions", "run_id", "metrics")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "blockers": json.dumps(row["blockers"]),
                         "missing_permissions": json.dumps(row["missing_permissions"]), "metrics": json.dumps(row["metrics"])})
    md = ["# Autopilot summary", "", f"Status: **{outcome}**", "", "| Model | Status | Blockers |", "|---|---|---|"]
    md.extend(f"| {row['model_id']} | {row['status']} | {'; '.join(row['blockers'])} |" for row in rows)
    _atomic_write(paths["summary_json"], json.dumps(summary, indent=2) + "\n")
    _atomic_write(paths["summary_md"], "\n".join(md) + "\n")
    _atomic_write(paths["csv"], buffer.getvalue())
    _atomic_write(paths["blockers"], json.dumps(blockers, indent=2) + "\n")
    _atomic_write(paths["manifest"], json.dumps(manifest, indent=2) + "\n")
    return {"exit_code": code, "summary": summary}


def _read_state(config: AutopilotConfig) -> dict[str, Any]:
    paths = _artifact_paths(config)
    if paths["summary_json"].is_file():
        summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
        return {"exit_code": int(summary.get("exit_code", 0)), "summary": summary}
    if paths["plan"].is_file():
        return {"exit_code": 0, "summary": json.loads(paths["plan"].read_text(encoding="utf-8"))}
    return {"exit_code": 1, "summary": {"status": "NOT_RUN", "models": [], "reason": "no autopilot state exists"}}


def run(config_path: Path, *, apply: bool, resume: bool, force_models: list[str], mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    if mode in {"status", "report"}:
        return _read_state(config)
    unknown = sorted(set(force_models) - set(config.selected_model_ids))
    if unknown:
        raise ValueError(f"--force-model must be selected in config: {', '.join(unknown)}")
    paths = _artifact_paths(config)
    previous = None
    if paths["plan"].exists():
        previous = json.loads(paths["plan"].read_text(encoding="utf-8"))
        if previous.get("execution_started") and previous.get("selected_models") != list(config.selected_model_ids):
            raise ValueError("refusing to change selected_models after autopilot execution started")
    plan = resolved_plan(config, dry_run=mode == "dry-run")
    if mode == "dry-run":
        return {"exit_code": 0, "summary": {"status": "NOT_RUN", "selected_models": list(config.selected_model_ids),
                                                "solver_verified": plan["solver_verified"], "plan": plan}}
    if previous and previous.get("execution_started"):
        plan["execution_started"] = True
    _atomic_write(paths["plan"], json.dumps(plan, indent=2) + "\n")
    if apply:
        plan["execution_started"] = True
        _atomic_write(paths["plan"], json.dumps(plan, indent=2) + "\n")
        execution = _execute(config, plan, resume=resume, force_models=force_models)
    else:
        execution = {"exit_code": None, "stages": {}, "missing_permissions": {}}
    return _write_reports(config, plan, *_classify(plan, execution, config), execution, planned=not apply)


def parser_for():
    import argparse
    parser = argparse.ArgumentParser(description="Run guarded workbench reproduction.")
    parser.add_argument("--config", type=Path, default=REPOSITORY / "workbench" / "configs" / "pc_autopilot.example.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true", help="read autopilot state")
    modes.add_argument("--report", action="store_true", help="read latest report")
    parser.add_argument("--apply", action="store_true", help="execute authorized stages")
    parser.add_argument("--resume", action="store_true", help="resume verified stages")
    parser.add_argument("--dry-run", action="store_true", help="resolve plan without artifacts or subprocesses")
    parser.add_argument("--force-model", action="append", default=[], help="force selected model stages on resume")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parser_for()
    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--dry-run cannot be combined with --apply")
    mode = "status" if args.status else "report" if args.report else "dry-run" if args.dry_run else "run"
    try:
        result = run(args.config, apply=args.apply, resume=args.resume, force_models=args.force_model, mode=mode)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"autopilot: {error}", file=sys.stderr)
        return 1
    if args.json or mode in {"status", "report"}:
        print(json.dumps(result["summary"], indent=2))
    else:
        print(f"autopilot {result['summary'].get('status', 'NOT_RUN')}: {len(result['summary'].get('models', []))} model(s)")
    return result["exit_code"]
