from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass, field, replace
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
CHECKPOINT_POLICIES = ("strict_clean", "include_flagged_conditions")
STATUSES = {"BLOCKED_SOURCE", "BLOCKED_DATASET", "BLOCKED_PREPARATION", "BLOCKED_CHECKPOINT", "BLOCKED_ENVIRONMENT", "BLOCKED_GPU", "BLOCKED_EVALUATOR", "BLOCKED_AUTHORIZATION", "FAILED_EVALUATION", "FAILED_OUTPUT_VALIDATION", "COMPLETED", "COMPLETED_WITH_LIMITATIONS", "NOT_RUN"}


@dataclass(frozen=True)
class AutopilotConfig:
    path: Path
    selected_model_ids: tuple[str, ...]
    paths: dict[str, Path]
    budgets: dict[str, int]
    permissions: dict[str, bool]
    registry: dict[str, Any]
    #: Which checkpoint variants this experiment may evaluate. `strict_clean` refuses anything
    #: whose recorded training condition is not exactly 0; `include_flagged_conditions` is the
    #: explicit opt-in for the exploratory secondary cohort of noise-trained weights. A per-model
    #: entry in `checkpoint_overrides` is an operator-resolved mapping and wins over the policy,
    #: but the affected model is badged in the report rather than presented as clean.
    checkpoint_policy: str = "strict_clean"
    checkpoint_overrides: dict[str, list[str]] = field(default_factory=dict)


def _mapping(value: Any, keys: set[str], name: str, *, optional: set[str] = frozenset()) -> None:
    """Reject unknown keys, and require every key that is not declared optional."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - keys
    missing = set(keys) - set(value) - set(optional)
    if unknown:
        raise ValueError(f"unknown {name} key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing {name} key(s): {', '.join(sorted(missing))}")


def load_config(path: Path, *, registry: dict[str, Any] | None = None) -> AutopilotConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # `checkpoint_policy` and `checkpoint_overrides` are optional-by-omission: an existing config
    # stays valid, and a config that names them is validated below.
    _mapping(raw, {"schema_version", "selected_models", "paths", "budgets", "permissions",
                   "checkpoint_policy", "checkpoint_overrides"}, "config",
             optional={"checkpoint_policy", "checkpoint_overrides"})
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
    # Checkpoint policy: validated here so a typo cannot silently change which weights are
    # evaluated. An override names checkpoint IDs; whether they are registered is checked against
    # the registry in the same place the selection is resolved.
    policy = raw.get("checkpoint_policy", "strict_clean")
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"checkpoint_policy must be one of {', '.join(CHECKPOINT_POLICIES)}")
    overrides = raw.get("checkpoint_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("checkpoint_overrides must be a mapping of model ID to checkpoint IDs")
    unknown_models = sorted(set(overrides) - set(ids))
    if unknown_models:
        raise ValueError(f"checkpoint_overrides names unknown model ID(s): {', '.join(unknown_models)}")
    for mid, value in overrides.items():
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"checkpoint_overrides.{mid} must be a non-empty list of checkpoint IDs")
    unselected = sorted(set(overrides) - set(selected))
    if unselected:
        raise ValueError(f"checkpoint_overrides names model ID(s) that are not selected: {', '.join(unselected)}")
    return AutopilotConfig(path, selected, paths, dict(budgets), dict(permissions), registry,
                           policy, {mid: list(value) for mid, value in overrides.items()})


def _configuration_digest(config: AutopilotConfig) -> str:
    value = {"selected_models": config.selected_model_ids, "paths": {key: str(path) for key, path in config.paths.items()},
             "budgets": config.budgets, "permissions": config.permissions}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _environment_plan(models: list[dict[str, Any]], *, dry_run: bool,
                      artifacts_root: Path | None = None,
                      third_party_root: Path | None = None) -> dict[str, Any]:
    if not dry_run:
        try:
            from workbench.backend.environment_planner import plan_environments
        except ImportError:
            pass
        else:
            kwargs = {"artifacts_root": artifacts_root}
            if third_party_root is not None:
                kwargs["third_party_root"] = third_party_root
                os.environ["WORKBENCH_THIRD_PARTY_ROOT"] = str(third_party_root)
            try:
                result = plan_environments(models, **kwargs)
            except TypeError:
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
    from workbench.backend.registry import (
        auxiliary_assets_for_model, auxiliary_destination, checkpoint_is_present,
        checkpoint_path, sha256_file,
    )

    models = {model["model_id"]: model for model in config.registry["models"]}
    remaining = config.budgets["max_download_bytes"]
    for row in rows:
        if row["status"] != "NOT_RUN":
            continue
        model = models[row["model_id"]]
        bundles = model.get("checkpoint_bundles") or []
        if bundles:
            required_ids = set()
            for b in bundles:
                required_ids.update(b.get("required_checkpoint_ids", []))
            selected_variants = [v for v in model.get("checkpoint_variants", []) if v.get("checkpoint_id") in required_ids]
        else:
            variants = model.get("checkpoint_variants", [])
            selected_variants = variants[:1] if variants else []

        required, unknown = 0, False
        unknown_reason = None
        for checkpoint in selected_variants:
            if not checkpoint.get("download_url"):
                continue
            destination = checkpoint_path(row["model_id"], checkpoint, config.paths["checkpoints"])
            try:
                if checkpoint_is_present(destination, checkpoint):
                    continue
            except Exception:
                pass
            size = checkpoint.get("expected_size_bytes")
            if type(size) is not int or size < 0:
                unknown = True
                unknown_reason = "download budget cannot be verified: checkpoint size unknown"
                break
            required += size

        if not unknown:
            try:
                for asset in auxiliary_assets_for_model(row["model_id"]):
                    if not asset.get("automatic_download_available"):
                        continue
                    dest = auxiliary_destination(asset, config.paths["third_party"], config.paths["dataset_root"])
                    if dest.is_file() and not dest.is_symlink() and dest.stat().st_size > 0:
                        exp_sha = asset.get("expected_sha256")
                        if not exp_sha or sha256_file(dest) == exp_sha:
                            continue
                    size = asset.get("expected_size_bytes")
                    if type(size) is not int or size < 0:
                        unknown = True
                        unknown_reason = "download budget cannot be verified: auxiliary asset size unknown"
                        break
                    required += size
            except Exception:
                pass

        if unknown:
            row.update(status="BLOCKED_CHECKPOINT", blockers=[unknown_reason or "download budget cannot be verified: checkpoint size unknown"])
        elif required > remaining:
            row.update(status="BLOCKED_CHECKPOINT", blockers=[f"download budget exceeded: requires {required} bytes, {remaining} remain"])
        else:
            remaining -= required


def resolved_plan(config: AutopilotConfig, *, dry_run: bool = False) -> dict[str, Any]:
    from workbench.backend.checkpoint_selection import select_checkpoints

    by_id = {model["model_id"]: model for model in config.registry["models"]}
    models = [by_id[mid] for mid in config.selected_model_ids]
    environment_plan = _environment_plan(models, dry_run=dry_run,
                                          artifacts_root=config.paths["artifacts"],
                                          third_party_root=config.paths["third_party"])
    # The checkpoint policy is resolved before any work starts, so a model whose weights the policy
    # refuses is reported as blocked rather than being evaluated with a variant nobody chose. This is
    # also what stops a model with two registered variants from having both of them evaluated.
    selections = {mid: select_checkpoints(by_id[mid], policy=config.checkpoint_policy,
                                          requested=config.checkpoint_overrides.get(mid),
                                          checkpoint_root=config.paths["checkpoints"])
                  for mid in config.selected_model_ids}
    blocked = {item["model_id"]: item["reason"] for item in environment_plan.get("blocked", [])}
    rows = []
    for model in models:
        mid = model["model_id"]
        selection = selections[mid]
        if mid in blocked:
            rows.append({"model_id": mid, "status": "BLOCKED_ENVIRONMENT",
                         "blockers": [blocked[mid]], "metrics": None,
                         "checkpoint_ids": list(selection.checkpoint_ids),
                         "checkpoint_selection": selection.as_dict()})
        elif selection.blocked:
            rows.append({"model_id": mid, "status": "BLOCKED_CHECKPOINT",
                         "blockers": [selection.reason], "metrics": None,
                         "checkpoint_ids": [], "checkpoint_selection": selection.as_dict()})
        else:
            rows.append({"model_id": mid, "status": "NOT_RUN", "blockers": [], "metrics": None,
                         "checkpoint_ids": list(selection.checkpoint_ids),
                         "checkpoint_selection": selection.as_dict()})
    plan = {"schema_version": 1, "generated_at": datetime.now(UTC).isoformat(),
            "configuration_digest": _configuration_digest(config), "selected_models": list(config.selected_model_ids),
            "execution_started": False, "solver_verified": environment_plan["solver_verified"],
            "checkpoint_policy": config.checkpoint_policy,
            "environment_plan": environment_plan, "models": rows}
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
    # Only the selected models do any work, and each one works on exactly the checkpoints the
    # policy selected. Without this the evaluator's own `candidates()` would return every
    # registered variant for a model with several, silently evaluating weights nobody chose.
    selected_ids = [mid for mid in config.selected_model_ids if mid not in env_blocked | plan_blocked]
    checkpoints = {mid: list(select["checkpoint_ids"]) for mid, select in
                   ((row["model_id"], row.get("checkpoint_selection") or {}) for row in plan["models"])
                   if select.get("checkpoint_ids")}
    if not models:
        return {"exit_code": 0, "stages": {}, "missing_permissions": {}, "selected_models": []}
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
        mode="all", model=None, selected_model_ids=selected_ids,
        checkpoint_ids=checkpoints,
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
        plan_cmd = ["plan_environments.py", "--output", str(environment_plan_path),
                    "--third-party-root", str(config.paths["third_party"])]
        for mid in selected_ids:
            plan_cmd.extend(["--model", mid])
        stages.insert(0, pipeline.run_stage(
            "environment-plan", pipeline.command(*plan_cmd),
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
    return {"exit_code": exit_code, "stages": records, "selected_models": selected_ids,
            "checkpoint_ids": checkpoints,
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

def _resolve_evaluation_metrics(
    model_id: str,
    related: list[dict[str, Any]],
    config: AutopilotConfig,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    from workbench.backend.metrics_extraction import extract_aggregate_metrics
    from workbench.backend.registry import sha256_file
    from workbench.scripts.evaluate_models import read_evidence_ledger

    models = {m["model_id"]: m for m in config.registry["models"]}
    model = models.get(model_id, {})
    method_name = model.get("method_name") or model_id
    native_protocol = model.get("native_protocol") or "unknown"
    env_groups = plan.get("environment_plan", {}).get("groups", [])
    env_group = next((g.get("environment_id") for g in env_groups if model_id in g.get("members", [])), None)

    eval_stage = next((item for item in related if isinstance(item, dict) and item.get("name") == "evaluation"), None)
    run_id = None
    if eval_stage:
        run_id = eval_stage.get("run_id")
        if not run_id and isinstance(eval_stage.get("evidence"), dict):
            run_id = eval_stage["evidence"].get("run_id")

    base_fields = {
        "model_id": model_id,
        "method_name": method_name,
        "checkpoint_id": None,
        "checkpoint_provenance": None,
        "environment_group": env_group,
        "native_protocol": native_protocol,
        "run_id": run_id,
        "dress_r10": None,
        "dress_r50": None,
        "shirt_r10": None,
        "shirt_r50": None,
        "toptee_r10": None,
        "toptee_r50": None,
        "macro_r10": None,
        "macro_r50": None,
        "paper_r10": model.get("reported_r10"),
        "paper_r50": model.get("reported_r50"),
        "paper_mean": model.get("reported_mean"),
        "measured_difference": None,
        "evaluation_duration_seconds": None,
        "completion_proof": None,
        "aggregate_report": None,
        "aggregate_only": True,
        "per_query_export_available": False,
        "evaluation_kind": "NOT_RUN",
        "missing_metrics_reason": "no evaluation run id recorded" if not run_id else None,
        "metrics": None,
    }

    if not eval_stage or not run_id:
        return base_fields, None

    if eval_stage.get("started_at") and eval_stage.get("finished_at"):
        try:
            t0 = datetime.fromisoformat(eval_stage["started_at"])
            t1 = datetime.fromisoformat(eval_stage["finished_at"])
            base_fields["evaluation_duration_seconds"] = round((t1 - t0).total_seconds(), 2)
        except Exception:
            pass

    evidence = eval_stage.get("evidence") if isinstance(eval_stage.get("evidence"), dict) else {}
    proof_path = None
    if evidence.get("completion_proof"):
        cand = Path(evidence["completion_proof"])
        if cand.is_file():
            proof_path = cand
    if proof_path is None:
        reports_dir = config.paths["artifacts"] / "reports"
        matches = list(reports_dir.glob(f"{model_id}_*_{run_id}_completion.json"))
        if not matches:
            matches = list(reports_dir.glob(f"*{run_id}*_completion.json"))
        if not matches and "results" in config.paths:
            matches = list(config.paths["results"].glob(f"{model_id}_*_{run_id}_completion.json"))
        if matches:
            proof_path = matches[0]

    if proof_path is None or not proof_path.is_file():
        base_fields["missing_metrics_reason"] = "completion proof missing"
        return base_fields, None

    base_fields["completion_proof"] = str(proof_path)

    try:
        proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    except Exception as err:
        return base_fields, f"corrupted completion proof: {err}"

    if not isinstance(proof_payload, dict):
        return base_fields, "corrupted completion proof: not a JSON object"

    if proof_payload.get("completion_status") != "COMPLETED" or proof_payload.get("return_code") != 0:
        return base_fields, f"completion proof records failure (status={proof_payload.get('completion_status')}, code={proof_payload.get('return_code')})"

    if str(proof_payload.get("run_id")) != str(run_id) or str(proof_payload.get("model_id")) != str(model_id):
        return base_fields, f"completion proof metadata mismatch: run={proof_payload.get('run_id')}, model={proof_payload.get('model_id')}"

    base_fields["checkpoint_id"] = proof_payload.get("checkpoint_id")
    base_fields["checkpoint_provenance"] = proof_payload.get("checkpoint_artifact_type") or proof_payload.get("checkpoint_mapping_status")

    report_path = None
    if proof_payload.get("report"):
        cand = Path(proof_payload["report"])
        if not cand.is_absolute():
            cand = proof_path.parent / cand
        if cand.is_file():
            report_path = cand
    if report_path is None and evidence.get("run_report"):
        cand = Path(evidence["run_report"])
        if cand.is_file():
            report_path = cand
    if report_path is None:
        reports_dir = config.paths["artifacts"] / "reports"
        matches = list(reports_dir.glob(f"{model_id}_*_{run_id}_aggregate.json"))
        if not matches and "results" in config.paths:
            matches = list(config.paths["results"].glob(f"{model_id}_*_{run_id}_aggregate.json"))
        if not matches and proof_payload.get("run_directory"):
            cand = Path(proof_payload["run_directory"]) / "aggregate_report.json"
            if cand.is_file():
                matches = [cand]
        if matches:
            report_path = matches[0]

    if report_path is None or not report_path.is_file():
        return base_fields, f"aggregate report missing for run {run_id}"

    base_fields["aggregate_report"] = str(report_path)

    declared_report_digest = proof_payload.get("report_digest")
    actual_report_digest = sha256_file(report_path)
    if declared_report_digest and actual_report_digest != declared_report_digest:
        return base_fields, f"corrupted evidence rejected: aggregate report digest mismatch (expected {declared_report_digest[:12]}, actual {actual_report_digest[:12]})"

    wb_cfg = _workbench_config(config)
    ledger = read_evidence_ledger(wb_cfg)
    if ledger is not None:
        matching = [item for item in ledger if item.get("run_id") == run_id]
        if not matching:
            return base_fields, f"corrupted evidence rejected: run {run_id} missing from evidence ledger"
        witness = matching[-1]
        claimed_entry = proof_payload.get("ledger_entry_digest")
        if claimed_entry and witness.get("entry_digest") != claimed_entry:
            return base_fields, "corrupted evidence rejected: proof ledger entry digest disagrees with ledger"
        if witness.get("return_code") not in (0, None):
            return base_fields, f"corrupted evidence rejected: evidence ledger records failed return code {witness.get('return_code')}"
        if witness.get("report_digest") and witness["report_digest"] != actual_report_digest:
            return base_fields, "corrupted evidence rejected: aggregate report digest disagrees with evidence ledger"

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as err:
        return base_fields, f"corrupted aggregate report: {err}"

    if not isinstance(report_payload, dict):
        return base_fields, "corrupted aggregate report: not a JSON object"

    if report_payload.get("execution_status") not in ("EXECUTION_COMPLETED", "COMPLETED", None):
        return base_fields, f"aggregate report records execution status {report_payload.get('execution_status')}"

    try:
        from workbench.scripts.evaluate_models import completion_validator
        ckpt_id = str(proof_payload.get("checkpoint_id") or "default")
        protocol = str(proof_payload.get("protocol_id") or model.get("native_protocol") or "fashioniq_original_split")
        pin = model.get("upstream_commit_sha")
        validator = completion_validator(model_id, ckpt_id, protocol, pin, wb_cfg)
        ok, val_reason, _ = validator({"run_id": run_id})
        if not ok:
            return base_fields, f"corrupted evidence rejected: {val_reason}"
    except Exception:
        pass

    is_synthetic = bool(
        model.get("synthetic_source")
        or model.get("source_type") == "synthetic"
        or model_id.startswith("synthetic")
        or report_payload.get("is_synthetic")
    )
    base_fields["evaluation_kind"] = "SYNTHETIC" if is_synthetic else "OFFICIAL_VERIFIED"

    run_dir = Path(report_payload.get("run_directory") or proof_payload.get("run_directory") or ".")
    stdout_file = run_dir / report_payload.get("stdout_log", "stdout.log")
    stderr_file = run_dir / report_payload.get("stderr_log", "stderr.log")
    stdout_text = stdout_file.read_text(encoding="utf-8") if stdout_file.is_file() else ""
    stderr_text = stderr_file.read_text(encoding="utf-8") if stderr_file.is_file() else ""

    paper_metrics = model.get("paper_metrics") or report_payload.get("paper_sanity_metrics") or {}
    extraction = extract_aggregate_metrics(
        model_id,
        stdout_text,
        stderr_text,
        report_payload.get("return_code", 0),
        paper_metrics=paper_metrics,
    )

    cat_metrics = None
    obs_metrics = None
    parity_status = "PARSER_UNAVAILABLE"
    parser_id = None

    if extraction.extraction_status == "EXTRACTED_VERIFIED":
        cat_metrics = extraction.category_metrics
        obs_metrics = extraction.observed_metrics
        parity_status = extraction.parity_status
        parser_id = extraction.parser_id
    elif is_synthetic and report_payload.get("category_metrics"):
        cat_metrics = report_payload["category_metrics"]
        obs_metrics = report_payload.get("observed_metrics")
        parser_id = report_payload.get("parser_id") or "synthetic_output"
        parity_status = report_payload.get("parity_status") or "PARITY_NOT_EVALUATED"
    elif report_payload.get("metric_extraction_available") and report_payload.get("category_metrics"):
        cat_metrics = report_payload["category_metrics"]
        obs_metrics = report_payload.get("observed_metrics")
        parser_id = report_payload.get("parser_id")
        parity_status = report_payload.get("parity_status") or "PARITY_NOT_EVALUATED"

    if cat_metrics and isinstance(cat_metrics, dict):
        base_fields["dress_r10"] = cat_metrics.get("dress", {}).get("r10")
        base_fields["dress_r50"] = cat_metrics.get("dress", {}).get("r50")
        base_fields["shirt_r10"] = cat_metrics.get("shirt", {}).get("r10")
        base_fields["shirt_r50"] = cat_metrics.get("shirt", {}).get("r50")
        base_fields["toptee_r10"] = cat_metrics.get("toptee", {}).get("r10")
        base_fields["toptee_r50"] = cat_metrics.get("toptee", {}).get("r50")

    if obs_metrics and isinstance(obs_metrics, dict):
        base_fields["macro_r10"] = obs_metrics.get("r10")
        base_fields["macro_r50"] = obs_metrics.get("r50")

    paper_r10 = base_fields["paper_r10"]
    paper_r50 = base_fields["paper_r50"]
    paper_mean = base_fields["paper_mean"]

    measured_diff = None
    if base_fields["macro_r10"] is not None and paper_r10 is not None:
        diff_10 = round(float(base_fields["macro_r10"]) - float(paper_r10), 4)
        diff_50 = round(float(base_fields["macro_r50"]) - float(paper_r50), 4) if base_fields["macro_r50"] is not None and paper_r50 is not None else None
        measured_diff = {
            "r10_diff": diff_10,
            "r50_diff": diff_50,
            "label": f"observed - paper (R@10: {diff_10:+.2f})" + (f", R@50: {diff_50:+.2f}" if diff_50 is not None else ""),
        }
    base_fields["measured_difference"] = measured_diff

    if obs_metrics is not None and cat_metrics is not None:
        base_fields["metrics"] = {
            "aggregate_only": True,
            "per_query_export_available": False,
            "evaluation_kind": base_fields["evaluation_kind"],
            "extraction_status": extraction.extraction_status if extraction.extraction_status == "EXTRACTED_VERIFIED" else "EXTRACTED_SYNTHETIC",
            "parser_id": parser_id,
            "parity_status": parity_status,
            "observed_metrics": obs_metrics,
            "category_metrics": cat_metrics,
            "paper_metrics": {"r10": paper_r10, "r50": paper_r50, "mean": paper_mean},
            "measured_difference": measured_diff,
        }
        base_fields["missing_metrics_reason"] = None
    else:
        base_fields["missing_metrics_reason"] = (
            extraction.extraction_status if extraction.extraction_status != "EXTRACTED_VERIFIED"
            else "no aggregate recall values extracted"
        )
        base_fields["metrics"] = None

    return base_fields, None


def _classify(plan: dict[str, Any], execution: dict[str, Any], config: AutopilotConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = {model["model_id"]: model for model in config.registry["models"]}
    plan_rows = {row["model_id"]: row for row in plan["models"]}
    records = execution.get("stages", {})
    global_failures = [item for item in records.values() if isinstance(item, dict) and item.get("model_id") is None
                       and item.get("status") not in {"COMPLETE", "RUNNING"}]
    stage_status = {"doctor": "BLOCKED_EVALUATOR", "environment-plan": "BLOCKED_ENVIRONMENT",
                    "sync": "BLOCKED_SOURCE", "dataset": "BLOCKED_DATASET", "preparation": "BLOCKED_PREPARATION",
                    "checkpoint": "BLOCKED_CHECKPOINT", "auxiliary-assets": "BLOCKED_CHECKPOINT",
                    "environment": "BLOCKED_ENVIRONMENT", "gpu-verification": "BLOCKED_GPU",
                    "evaluation": "FAILED_EVALUATION", "validation-index": "FAILED_OUTPUT_VALIDATION"}
    rows, blockers = [], []
    for model_id in config.selected_model_ids:
        prior = plan_rows[model_id]
        status, reasons = prior["status"], list(prior.get("blockers", []))
        related = [item for item in records.values() if isinstance(item, dict) and item.get("model_id") == model_id]
        if status == "NOT_RUN" and execution.get("exit_code") is not None:
            failed = [item for item in related if item.get("status") not in {"COMPLETE", "RUNNING"}]
            successful_eval = any(item.get("name") == "evaluation" and item.get("status") == "COMPLETE" for item in related)
            if not failed and successful_eval:
                status = "COMPLETED_WITH_LIMITATIONS" if not plan["solver_verified"] else "COMPLETED"
            else:
                relevant_globals = [item for item in global_failures if item.get("name") != "validation-index"]
                failed = failed or relevant_globals
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

        eval_info, eval_err = _resolve_evaluation_metrics(model_id, related, config, plan)
        if status in ("COMPLETED", "COMPLETED_WITH_LIMITATIONS") and eval_err:
            status = "FAILED_OUTPUT_VALIDATION"
            reasons.append(eval_err)
            eval_info["metrics"] = None
            eval_info["missing_metrics_reason"] = eval_err
        elif status not in ("COMPLETED", "COMPLETED_WITH_LIMITATIONS"):
            eval_info["metrics"] = None
            if not eval_info.get("missing_metrics_reason"):
                eval_info["missing_metrics_reason"] = f"model status is {status}"

        val_idx_complete = records.get("validation-index", {}).get("status") == "COMPLETE"
        index_available = bool(val_idx_complete and eval_info.get("per_query_export_available"))
        row = {
            **eval_info,
            "model_id": model_id,
            "status": status,
            "blockers": reasons,
            "missing_permissions": missing,
            "has_aggregate_results": bool(eval_info.get("macro_r10") is not None or eval_info.get("metrics") is not None),
            "has_per_query_results": bool(eval_info.get("per_query_export_available")),
            "index_available": index_available,
        }
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
    records = execution.get("stages", {})
    val_stage = records.get("validation-index") or next(
        (item for item in records.values() if isinstance(item, dict) and item.get("name") == "validation-index"), None
    )
    any_query_export = any(row.get("per_query_export_available") for row in rows)
    if planned or not any_query_export:
        global_index_status = "NOT_APPLICABLE"
    elif val_stage and val_stage.get("status") == "COMPLETE":
        global_index_status = "COMPLETED"
    elif any(row.get("index_available") for row in rows):
        global_index_status = "PARTIAL"
    else:
        global_index_status = "PARTIAL" if any(row.get("status") in ("COMPLETED", "COMPLETED_WITH_LIMITATIONS") for row in rows) else "NOT_APPLICABLE"

    summary = {"schema_version": 1, "status": outcome, "exit_code": code, "generated_at": generated,
               "selected_models": list(config.selected_model_ids), "solver_verified": plan["solver_verified"],
               "global_index_status": global_index_status,
               "counts": {"selected": len(rows), "completed": completed, "completed_with_limitations": limited,
                          "blocked_or_failed": failed}, "models": rows}
    manifest = {"schema_version": 1, "configuration_digest": plan["configuration_digest"],
                "selected_models": list(config.selected_model_ids), "execution_started": not planned,
                "finished_at": generated, "pipeline_exit_code": execution.get("exit_code"), "outcome": outcome,
                "global_index_status": global_index_status}
    buffer = io.StringIO(newline="")
    fields = (
        "model_id", "method_name", "status", "blockers", "missing_permissions", "run_id",
        "checkpoint_id", "checkpoint_provenance", "environment_group", "native_protocol",
        "evaluation_kind", "dress_r10", "dress_r50", "shirt_r10", "shirt_r50", "toptee_r10", "toptee_r50",
        "macro_r10", "macro_r50", "paper_r10", "paper_r50", "paper_mean", "measured_difference",
        "evaluation_duration_seconds", "completion_proof", "aggregate_report",
        "aggregate_only", "per_query_export_available", "index_available", "missing_metrics_reason", "metrics",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        csv_row = dict(row)
        csv_row["blockers"] = json.dumps(row.get("blockers", []))
        csv_row["missing_permissions"] = json.dumps(row.get("missing_permissions", []))
        csv_row["metrics"] = json.dumps(row.get("metrics"))
        csv_row["measured_difference"] = json.dumps(row.get("measured_difference")) if row.get("measured_difference") is not None else ""
        for k in ("dress_r10", "dress_r50", "shirt_r10", "shirt_r50", "toptee_r10", "toptee_r50", "macro_r10", "macro_r50", "paper_r10", "paper_r50", "paper_mean"):
            csv_row[k] = f"{row[k]:.2f}" if row.get(k) is not None else "null"
        writer.writerow({k: csv_row.get(k, "") for k in fields})

    md = [
        "# Autopilot summary",
        "",
        f"Status: **{outcome}**",
        f"Global index status: **{global_index_status}**",
        "## Model Execution Status",
        "",
        "| Model | Method | Status | Blockers | Run ID |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        blockers_str = "; ".join(row.get("blockers") or []) or "none"
        run_str = row.get("run_id") or "none"
        md.append(f"| {row['model_id']} | {row.get('method_name', row['model_id'])} | {row['status']} | {blockers_str} | {run_str} |")

    md.extend([
        "",
        "## Category Recall Evaluation (Dress / Shirt / Toptee)",
        "",
        "| Model | Kind | Dress R@10 | Dress R@50 | Shirt R@10 | Shirt R@50 | Toptee R@10 | Toptee R@50 | Macro R@10 | Macro R@50 | Parity / Diff | Proof Reference |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for row in rows:
        kind = row.get("evaluation_kind") or "NOT_RUN"
        if kind == "SYNTHETIC":
            kind_str = "**SYNTHETIC**"
        elif kind == "OFFICIAL_VERIFIED":
            kind_str = "OFFICIAL_VERIFIED"
        else:
            kind_str = kind

        def _fmt(val: Any, reason: Any) -> str:
            if val is not None:
                return f"{val:.2f}%"
            return f"null ({reason or 'unavailable'})"

        reason = row.get("missing_metrics_reason") or row.get("status")
        d10 = _fmt(row.get("dress_r10"), reason)
        d50 = _fmt(row.get("dress_r50"), reason)
        s10 = _fmt(row.get("shirt_r10"), reason)
        s50 = _fmt(row.get("shirt_r50"), reason)
        t10 = _fmt(row.get("toptee_r10"), reason)
        t50 = _fmt(row.get("toptee_r50"), reason)
        m10 = _fmt(row.get("macro_r10"), reason)
        m50 = _fmt(row.get("macro_r50"), reason)

        diff = row.get("measured_difference")
        if diff and isinstance(diff, dict):
            diff_str = diff.get("label", "computed diff")
        elif row.get("metrics") and row["metrics"].get("parity_status"):
            diff_str = row["metrics"]["parity_status"]
        else:
            diff_str = "null (no paper comparison)"

        proof_ref = Path(row["completion_proof"]).name if row.get("completion_proof") else "null (no proof)"

        md.append(f"| {row['model_id']} | {kind_str} | {d10} | {d50} | {s10} | {s50} | {t10} | {t50} | {m10} | {m50} | {diff_str} | {proof_ref} |")

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
