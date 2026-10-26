from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workbench.backend import autopilot, environment_planner
from test_autopilot_config import fixture_config


def _model(model_id: str, *, python: str, noise: int = 0, size: int = 0) -> dict:
    return {
        "model_id": model_id,
        "method_name": model_id,
        "native_protocol": "fashioniq_original_split",
        "supported_protocols": ["fashioniq_original_split"],
        "source_available": True,
        "environment_required": True,
        "environment": {"python": python, "pytorch": "2.0.1", "torchvision": "0.15.2",
                        "cuda": "UNKNOWN", "packages": [], "confidence": "TEST_SOURCE"},
        "checkpoint_variants": [{
            "checkpoint_id": "checkpoint",
            "filename": f"{model_id}.pt",
            "download_url": f"https://example.invalid/{model_id}.pt",
            "checkpoint_training_noise_pct": noise,
            "expected_size_bytes": size,
            "checkpoint_mapping_status": "VERIFIED_METADATA",
        }],
        "checkpoint_bundles": [],
    }


def _config(tmp_path: Path, models: list[dict]):
    ids = [model["model_id"] for model in models]
    path, document, _ = fixture_config(tmp_path, ids)
    document["budgets"]["max_download_bytes"] = 0
    document["permissions"].update(allow_network=True, allow_large_downloads=True,
                                   allow_env_install=True)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return autopilot.load_config(path, registry={"models": models})


def _capture_environment_planning(monkeypatch):
    monkeypatch.setattr(environment_planner, "_solver_command", lambda: None)
    monkeypatch.setattr(environment_planner, "_reusable_interpreter", lambda *args, **kwargs: None)
    monkeypatch.setattr(environment_planner, "_group_status",
                        lambda *args, **kwargs: ("NOT_INSTALLED", "SOLVER_UNAVAILABLE", None))
    calls = []
    real_environment_plan = autopilot._environment_plan

    def capture(selected, **kwargs):
        calls.append({model["model_id"] for model in selected})
        return real_environment_plan(selected, **kwargs)

    monkeypatch.setattr(autopilot, "_environment_plan", capture)
    return calls


def _execute_as_completed(monkeypatch, config, plan):
    from workbench.scripts import pipeline

    stages = [pipeline.Stage(name="evaluation", model_id=mid) for mid in config.selected_model_ids]
    records = {pipeline.stage_key(stage): {"name": "evaluation", "model_id": stage.model_id,
                                           "status": "COMPLETE"} for stage in stages}
    monkeypatch.setattr(pipeline, "all_stages", lambda args, cfg: stages)
    monkeypatch.setattr(pipeline, "run_stages", lambda *args, **kwargs: 0)
    monkeypatch.setattr(pipeline, "load_pipeline_state", lambda path: {"stages": records})
    monkeypatch.setattr(environment_planner, "provision_environment_plan",
                        lambda models, environment_plan, **kwargs: {
                            "plan": environment_plan, "models": {}, "groups": environment_plan["groups"]})
    return autopilot._execute(config, plan, resume=False, force_models=[])



def test_replacement_plan_solver_verification_drives_classification_and_report(tmp_path, monkeypatch):
    model = _model("solver-replaced", python="3.9")
    config = _config(tmp_path, [model])
    planning_calls = _capture_environment_planning(monkeypatch)
    plan = autopilot.resolved_plan(config)
    plan["solver_verified"] = False
    plan["environment_plan"]["groups"][0]["environment_id"] = "stale-environment-id"
    planning_calls.clear()
    real_environment_plan = autopilot._environment_plan

    def verified_replacement(selected, **kwargs):
        replacement = real_environment_plan(selected, **kwargs)
        replacement["solver_verified"] = True
        return replacement

    monkeypatch.setattr(autopilot, "_environment_plan", verified_replacement)
    execution = _execute_as_completed(monkeypatch, config, plan)

    assert plan["solver_verified"] is plan["environment_plan"]["solver_verified"] is True
    rows, blockers = autopilot._classify(plan, execution, config)
    summary = autopilot._write_reports(config, plan, rows, blockers, execution, planned=False)["summary"]
    assert summary["solver_verified"] is True
    assert summary["models"][0]["status"] == "COMPLETED"


def test_equal_membership_with_stale_group_identity_replans(tmp_path, monkeypatch):
    models = [_model("stale-a", python="3.9"), _model("stale-b", python="3.9")]
    config = _config(tmp_path, models)
    planning_calls = _capture_environment_planning(monkeypatch)
    plan = autopilot.resolved_plan(config)
    planning_calls.clear()
    groups = plan["environment_plan"]["groups"]
    assert {mid for group in groups for mid in group["members"]} == set(config.selected_model_ids)
    groups[0]["constraints"] = {"corrupted": True}
    groups[0]["environment_id"] = "stale-environment-id"

    _execute_as_completed(monkeypatch, config, plan)

    executable_ids = set(config.selected_model_ids)
    assert planning_calls == [executable_ids]
    specs = {model["model_id"]: environment_planner.model_constraints(
        model, third_party_root=config.paths["third_party"]) for model in models}
    for group in plan["environment_plan"]["groups"]:
        members = sorted(group["members"])
        rebuilt_constraints, conflict = environment_planner._constraints(
            [specs[mid] for mid in members])
        assert conflict is None
        assert group["constraints"] == rebuilt_constraints
        assert group["environment_id"] == environment_planner._environment_id(
            members, rebuilt_constraints)


def test_matching_plan_normalizes_outer_solver_fact_for_classification(tmp_path, monkeypatch):
    model = _model("solver-matching", python="3.9")
    config = _config(tmp_path, [model])
    planning_calls = _capture_environment_planning(monkeypatch)
    plan = autopilot.resolved_plan(config)
    plan["solver_verified"] = not plan["environment_plan"]["solver_verified"]
    planning_calls.clear()

    execution = _execute_as_completed(monkeypatch, config, plan)

    assert planning_calls == []
    assert plan["solver_verified"] == plan["environment_plan"]["solver_verified"]
    rows, blockers = autopilot._classify(plan, execution, config)
    summary = autopilot._write_reports(config, plan, rows, blockers, execution, planned=False)["summary"]
    assert summary["solver_verified"] is False
    assert summary["models"][0]["status"] == "COMPLETED_WITH_LIMITATIONS"


def test_stale_interpreter_mapping_replans(tmp_path, monkeypatch):
    model = _model("mapping-stale", python="3.9")
    config = _config(tmp_path, [model])
    planning_calls = _capture_environment_planning(monkeypatch)
    plan = autopilot.resolved_plan(config)
    plan["environment_plan"]["interpreter_mapping"][model["model_id"]] = "/stale/python"
    planning_calls.clear()

    _execute_as_completed(monkeypatch, config, plan)

    assert planning_calls == [{model["model_id"]}]
    assert plan["environment_plan"]["interpreter_mapping"] == environment_planner.apply_interpreter_mapping(
        [model], plan["environment_plan"])


def test_environment_assignment_reason_appears_in_classification(tmp_path):
    model = _model("named-environment-failure", python="3.9")
    config = _config(tmp_path, [model])
    plan = autopilot.resolved_plan(config)
    execution = {
        "exit_code": 1,
        "stages": {"environment:named-environment-failure": {
            "name": "environment", "model_id": model["model_id"], "status": "FAILED"}},
        "environment_assignment": {"models": {model["model_id"]: {
            "reason": "named environment provisioning cause"}}},
    }

    rows, _ = autopilot._classify(plan, execution, config)

    assert rows[0]["status"] == "BLOCKED_ENVIRONMENT"
    assert "named environment provisioning cause" in rows[0]["blockers"]


def test_registry_only_fallback_replans_at_most_once(tmp_path, monkeypatch):
    model = _model("registry-only", python="3.9")
    config = _config(tmp_path, [model])
    calls = []

    def registry_only(selected, **kwargs):
        calls.append({item["model_id"] for item in selected})
        return {"solver_verified": False, "groups": [
            {"environment_id": item["model_id"], "members": [item["model_id"]],
             "interpreter": None, "lock_digest": "registry-only-unverified"}
            for item in selected]}

    monkeypatch.setattr(autopilot, "_environment_plan", registry_only)
    plan = autopilot.resolved_plan(config)
    calls.clear()

    _execute_as_completed(monkeypatch, config, plan)

    assert calls == [{model["model_id"]}]


def test_consistent_environment_plan_does_not_replan(tmp_path, monkeypatch):
    model = _model("consistent", python="3.9")
    config = _config(tmp_path, [model])
    planning_calls = _capture_environment_planning(monkeypatch)
    plan = autopilot.resolved_plan(config)
    planning_calls.clear()

    _execute_as_completed(monkeypatch, config, plan)

    assert planning_calls == []


def test_checkpoint_and_source_blockers_leave_only_executable_groups_to_provision(tmp_path, monkeypatch):
    from workbench.scripts import pipeline

    models = [
        _model("shared-a", python="3.9"),
        _model("shared-b", python="3.9"),
        _model("provision-fails", python="3.10"),
        _model("source-blocked", python="3.9"),
        _model("budget-blocked", python="3.9", size=1),
        _model("checkpoint-blocked", python="3.9", noise=50),
    ]
    config = _config(tmp_path, models)
    monkeypatch.setattr(environment_planner, "_solver_command", lambda: None)
    monkeypatch.setattr(environment_planner, "_reusable_interpreter", lambda *args, **kwargs: None)
    monkeypatch.setattr(environment_planner, "_group_status",
                        lambda *args, **kwargs: ("NOT_INSTALLED", "SOLVER_UNAVAILABLE", None))

    planning_calls = []
    real_environment_plan = autopilot._environment_plan

    def capture_environment_plan(selected, **kwargs):
        planning_calls.append({model["model_id"] for model in selected})
        return real_environment_plan(selected, **kwargs)

    monkeypatch.setattr(autopilot, "_environment_plan", capture_environment_plan)
    plan = autopilot.resolved_plan(config)
    rows = {row["model_id"]: row for row in plan["models"]}
    assert rows["budget-blocked"]["status"] == "BLOCKED_CHECKPOINT"
    assert "budget exceeded" in rows["budget-blocked"]["blockers"][0]
    assert rows["checkpoint-blocked"]["status"] == "BLOCKED_CHECKPOINT"
    assert rows["checkpoint-blocked"]["blockers"]
    assert {"budget-blocked", "checkpoint-blocked"}.isdisjoint(
        {mid for group in plan["environment_plan"]["groups"] for mid in group["members"]})

    assert planning_calls == [{"shared-a", "shared-b", "source-blocked", "provision-fails"}]
    rows["source-blocked"].update(status="BLOCKED_SOURCE", blockers=["pinned source unavailable"])
    provision_calls = []
    real_provision = environment_planner.provision_environment_plan

    def capture_provision(selected, environment_plan, **kwargs):
        provision_calls.append({"models": {model["model_id"] for model in selected},
                                "members": {mid for group in environment_plan["groups"]
                                            for mid in group["members"]}})
        return real_provision(selected, environment_plan, **kwargs)

    monkeypatch.setattr(environment_planner, "provision_environment_plan", capture_provision)

    def provision_group(group, by_id, specs, artifacts_root, third_party_root, allowed, solver_cmd,
                        network_allowed=True):
        provision_calls.append({"group_call": set(group["members"])})
        assert set(group["members"]) <= set(by_id) == set(specs)
        if "provision-fails" in group["members"]:
            return False, "controlled provisioning failure"
        interpreter = tmp_path / "fake-envs" / group["environment_id"] / "bin" / "python"
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        interpreter.chmod(0o755)
        group.update(conda_env_path=str(interpreter.parent.parent), interpreter=str(interpreter),
                     installation_status="INSTALLED", runtime_verification_status="READY",
                     state="RUNTIME_VERIFIED")
        return True, "controlled success"

    monkeypatch.setattr(environment_planner, "_provision_group", provision_group)
    records = {
        "evaluation:shared-a": {"name": "evaluation", "model_id": "shared-a", "status": "COMPLETE"},
        "evaluation:shared-b": {"name": "evaluation", "model_id": "shared-b", "status": "COMPLETE"},
        "environment:provision-fails": {
            "name": "environment", "model_id": "provision-fails", "status": "FAILED",
            "failure_reason": "controlled provisioning failure",
        },
    }
    monkeypatch.setattr(pipeline, "all_stages", lambda args, cfg: [
        pipeline.Stage(name="evaluation", model_id=mid) for mid in ("shared-a", "shared-b")
    ] + [pipeline.Stage(name="environment", model_id="provision-fails")])
    monkeypatch.setattr(pipeline, "run_stages", lambda *args, **kwargs: 1)
    monkeypatch.setattr(pipeline, "load_pipeline_state", lambda path: {"stages": records})

    execution = autopilot._execute(config, plan, resume=False, force_models=[])
    executable = {"shared-a", "shared-b", "provision-fails"}
    assert planning_calls[-1] == executable
    assert provision_calls[0] == {"models": executable, "members": executable}
    assert {"group_call": {"shared-a", "shared-b"}} in provision_calls
    assert {"group_call": {"provision-fails"}} in provision_calls
    assert all("checkpoint-blocked" not in call.get("group_call", set()) and
               "source-blocked" not in call.get("group_call", set()) for call in provision_calls)

    final_plan = plan["environment_plan"]
    groups = final_plan["groups"]
    assert {mid for group in groups for mid in group["members"]} == executable
    specs = {model["model_id"]: environment_planner.model_constraints(
        model, third_party_root=config.paths["third_party"]) for model in models}
    for group in groups:
        members = group["members"]
        assert set(members) <= executable
        constraints, conflict = environment_planner._constraints([specs[mid] for mid in members])
        assert conflict is None
        assert group["constraints"] == constraints
        assert group["environment_id"] == environment_planner._environment_id(members, constraints)

    shared = next(group for group in groups if "shared-a" in group["members"])
    failed = next(group for group in groups if "provision-fails" in group["members"])
    assert shared["state"] == "RUNTIME_VERIFIED"
    assert failed["state"] == "FAILED"
    assert any("controlled provisioning failure" in reason for reason in failed["unresolved"])
    assert plan["solver_verified"] is False
    assert execution["environment_assignment"]["models"]["shared-a"]["state"] == shared["state"]
    assert set(final_plan["interpreter_mapping"]) == executable
    assert set(execution["environment_assignment"]["models"]) == executable
    assert failed["interpreter"] is None
    assert execution["environment_assignment"]["models"]["provision-fails"]["state"] == "FAILED"
    assert "controlled provisioning failure" in execution["environment_assignment"]["models"][
        "provision-fails"]["reason"]

    report_rows, blockers = autopilot._classify(plan, execution, config)
    summary = autopilot._write_reports(config, plan, report_rows, blockers, execution,
                                       planned=False)["summary"]
    reported = {row["model_id"]: row for row in summary["models"]}
    assert set(reported) == set(config.selected_model_ids)
    assert reported["shared-a"]["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert reported["shared-b"]["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert reported["provision-fails"]["status"] == "BLOCKED_ENVIRONMENT"
    assert "controlled provisioning failure" in reported["provision-fails"]["blockers"][0]
    assert reported["source-blocked"]["status"] == "BLOCKED_SOURCE"
    assert reported["source-blocked"]["blockers"] == ["pinned source unavailable"]
    assert reported["budget-blocked"]["status"] == "BLOCKED_CHECKPOINT"
    assert "budget exceeded" in reported["budget-blocked"]["blockers"][0]
    assert reported["checkpoint-blocked"]["status"] == "BLOCKED_CHECKPOINT"
    assert reported["checkpoint-blocked"]["blockers"]


def test_empty_executable_cohort_reports_checkpoint_blocker_without_provisioning(tmp_path, monkeypatch):
    model = _model("blocked-only", python="3.9", noise=50)
    config = _config(tmp_path, [model])
    plan = autopilot.resolved_plan(config)
    row = plan["models"][0]
    assert row["status"] == "BLOCKED_CHECKPOINT"
    assert row["blockers"]
    assert plan["environment_plan"]["groups"] == []
    monkeypatch.setattr(environment_planner, "provision_environment_plan",
                        lambda *args, **kwargs: pytest.fail("empty cohort reached provisioning"))
    result = autopilot._execute(config, plan, resume=False, force_models=[])
    assert result["selected_models"] == []
    rows, blockers = autopilot._classify(plan, result, config)
    summary = autopilot._write_reports(config, plan, rows, blockers, result, planned=False)["summary"]
    assert summary["models"][0]["model_id"] == "blocked-only"
    assert summary["models"][0]["status"] == "BLOCKED_CHECKPOINT"
    assert summary["models"][0]["blockers"]
