from __future__ import annotations

import os
import sys
from types import SimpleNamespace


from workbench.backend import environment_planner as planner


def model(model_id, *, python="3.9", torch="UNKNOWN", torchvision="UNKNOWN", packages=None,
          ranges=None, env_var=None):
    declared = {"python": python, "pytorch": torch, "torchvision": torchvision,
                "cuda": "UNKNOWN", "packages": packages or [], "confidence": "TEST_SOURCE"}
    if env_var:
        declared["python_env_var"] = env_var
    return {"model_id": model_id, "environment": declared}


def accepted(*args, **kwargs):
    return True


def groups(plan):
    return [set(group["members"]) for group in plan["groups"]]


def test_two_compatible_models_share_one_group():
    models = [model("alpha", python="3.9", torch="2.0.1", torchvision="0.15.2"),
              model("beta", python="3.9", torch="2.0.1", torchvision="0.15.2")]
    assert groups(planner.plan_environments(models, solver=accepted)) == [{"alpha", "beta"}]


def test_incompatible_python_versions_split():
    result = planner.plan_environments([model("py38", python="3.8"), model("py39", python="3.9")], solver=accepted)
    assert len(result["groups"]) == 2
    assert any("python" in reason.lower() and "source:" in reason.lower()
               for group in result["groups"] for reason in group["split_reasons"])


def test_conflicting_exact_pytorch_versions_split():
    models = [model("old", torch="1.12.1", torchvision="0.13.1"),
              model("new", torch="1.13.1", torchvision="0.14.1")]
    result = planner.plan_environments(models, solver=accepted)
    assert len(result["groups"]) == 2
    assert any("torch" in reason.lower() and "source:" in reason.lower()
               for group in result["groups"] for reason in group["split_reasons"])


def test_compatible_package_ranges_may_share():
    models = [model("low", packages=["numpy>=1.20,<2"]),
              model("high", packages=["numpy>=1.21,<1.23"])]
    assert groups(planner.plan_environments(models, solver=accepted)) == [{"high", "low"}]


def test_pairwise_accepted_constraints_do_not_override_collective_solver_rejection():
    models = [model("alpha", packages=["alpha-dep==1"]),
              model("beta", packages=["beta-dep==1"]),
              model("gamma", packages=["gamma-dep==1"])]

    def solver(constraints):
        # Every pair is accepted; the full three-package solve is not.
        return len(constraints["packages"]) < 3

    result = planner.plan_environments(models, solver=solver)
    assert len(result["groups"]) == 2
    assert any("solver rejected combined" in reason and "source:" in reason
               for group in result["groups"] for reason in group["split_reasons"])


def test_solver_failure_splits_with_source_conflict():
    models = [model("first", packages=["shared==1", "first-only==1"]),
              model("second", packages=["shared==1", "second-only==1"])]

    def solver(constraints):
        return not {"first-only", "second-only"}.issubset(constraints["packages"])

    result = planner.plan_environments(models, solver=solver)
    assert len(result["groups"]) == 2
    reasons = [reason for group in result["groups"] for reason in group["split_reasons"]]
    assert any("SOLVER_ERROR" in item and "first-only" in item and "source:" in item for item in reasons)


def test_channel_gap_is_recorded_in_group_status_and_split_reason():
    def solver(constraints):
        if {"alpha-pkg", "beta-pkg"}.issubset(constraints["packages"]):
            return False, "PackagesNotFoundError: beta-pkg not available from current channels"
        return True

    result = planner.plan_environments([
        model("alpha", packages=["alpha-pkg==1"]), model("beta", packages=["beta-pkg==1"])],
        solver=solver)
    group = next(group for group in result["groups"] if "alpha" in group["members"])
    assert "CHANNEL_GAP" in group["solver_outcomes"]
    assert any("CHANNEL_GAP" in reason and "channels searched: none (injected solver)" in reason
               for reason in group["split_reasons"])


def test_solver_outcomes_distinguish_channel_gap_from_version_conflict():
    constraints = {"python": [], "torch": [], "torchvision": [], "cuda": [],
                   "packages": {}, "ranges": {}}
    gap, gap_reason = planner._solver_reason(
        "PackagesNotFoundError: package not available from current channels", constraints, "standalone")
    conflict, conflict_reason = planner._solver_reason(
        "UnsatisfiableError: python conflicts with numpy", constraints, "standalone")
    assert gap == "CHANNEL_GAP" and "channel" in gap_reason.lower()
    assert conflict == "VERSION_CONFLICT" and "version" in conflict_reason.lower()
    assert "conda-forge, pytorch, defaults" in gap_reason
    assert "conda-forge, pytorch, defaults" in conflict_reason


def test_channel_gap_and_version_conflict_are_named_in_plan(monkeypatch):
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    messages = iter(("PackagesNotFoundError: package not available from current channels",
                     "PackagesNotFoundError: package not available from current channels"))
    monkeypatch.setattr(planner, "_solver_solve", lambda command, constraints: (False, next(messages)))
    gap_plan = planner.plan_environments([model("channel-block", packages=["channel-block-pkg==1"])])
    assert gap_plan["blocked"][0]["solver_outcome"] == "CHANNEL_GAP"
    assert "channels searched" in gap_plan["blocked"][0]["reason"]

    conflict_plan = planner.plan_environments(
        [model("conflict-block", packages=["conflict-block-pkg==1"])],
        solver=lambda constraints: (False, "UnsatisfiableError: python conflicts with numpy"))
    assert conflict_plan["blocked"][0]["solver_outcome"] == "VERSION_CONFLICT"
    assert "version" in conflict_plan["blocked"][0]["reason"].lower()



def test_missing_requirements_remain_unknown_and_unresolved():
    unknown = model("unknown", python="UNKNOWN")
    spec = planner.model_constraints(unknown)
    assert spec["python"] == "UNKNOWN"
    result = planner.plan_environments([unknown], solver=accepted)
    assert any("python constraint UNKNOWN" in item for item in result["groups"][0]["unresolved"])


def test_existing_valid_environment_is_reused_not_reinstalled(monkeypatch, tmp_path):
    interpreter = tmp_path / "env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    item = model("existing", env_var="EXISTING_PYTHON")
    monkeypatch.setenv("EXISTING_PYTHON", str(interpreter))
    monkeypatch.setattr(planner.runtime, "verify_environment", lambda *a, **k: SimpleNamespace(
        tier="RUNTIME_READY", status="DEPENDENCIES_OK"))
    result = planner.plan_environments([item], solver=accepted)
    group = result["groups"][0]
    assert group["installation_status"] == "REUSED"
    assert group["runtime_verification_status"] == "READY"
    assert group["interpreter"] == str(interpreter)


def test_changed_lock_changes_environment_identity_and_rechecks_readiness():
    first = model("lock", packages=["sample==1.0"])
    changed = model("lock", packages=["sample==2.0"])
    old_plan = planner.plan_environments([first], solver=accepted)
    new_plan = planner.plan_environments([changed], solver=accepted)
    assert old_plan["groups"][0]["lock_digest"] != new_plan["groups"][0]["lock_digest"]
    assert old_plan["groups"][0]["environment_id"] != new_plan["groups"][0]["environment_id"]
    assert new_plan["groups"][0]["runtime_verification_status"] != "READY"


def test_member_import_failure_does_not_mark_group_ready(monkeypatch, tmp_path):
    interpreter = tmp_path / "env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    models = [model("one", env_var="ONE_PYTHON"), model("two", env_var="TWO_PYTHON")]
    monkeypatch.setenv("ONE_PYTHON", str(interpreter))
    monkeypatch.setenv("TWO_PYTHON", str(interpreter))
    monkeypatch.setattr(planner.runtime, "verify_environment", lambda item, **kwargs: SimpleNamespace(
        tier="RUNTIME_READY", status="DEPENDENCIES_OK" if item["model_id"] == "one" else "MODEL_IMPORT_FAILED"))
    result = planner.plan_environments(models, solver=accepted)
    group = result["groups"][0]
    assert group["installation_status"] != "REUSED"
    assert group["runtime_verification_status"] == "IMPORT_CHECK_FAILED"


def test_system_python_is_never_a_planned_environment_target():
    result = planner.plan_environments([model("isolated-target")], solver=accepted)
    assert result["groups"][0]["interpreter"] != sys.executable
    assert "/artifacts/environments/" in result["groups"][0]["interpreter"]


def test_source_requirements_and_registry_declarations_are_read_from_real_checkouts():
    from workbench.backend.registry import load_registry

    models = {item["model_id"]: item for item in load_registry()["models"]}
    limn = planner.model_constraints(models["limn"])
    assert limn["torch"] == "1.12.1"
    assert limn["torchvision"] == "0.13.1"
    assert limn["packages"]["open-clip-torch"] == "2.20.0"
    sprc = planner.model_constraints(models["sprc"])
    assert sprc["torch"] == "2.0.1"
    assert sprc["packages"]["transformers"] == "4.36.2"
    assert sprc["python"] == "3.9"
    assert sprc["cuda"] == "UNKNOWN"


def _write_import_fixture(monkeypatch, tmp_path, script):
    from workbench.backend.adapters.models import ADAPTERS

    source = tmp_path / "third_party" / "ENCODER"
    source.mkdir(parents=True)
    (source / "evaluate_model.py").write_text(script, encoding="utf-8")
    monkeypatch.setattr(planner, "WORKBENCH_ROOT", tmp_path)
    monkeypatch.setattr(ADAPTERS["encoder"], "script", "evaluate_model.py")
    return source


def test_evaluator_imports_keep_pins_and_drop_training_only_packages(monkeypatch, tmp_path):
    source = _write_import_fixture(monkeypatch, tmp_path, "import numpy\nimport helper\n")
    (source / "helper.py").write_text("import pandas\n", encoding="utf-8")
    (source / "requirements.txt").write_text(
        "numpy==1.22.3\npandas==2.0.0\ncomet-ml==3.21.0\n", encoding="utf-8")

    spec = planner.model_constraints({"model_id": "encoder", "source_dir": "ENCODER",
                                     "environment": {"python": "3.9", "pytorch": "UNKNOWN",
                                                     "torchvision": "UNKNOWN", "cuda": "UNKNOWN"}})

    assert spec["packages"] == {"numpy": "1.22.3", "pandas": "2.0.0"}


def test_unpinned_evaluator_import_is_reported_unknown(monkeypatch, tmp_path):
    source = _write_import_fixture(monkeypatch, tmp_path, "import open_clip\n")
    (source / "requirements.txt").write_text("training-only==1.0\n", encoding="utf-8")

    spec = planner.model_constraints({"model_id": "encoder", "source_dir": "ENCODER",
                                     "environment": {"python": "3.9", "pytorch": "UNKNOWN",
                                                     "torchvision": "UNKNOWN", "cuda": "UNKNOWN"}})

    assert spec["packages"] == {"open-clip-torch": "UNKNOWN"}


def test_narrowed_numpy_pin_still_refuses_conflicting_python(monkeypatch, tmp_path):
    source = _write_import_fixture(monkeypatch, tmp_path, "import numpy\n")
    (source / "requirements.txt").write_text(
        "numpy==1.22.3\ntraining-only==9.9\n", encoding="utf-8")
    item = {"model_id": "encoder", "source_dir": "ENCODER", "source_available": True,
            "environment": {"python": "3.7.6", "pytorch": "UNKNOWN",
                            "torchvision": "UNKNOWN", "cuda": "UNKNOWN"}}
    calls = []

    def reject_real_numpy_conflict(constraints):
        calls.append(constraints)
        return False, "python ==3.7.6 conflicts with numpy ==1.22.3"

    result = planner.plan_environments([item], solver=reject_real_numpy_conflict)

    assert calls[0]["python"] == [("==", "3.7.6")]
    assert calls[0]["packages"] == {"numpy": [("==", "1.22.3")]}
    assert result["blocked"][0]["model_id"] == "encoder"
    assert "python" in result["blocked"][0]["reason"] and "numpy" in result["blocked"][0]["reason"]
    assert "training-only" not in result["blocked"][0]["reason"]



def test_operator_interpreter_override_wins_mapping():
    item = model("override", env_var="OPERATOR_PYTHON")
    plan = planner.plan_environments([item], solver=accepted)
    mapped = planner.apply_interpreter_mapping([item], plan, {"OPERATOR_PYTHON": "/operator/env/bin/python"})
    assert mapped["override"] == "/operator/env/bin/python"


def test_absent_solver_is_never_reported_verified(monkeypatch):
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    result = planner.plan_environments([model("offline")])
    assert result["solver"] == "unavailable"
    assert result["groups"][0]["runtime_verification_status"] == "SOLVER_UNAVAILABLE"


def test_solver_timeout_is_reported_as_refusal():
    def timed_out(constraints):
        raise planner.subprocess.TimeoutExpired("solver", 7)

    result = planner.plan_environments([model("timeout", packages=["timeout-pkg==1"])], solver=timed_out)
    assert result["blocked"][0]["model_id"] == "timeout"
    assert "timeout" in result["blocked"][0]["reason"].lower()


def test_conda_solver_excludes_pytorch_index_packages(monkeypatch):
    calls = []
    monkeypatch.setattr(planner.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(
        returncode=0, stderr="", stdout=""))
    constraints = {"python": [("==", "3.9")], "packages": {
        "torch": [("==", "1.13.1")], "torchvision": [("==", "0.14.1")],
        "open-clip-torch": [("==", "2.24.0")], "numpy": [("==", "1.21.6")]}, "ranges": {}}

    assert planner._solver_solve("conda", constraints)[0]
    specs = calls[0]
    assert "python=3.9" in specs and "numpy=1.21.6" in specs
    assert not any(name in specs for name in ("torch=1.13.1", "torchvision=0.14.1", "open-clip-torch=2.24.0"))


def test_conda_solver_uses_configured_channels(monkeypatch):
    calls = []
    monkeypatch.setattr(planner.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(
        returncode=0, stderr="", stdout=""))
    constraints = {"python": [("==", "3.9")], "packages": {"numpy": [("==", "1.21.6")]},
                   "ranges": {}}

    assert planner._solver_solve("conda", constraints)[0]
    argv = calls[0]
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"] == list(planner.CONDA_CHANNELS)


def test_group_evidence_records_channels_used(monkeypatch):
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(planner, "_solver_solve", lambda command, constraints: (True, "ok"))
    result = planner.plan_environments([model("channel-evidence")])
    assert f"Conda channels searched: {', '.join(planner.CONDA_CHANNELS)}" in result["groups"][0]["evidence"]


def test_injected_solver_bypasses_real_solve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("injected solver must bypass conda command")

    monkeypatch.setattr(planner, "_solver_command", forbidden)
    monkeypatch.setattr(planner, "_solver_solve", forbidden)
    assert groups(planner.plan_environments([model("injected")], solver=accepted)) == [{"injected"}]


def test_conda_solver_rejects_conflicting_resolvable_packages(monkeypatch, tmp_path):
    # Keep the real TemporaryDirectory: the point of the test is the solve result, not the
    # directory lifecycle, and a bare Path is not a context manager.

    monkeypatch.setattr(planner.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(
        returncode=1, stderr="LibMambaUnsatisfiableError: python conflicts with numpy", stdout=""))
    constraints = {"python": [("==", "3.7.6")], "packages": {
        "numpy": [("==", "1.22.3")]}, "ranges": {}}

    ok, detail = planner._solver_solve("conda", constraints)
    assert not ok and "python conflicts with numpy" in detail


def test_identical_solve_is_memoized_across_plan_calls():
    calls = []

    def solver(constraints):
        calls.append(planner.json.dumps(constraints, sort_keys=True))
        return True

    models = [model("memo-a", packages=["memo==1"]), model("memo-b", packages=["memo==1"]),
              model("memo-c", packages=["memo==1"])]
    planner.plan_environments(models, solver=solver)
    first_call_count = len(calls)
    planner.plan_environments(models, solver=solver)
    assert first_call_count == 1
    assert len(calls) == first_call_count


def test_model_without_source_or_checkpoint_is_blocked():
    item = {"model_id": "unavailable", "source_available": False, "source_dir": None,
            "checkpoint_variants": [], "environment": {}}
    result = planner.plan_environments([item], solver=accepted)
    assert not result["groups"]
    assert result["blocked"][0]["model_id"] == "unavailable"
    assert "source" in result["blocked"][0]["reason"].lower()
    assert "checkpoint" in result["blocked"][0]["reason"].lower()


def test_verified_shared_interpreter_bypasses_solver(monkeypatch):
    monkeypatch.setenv("SHARED_TEST_PYTHON", "/verified/env/bin/python")
    monkeypatch.setattr(planner.runtime, "verify_environment", lambda *args, **kwargs: SimpleNamespace(
        tier="RUNTIME_READY", status="DEPENDENCIES_OK"))

    def forbidden_solver(constraints):
        raise AssertionError("verified interpreter must bypass solver")

    models = [model("reuse-a", env_var="SHARED_TEST_PYTHON"),
              model("reuse-b", env_var="SHARED_TEST_PYTHON")]
    result = planner.plan_environments(models, solver=forbidden_solver)
    assert groups(result) == [{"reuse-a", "reuse-b"}]
    group = result["groups"][0]
    assert group["installation_status"] == "REUSED"
    assert group["runtime_verification_status"] == "READY"
    assert group["solver_verification_status"] == "NOT_RUN_REUSED"
    assert result["solver_verification_status"] == "NOT_RUN"
    assert "/verified/env/bin/python" in group["runtime_evidence"]


def test_registry_planning_does_not_group_unavailable_model_and_finishes():
    from time import monotonic
    from workbench.backend.registry import load_registry

    started = monotonic()
    result = planner.plan_environments(load_registry()["models"])
    elapsed = monotonic() - started
    assert elapsed < 120
    assert not any("ptha_mtst" in group["members"] for group in result["groups"])
    source_backed = {item["model_id"] for item in load_registry()["models"]
                     if item.get("source_available") or item.get("source_dir") or
                     item.get("checkpoint_variants") or item.get("checkpoint_bundles")}
    grouped = {mid for group in result["groups"] for mid in group["members"]}
    blocked = {item["model_id"]: item["reason"] for item in result["blocked"]}
    assert source_backed <= grouped | blocked.keys()
    assert all(blocked[mid].strip() for mid in source_backed - grouped)
    assert all("source:" in blocked[mid].lower() for mid in source_backed - grouped)
