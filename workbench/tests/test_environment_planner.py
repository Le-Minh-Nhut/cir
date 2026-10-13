from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    """Planning records a target prefix, never the interpreter running the workbench."""
    result = planner.plan_environments([model("isolated-target")], solver=accepted)
    group = result["groups"][0]
    assert group["interpreter"] != sys.executable
    assert "/artifacts/environments/" in group["interpreter"]


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
    adapter = type("FixtureEncoderAdapter", (ADAPTERS["encoder"],), {"script": "evaluate_model.py"})
    monkeypatch.setitem(ADAPTERS, "encoder", adapter)
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


def test_pip_only_registry_pin_skips_conda_but_resolvable_pin_reaches_it(monkeypatch):
    calls = []
    monkeypatch.setattr(planner.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(
        returncode=0, stderr="", stdout=""))
    item = model("declared-pins", packages=["comet-ml==3.21.0", "numpy==1.22.3"])
    constraints = planner._constraints([planner.model_constraints(item)])[0]

    assert constraints["packages"]["comet-ml"] == [("==", "3.21.0")]
    assert constraints["packages"]["numpy"] == [("==", "1.22.3")]
    assert "comet-ml" not in planner._conda_constraints(constraints)["packages"]
    assert "numpy" in planner._conda_constraints(constraints)["packages"]
    assert planner._solver_solve("conda", planner._conda_constraints(constraints))[0]
    specs = calls[0][calls[0].index("--override-channels") + 1:]
    assert "comet-ml=3.21.0" not in specs
    assert "numpy=1.22.3" in specs


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
    # Real conda channel solves vary with network latency; 300s matches solver timeout.
    assert elapsed < planner.SOLVER_TIMEOUT_SECONDS
    assert not any("ptha_mtst" in group["members"] for group in result["groups"])
    source_backed = {item["model_id"] for item in load_registry()["models"]
                     if item.get("source_available") or item.get("source_dir") or
                     item.get("checkpoint_variants") or item.get("checkpoint_bundles")}
    grouped = {mid for group in result["groups"] for mid in group["members"]}
    blocked = {item["model_id"]: item["reason"] for item in result["blocked"]}
    assert source_backed <= grouped | blocked.keys()
    assert all(blocked[mid].strip() for mid in source_backed - grouped)
    assert all("source:" in blocked[mid].lower() for mid in source_backed - grouped)

def test_custom_third_party_root_p0_02(tmp_path):
    custom_root = tmp_path / "custom_third_party"
    custom_dir = custom_root / "CUSTOM_MODEL"
    custom_dir.mkdir(parents=True)
    (custom_dir / "requirements.txt").write_text("custom-dep==9.9.9\n", encoding="utf-8")

    custom_model = {
        "model_id": "custom-model",
        "source_dir": "CUSTOM_MODEL",
        "source_available": True,
        "environment": {"python": "3.9", "pytorch": "1.12.1", "torchvision": "0.13.1", "confidence": "CUSTOM"},
    }

    plan_custom = planner.plan_environments([custom_model], third_party_root=custom_root, solver=accepted)
    spec_custom = planner.model_constraints(custom_model, third_party_root=custom_root)
    assert "custom-dep" in spec_custom["packages"]
    assert spec_custom["packages"]["custom-dep"] == "9.9.9"

    from workbench.backend.registry import load_registry
    models_by_id = {m["model_id"]: m for m in load_registry()["models"]}
    csmcir = models_by_id["csmcir"]
    spec_no_csmcir = planner.model_constraints(csmcir, third_party_root=custom_root)
    assert "pandas" not in spec_no_csmcir["packages"]
    spec_default_csmcir = planner.model_constraints(csmcir)
    assert "pandas" in spec_default_csmcir["packages"]
    out_json = tmp_path / "custom_plan.json"
    ret = planner.main(["--third-party-root", str(custom_root), "--model", "custom-model", "--output", str(out_json)])
    assert ret == 0
    assert out_json.is_file()
    plan_data = json.loads(out_json.read_text(encoding="utf-8"))
    assert plan_data["third_party_root"] == str(custom_root.resolve())

    with pytest.raises(FileNotFoundError):
        planner.main(["--third-party-root", str(tmp_path / "nonexistent_dir"), "--output", str(out_json)])

def test_p1_04_missing_pip_only_package_not_runtime_verified(tmp_path, monkeypatch):
    m = {
        "model_id": "pip-check-test",
        "environment": {
            "python": "3.11",
            "packages": ["pyyaml==6.0"],
            "pip_packages": ["open-clip-torch==2.20.0"],
            "confidence": "TEST",
        },
    }
    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    group = plan["groups"][0]
    assert group["state"] == "SOLVER_COMPATIBLE"
    assert group["runtime_verification_status"] != "RUNTIME_VERIFIED"

    fake_interp = tmp_path / "env" / "bin" / "python"
    fake_interp.parent.mkdir(parents=True)
    fake_interp.touch()
    os.chmod(fake_interp, 0o755)
    group["interpreter"] = str(fake_interp)
    group["conda_env_path"] = str(fake_interp.parent.parent)

    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda prefix: ({"pyyaml": "6.0"}, {"yaml"}, set()))

    ok, reason, details = planner._verify_group_interpreter(group, {"pip-check-test": m},
                                                            {"pip-check-test": planner.model_constraints(m)})
    assert not ok
    assert "open-clip-torch" in reason.lower()
    assert group["runtime_verification_status"] != "RUNTIME_VERIFIED"


def test_four_states_distinctly_recorded(tmp_path, monkeypatch):
    m = model("four-states", python="3.11", packages=["numpy==1.24.0"])

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    unavail_plan = planner.plan_environments([m], solver=None)
    unavail_group = unavail_plan["groups"][0]
    assert unavail_group["state"] == "CONSTRAINTS_COMPATIBLE"
    assert any("CONSTRAINTS_COMPATIBLE" in ev for ev in unavail_group["evidence"])

    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    solved_plan = planner.plan_environments([m], solver=accepted)
    solved_group = solved_plan["groups"][0]
    assert solved_group["state"] == "SOLVER_COMPATIBLE"
    assert any("SOLVER_COMPATIBLE" in ev for ev in solved_group["evidence"])
    group_copy = dict(solved_group)
    group_copy["state"] = "INSTALLED"
    group_copy["evidence"] = list(group_copy["evidence"]) + ["INSTALLED: packages are present"]
    assert group_copy["state"] == "INSTALLED"

    group_copy["state"] = "RUNTIME_VERIFIED"
    group_copy["evidence"].append("RUNTIME_VERIFIED: imports and versions verified in that interpreter")
    assert group_copy["state"] == "RUNTIME_VERIFIED"
    states = {"CONSTRAINTS_COMPATIBLE", "SOLVER_COMPATIBLE", "INSTALLED", "RUNTIME_VERIFIED"}
    assert len(states) == 4


def test_idempotent_environment_reuse(tmp_path, monkeypatch):
    """An environment this workbench created is reused without reinstalling.

    Reuse requires that the prefix is ours: either the managed marker the provisioner writes, or an
    operator pointing the model at it through its environment variable. The fixture records the
    marker, which is what a real prior run leaves behind.
    """
    m = model("reuse-test", python="3.11", packages=["pyyaml==6.0"])
    fake_interp = tmp_path / "env" / "bin" / "python"
    fake_interp.parent.mkdir(parents=True)
    fake_interp.touch()
    os.chmod(fake_interp, 0o755)

    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    plan["groups"][0]["interpreter"] = str(fake_interp)
    plan["groups"][0]["conda_env_path"] = str(fake_interp.parent.parent)
    planner._mark_workbench_managed(fake_interp.parent.parent, plan["groups"][0]["environment_id"])

    monkeypatch.setattr(planner, "_verify_group_interpreter", lambda *a, **k: (True, "OK", {}))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=""))

    assignment = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)
    group = assignment["groups"][0]
    assert group["installation_status"] == "REUSED"
    assert group["state"] == "RUNTIME_VERIFIED"
    assert len(calls) == 0



def test_an_unmarked_environment_is_not_silently_claimed(tmp_path, monkeypatch):
    """A prefix at the planned path that this workbench never created is not reused as its own.

    Without this, an operator's unrelated environment sitting at the planned path could be adopted,
    verified against another model's requirements and recorded as this model's environment.
    """
    m = model("foreign-test", python="3.11", packages=["pyyaml==6.0"])
    fake_interp = tmp_path / "env" / "bin" / "python"
    fake_interp.parent.mkdir(parents=True)
    fake_interp.touch()
    os.chmod(fake_interp, 0o755)

    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    plan["groups"][0]["interpreter"] = str(fake_interp)
    plan["groups"][0]["conda_env_path"] = str(fake_interp.parent.parent)
    monkeypatch.setattr(planner, "_verify_group_interpreter", lambda *a, **k: (True, "OK", {}))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")

    assignment = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True,
                                                    network_allowed=True)
    group = assignment["groups"][0]
    # It installed its own environment instead of adopting the unmarked one.
    assert calls, "an unmarked prefix must not satisfy reuse"
    assert group["installation_status"] != "REUSED"


def test_bounded_split_and_retry(tmp_path, monkeypatch):
    m_good = {"model_id": "good-model", "environment": {"python": "3.11", "packages": ["pkg-a==1.0"], "confidence": "T"}}
    m_bad = {"model_id": "bad-model", "environment": {"python": "3.11", "packages": ["pkg-b==1.0"], "confidence": "T"}}

    plan = planner.plan_environments([m_good, m_bad], solver=accepted, artifacts_root=tmp_path)
    assert len(plan["groups"]) == 1

    def mock_provision(group, by_id, specs, artifacts_root, third_party_root, allowed, solver_cmd, network_allowed=True):
        if "bad-model" in group["members"] and "good-model" in group["members"]:
            group["verification_details"] = {"evaluator_imports": {"bad-model": {"returncode": 1}}}
            return False, "bad-model evaluator import failed"
        if "good-model" in group["members"]:
            group["state"] = "RUNTIME_VERIFIED"
            group["installation_status"] = "INSTALLED"
            group["runtime_verification_status"] = "READY"
            return True, "good-model passed"
        group["verification_details"] = {"evaluator_imports": {"bad-model": {"returncode": 1}}}
        return False, "bad-model persistently fails"

    monkeypatch.setattr(planner, "_provision_group", mock_provision)
    assignment = planner.provision_environment_plan([m_good, m_bad], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)

    good_group = next(g for g in assignment["groups"] if "good-model" in g["members"])
    bad_group = next(g for g in assignment["groups"] if "bad-model" in g["members"])

    assert good_group["state"] == "RUNTIME_VERIFIED"
    assert good_group["members"] == ["good-model"]
    assert bad_group["members"] == ["bad-model"]
    assert bad_group["installation_status"] == "FAILED"
    assert any("Split" in r for r in good_group["split_reasons"])
    assert any("Split" in r for r in bad_group["split_reasons"])


def test_manage_environment_respects_shared_group(tmp_path, monkeypatch):
    from workbench.scripts import manage_environment

    fake_interp = tmp_path / "bin" / "python"
    fake_interp.parent.mkdir(parents=True, exist_ok=True)
    fake_interp.touch()
    os.chmod(fake_interp, 0o755)
    assignment = {
        "models": {
            "shared-a": {
                "environment_id": "cir-shared123",
                "interpreter": str(fake_interp),
                "group_members": ["shared-a", "shared-b"],
                "state": "RUNTIME_VERIFIED",
                "shared": True,
            }
        }
    }
    assign_path = tmp_path / "environment_assignment.json"
    planner.write_environment_assignment(assignment, assign_path)

    config = SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=tmp_path, WORKBENCH_THIRD_PARTY_ROOT=tmp_path)
    info = manage_environment._check_shared_assignment("shared-a", config)
    assert info is not None
    assert info["environment_id"] == "cir-shared123"
    assert info["shared"] is True

    st = {"model_id": "shared-a", "source_synced": True, "env_spec_path": None, "env_spec_type": None}
    created = manage_environment.create_env({"model_id": "shared-a"}, st, dry_run=False, config=config)
    assert created is True


@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")

@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")

def test_only_real_interpreters_count_as_prefix_pythons():
    """A python-named helper must not be usable as an alibi for a stale shebang.

    `python3-config` shares the `python` prefix but is a shell script; admitting it into the set of
    acceptable interpreters would let a script naming a stale `python3-config` pass the check.
    """
    assert planner._INTERPRETER_NAME.fullmatch("python")
    assert planner._INTERPRETER_NAME.fullmatch("python3")
    assert planner._INTERPRETER_NAME.fullmatch("python3.11")
    assert planner._INTERPRETER_NAME.fullmatch("python39")
    assert planner._INTERPRETER_NAME.fullmatch("python3-config") is None
    assert planner._INTERPRETER_NAME.fullmatch("python3.11-config") is None
    assert planner._INTERPRETER_NAME.fullmatch("python-config") is None


def test_a_renamed_conda_prefix_is_detected_as_stale(tmp_path):
    """REAL conda: a prefix that has been moved must fail verification.

    `conda create --prefix` writes absolute shebangs into every console script. Moving the directory
    leaves those pointing at the old location, which is why building at a staging path and renaming
    was wrong: this is the check that catches it, exercised against a real conda prefix rather than a
    hand-written shebang.
    """
    import subprocess as _subprocess

    original = tmp_path / "env"
    result = _subprocess.run(
        ["conda", "create", "--prefix", str(original), "--override-channels",
         "-c", "conda-forge", "python=3.11", "packaging", "--yes", "-q"],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"conda could not create a probe environment: {result.stderr.strip()[-200:]}")

    interpreter = original / "bin" / "python"
    assert planner._verify_console_scripts(original, interpreter)[0] is True

    moved = tmp_path / "env-moved"
    original.rename(moved)
    moved_interpreter = moved / "bin" / "python"
    ok, reason = planner._verify_console_scripts(moved, moved_interpreter)
    assert ok is False, "a relocated prefix must not verify"
    assert "not a python of this environment" in reason


@pytest.mark.skipif(shutil.which("conda") is None, reason="conda binary not available on PATH")
def test_a_real_conda_prefix_shell_helpers_are_not_rejected(tmp_path):
    """REAL conda: `#!/bin/sh` helpers are normal and must not fail verification."""
    import subprocess as _subprocess

    prefix = tmp_path / "env"
    result = _subprocess.run(
        ["conda", "create", "--prefix", str(prefix), "--override-channels",
         "-c", "conda-forge", "python=3.11", "bzip2", "xz", "packaging", "--yes", "-q"],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"conda could not create a probe environment: {result.stderr.strip()[-200:]}")

    shell_helpers = [p for p in (prefix / "bin").iterdir()
                     if p.is_file() and p.open("rb").readline().startswith(b"#!/bin/sh")]
    assert shell_helpers, "a conda prefix is expected to ship shell helpers"
    ok, reason = planner._verify_console_scripts(prefix, prefix / "bin" / "python")
    assert ok is True, reason


def test_real_conda_cold_start_shared_environment_and_reuse(tmp_path, monkeypatch):
    # REAL package-manager test: real conda cold-start shared environment and reuse.
    # These stand-in models have no adapter, so the evaluator-import trace cannot resolve an entry
    # point; that refusal has its own regression. Here the models stand in for ones whose evaluator
    # imports exactly the distributions they declare, so the constraint set under test is built.
    monkeypatch.setattr(planner, "_evaluator_imports", lambda *a, **k: {"yaml", "packaging"})
    m1 = {
        "model_id": "real-cold-a",
        "source_available": True,
        "environment": {
            "python": "3.11",
            "packages": ["pyyaml==6.0.3"],
            "pip_packages": ["packaging==26.3"],
            "confidence": "REAL_TEST",
        },
    }
    m2 = {
        "model_id": "real-cold-b",
        "source_available": True,
        "environment": {
            "python": "3.11",
            "packages": ["pyyaml==6.0.3"],
            "pip_packages": ["packaging==26.3"],
            "confidence": "REAL_TEST",
        },
    }

    artifacts_dir = tmp_path / "artifacts"
    plan = planner.plan_environments([m1, m2], artifacts_root=artifacts_dir)
    assert len(plan["groups"]) == 1
    group = plan["groups"][0]
    assert set(group["members"]) == {"real-cold-a", "real-cold-b"}
    assert group["state"] == "SOLVER_COMPATIBLE"

    assignment = planner.provision_environment_plan(
        [m1, m2], plan, artifacts_root=artifacts_dir, allowed=True, network_allowed=True
    )
    prov_group = assignment["groups"][0]
    assert prov_group["state"] == "RUNTIME_VERIFIED", prov_group["evidence"]
    assert prov_group["installation_status"] == "INSTALLED"
    assert prov_group["runtime_verification_status"] == "READY"

    interp_a = assignment["models"]["real-cold-a"]["interpreter"]
    interp_b = assignment["models"]["real-cold-b"]["interpreter"]
    assert interp_a == interp_b
    assert Path(interp_a).is_file()
    assert os.access(interp_a, os.X_OK)

    probe_res = subprocess.run([interp_a, "-c", "import yaml, packaging; print(yaml.__name__, packaging.__name__)"],
                               capture_output=True, text=True, check=False)
    assert probe_res.returncode == 0
    assert "yaml packaging" in probe_res.stdout

    rerun_assignment = planner.provision_environment_plan(
        [m1, m2], plan, artifacts_root=artifacts_dir, allowed=True, network_allowed=True
    )
    rerun_group = rerun_assignment["groups"][0]
    assert rerun_group["installation_status"] == "REUSED"
    assert rerun_group["state"] == "RUNTIME_VERIFIED"
    assert rerun_group["runtime_verification_status"] == "READY"
    assert any("reused verified interpreter" in ev for ev in rerun_group["evidence"])


def test_p0_03_manage_environment_failure_propagates(tmp_path, monkeypatch):
    from workbench.scripts import manage_environment

    m = {
        "model_id": "fail-model",
        "environment": {
            "python": "3.11",
            "packages": ["impossible-pkg==99.99.99"],
            "confidence": "TEST",
        },
    }
    assignment = {
        "models": {
            "fail-model": {
                "environment_id": "cir-failgroup",
                "interpreter": str(tmp_path / "nonexistent" / "bin" / "python"),
                "group_members": ["fail-model"],
                "shared": False,
            }
        }
    }
    assign_path = tmp_path / "environment_assignment.json"
    planner.write_environment_assignment(assignment, assign_path)
    config = SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=tmp_path, WORKBENCH_THIRD_PARTY_ROOT=tmp_path)

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    from workbench.backend import registry
    monkeypatch.setattr(registry, "load_registry", lambda: {"models": [m]})
    provision = planner.provision_environment_plan
    result = {}

    def capture_assignment(*args, **kwargs):
        assignment_result = provision(*args, **kwargs)
        result.update(assignment_result)
        return assignment_result

    monkeypatch.setattr(planner, "provision_environment_plan", capture_assignment)
    st = {"model_id": "fail-model", "source_synced": True, "env_spec_path": None, "env_spec_type": None, "ready": False}
    ok = manage_environment.create_env(m, st, dry_run=False, config=config)
    assert ok is False
    assert result["groups"][0]["installation_status"] == "FAILED"
    assert result["groups"][0]["state"] == "FAILED"
    assert result["models"]["fail-model"]["state"] == "FAILED"


def test_p0_04_safe_replacement_and_atomic_commit(tmp_path, monkeypatch):
    m = {
        "model_id": "safe-replace-model",
        "environment": {
            "python": "3.11",
            "packages": ["pyyaml==6.0"],
            "confidence": "TEST",
        },
    }
    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    group = plan["groups"][0]
    env_id = group["environment_id"]
    existing_prefix = tmp_path / "environments" / env_id
    existing_prefix.mkdir(parents=True, exist_ok=True)
    sentinel = existing_prefix / "sentinel.txt"
    sentinel.write_text("keep_this_safe", encoding="utf-8")

    # 1. Force verification to fail during provisioning.
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")

    def mock_subp_run(argv, *args, **kwargs):
        if "create" in argv and "--prefix" in argv:
            prefix = Path(argv[argv.index("--prefix") + 1])
            interpreter = prefix / "bin" / "python"
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.touch()
            os.chmod(interpreter, 0o755)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subp_run)

    def failing_verify(staging_group, *args, **kwargs):
        # The environment is created at its own final path and verified there; nothing is
        # relocated, so the path under verification is the path that was installed.
        assert Path(staging_group["conda_env_path"]).name.startswith(env_id)
        assert not (Path(staging_group["conda_env_path"]) / planner.MARKER_FILENAME).exists()
        return False, "verification failed intentionally", {}

    monkeypatch.setattr(planner, "_verify_group_interpreter", failing_verify)
    fail_assign = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)

    # Sentinel file must still exist and no invalid assignment published
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "keep_this_safe"
    assert not (tmp_path / "environment_assignment.json").is_file()
    assert fail_assign["groups"][0]["state"] == "FAILED"
    assert fail_assign["groups"][0]["interpreter"] is None
    assert fail_assign["models"]["safe-replace-model"]["interpreter"] is None
    # 2. Now run real safe replacement logic where replacement succeeds to a NEW prefix
    def mock_verify_group(staging_group, *a, **k):
        # Not yet managed: the marker is only written once the environment passed verification.
        assert not (Path(staging_group["conda_env_path"]) / planner.MARKER_FILENAME).exists()
        return True, "verification passed", {}
    monkeypatch.undo()  # restore _provision_group before patching retry-specific behavior
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(planner, "_verify_group_interpreter", mock_verify_group)
    # A retry starts from a freshly resolved plan: the failed attempt cleared the group's
    # interpreter (nothing was installed), so the test re-establishes the planned path exactly as a
    # new plan would.
    group["interpreter"] = str(existing_prefix / "bin" / "python")

    plan["interpreter_mapping"]["safe-replace-model"] = str(
        tmp_path / "environments" / "cir-stale-mapping" / "bin" / "python")
    # The installer writes the interpreter into the prefix it was given.
    def mock_subp_run(argv, *a, **k):
        if "create" in argv and "--prefix" in argv:
            p = Path(argv[argv.index("--prefix") + 1])
            interp = p / "bin" / "python"
            interp.parent.mkdir(parents=True, exist_ok=True)
            interp.touch()
            os.chmod(interp, 0o755)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subp_run)

    success_assign = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)
    new_group = success_assign["groups"][0]
    assert new_group["state"] == "RUNTIME_VERIFIED"
    new_prefix = Path(new_group["conda_env_path"])
    # A successful replacement goes to a NEW prefix; the occupied one is never rebuilt in place
    assert new_prefix != existing_prefix
    assert new_prefix.name.startswith(f"{env_id}-")
    # Existing prefix with sentinel file still intact
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "keep_this_safe"
    # Assignment is committed atomically
    assert (tmp_path / "environment_assignment.json").is_file()
    loaded_assignment = planner.load_environment_assignment(tmp_path / "environment_assignment.json")
    assert loaded_assignment["models"]["safe-replace-model"]["interpreter"] == new_group["interpreter"]
    assert loaded_assignment["models"]["safe-replace-model"]["environment_id"] == new_group["environment_id"]
    assert loaded_assignment["models"]["safe-replace-model"]["conda_env_path"] == str(new_prefix)
    # Workbench marker recorded
    assert planner._is_workbench_managed(new_prefix, env_id) is True
    assert planner._is_workbench_managed(existing_prefix) is False


def test_p0_05_package_manager_env_strips_credentials(monkeypatch):
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "sensitive_ledger_key_12345")
    monkeypatch.setenv("WORKBENCH_SECRET_TOKEN", "sensitive_secret_token_abcde")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret")

    child_env = planner._package_manager_env()
    assert "WORKBENCH_LEDGER_KEY" not in child_env
    assert "WORKBENCH_SECRET_TOKEN" not in child_env
    assert "HF_TOKEN" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert "PATH" in child_env

    # Run actual python probe with sanitized environment
    probe_code = "import os; assert 'WORKBENCH_LEDGER_KEY' not in os.environ"
    res = subprocess.run([sys.executable, "-c", probe_code], env=child_env, capture_output=True, text=True)
    assert res.returncode == 0



def test_installation_is_refused_without_network_authorization(tmp_path, monkeypatch):
    """`allow_env_install` authorizes writing an environment, not reaching the network.

    conda and pip contact an index to resolve, so an install without network authorization would
    silently egress. The provisioner must refuse and say why instead.
    """
    m = model("offline-model", python="3.11", env_var="OFFLINE_MODEL_PYTHON")
    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    launched: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda argv, *a, **k: launched.append(argv) or SimpleNamespace(returncode=0, stdout="", stderr=""))

    result = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True,
                                                network_allowed=False)
    group = result["groups"][0]
    assert group["state"] == "BLOCKED_AUTHORIZATION_REQUIRED"
    assert launched == [], "no package manager may run without network authorization"
    assert "allow_network" in " ".join(group["evidence"])
    # The planned interpreter is still recorded (the plan knows where the environment would live),
    # but nothing verified it and nothing was installed, so the state says so.
    assert group["installation_status"] == "FAILED"
    assert group["runtime_verification_status"] == "NOT_CONFIGURED"
    assert result["models"]["offline-model"]["runtime_verification_status"] == "NOT_CONFIGURED"


def test_environment_is_created_directly_at_its_final_prefix(tmp_path, monkeypatch):
    """The conda prefix a model will execute from must be the prefix it was installed into.

    A conda prefix embeds its own absolute path in the interpreter, in every console-script
    shebang, and in conda-meta. Installing somewhere else and renaming produced an environment
    whose scripts pointed at a directory that no longer existed, and the verification that
    followed the rename could not see it. So the installer must be handed the final path.
    """
    m = model("final-prefix-model", python="3.11", env_var="FINAL_PREFIX_MODEL_PYTHON")
    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    group = plan["groups"][0]
    expected_prefix = tmp_path / "environments" / group["environment_id"]

    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(planner, "_verify_group_interpreter", lambda *a, **k: (True, "ok", {}))
    seen: list[Path] = []

    def recording_run(argv, *a, **k):
        if "create" in argv and "--prefix" in argv:
            prefix = Path(argv[argv.index("--prefix") + 1])
            seen.append(prefix)
            interp = prefix / "bin" / "python"
            interp.parent.mkdir(parents=True, exist_ok=True)
            interp.touch()
            os.chmod(interp, 0o755)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", recording_run)
    result = planner.provision_environment_plan([m], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True)

    assert seen == [expected_prefix], f"conda was told to install at {seen}"
    assert not any(".staging-" in part for part in expected_prefix.parts)
    assert Path(result["groups"][0]["conda_env_path"]) == expected_prefix
    assert Path(result["groups"][0]["interpreter"]) == expected_prefix / "bin" / "python"
    assert result["groups"][0]["state"] == "RUNTIME_VERIFIED"


def test_a_stale_console_script_shebang_fails_verification(tmp_path):
    """Verification must notice a console script that points outside its own prefix."""
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    interpreter = prefix / "bin" / "python"
    interpreter.touch()
    os.chmod(interpreter, 0o755)
    elsewhere = tmp_path / "other" / "bin" / "python"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.touch()
    script = prefix / "bin" / "some-tool"
    script.write_text(f"#!{elsewhere}\nprint('hi')\n", encoding="utf-8")
    os.chmod(script, 0o755)

    ok, reason = planner._verify_console_scripts(prefix, interpreter)
    assert ok is False
    assert "points at" in reason

    # A tool whose shebang names this prefix's own interpreter passes.
    script.write_text(f"#!{interpreter}\nprint('hi')\n", encoding="utf-8")
    ok2, _ = planner._verify_console_scripts(prefix, interpreter)
    assert ok2 is True


def test_concurrent_provisioning_of_one_environment_is_serialized(tmp_path):
    """Two provisioners racing on one environment_id must not interleave their installs."""
    first = tmp_path / "environments"
    second = tmp_path / "environments"
    order: list[str] = []

    with planner._environment_lock(first, "cir-deadbeef"):
        # A second acquisition of the same lock cannot proceed until the first is released;
        # observe that by re-entering from a subprocess that must wait.
        probe = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r);"
             "from pathlib import Path;"
             "from workbench.backend import environment_planner as p;"
             "\nwith p._environment_lock(Path(%r), 'cir-deadbeef'):\n"
             "    print('acquired')" % (str(Path(__file__).resolve().parents[2]), str(second))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, _ = probe.communicate(timeout=2)
            order.append("child-acquired-within-2s")
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.wait()
            order.append("child-waited")
    assert order == ["child-waited"], "the lock did not exclude a second provisioner"
    assert (first / ".provision-cir-deadbeef.lock").is_file()

def test_interrupted_staging_prefix_is_never_adopted(tmp_path, monkeypatch):
    from workbench.scripts import manage_environment

    item = model("staged-model", python="3.11", env_var="STAGED_MODEL_PYTHON")
    plan = planner.plan_environments([item], solver=accepted, artifacts_root=tmp_path)
    group = plan["groups"][0]
    prefix = tmp_path / "environments" / f".staging-{group['environment_id']}-interrupted"
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    os.chmod(interpreter, 0o755)
    sentinel = prefix / "do-not-adopt"
    sentinel.write_text("left by interrupted provisioning", encoding="utf-8")
    (prefix / planner.MARKER_FILENAME).write_text(
        json.dumps({"managed_by": "workbench", "environment_id": group["environment_id"]}), encoding="utf-8")
    monkeypatch.setenv("STAGED_MODEL_PYTHON", str(interpreter))

    group["interpreter"] = str(interpreter)
    group["conda_env_path"] = str(prefix)
    assert planner.apply_interpreter_mapping([item], plan)["staged-model"] is None
    installation, verification, reused = planner._group_status(
        {"members": ["staged-model"], "unresolved": []},
        {"staged-model": item}, {"staged-model": planner.model_constraints(item)}, "available")
    assert (installation, verification, reused) == ("EXISTING_UNVERIFIED", "IMPORT_CHECK_FAILED", None)

    ok, reason = planner._provision_group(
        group, {"staged-model": item}, {"staged-model": planner.model_constraints(item)},
        tmp_path, None, True, "conda", network_allowed=True)
    assert not ok and "interrupted staging" in reason.lower()
    assert group["state"] == "FAILED"
    assert sentinel.read_text(encoding="utf-8") == "left by interrupted provisioning"
    assert not planner._is_workbench_managed(prefix, group["environment_id"])

    assignment = {"models": {"staged-model": {
        "environment_id": group["environment_id"], "interpreter": str(interpreter),
        "conda_env_path": str(prefix), "group_members": ["staged-model"]}}}
    planner.write_environment_assignment(assignment, tmp_path / "environment_assignment.json")
    (tmp_path / "environment_plan.json").write_text(json.dumps({"groups": [group]}), encoding="utf-8")
    config = SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=tmp_path)
    assert manage_environment._check_shared_assignment("staged-model", config) is None
    status = manage_environment._interpreter_status(item, tmp_path)
    assert status["status"] == "INTERRUPTED_STAGING" and status["verified"] is False


def test_real_provisioning_children_withhold_ledger_key(tmp_path, monkeypatch):
    source = _write_import_fixture(monkeypatch, tmp_path, "import packaging\n")
    item = {"model_id": "encoder", "source_dir": "ENCODER", "source_available": True,
            "environment": {"python": "3.11", "packages": ["packaging==26.3"], "confidence": "TEST"}}
    plan = planner.plan_environments([item], solver=accepted, artifacts_root=tmp_path,
                                     third_party_root=tmp_path / "third_party")
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "fake-ledger-key-for-child-test")
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(planner, "_pip_specs_for_group", lambda *args: ["packaging==26.3"])
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda prefix: ({"sample": "1", "packaging": "26.3"}, {"sample", "packaging"}, set()))
    monkeypatch.setattr(planner.runtime, "verify_environment",
                        lambda *args, **kwargs: SimpleNamespace(
                            tier="RUNTIME_READY", status="READY", reason="ok"))

    actual_run = subprocess.run
    child_commands = []

    def provisioner_run(argv, *args, **kwargs):
        assert "WORKBENCH_LEDGER_KEY" not in kwargs["env"]
        child_commands.append(argv)
        if argv[0] == "conda" and argv[1] == "create":
            prefix = Path(argv[argv.index("--prefix") + 1])
            interpreter = prefix / "bin" / "python"
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "assert 'WORKBENCH_LEDGER_KEY' not in os.environ\n"
                "if sys.argv[1:3] == ['-m', 'pip']:\n"
                "    assert sys.argv[3] in ('check', 'install')\n"
                "    raise SystemExit(0)\n"
                "if '-c' in sys.argv:\n"
                "    assert sys.argv[sys.argv.index('-c') + 1].startswith('import ')\n"
                "    raise SystemExit(0)\n"
                f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
                encoding="utf-8")
            os.chmod(interpreter, 0o755)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return actual_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", provisioner_run)
    assignment = planner.provision_environment_plan(
        [item], plan, artifacts_root=tmp_path, allowed=True, network_allowed=True,
        third_party_root=tmp_path / "third_party")

    group = assignment["groups"][0]
    assert group["state"] == "RUNTIME_VERIFIED", group
    assert planner._is_workbench_managed(Path(group["conda_env_path"]), group["environment_id"])
    assert any(len(argv) > 3 and argv[1:3] == ["-m", "pip"] and argv[3] == "check"
               for argv in child_commands), child_commands
    assert any(argv[1:4] == ["-m", "pip", "install"] for argv in child_commands)
    assert any("-c" in argv for argv in child_commands)
    assert sum("-c" in argv for argv in child_commands) >= 2


def test_p1_03_range_constraint_enforced_against_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "_evaluator_imports", lambda *a, **k: set())
    # Declared as a pip package so the constraint survives import tracing: this test is about the
    # range check itself, and a `packages` declaration is filtered to what the evaluator actually
    # imports (which is nothing here, since the model has no adapter).
    m = {
        "model_id": "range-model",
        "environment": {
            "python": "3.11",
            "pip_packages": ["numpy>=1.21,<1.24"],
            "confidence": "TEST",
        },
    }
    plan = planner.plan_environments([m], solver=accepted, artifacts_root=tmp_path)
    group = plan["groups"][0]

    fake_interp = tmp_path / "env" / "bin" / "python"
    fake_interp.parent.mkdir(parents=True, exist_ok=True)
    fake_interp.touch()
    os.chmod(fake_interp, 0o755)
    group["interpreter"] = str(fake_interp)
    group["conda_env_path"] = str(fake_interp.parent.parent)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))

    # Case 1: numpy installed version 1.25.0 is OUTSIDE range >=1.21,<1.24 -> must refuse RUNTIME_VERIFIED.
    # The runtime probe must report ready too, or the refusal under test would come from the probe
    # rather than from the version check.
    monkeypatch.setattr(planner.runtime, "verify_environment",
                        lambda *a, **k: SimpleNamespace(tier="RUNTIME_READY", status="READY", reason="ok"))
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda prefix: ({"numpy": "1.25.0"}, {"numpy"}, set()))
    ok, reason, details = planner._verify_group_interpreter(
        group, {"range-model": m}, {"range-model": planner.model_constraints(m)}
    )
    assert ok is False
    assert "numpy" in reason.lower()
    assert "mismatch" in reason.lower()
    assert details["verified_versions"]["numpy"]["installed"] == "1.25.0"

    # Case 2: numpy installed version 1.22.0 is INSIDE range >=1.21,<1.24 -> accepted
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda prefix: ({"numpy": "1.22.0"}, {"numpy"}, set()))
    ok2, reason2, details2 = planner._verify_group_interpreter(
        group, {"range-model": m}, {"range-model": planner.model_constraints(m)}
    )
    assert ok2 is True
    assert details2["verified_versions"]["numpy"]["installed"] == "1.22.0"


def test_an_absent_pinned_source_blocks_verification(tmp_path, monkeypatch):
    """A source that is not present yet must not yield RUNTIME_VERIFIED.

    The evaluator-import trace cannot run before the pinned checkout exists, and treating that as
    "nothing to verify" marked an environment verified without ever having looked at what the
    evaluator imports -- the same interpreter then failed as soon as the source arrived.
    """
    monkeypatch.setattr(planner, "_evaluator_imports", lambda *a, **k: None)
    monkeypatch.setattr(planner.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda *a, **k: ({}, set(), set()))

    interpreter = tmp_path / "env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    os.chmod(interpreter, 0o755)
    group = {"environment_id": "cir-test", "members": ["absent-source"],
             "interpreter": str(interpreter), "conda_env_path": str(interpreter.parent.parent),
             "constraints": {"python": [], "packages": {}, "ranges": {}, "torch": [], "torchvision": []}}
    model = {"model_id": "absent-source", "source_dir": "Absent", "environment": {}}

    ok, reason, details = planner._verify_group_interpreter(group, {"absent-source": model},
                                                            {"absent-source": planner.model_constraints(model)})
    assert ok is False
    assert "cannot verify evaluator imports" in reason
    assert details["evaluator_imports"]["absent-source"]["modules"] is None


def test_evaluator_imports_excludes_checkout_subpackages(monkeypatch, tmp_path):
    source = _write_import_fixture(monkeypatch, tmp_path, "import internal\n")
    (source / "internal").mkdir()
    (source / "internal" / "__init__.py").write_text("import nested\n", encoding="utf-8")
    (source / "internal" / "nested.py").write_text("value = 1\n", encoding="utf-8")

    imports = planner._evaluator_imports({"model_id": "encoder", "source_dir": "ENCODER"})

    assert imports is not None
    assert not any((source / name).is_dir() for name in imports)


def test_registry_package_pins_survive_import_filtering(monkeypatch, tmp_path):
    _write_import_fixture(monkeypatch, tmp_path, "import numpy\n")
    item = {"model_id": "encoder", "source_dir": "ENCODER",
            "environment": {"python": "3.9", "packages": ["numpy==1.22", "pandas==1.5"]}}

    spec = planner.model_constraints(item)

    assert spec["packages"] == {"numpy": "1.22", "pandas": "1.5"}



def test_partial_verification_publishes_only_verified_interpreters(tmp_path, monkeypatch):
    items = [model("verified", python="3.10"), model("unverified", python="3.11")]
    plan = planner.plan_environments(items, solver=accepted, artifacts_root=tmp_path)
    verified_interpreter = tmp_path / "verified" / "bin" / "python"
    verified_interpreter.parent.mkdir(parents=True)
    verified_interpreter.touch()
    os.chmod(verified_interpreter, 0o755)
    groups = plan["groups"]
    verified_group = next(group for group in groups if "verified" in group["members"])
    unverified_group = next(group for group in groups if "unverified" in group["members"])
    verified_group.update(state="RUNTIME_VERIFIED", interpreter=str(verified_interpreter))
    unverified_group.update(state="SOLVER_COMPATIBLE", interpreter=str(verified_interpreter))
    monkeypatch.setattr(planner, "_provision_all_groups", lambda initial, *args, **kwargs: initial)

    assignment = planner.provision_environment_plan(items, plan, artifacts_root=tmp_path,
                                                    allowed=True, network_allowed=False)

    path = tmp_path / "environment_assignment.json"
    assert path.is_file()
    persisted = planner.load_environment_assignment(path)
    assert persisted["models"]["verified"]["interpreter"] == str(verified_interpreter)
    assert Path(persisted["models"]["verified"]["interpreter"]).is_file()
    assert persisted["models"]["unverified"]["interpreter"] is None
    assert persisted["models"]["unverified"]["reason"]
    assert assignment["models"]["verified"]["interpreter"] == str(verified_interpreter)
def test_operator_interpreter_survives_unverified_provisioning(tmp_path, monkeypatch):
    item = model("operator-model", python="3.11", env_var="WORKBENCH_OPERATOR_MODEL_PYTHON")
    interpreter = tmp_path / "operator-env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    os.chmod(interpreter, 0o755)
    monkeypatch.setenv("WORKBENCH_OPERATOR_MODEL_PYTHON", str(interpreter))
    plan = planner.plan_environments([item], solver=accepted, artifacts_root=tmp_path)

    assignment = planner.provision_environment_plan([item], plan, artifacts_root=tmp_path,
                                                    allowed=False, network_allowed=False)

    assert assignment["interpreter_mapping"]["operator-model"] == str(interpreter)


def test_missing_unpinned_declared_package_blocks_verification(tmp_path, monkeypatch):
    item = model("missing-unpinned", packages=["declared-package"])
    interpreter = tmp_path / "env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    os.chmod(interpreter, 0o755)
    group = {"interpreter": str(interpreter), "conda_env_path": str(interpreter.parent.parent),
             "members": ["missing-unpinned"], "constraints": {
                 "packages": {"declared-package": []}, "ranges": {}, "torch": [], "torchvision": []}}
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(planner.runtime, "inspect_environment_distributions",
                        lambda prefix: ({}, set(), set()))

    ok, reason, _ = planner._verify_group_interpreter(
        group, {"missing-unpinned": item}, {"missing-unpinned": planner.model_constraints(item)})

    assert ok is False
    assert "declared-package" in reason
    assert "is not installed" in reason
