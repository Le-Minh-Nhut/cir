"""A32: provisioning exit-code truthfulness and cross-scope assignment preservation.

The planner is driven through its real entry points (`main`, `provision_environment_plan`,
`write_environment_assignment`, `runtime.resolve_model_interpreter`); only the provisioning
internals that would touch a package manager are replaced, so the assignment writer, loader and
resolver under test are the real ones.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.backend import environment_planner as planner
from workbench.backend import model_settings
from workbench.backend import registry as registry_module
from workbench.backend import runtime

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN_SCRIPT = REPOSITORY / "workbench" / "scripts" / "plan_environments.py"


def _model(model_id: str, *, python: str = "3.11", var: str | None = None) -> dict:
    return {"model_id": model_id, "source_available": True, "source_dir": model_id.upper(),
            "environment": {"python_env_var": var or f"WORKBENCH_{model_id.upper()}_PYTHON",
                            "python": python, "packages": [f"{model_id}-dep==1.0"],
                            "confidence": "TEST_CONTRACT"}}


def _registry(models: list[dict]):
    return SimpleNamespace(load_registry=lambda: {"models": models})


def _config(root: Path):
    return SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=root)


def _fake_interpreter(root: Path, name: str) -> Path:
    interp = root / "environments" / name / "bin" / "python"
    interp.parent.mkdir(parents=True, exist_ok=True)
    interp.write_text("#!/bin/sh\n", encoding="utf-8")
    interp.chmod(0o755)
    return interp


def _stub_provisioner(by_member: dict[str, Path]):
    """A provisioner that verifies whatever group it is given, without installing anything."""

    def provision(group, *_args, **_kwargs):
        interp = next(path for mid, path in by_member.items() if mid in group["members"])
        group.update(interpreter=str(interp), conda_env_path=str(interp.parent.parent),
                     installation_status="INSTALLED", runtime_verification_status="READY",
                     state="RUNTIME_VERIFIED")
        return True, "stub provisioner"

    return provision


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No operator setting and no solver cache may leak in from the host."""
    for name in list(os.environ):
        if name.startswith("WORKBENCH_") and name.endswith("_PYTHON"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_settings, "file_settings", lambda: {})
    planner._SOLVE_CACHE.clear()
    yield
    planner._SOLVE_CACHE.clear()


# ------------------------------------------------------------------ A32-01

def test_provisioning_failure_exits_nonzero_and_keeps_json(monkeypatch, tmp_path, capsys):
    """A requested provisioning run with no solver must not report success."""
    model = _model("alpha-model")
    monkeypatch.setattr(registry_module, "load_registry", _registry([model]).load_registry)
    monkeypatch.setattr(planner, "_solver_command", lambda: None)

    out = tmp_path / "environment_plan.json"
    code = planner.main(["--provision", "--allow-network", "--json", "--model", "alpha-model",
                         "--artifacts-root", str(tmp_path), "--output", str(out)])

    assert code != 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["groups"][0]["state"] == "FAILED"
    assert plan["groups"][0]["installation_status"] == "FAILED"
    assert plan["groups"][0]["runtime_verification_status"] == "FAILED"
    assert "no conda/mamba/micromamba" in json.dumps(plan["groups"][0]["unresolved"]).lower()
    assert "did not complete" in capsys.readouterr().err


def test_provisioning_success_exits_zero(monkeypatch, tmp_path):
    model = _model("alpha-model")
    monkeypatch.setattr(registry_module, "load_registry", _registry([model]).load_registry)
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group",
                        _stub_provisioner({"alpha-model": _fake_interpreter(tmp_path, "cir-alpha")}))

    out = tmp_path / "environment_plan.json"
    code = planner.main(["--provision", "--allow-network", "--json", "--model", "alpha-model",
                         "--artifacts-root", str(tmp_path), "--output", str(out)])

    assert code == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["groups"][0]["state"] == "RUNTIME_VERIFIED"
    assert (tmp_path / "environment_assignment.json").is_file()


def test_group_refused_at_install_exits_nonzero(monkeypatch, tmp_path):
    """A group that refuses to install is unfinished requested work, not a pass.

    The refusal shape mirrors `_provision_group`'s authorization branch; that branch's own
    semantics are covered by the authorization-policy tests, so what this test measures is that a
    non-verifying group in the requested scope reaches the exit code.
    """

    def refuse(group, *_args, **_kwargs):
        group.update(installation_status="FAILED", runtime_verification_status="NOT_CONFIGURED",
                     state="BLOCKED_AUTHORIZATION_REQUIRED")
        return False, "environment installation requires network authorization (allow_network)"

    model = _model("alpha-model")
    monkeypatch.setattr(registry_module, "load_registry", _registry([model]).load_registry)
    monkeypatch.setattr(planner, "_solver_command", lambda: "conda")
    monkeypatch.setattr(planner, "_solve", lambda *_a, **_k: (True, "accepted", True))
    monkeypatch.setattr(planner, "_provision_group", refuse)

    out = tmp_path / "environment_plan.json"
    code = planner.main(["--provision", "--json", "--model", "alpha-model",
                         "--artifacts-root", str(tmp_path), "--output", str(out)])

    assert code != 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["groups"][0]["state"] == "BLOCKED_AUTHORIZATION_REQUIRED"


def test_plan_only_and_list_stay_zero(monkeypatch, tmp_path):
    """Planning without a provisioning request keeps its read-only contract and exit code."""
    model = _model("alpha-model")
    monkeypatch.setattr(registry_module, "load_registry", _registry([model]).load_registry)
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    out = tmp_path / "environment_plan.json"

    assert planner.main(["--json", "--model", "alpha-model", "--artifacts-root", str(tmp_path),
                         "--output", str(out)]) == 0
    assert planner.main(["--list", "--model", "alpha-model", "--artifacts-root", str(tmp_path),
                         "--output", str(out)]) == 0


def test_unrelated_blocked_model_cannot_fail_a_successful_request(monkeypatch, tmp_path):
    """A model outside the selection being blocked must not fail the scoped request."""
    good = _model("alpha-model")
    blocked = {"model_id": "blocked-model", "source_available": False, "source_dir": None,
               "environment": {"python_env_var": "WORKBENCH_BLOCKED_MODEL_PYTHON",
                               "confidence": "TEST_CONTRACT"}}
    monkeypatch.setattr(registry_module, "load_registry", _registry([good, blocked]).load_registry)
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group",
                        _stub_provisioner({"alpha-model": _fake_interpreter(tmp_path, "cir-alpha")}))

    out = tmp_path / "environment_plan.json"
    code = planner.main(["--provision", "--allow-network", "--json", "--model", "alpha-model",
                         "--artifacts-root", str(tmp_path), "--output", str(out)])

    assert code == 0, out.read_text(encoding="utf-8")[:2000]


def test_cli_boundary_exit_code_is_observable_without_a_solver(tmp_path):
    """Process-level boundary: the exit code an operator or automation actually sees."""
    environment = {key: value for key, value in os.environ.items()
                   if not (key.startswith("WORKBENCH_") and key.endswith("_PYTHON"))}
    environment.update({"PATH": "/usr/bin:/bin", "WORKBENCH_ARTIFACTS_ROOT": str(tmp_path)})
    for solver in ("conda", "mamba", "micromamba"):
        assert subprocess.run(["which", solver], capture_output=True,
                              env=environment, check=False).returncode != 0, \
            "this boundary test requires a host PATH without conda/mamba/micromamba"

    result = subprocess.run(
        [sys.executable, str(PLAN_SCRIPT), "--provision", "--allow-network", "--json",
         "--model", "encoder", "--artifacts-root", str(tmp_path),
         "--output", str(tmp_path / "environment_plan.json")],
        capture_output=True, text=True, env=environment, cwd=REPOSITORY, timeout=300, check=False)

    assert result.returncode != 0, result.stdout
    plan = json.loads((tmp_path / "environment_plan.json").read_text(encoding="utf-8"))
    assert plan["groups"], result.stderr
    assert plan["groups"][0]["installation_status"] == "FAILED"


# ------------------------------------------------------------------ A32-02

def _provision(models: list[dict], root: Path, *, keep_id: str | None = None) -> dict:
    plan = planner.plan_environments(models, artifacts_root=root)
    if keep_id is not None:
        for group in plan.get("groups") or []:
            group["environment_id"] = keep_id
        plan["groups_by_id"] = {group["environment_id"]: group
                                for group in plan.get("groups") or []}
    return planner.provision_environment_plan(models, plan, artifacts_root=root, allowed=True,
                                              network_allowed=True)


def test_second_scope_keeps_the_first_scope_resolvable(monkeypatch, tmp_path):
    """Provisioning B must not drop A's interpreter route from the shared assignment."""
    limn, encoder = _model("limn"), _model("encoder")
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group",
                        _stub_provisioner({"limn": _fake_interpreter(tmp_path, "cir-limn"),
                                           "encoder": _fake_interpreter(tmp_path, "cir-enc")}))

    _provision([limn], tmp_path)
    first = runtime.resolve_model_interpreter(limn, _config(tmp_path))
    assert first is not None, "the first scope must publish a resolvable interpreter"

    _provision([encoder], tmp_path)

    assert runtime.resolve_model_interpreter(limn, _config(tmp_path)) == first
    assert runtime.resolve_model_interpreter(encoder, _config(tmp_path)) is not None
    assignment = planner.load_environment_assignment(tmp_path / "environment_assignment.json")
    assert set(assignment["models"]) == {"limn", "encoder"}
    assert set(assignment["interpreter_mapping"]) == {"limn", "encoder"}
    # No model may name a group the document does not carry.
    assert {info["environment_id"] for info in assignment["models"].values()} <= set(assignment["groups_by_id"])


def test_reprovisioning_a_scope_keeps_the_other_scope(monkeypatch, tmp_path):
    limn, encoder = _model("limn"), _model("encoder")
    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group",
                        _stub_provisioner({"limn": _fake_interpreter(tmp_path, "cir-limn"),
                                           "encoder": _fake_interpreter(tmp_path, "cir-enc")}))
    _provision([limn], tmp_path)
    _provision([encoder], tmp_path)

    _provision([limn], tmp_path)          # a valid reuse of the first scope

    assert runtime.resolve_model_interpreter(encoder, _config(tmp_path)) is not None
    assert runtime.resolve_model_interpreter(limn, _config(tmp_path)) is not None


def test_failed_repair_does_not_leave_a_stale_verified_route(monkeypatch, tmp_path):
    """A failed requested repair must not keep advertising the old verified interpreter."""
    limn, encoder = _model("limn"), _model("encoder")
    shared = _fake_interpreter(tmp_path, "cir-shared")
    group = {"environment_id": "cir-shared", "members": ["encoder", "limn"],
             "conda_env_path": str(shared.parent.parent), "interpreter": str(shared),
             "state": "RUNTIME_VERIFIED", "installation_status": "INSTALLED",
             "runtime_verification_status": "READY", "group_members": ["encoder", "limn"]}
    planner.write_environment_assignment({
        "schema_version": 1, "artifacts_root": str(tmp_path), "groups": [group],
        "groups_by_id": {"cir-shared": group},
        "models": {mid: {"environment_id": "cir-shared", "interpreter": str(shared),
                         "conda_env_path": str(shared.parent.parent), "state": "RUNTIME_VERIFIED",
                         "installation_status": "INSTALLED", "runtime_verification_status": "READY",
                         "shared": True, "group_members": ["encoder", "limn"]}
                   for mid in ("encoder", "limn")},
        "interpreter_mapping": {"encoder": str(shared), "limn": str(shared)},
        "blocked": [], "plan": {},
    }, tmp_path / "environment_assignment.json")

    def failing(group_, *_args, **_kwargs):
        group_.update(installation_status="FAILED", runtime_verification_status="FAILED",
                      state="FAILED")
        return False, "requested repair failed"

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group", failing)

    _provision([limn], tmp_path)

    assignment = planner.load_environment_assignment(tmp_path / "environment_assignment.json")
    assert assignment["models"]["limn"]["state"] == "FAILED"
    assert runtime.resolve_model_interpreter(limn, _config(tmp_path)) is None
    # The unrelated member of the shared group keeps the environment it still has.
    assert assignment["models"]["encoder"]["state"] == "RUNTIME_VERIFIED"
    encoder_group = assignment["groups_by_id"][assignment["models"]["encoder"]["environment_id"]]
    assert encoder_group["members"] == ["encoder"]
    assert assignment["models"]["encoder"]["group_members"] == ["encoder"]
    assert not [mid for mid, info in assignment["models"].items()
                if info["environment_id"] not in assignment["groups_by_id"]]


def test_total_failure_still_rewrites_and_carries_untouched_models(monkeypatch, tmp_path):
    """Every requested group failing must still zero the failed route without dropping others.

    The widened publish gate writes in this case on purpose: the previous document advertised a
    verified interpreter for a group this run could not verify. What must not happen is the write
    discarding models the call never touched, or leaving a model citing a group that is gone.
    """
    other, target = _model("other-model"), _model("target-model")
    survivor = _fake_interpreter(tmp_path, "cir-other")
    other_group = {"environment_id": "cir-other", "members": ["other-model"],
                   "conda_env_path": str(survivor.parent.parent), "interpreter": str(survivor),
                   "state": "RUNTIME_VERIFIED", "installation_status": "INSTALLED",
                   "runtime_verification_status": "READY", "group_members": ["other-model"]}
    target_group = {"environment_id": "cir-target", "members": ["target-model"],
                    "conda_env_path": str(survivor.parent.parent), "interpreter": str(survivor),
                    "state": "RUNTIME_VERIFIED", "installation_status": "INSTALLED",
                    "runtime_verification_status": "READY", "group_members": ["target-model"]}
    planner.write_environment_assignment({
        "schema_version": 1, "artifacts_root": str(tmp_path),
        "groups": [other_group, target_group],
        "groups_by_id": {"cir-other": other_group, "cir-target": target_group},
        "models": {"other-model": {**other_group, "shared": False},
                   "target-model": {**target_group, "shared": False}},
        "interpreter_mapping": {"other-model": str(survivor), "target-model": str(survivor)},
        "blocked": [], "plan": {},
    }, tmp_path / "environment_assignment.json")

    def failing(group_, *_args, **_kwargs):
        group_.update(installation_status="FAILED", runtime_verification_status="FAILED",
                      state="FAILED")
        return False, "provisioning failed"

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group", failing)

    _provision([target], tmp_path)

    assignment = planner.load_environment_assignment(tmp_path / "environment_assignment.json")
    assert assignment["models"]["target-model"]["state"] == "FAILED"
    assert assignment["interpreter_mapping"]["target-model"] is None
    assert runtime.resolve_model_interpreter(target, _config(tmp_path)) is None
    # The untouched model survives verbatim, with its group and route intact.
    assert assignment["models"]["other-model"]["state"] == "RUNTIME_VERIFIED"
    assert assignment["groups_by_id"]["cir-other"]["members"] == ["other-model"]
    assert runtime.resolve_model_interpreter(other, _config(tmp_path)) is not None
    # Every model still cites a group that exists, and ids are unique.
    ids = [group["environment_id"] for group in assignment["groups"]]
    assert len(ids) == len(set(ids))
    assert assignment["groups_by_id"] == {group["environment_id"]: group
                                          for group in assignment["groups"]}
    assert not [mid for mid, info in assignment["models"].items()
                if info["environment_id"] not in assignment["groups_by_id"]]


def test_reusing_a_group_id_does_not_duplicate_members(monkeypatch, tmp_path):
    """A repair that lands on the same content-derived environment id merges membership once.

    The owner's members all come from this call, and the carried-over members are by definition
    outside it, so the concatenation cannot repeat a model. This pins that invariant.
    """
    other, target = _model("other-model"), _model("target-model")
    shared = _fake_interpreter(tmp_path, "cir-shared")
    group = {"environment_id": "cir-shared", "members": ["other-model", "target-model"],
             "conda_env_path": str(shared.parent.parent), "interpreter": str(shared),
             "state": "RUNTIME_VERIFIED", "installation_status": "INSTALLED",
             "runtime_verification_status": "READY", "group_members": ["other-model", "target-model"]}
    planner.write_environment_assignment({
        "schema_version": 1, "artifacts_root": str(tmp_path), "groups": [group],
        "groups_by_id": {"cir-shared": group},
        "models": {mid: {**group, "shared": True} for mid in ("other-model", "target-model")},
        "interpreter_mapping": {mid: str(shared) for mid in ("other-model", "target-model")},
        "blocked": [], "plan": {},
    }, tmp_path / "environment_assignment.json")

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "_provision_group",
                        _stub_provisioner({"target-model": shared}))

    # Provisioning target-model alone forms cir-shared again (same content-derived id).
    _provision([target], tmp_path, keep_id="cir-shared")

    assignment = planner.load_environment_assignment(tmp_path / "environment_assignment.json")
    for member in assignment["groups"]:
        assert len(member["members"]) == len(set(member["members"])), member
    assert len([g for g in assignment["groups"] if g["environment_id"] == "cir-shared"]) == 1
    assert set(assignment["models"]) == {"other-model", "target-model"}
    assert runtime.resolve_model_interpreter(other, _config(tmp_path)) is not None
    assert runtime.resolve_model_interpreter(target, _config(tmp_path)) is not None
