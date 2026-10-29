"""A30 regressions: resolved-contract verification, selection lookup, stale assignments.

Three independent defects, one theme: a persisted or raw record was allowed to outvote the CURRENT
resolved contract. These tests drive the production paths against real interpreters.

- A30-01: `verify_environment` checked the interpreter's Python against the HISTORICAL author pin
  even when the operator selected a declared adaptation whose resolved contract demands a different
  Python (encoder: historical 3.9, adapted 3.10).
- A30-02: `manage_environment.create_env` read `os.environ` directly (a file-only setting was
  invisible) and installed the raw historical freeze instead of the resolved constraints.
- A30-03: a persisted `state == "RUNTIME_VERIFIED"` assignment was reused without re-verifying, so
  `status` reported ready over a broken environment and `--create` refused to repair it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.registry import model_by_id

REPO = Path(__file__).resolve().parents[2]
_ENCODER_FREEZE = REPO / "workbench" / "third_party" / "ENCODER" / "requirements.txt"
_TME_FREEZE = REPO / "workbench" / "third_party" / "TME" / "requirements.txt"
_HOST_SERIES = f"{sys.version_info.major}.{sys.version_info.minor}"

#: A real CPython 3.10. The host interpreter comes first: the audit's venv cannot itself create a
#: nested venv whose identity probe succeeds, so it is only a fallback for the direct-probe case.
_PY310 = next((candidate for candidate in (
    Path("/usr/local/bin/python3.10"),
    Path("/tmp/a30-verify/venv310/bin/python"),
) if candidate.exists() and os.access(candidate, os.X_OK)), None)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Every test starts with no adaptation selected and no file settings.

    The operator file itself is repointed so a developer's real `workbench/config/workbench.env`
    (which sets WORKBENCH_*_PYTHON) cannot leak into these tests; `tmp_path` is this test's own
    directory, and the tests that need a file write to `tmp_path/"workbench.env"`.
    """
    from workbench.backend import model_settings, operator_config

    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", tmp_path / "workbench.env")
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()
    yield
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()


def _venv(root: Path, name: str) -> Path:
    """A real isolated venv running this suite's interpreter."""
    import venv as _venv_module

    env_dir = root / name
    _venv_module.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(env_dir)
    return env_dir / "bin" / "python"


def _python310(root: Path, name: str) -> Path:
    """A real venv whose interpreter IS CPython 3.10 — the adapted encoder/TME contract.

    `venv.EnvBuilder` always bases the environment on `sys.executable`, which here is the 3.13
    orchestrator, so the 3.10 interpreter is invoked explicitly.
    """
    if _PY310 is None:
        pytest.skip("no CPython 3.10 available on this host")
    import subprocess

    env_dir = root / name
    result = subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(env_dir)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not build a CPython 3.10 venv: {result.stderr.strip()[-200:]}")
    return env_dir / "bin" / "python"


def _config(tmp_path: Path):
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config

    third_party = tmp_path / "third_party"
    third_party.mkdir(parents=True, exist_ok=True)
    return replace(resolve_config(), WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
                   WORKBENCH_THIRD_PARTY_ROOT=third_party)


def _manage(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, REPO / "workbench" / "scripts" / "manage_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_stub(interpreter: Path, name: str, version: str) -> None:
    """A real, importable distribution in the venv's own site-packages."""
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        module = site_packages / f"{name}.py"
        module.write_text(f"__version__ = {version!r}\n")
        dist = site_packages / f"{name}-{version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (dist / "RECORD").write_text(
            f"{name}.py,sha256=test,{module.stat().st_size}\n"
            f"{name}-{version}.dist-info/METADATA,sha256=test,60\n")
        return
    raise AssertionError("no site-packages directory in the test environment")


# --------------------------------------------------------------------------- A30-01


@pytest.mark.parametrize("model_id,resolved_python", [
    ("csmcir", "3.10"),
    ("conesep", "3.9"),
    ("encoder", "3.10"),
])
def test_a30_01_adapted_contract_is_checked_against_the_resolved_python(
        model_id, resolved_python, tmp_path, monkeypatch):
    """With an adaptation selected, the python check must use the RESOLVED contract's python.

    Historical and resolved differ for every model here, so the refusal (when the interpreter does
    not match) must NAME the resolved series. Naming the historical one is the defect.
    """
    interpreter = _venv(tmp_path, f"{model_id}_env")
    model = model_by_id(model_id)
    historical_python = model["environment"]["python"]
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    monkeypatch.setenv(model["environment"]["python_env_var"], str(interpreter))

    resolved = runtime.resolved_runtime_environment(model)
    assert resolved["python"] == resolved_python != historical_python
    assert runtime._interpreter_identity(interpreter)["version"].split(".")[:2] == _HOST_SERIES.split(".")

    report = runtime.verify_environment(model, probe=True)
    if _HOST_SERIES == resolved_python:
        # The resolved contract is satisfied: the check must NOT refuse this interpreter.
        assert report.status != "PYTHON_VERSION_MISMATCH", report.reason
    else:
        assert report.status == "PYTHON_VERSION_MISMATCH"
        assert f"expected Python {resolved_python}" in report.reason, report.reason
        assert f"expected Python {historical_python}" not in report.reason, report.reason


@pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")
def test_a30_01_real_3_10_interpreter_reaches_dependency_checks(tmp_path, monkeypatch):
    """The audit's reproduction: encoder + sm_120 + a real CPython 3.10 must pass the python step.

    Missing dependencies are the EXPECTED outcome on an empty venv — a python mismatch is not.
    """
    interpreter = _python310(tmp_path, "encoder310")
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", str(interpreter))
    report = runtime.verify_environment(model_by_id("encoder"), probe=True)
    assert report.status != "PYTHON_VERSION_MISMATCH", report.reason
    assert report.status.startswith("DEPENDENCY") or report.tier in runtime.READY_TIERS, report.reason
    assert report.python_version.startswith("3.10.")


@pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")
def test_a30_01_historical_contract_still_rejects_a_3_10_interpreter(tmp_path, monkeypatch):
    """Unset adaptation keeps the historical contract: a 3.10 interpreter is genuinely wrong."""
    interpreter = _python310(tmp_path, "historical_mismatch")
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", str(interpreter))
    model = model_by_id("encoder")
    assert runtime.resolved_runtime_environment(model)["python"] == "3.9"
    report = runtime.verify_environment(model, probe=True)
    assert report.status == "PYTHON_VERSION_MISMATCH"
    assert report.tier == "BLOCKED"
    assert "expected Python 3.9" in report.reason


def test_a30_01_no_adaptation_uses_the_historical_declaration_exactly(tmp_path, monkeypatch):
    """With no adaptation the resolved block IS the historical block — no behaviour change."""
    interpreter = _venv(tmp_path, "historical_env")
    model = model_by_id("encoder")
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", str(interpreter))
    resolved = runtime.resolved_runtime_environment(model)
    assert resolved["python"] == model["environment"]["python"] == "3.9"
    report = runtime.verify_environment(model, probe=True)
    assert report.status == "PYTHON_VERSION_MISMATCH"
    assert "expected Python 3.9" in report.reason


# --------------------------------------------------------------------------- A30-02


def _create_dry_run(module, model, env_var, spec, config):
    status = {"model_id": model["model_id"], "source_synced": True, "env_var": env_var,
              "env_spec_path": str(spec), "env_spec_type": "requirements.txt", "ready": False}
    return module.create_env(model, status, dry_run=True, config=config, network_allowed=True)


def test_a30_02_file_only_selector_takes_the_same_path_as_an_export(tmp_path, monkeypatch, capsys):
    """A file-only `workbench.env` selector must select the venv branch, exactly like an export."""
    from workbench.backend import model_settings, operator_config

    module = _manage("manage_environment_a30_preset")
    config = _config(tmp_path)
    interpreter = _venv(tmp_path, "preset")
    var = "WORKBENCH_ENCODER_PYTHON"
    spec = tmp_path / "requirements.txt"
    spec.write_text("some-package==1.0\n", encoding="utf-8")
    model = model_by_id("encoder")

    planned = []
    monkeypatch.setattr("workbench.backend.environment_planner.plan_environments",
                        lambda *a, **k: planned.append(1) or {"groups": [{"environment_id": "P"}]})
    local_env = tmp_path / "workbench.env"
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", local_env)
    monkeypatch.delenv(var, raising=False)

    for label in ("file-only", "exported"):
        planned.clear()
        local_env.write_text(f"{var}={interpreter}\n", encoding="utf-8")
        if label == "exported":
            monkeypatch.setenv(var, str(interpreter))
        else:
            monkeypatch.delenv(var, raising=False)
        model_settings.file_settings.cache_clear()
        assert _create_dry_run(module, model, var, spec, config) is True
        out, err = capsys.readouterr()
        assert not planned, f"{label}: the planner branch must not be taken — {err}"
        assert "pip install" in out and str(interpreter) in out, f"{label}: {out}"


def test_a30_02_conflicting_selector_export_wins_over_the_file(tmp_path, monkeypatch, capsys):
    """The documented precedence holds: the process environment beats `workbench.env`."""
    from workbench.backend import model_settings, operator_config

    module = _manage("manage_environment_a30_conflict")
    config = _config(tmp_path)
    exported = _venv(tmp_path, "exported")
    from_file = _venv(tmp_path, "from_file")
    var = "WORKBENCH_ENCODER_PYTHON"
    spec = tmp_path / "requirements.txt"
    spec.write_text("some-package==1.0\n", encoding="utf-8")
    local_env = tmp_path / "workbench.env"
    local_env.write_text(f"{var}={from_file}\n", encoding="utf-8")
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", local_env)
    monkeypatch.setenv(var, str(exported))
    model_settings.file_settings.cache_clear()

    assert _create_dry_run(module, model_by_id("encoder"), var, spec, config) is True
    out = capsys.readouterr().out
    assert str(exported) in out and str(from_file) not in out, out


def test_a30_02_adapted_model_installs_the_resolved_spec_not_the_historical_freeze(
        tmp_path, monkeypatch, capsys):
    """The adapted install must carry the resolved pins + the cu128 index, never the raw freeze."""
    module = _manage("manage_environment_a30_spec")
    config = _config(tmp_path)
    interpreter = _venv(tmp_path, "spec")
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", str(interpreter))
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    site_packages = next((interpreter.parent.parent / "lib").glob("python*/site-packages"))
    before = sorted(p.name for p in site_packages.iterdir())

    assert _create_dry_run(module, model_by_id("encoder"), "WORKBENCH_ENCODER_PYTHON",
                           _ENCODER_FREEZE, config) is True
    out = capsys.readouterr().out
    assert str(_ENCODER_FREEZE) not in out, "the raw historical freeze must not be installed"
    assert " -r " not in out
    assert "torch==2.7.1" in out and "torchvision==0.22.1" in out and "open-clip-torch==2.24.0" in out
    assert "--extra-index-url" in out and "download.pytorch.org/whl/cu128" in out
    # Dry-run is read-only.
    assert sorted(p.name for p in site_packages.iterdir()) == before


def test_a30_02_unset_adaptation_keeps_the_historical_freeze(tmp_path, monkeypatch, capsys):
    """No adaptation → the historical `-r requirements.txt` invocation, unchanged."""
    module = _manage("manage_environment_a30_spec_hist")
    config = _config(tmp_path)
    interpreter = _venv(tmp_path, "spec_hist")
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", str(interpreter))
    assert _create_dry_run(module, model_by_id("encoder"), "WORKBENCH_ENCODER_PYTHON",
                           _ENCODER_FREEZE, config) is True
    out = capsys.readouterr().out
    assert "pip install -r" in out and str(_ENCODER_FREEZE) in out
    assert "torch==2.7.1" not in out and "--extra-index-url" not in out


def test_a30_02_unsafe_interpreter_is_still_refused(tmp_path, monkeypatch, capsys):
    """The isolation refusal is not weakened by the resolved-spec route."""
    module = _manage("manage_environment_a30_unsafe")
    config = _config(tmp_path)
    monkeypatch.setenv("WORKBENCH_ENCODER_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    spec = tmp_path / "requirements.txt"
    spec.write_text("x\n", encoding="utf-8")
    assert _create_dry_run(module, model_by_id("encoder"), "WORKBENCH_ENCODER_PYTHON",
                           spec, config) is False
    assert "unsafe" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- A30-03


def _stateful_model(*, python: str = "3.10", packages: list[str] | None = None) -> dict:
    """TME's identity with a contract small enough to actually satisfy in a test venv.

    TME declares python 3.10 in BOTH its historical and its adapted contract, so this isolates the
    stale-assignment behaviour from A30-01 while keeping the model_id the assignment is keyed by.
    """
    model = dict(model_by_id("tme"))
    model["environment"] = {"python_env_var": "WORKBENCH_TME_PYTHON", "python": python,
                            "packages": list(packages or []), "confidence": "TEST_CONTRACT"}
    return model


def _write_assignment(root: Path, model_id: str, interpreter: Path) -> Path:
    path = root / "environment_assignment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "models": {model_id: {"environment_id": "old-group", "interpreter": str(interpreter),
                              "group_members": [model_id], "shared": False,
                              "state": "RUNTIME_VERIFIED"}},
        "groups": [{"environment_id": "old-group", "members": [model_id],
                    "interpreter": str(interpreter), "state": "RUNTIME_VERIFIED"}],
    }), encoding="utf-8")
    return path


@pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")
def test_a30_03_stale_assignment_is_not_reported_ready(tmp_path, monkeypatch):
    """A stale RUNTIME_VERIFIED record over a broken environment must report NOT ready."""
    module = _manage("manage_environment_a30_stale")
    config = _config(tmp_path)
    interpreter = _python310(tmp_path, "stale_env")  # empty: the declared dependency is absent
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], interpreter)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(interpreter))

    status = module.status(model, config)
    assert status["ready"] is False, status
    assert status["runtime_verified"] is False
    # One truthful verdict: never GROUP_READY beside a BLOCKED tier.
    assert status["interpreter_status"] != "GROUP_READY"
    assert status["interpreter_status"] == "DEPENDENCY_MISSING", status
    assert status["interpreter_tier"] == "BLOCKED", status
    assert status["runtime_reason"] == "missing package: stubdep (no importable installation found in the environment)"


@pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")
def test_a30_03_stale_assignment_does_not_early_return_from_create(tmp_path, monkeypatch, capsys):
    """A stale assignment: the dry run stays read-only/result-stable, the real run repairs.

    The dry run provisions nothing, so it does not drop the assignment either: it reports the
    planned group, exactly as before the current-contract check existed. A REAL run must not
    early-return on the stale record — it reaches the provisioning (repair) path.
    """
    from workbench.backend import environment_planner as planner

    module = _manage("manage_environment_a30_stale_create")
    config = _config(tmp_path)
    interpreter = _python310(tmp_path, "stale_create")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], interpreter)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(interpreter))

    assert _create_dry_run(module, model, "WORKBENCH_TME_PYTHON", _TME_FREEZE, config) is True
    out = capsys.readouterr().out
    assert "[NOTICE]" in out and "would provision or reuse group old-group" in out, out
    assert "pip install" not in out, out

    repaired = []

    def fake_provision(group_models, scoped, **kwargs):
        repaired.append(sorted(m["model_id"] for m in group_models))
        return {"models": {model["model_id"]: {"state": "RUNTIME_VERIFIED",
                                               "interpreter": str(interpreter),
                                               "installation_status": "INSTALLED"}},
                "groups": [{"environment_id": "old-group", "members": [model["model_id"]],
                            "interpreter": str(interpreter), "state": "RUNTIME_VERIFIED",
                            "installation_status": "INSTALLED",
                            "runtime_verification_status": "READY"}]}

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "provision_environment_plan", fake_provision)
    status = {"model_id": model["model_id"], "source_synced": True, "env_var": "WORKBENCH_TME_PYTHON",
              "env_spec_path": str(_TME_FREEZE), "env_spec_type": "requirements.txt", "ready": False}
    assert module.create_env(model, status, dry_run=False, config=config, network_allowed=True) is True
    assert repaired == [["tme"]], repaired
    capsys.readouterr()


@pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")
def test_a30_03_valid_unchanged_assignment_is_reused_without_reinstall(tmp_path, monkeypatch, capsys):
    """A valid, unchanged assignment is still reused: no install argv, no re-provisioning."""
    module = _manage("manage_environment_a30_valid")
    config = _config(tmp_path)
    interpreter = _python310(tmp_path, "valid_env")
    _install_stub(interpreter, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], interpreter)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(interpreter))

    status = module.status(model, config)
    assert status["ready"] is True, status
    assert status["runtime_verified"] is True

    assert _create_dry_run(module, model, "WORKBENCH_TME_PYTHON", _TME_FREEZE, config) is True
    out = capsys.readouterr().out
    assert "reuse group" in out, out
    assert "pip install" not in out, out


def test_a30_03_model_without_a_declared_variable_keeps_the_record(tmp_path):
    """No declared contract to verify against → the persisted record stays the evidence."""
    module = _manage("manage_environment_a30_novar")
    config = _config(tmp_path)
    interpreter = _venv(tmp_path, "novar")
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, "tme", interpreter)
    status = module.status({"model_id": "tme", "environment": {}}, config)
    assert status["shared_group"] == "old-group"
    assert status["interpreter_status"] == "GROUP_READY"
    assert status["runtime_verified"] is True and status["ready"] is True


# --------------------------------------------------------------------------- A30-B1


@pytest.mark.parametrize("adaptation", ["UNSET", "sm_120"])
def test_a30_b1_dry_run_with_a_persisted_assignment_is_result_stable(
        tmp_path, monkeypatch, capsys, adaptation):
    """`--dry-run --create` must be read-only AND result-stable for ANY persisted assignment.

    The assignment here comes from `environment_plan.json` alone: no interpreter, no state, so
    nothing about it is "stale" to repair and nothing needs the network. Dropping it sent the dry
    run down the install path, where the verdict became a function of `--allow-network` (BLOCKED
    without it). The planner branch must report the planned group either way, and the neutral
    `workbench.env` settings are absent so no preset interpreter can route around it.
    """
    module = _manage(f"manage_environment_a30_b1_{adaptation}")
    config = _config(tmp_path)
    root = config.WORKBENCH_ARTIFACTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / "environment_plan.json").write_text(json.dumps({
        "groups": [{"environment_id": "cir-x", "members": ["encoder"], "interpreter": None}],
    }), encoding="utf-8")
    if adaptation != "UNSET":
        monkeypatch.setenv(runtime.ADAPTATION_ENV, adaptation)

    status = {"model_id": "encoder", "source_synced": True,
              "env_spec_path": str(_ENCODER_FREEZE), "env_spec_type": "requirements.txt",
              "ready": False}
    assert module.create_env(model_by_id("encoder"), status, dry_run=True, config=config,
                             network_allowed=False) is True
    out, err = capsys.readouterr()
    assert "would provision or reuse group cir-x" in out, out
    assert "requires network authorization" not in out + err, out + err
