"""A31 regressions: the EFFECTIVE interpreter, and the contract a repair is planned from.

Two defects, one theme: a persisted record was allowed to outvote the operator's explicit selection,
and a stored plan was allowed to outvote the CURRENT contract.

- A31-01: `manage_environment.status` inspected the explicit selection (process env, then
  `workbench.env`) and then unconditionally overwrote `interpreter`/verdict from the persisted
  assignment; `create_env` reached the assignment branch before the selection, so it repaired a
  group the operator had moved away from. Once the assigned executable had been deleted the row
  read `ready: true` beside a path that does not exist. Contract: process env > file setting >
  persisted assignment; the assignment stays visible as metadata but never decides the target.
- A31-02: `create_env`'s repair branch parsed the stored `environment_plan.json` and reused it
  verbatim, so repairing after a profile change reinstalled a superseded freeze (Python 3.9 /
  torch==1.13.1) that the current verifier then rejects. Repair must be planned from the CURRENT
  contract via the planner.

These tests drive the production entry points against real interpreters and assert only on
observable output (status dicts, argv, stdout) — never on the patch's internal helpers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from workbench.backend import runtime

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TME_FREEZE = REPO / "workbench" / "third_party" / "TME" / "requirements.txt"

#: A real CPython 3.10 — the adapted TME contract's interpreter.
_PY310 = next((candidate for candidate in (
    Path(f"/usr/local/bin/python{sys.version_info.major}.{sys.version_info.minor}"),
    Path("/usr/local/bin/python3.10"),
    Path("/tmp/a30-verify/venv310/bin/python"),
) if candidate.exists() and os.access(candidate, os.X_OK)
    and subprocess.run([str(candidate), "-c", "import sys;print(sys.version_info[:2])"],
                       capture_output=True, text=True).stdout.strip() == "(3, 10)"), None)

_SKIP = pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Repoint the operator file and clear the caches, so a developer's real settings cannot leak."""
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
    _venv_module.EnvBuilder(with_pip=False, system_site_packages=False,
                            symlinks=True).create(env_dir)
    return env_dir / "bin" / "python"


def _python310(root: Path, name: str) -> Path:
    """A real venv whose interpreter IS CPython 3.10 — the adapted TME contract."""
    if _PY310 is None:
        pytest.skip("no CPython 3.10 available on this host")
    env_dir = root / name
    result = subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(env_dir)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not build a CPython 3.10 venv: {result.stderr.strip()[-200:]}")
    return env_dir / "bin" / "python"


def _config(tmp_path: Path):
    from workbench.backend.operator_config import resolve_config

    third_party = tmp_path / "third_party"
    (third_party / "TME").mkdir(parents=True, exist_ok=True)
    # The planner reads manifests from the configured root, so the model needs one there.
    (third_party / "TME" / "requirements.txt").write_text("stubdep==1.2.3\n", encoding="utf-8")
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


def _stateful_model(*, python: str = "3.10", packages: list[str] | None = None) -> dict:
    """TME's model_id and variable with a contract small enough to satisfy in a test venv."""
    from workbench.backend.registry import model_by_id

    model = dict(model_by_id("tme"))
    model["environment"] = {"python_env_var": "WORKBENCH_TME_PYTHON", "python": python,
                            "packages": list(packages or []), "confidence": "TEST_CONTRACT"}
    model["environment_required"] = False
    return model


def _write_assignment(root: Path, model_id: str, interpreter) -> Path:
    """A persisted assignment naming `interpreter`; pass a missing path to model a deleted one."""
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


def _create_status(model: dict, spec: Path | None, env_var: str | None = None) -> dict:
    """The status dict `create_env` receives from the CLI, with the fields it actually reads."""
    return {"model_id": model["model_id"], "source_synced": True, "ready": False,
            "env_var": env_var or (model.get("environment") or {}).get("python_env_var"),
            "env_spec_path": str(spec) if spec else None,
            "env_spec_type": spec.name if spec else None}


class _ProvisionSpy:
    """Records the arguments the repair branch hands the planner, and answers a clean repair."""

    def __init__(self, interpreter: str, verification: str = "READY"):
        self.calls: list[dict] = []
        self._interpreter = interpreter
        self._verification = verification

    def __call__(self, group_models, scoped, **kwargs):
        self.calls.append({
            "members": sorted(m["model_id"] for m in group_models),
            "scoped_members": sorted({mid for g in scoped.get("groups", [])
                                      for mid in g.get("members", [])}),
            "scoped_groups": [{"python": g.get("python"),
                               "torch": (g.get("packages") or {}).get("torch")}
                              for g in scoped.get("groups", [])],
            "third_party_root": kwargs.get("third_party_root"),
        })
        mark = ("BLOCKED" if self._verification == "BLOCKED" else "INSTALLED")
        return {
            "models": {"tme": {"state": "RUNTIME_VERIFIED", "interpreter": self._interpreter,
                               "installation_status": mark}},
            "groups": [{"environment_id": "repaired", "members": ["tme"],
                        "interpreter": self._interpreter, "state": "RUNTIME_VERIFIED",
                        "installation_status": mark,
                        "runtime_verification_status": self._verification}],
        }


@pytest.fixture
def _no_solver(monkeypatch):
    """Keep the repair planning offline: the solver would otherwise shell out to pip."""
    from workbench.backend import environment_planner as planner

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    return planner


# --------------------------------------------------------------------------- A31-01

@_SKIP
def test_a31_01_export_only_selection_beats_a_broken_assignment(tmp_path, monkeypatch):
    """Process env wins over the assignment: the selected venv's verdict, not the assigned one."""
    module = _manage("manage_environment_a31_export")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")           # no stubdep
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)

    assert status["interpreter"] == str(selected), status
    assert status["interpreter_status"] == "RUNTIME_READY", status
    assert status["ready"] is True, status
    # The superseded assignment stays visible as metadata.
    assert status["shared_group"] == "old-group", status
    assert status["shared_interpreter"] == str(broken), status


@_SKIP
def test_a31_01_file_only_selection_beats_a_broken_assignment(tmp_path, monkeypatch):
    """`workbench.env` alone selects the same way a process export does."""
    module = _manage("manage_environment_a31_file")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)
    (tmp_path / "workbench.env").write_text(f"WORKBENCH_TME_PYTHON={selected}\n", encoding="utf-8")
    from workbench.backend import model_settings
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()

    status = module.status(model, config)

    assert status["interpreter"] == str(selected), status
    assert status["ready"] is True and status["interpreter_status"] == "RUNTIME_READY", status


@_SKIP
def test_a31_01_export_conflicting_with_the_file_wins(tmp_path, monkeypatch):
    """Documented precedence: the process environment beats `workbench.env`; assignment loses both."""
    module = _manage("manage_environment_a31_conflict")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    exported = _python310(tmp_path, "exported")
    from_file = _python310(tmp_path, "from_file")
    _install_stub(exported, "stubdep", "1.2.3")
    _install_stub(from_file, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    (tmp_path / "workbench.env").write_text(f"WORKBENCH_TME_PYTHON={from_file}\n", encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(exported))
    from workbench.backend import model_settings
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()

    status = module.status(model, config)

    assert status["interpreter"] == str(exported), status
    assert str(from_file) not in json.dumps(status), status
    assert status["ready"] is True, status


@_SKIP
def test_a31_01_old_assignment_missing_dependency_is_not_ready_once_selected(tmp_path, monkeypatch):
    """The explicit selection decides: ready when it verifies, even though the assignment is broken."""
    module = _manage("manage_environment_a31_missing_dep")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)

    # With no selection the persisted (broken) assignment stands and is truthfully NOT ready.
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)
    unselected = module.status(model, config)
    assert unselected["ready"] is False, unselected
    assert unselected["interpreter"] == str(broken), unselected

    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))
    status = module.status(model, config)
    assert status["ready"] is True and status["interpreter_status"] == "RUNTIME_READY", status
    assert "DEPENDENCY_MISSING" not in status["interpreter_status"], status


@_SKIP
def test_a31_01_deleted_assignment_executable_is_never_ready_with_a_dangling_path(
        tmp_path, monkeypatch):
    """The exact reported claim: ready=true beside an interpreter that does not exist."""
    module = _manage("manage_environment_a31_deleted")
    config = _config(tmp_path)
    deleted = _python310(tmp_path, "gone_env")
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], deleted)
    os.remove(deleted)

    # No selection: the assignment is gone, so the row must not claim a verified missing path.
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)
    unselected = module.status(model, config)
    assert unselected["ready"] is False, unselected
    assert not (unselected["interpreter"] and not Path(unselected["interpreter"]).exists()
                and unselected["ready"]), unselected

    # Explicit selection: the SELECTED path is reported and carries the verdict; the dangling
    # assigned path is metadata only.
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))
    status = module.status(model, config)
    assert status["interpreter"] == str(selected), status
    assert Path(status["interpreter"]).exists(), status
    assert status["ready"] is True, status
    assert status["shared_interpreter"] == str(deleted), status


@_SKIP
def test_a31_01_missing_explicit_interpreter_is_reported_truthfully(tmp_path, monkeypatch):
    """A selected path that does not exist must never silently fall back to a valid assignment."""
    module = _manage("manage_environment_a31_absent")
    config = _config(tmp_path)
    valid = _python310(tmp_path, "valid_env")
    _install_stub(valid, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], valid)
    absent = tmp_path / "absent" / "bin" / "python"
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(absent))

    status = module.status(model, config)

    assert status["ready"] is False, status
    assert status["interpreter_status"] == "CONFIGURED_MISSING", status
    assert status["interpreter"] == str(absent), status
    assert status["runtime_verified"] is False, status


@_SKIP
def test_a31_01_valid_unchanged_assignment_is_reused_without_provisioning(
        tmp_path, monkeypatch, capsys, _no_solver):
    """A genuinely valid, unchanged assignment is still reused: no planning, no install argv."""
    module = _manage("manage_environment_a31_valid")
    config = _config(tmp_path)
    valid = _python310(tmp_path, "valid_env")
    _install_stub(valid, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    assignment = _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], valid)
    before = assignment.read_text(encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(valid))

    status = module.status(model, config)
    assert status["ready"] is True and status["runtime_verified"] is True, status

    spy = _ProvisionSpy(str(valid))
    monkeypatch.setattr(_no_solver, "provision_environment_plan", spy)
    assert module.create_env(model, _create_status(model, _TME_FREEZE), dry_run=True,
                             config=config, network_allowed=True) is True
    out = capsys.readouterr().out
    assert "reuse group" in out, out
    assert "pip install" not in out, out
    assert spy.calls == [], spy.calls
    assert assignment.read_text(encoding="utf-8") == before


@_SKIP
def test_a31_01_dry_run_is_read_only_and_network_independent_without_a_selection(
        tmp_path, monkeypatch, capsys, _no_solver):
    """A persisted assignment + no selection: the dry run reports the group, network-agnostic."""
    module = _manage("manage_environment_a31_dryrun_assigned")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    assignment = _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    before = assignment.read_text(encoding="utf-8")
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    for network_allowed in (True, False):
        capsys.readouterr()
        assert module.create_env(model, _create_status(model, _TME_FREEZE), dry_run=True,
                                 config=config, network_allowed=network_allowed) is True
        out, err = capsys.readouterr()
        assert "would provision or reuse group old-group" in out, (out, err)
        assert "requires network authorization" not in out + err, out + err
        assert "[BLOCKED]" not in out + err, out + err
    assert assignment.read_text(encoding="utf-8") == before


@_SKIP
def test_a31_01_dry_run_with_an_override_is_read_only_and_network_independent(
        tmp_path, monkeypatch, capsys, _no_solver):
    """The override path must not turn a dry run into a network-gated install (A30-B1 lesson)."""
    module = _manage("manage_environment_a31_dryrun_override")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    assignment = _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    before = assignment.read_text(encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    for network_allowed in (True, False):
        capsys.readouterr()
        assert module.create_env(model, _create_status(model, _TME_FREEZE), dry_run=True,
                                 config=config, network_allowed=network_allowed) is True
        out, err = capsys.readouterr()
        assert "overrides assigned group" in out, (out, err)
        assert str(selected) in out, out
        assert "requires network authorization" not in out + err, out + err
        assert "[BLOCKED]" not in out + err, out + err
        assert "[GROUP]" not in out, out       # the assigned group is not the reported target
    assert assignment.read_text(encoding="utf-8") == before


@_SKIP
def test_a31_01_real_run_override_targets_the_selected_interpreter(
        tmp_path, monkeypatch, capsys, _no_solver):
    """A real run with an override must leave the assignment branch: pip runs against the SELECTED
    venv and the old group is never provisioned."""
    module = _manage("manage_environment_a31_real_override")
    config = _config(tmp_path)
    broken = _python310(tmp_path, "old_env")
    selected = _python310(tmp_path, "new_env")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], broken)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))
    spec = tmp_path / "requirements.txt"
    spec.write_text("stubdep==1.2.3\n", encoding="utf-8")

    installed: list[list[str]] = []
    real_run = module.subprocess.run

    def fake_run(argv, **kwargs):
        if "-m" in argv and "pip" in argv:      # fake only the install; probes must stay real
            installed.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("the assigned group must not be provisioned for an overridden selection")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(_no_solver, "provision_environment_plan", forbidden)

    assert module.create_env(model, _create_status(model, spec), dry_run=False, config=config,
                             network_allowed=True) is True
    out = capsys.readouterr().out
    assert installed and str(selected) in installed[0][0], (installed, out)
    assert str(broken) not in out, out
    assert "[GROUP]" not in out, out            # the assignment branch was never taken


# --------------------------------------------------------------------------- A31-02

def _write_historical_plan(root: Path, interpreter: str) -> Path:
    """The stored plan a superseded run wrote: the ENCODER freeze, Python 3.9 / torch==1.13.1."""
    path = root / "environment_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "groups": [{"environment_id": "old-group", "members": ["tme"], "interpreter": interpreter,
                    "python": "3.9", "packages": {"torch": "1.13.1"}, "state": "RUNTIME_VERIFIED",
                    "constraints": {"python": [["==", "3.9"]], "torch": [["==", "1.13.1"]],
                                    "torchvision": [], "cuda": [], "packages": {}, "ranges": {},
                                    "pip_only": [], "evidence": [], "pip_index_url": None}}],
    }), encoding="utf-8")
    return path


def test_a31_02_repair_plans_the_current_contract_not_the_stored_plan(
        tmp_path, monkeypatch, capsys, _no_solver):
    """A stored historical plan must not decide the repair target: the CURRENT contract does."""
    module = _manage("manage_environment_a31_repair")
    config = _config(tmp_path)
    model = _stateful_model(packages=["stubdep==1.2.3"])
    stale = tmp_path / "stale_env" / "bin" / "python"       # the deleted assignment target
    repaired = _venv(tmp_path, "repaired_env")
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], stale)
    _write_historical_plan(config.WORKBENCH_ARTIFACTS_ROOT, str(stale))
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    spy = _ProvisionSpy(str(repaired))
    monkeypatch.setattr(_no_solver, "provision_environment_plan", spy)
    assert module.create_env(model, _create_status(model, _TME_FREEZE), dry_run=False,
                             config=config, network_allowed=True) is True

    assert spy.calls, spy.calls
    call = spy.calls[0]
    planned = json.dumps(call["scoped_groups"])
    assert "3.9" not in planned, planned                     # the superseded freeze is not reused
    assert "1.13.1" not in planned, planned
    assert str(call["third_party_root"]) == str(config.WORKBENCH_THIRD_PARTY_ROOT), call
    assert call["members"] == ["tme"], call                  # the assignment's actual members
    assert call["scoped_members"] == ["tme"], call           # only the requested group is installed


@_SKIP
def test_a31_02_repair_verification_failure_is_reported_failed(
        tmp_path, monkeypatch, capsys, _no_solver):
    """A repair whose verification fails must not be published as a success."""
    module = _manage("manage_environment_a31_repair_failed")
    config = _config(tmp_path)
    interpreter = _python310(tmp_path, "repair_env")
    model = _stateful_model(packages=["stubdep==1.2.3"])
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], interpreter)
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    spy = _ProvisionSpy(str(interpreter), verification="BLOCKED")
    monkeypatch.setattr(_no_solver, "provision_environment_plan", spy)
    assert module.create_env(model, _create_status(model, _TME_FREEZE), dry_run=False,
                             config=config, network_allowed=True) is False
    out, err = capsys.readouterr()
    assert "[FAILED]" in err, (out, err)
