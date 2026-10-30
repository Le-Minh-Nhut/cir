"""A34-01ext: ONE effective-target decision, taken before assignment reuse/repair.

The audit's 48-cell matrix (process/file settings x 6 assignment states x 4 selection states, real
CPython 3.10) found `status` and `create_env` still disagreeing with the launch resolver on four
branches. All four are the same bug: the ASSIGNMENT was consulted before the operator's selection.

A. missing selected alias, live assigned alias, SAME prefix: `status` reported the assigned path as
   ready; `create` could return reuse success; the launch resolver used the missing selection.
B. non-executable selected file, live assigned alias: `status` blocked, `create` could still report
   reuse success through the assignment.
C. assignment FAILED/BLOCKED with `interpreter: None` + explicit selection: the override branch
   required a truthy assigned interpreter, so `create` fell into shared-group repair and planned the
   peers of a group the operator had already left.
D. a selected `.staging-*` environment: `status` says INTERRUPTED_STAGING (not ready), but
   `create`'s no-assignment and different-prefix routes still emitted pip argv INTO the staging
   target.

Contract: process environment > `workbench.env` > assignment, decided once (`_effective_target`) and
shared by `status` and `create_env`, so neither can disagree with the launch resolver or with each
other. An explicit selection is the target even when it is unusable; the existing
INTERRUPTED_STAGING rule applies on every real-create route; dry runs stay read-only and
network-independent; intentional shared-group repair and the A30/A31/A33 behaviours are preserved.
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
from workbench.backend.registry import model_by_id

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SERIES_PROBE = "import sys;print(sys.version_info[:2])"

def _discover_python310():
    """A real CPython 3.10 on this host, discovered rather than assumed at one path."""
    import shutil

    candidates = [Path("/usr/local/bin/python3.10"), Path("/usr/bin/python3.10"),
                  Path("/tmp/a30-verify/venv310/bin/python3.10")]
    which = shutil.which("python3.10")
    if which:
        candidates.append(Path(which))
    for directory in ("/usr/local/bin", "/usr/bin", "/opt/python/bin"):
        candidates.extend(sorted(Path(directory).glob("python3.10*")))
    for candidate in candidates:
        try:
            if (candidate.is_file() and os.access(candidate, os.X_OK)
                    and subprocess.run([str(candidate), "-c", SERIES_PROBE], capture_output=True,
                                       text=True).stdout.strip() == "(3, 10)"):
                return candidate
        except OSError:
            continue
    return None

_PY310 = _discover_python310()

_SKIP = pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")

@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Repoint the operator file and clear caches, so real developer settings cannot leak."""
    from workbench.backend import model_settings, operator_config

    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", tmp_path / "workbench.env")
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()
    yield
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()

def _config(tmp_path: Path):
    third_party = tmp_path / "third_party"
    (third_party / "TME").mkdir(parents=True, exist_ok=True)
    (third_party / "TME" / "requirements.txt").write_text("stubdep==1.2.3\n", encoding="utf-8")
    from workbench.backend.operator_config import resolve_config

    return replace(resolve_config(), WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
                   WORKBENCH_THIRD_PARTY_ROOT=third_party)

def _manage(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, REPO / "workbench" / "scripts" / "manage_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _venv(root: Path, name: str) -> Path:
    """A real CPython 3.10 venv whose `bin/python` and `bin/python3.10` are INDEPENDENT files.

    `venv`'s alias layout depends on the base interpreter's name: with an unversioned base it
    creates `python3.10 -> python`, so deleting one alias destroys the other. Both are materialized
    as real copies, making every alias in these tests independently viable.
    """
    if _PY310 is None:
        pytest.skip("no CPython 3.10 available on this host")
    import shutil

    env_dir = root / name
    result = subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(env_dir)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not build a CPython 3.10 venv: {result.stderr.strip()[-200:]}")
    for alias in ("python", "python3.10"):
        staged = env_dir / "bin" / f".{alias}.materialized"
        shutil.copy2(_PY310, staged)
        staged.chmod(0o755)
        target = env_dir / "bin" / alias
        target.unlink(missing_ok=True)
        staged.rename(target)
    return env_dir / "bin" / "python"

def _install_stub(interpreter: Path, name: str, version: str) -> None:
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        module = site_packages / f"{name}.py"
        module.write_text(f"__version__ = {version!r}\n")
        dist = site_packages / f"{name}-{version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (dist / "RECORD").write_text(
            f"{name}.py,sha256=test,{module.stat().st_size}\n"
            f"{name}-{version}.dist-info/METADATA,sha256=test,60\n")
        return
    raise AssertionError("no site-packages directory in the test environment")

def _model(*, packages: list[str] | None = None) -> dict:
    model = dict(model_by_id("tme"))
    model["environment"] = {"python_env_var": "WORKBENCH_TME_PYTHON", "python": "3.10",
                            "packages": list(packages or ["stubdep==1.2.3"]),
                            "confidence": "TEST_CONTRACT"}
    model["environment_required"] = False
    return model

def _write_assignment(root: Path, model_id: str, interpreter, *, state: str = "RUNTIME_VERIFIED",
                      members: list[str] | None = None) -> Path:
    """A persisted assignment; `interpreter=None` models a FAILED/BLOCKED group."""
    path = root / "environment_assignment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    members = members or [model_id]
    path.write_text(json.dumps({
        "models": {model_id: {"environment_id": "old-group", "interpreter":
                              str(interpreter) if interpreter else None,
                              "group_members": members, "state": state}},
        "groups": [{"environment_id": "old-group", "members": members,
                    "interpreter": str(interpreter) if interpreter else None, "state": state}],
    }), encoding="utf-8")
    return path

def _selector(kind: str, monkeypatch, tmp_path: Path, var: str, value: Path) -> None:
    """Set the selection through the process environment or through `workbench.env`."""
    from workbench.backend import model_settings

    if kind == "process":
        monkeypatch.delenv(var, raising=False)
        (tmp_path / "workbench.env").write_text(f"{var}={value}\n", encoding="utf-8")
    else:
        monkeypatch.setenv(var, str(value))
    model_settings.file_settings.cache_clear()

class _InstallSpy:
    """Fakes only pip installs, records their argv, and answers success."""

    def __init__(self, module):
        self.argv: list[list[str]] = []
        self._real = module.subprocess.run

    def __call__(self, argv, **kwargs):
        if "-m" in argv and "pip" in argv:
            self.argv.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return self._real(argv, **kwargs)

class _PlannerSpy:
    """Records what the shared-group repair branch planned, and answers a clean repair."""

    def __init__(self, interpreter: str | None, members: list[str] | None = None):
        self.calls: list[dict] = []
        self._interpreter = interpreter
        self._members = members or []

    def __call__(self, group_models, scoped, **kwargs):
        members = sorted(m["model_id"] for m in group_models)
        self.calls.append({"members": members,
                           "scoped_members": sorted({mid for g in scoped.get("groups", [])
                                                     for mid in g.get("members", [])})})
        return {
            "models": {"tme": {"state": "RUNTIME_VERIFIED", "interpreter": self._interpreter,
                               "installation_status": "INSTALLED"}},
            "groups": [{"environment_id": "repaired", "members": members or self._members,
                        "interpreter": self._interpreter, "state": "RUNTIME_VERIFIED",
                        "installation_status": "INSTALLED",
                        "runtime_verification_status": "READY"}],
        }

def _status_for_create(model: dict, spec: Path, env_var: str = "WORKBENCH_TME_PYTHON") -> dict:
    return {"model_id": model["model_id"], "source_synced": True, "ready": False,
            "env_var": env_var, "env_spec_path": str(spec), "env_spec_type": "requirements.txt"}

@pytest.fixture
def spec(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text("stubdep==1.2.3\n", encoding="utf-8")
    return path

@pytest.fixture
def no_solver(monkeypatch):
    from workbench.backend import environment_planner as planner

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    return planner

# --------------------------------------------------------------------- A. missing selection

@_SKIP
@pytest.mark.parametrize("kind", ["process", "file"])
def test_a34_a_missing_selection_is_the_target_not_the_live_assignment(
        kind, tmp_path, monkeypatch, capsys, no_solver, spec):
    """Missing selected alias + live assigned alias, SAME prefix: status blocked, create refuses."""
    module = _manage(f"manage_environment_a34_missing_{kind}")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "env")
    selected = assigned.parent / "python3.10"
    _install_stub(assigned, "stubdep", "1.2.3")
    missing = assigned.parent / "gone3.10"
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    _selector(kind, monkeypatch, tmp_path, "WORKBENCH_TME_PYTHON", missing)

    status = module.status(model, config)
    assert status["interpreter"] == str(missing), status
    assert status["ready"] is False and status["interpreter_status"] == "CONFIGURED_MISSING", status
    assert status["shared_interpreter"] == str(assigned), status      # metadata only
    # The reported path IS the one the launcher will use.
    assert runtime.resolve_model_interpreter(model, config) == str(missing)

    spy = _PlannerSpy(str(assigned))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)

    assert module.create_env(model, _status_for_create(model, spec), dry_run=False, config=config,
                             network_allowed=True) is False
    out, err = capsys.readouterr()
    assert "[BLOCKED]" in err and "missing" in err, (out, err)
    assert installs.argv == [], installs.argv          # nothing installed anywhere
    assert spy.calls == [], spy.calls                  # no repair of the live assignment either
    assert selected.is_file()                          # the fixture's other alias is untouched

# ------------------------------------------------------------------ B. non-executable selection

@_SKIP
def test_a34_b_non_executable_selection_blocks_status_and_create(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """An existing but NON-EXECUTABLE selected alias: no reuse success through the assignment."""
    module = _manage("manage_environment_a34_nonexec")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "env")
    selected = assigned.parent / "python3.10"
    _install_stub(assigned, "stubdep", "1.2.3")
    selected.chmod(0o644)                              # present, not executable
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)
    assert status["interpreter"] == str(selected), status
    assert status["ready"] is False and status["interpreter_status"] == "CONFIGURED_MISSING", status
    assert runtime.resolve_model_interpreter(model, config) == str(selected)

    spy = _PlannerSpy(str(assigned))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)

    assert module.create_env(model, _status_for_create(model, spec), dry_run=False, config=config,
                             network_allowed=True) is False
    out, err = capsys.readouterr()
    assert "[BLOCKED]" in err and "not executable" in err, (out, err)
    assert installs.argv == [] and spy.calls == [], (installs.argv, spy.calls)
    assert assigned.is_file()                          # the assigned alias is real and untouched

# --------------------------------------- C. FAILED/BLOCKED assignment with interpreter None

@_SKIP
@pytest.mark.parametrize("state", ["FAILED", "BLOCKED"])
def test_a34_c_assignment_without_an_interpreter_never_plans_the_group(
        state, tmp_path, monkeypatch, capsys, no_solver, spec):
    """A FAILED/BLOCKED assignment with no interpreter + a selection: target the selection."""
    module = _manage(f"manage_environment_a34_none_{state}")
    config = _config(tmp_path)
    selected = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "selected")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], None, state=state,
                      members=[model["model_id"], "encoder"])
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)
    assert status["interpreter"] == str(selected) and status["ready"] is True, status
    assert status["shared_group"] == "old-group", status
    assert runtime.resolve_model_interpreter(model, config) == str(selected)

    spy = _PlannerSpy(str(selected))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)

    assert module.create_env(model, _status_for_create(model, spec), dry_run=False, config=config,
                             network_allowed=True) is True
    out = capsys.readouterr().out
    assert spy.calls == [], spy.calls                  # the peer ("encoder") is never planned
    assert installs.argv and installs.argv[0][0] == str(selected), installs.argv
    assert "[GROUP]" not in out, out

# ---------------------------------------------------------------------------- D. staging

@_SKIP
def test_a34_d1_staging_selection_with_no_assignment_is_refused(
        tmp_path, monkeypatch, capsys, spec):
    """No assignment at all: a `.staging-*` selection must not reach the install path."""
    module = _manage("manage_environment_a34_staging_none")
    config = _config(tmp_path)
    staging = config.WORKBENCH_ARTIFACTS_ROOT / "environments" / ".staging-groups" / "cir-abc"
    interpreter = _venv(config.WORKBENCH_ARTIFACTS_ROOT / "environments", ".staging-groups/cir-abc")
    model = _model()
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(interpreter))
    assert module._is_staging_path(str(interpreter)), interpreter

    status = module.status(model, config)
    assert status["interpreter_status"] == "INTERRUPTED_STAGING", status
    assert status["ready"] is False, status
    assert runtime.resolve_model_interpreter(model, config) == str(interpreter)

    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)
    for network_allowed in (True, False):
        assert module.create_env(model, _status_for_create(model, spec), dry_run=False,
                                 config=config, network_allowed=network_allowed) is False
        # A dry run must not claim success either: `status` reports this selection not ready.
        assert module.create_env(model, _status_for_create(model, spec), dry_run=True,
                                 config=config, network_allowed=network_allowed) is False
    assert installs.argv == [], installs.argv
    assert interpreter.parent.parent == staging

@_SKIP
def test_a34_d2_staging_selection_in_a_different_prefix_is_refused(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """An assignment in a healthy prefix must not be replaced by a staging selection."""
    module = _manage("manage_environment_a34_staging_other")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "env")
    _install_stub(assigned, "stubdep", "1.2.3")
    staging = _venv(config.WORKBENCH_ARTIFACTS_ROOT / "environments", ".staging-groups/cir-abc")
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(staging))
    assert module._is_staging_path(str(staging)), staging

    status = module.status(model, config)
    assert status["interpreter_status"] == "INTERRUPTED_STAGING", status
    assert status["ready"] is False and status["shared_interpreter"] == str(assigned), status

    spy = _PlannerSpy(str(assigned))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)
    for network_allowed in (True, False):
        assert module.create_env(model, _status_for_create(model, spec), dry_run=False,
                                 config=config, network_allowed=network_allowed) is False
    assert installs.argv == [], installs.argv
    assert spy.calls == [], spy.calls

@_SKIP
def test_a34_d3_staging_selection_sharing_the_assigned_prefix_still_refuses(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """Same prefix as the assignment (the A33 shape): still no pip into `.staging-*`."""
    module = _manage("manage_environment_a34_staging_same")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT / "environments", ".staging-groups/cir-abc")
    selected = assigned.parent / "python3.10"
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    os.remove(assigned)
    assert selected.is_file() and os.access(selected, os.X_OK), selected
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    # The REAL assignment branch handles this shape (as it did at HEAD): it refuses to adopt the
    # interrupted staging prefix, so nothing is installed. Only the pip argv is faked.
    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)

    assert module.create_env(model, _status_for_create(model, spec), dry_run=False, config=config,
                             network_allowed=True) is False
    out, err = capsys.readouterr()
    assert installs.argv == [], (installs.argv, out, err)
    assert "[GROUP]" in out, out
    assert str(selected) not in out, (out, err)

# -------------------------------------------------------------- E. no-override assignment reuse

@_SKIP
def test_a34_e_valid_assignment_is_reused_without_a_selection(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """No selection: the assignment decides (A30 semantics), and a healthy one is reused."""
    module = _manage("manage_environment_a34_reuse")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "env")
    _install_stub(assigned, "stubdep", "1.2.3")
    model = _model()
    assignment = _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    before = assignment.read_text(encoding="utf-8")
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    status = module.status(model, config)
    assert status["interpreter"] == str(assigned) and status["ready"] is True, status
    assert runtime.resolve_model_interpreter(model, config) == str(assigned)

    spy = _PlannerSpy(str(assigned))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    assert module.create_env(model, _status_for_create(model, spec), dry_run=True, config=config,
                             network_allowed=True) is True
    out = capsys.readouterr().out
    assert "reuse group old-group" in out, out
    assert spy.calls == [], spy.calls
    assert assignment.read_text(encoding="utf-8") == before

# ------------------------------------------------------------------- F. intentional repair

@_SKIP
def test_a34_f_shared_group_repair_is_still_reachable_without_a_selection(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """Control: with NO selection, a FAILED group (interpreter None) is still repaired."""
    module = _manage("manage_environment_a34_repair")
    config = _config(tmp_path)
    repaired = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "repaired")
    model = _model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], None, state="FAILED")
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    status = module.status(model, config)
    assert status["ready"] is False and "FAILED" in (status["runtime_reason"] or ""), status

    spy = _PlannerSpy(str(repaired))
    monkeypatch.setattr(no_solver, "provision_environment_plan", spy)
    assert module.create_env(model, _status_for_create(model, spec), dry_run=False, config=config,
                             network_allowed=True) is True
    assert spy.calls and spy.calls[0]["members"] == ["tme"], spy.calls
    assert spy.calls[0]["scoped_members"] == ["tme"], spy.calls

# --------------------------------------------------- G. dry-run read-only / network-independent

@_SKIP
def test_a34_g_dry_run_over_an_unusable_selection_is_read_only_and_stable(
        tmp_path, monkeypatch, capsys, no_solver, spec):
    """A dry run with a missing selection: same verdict with/without network, nothing installed."""
    module = _manage("manage_environment_a34_dryrun")
    config = _config(tmp_path)
    assigned = _venv(config.WORKBENCH_ARTIFACTS_ROOT, "env")
    _install_stub(assigned, "stubdep", "1.2.3")
    model = _model()
    assignment = _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    before = assignment.read_text(encoding="utf-8")
    missing = assigned.parent / "gone3.10"
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(missing))

    installs = _InstallSpy(module)
    monkeypatch.setattr(module.subprocess, "run", installs)
    verdicts = []
    for network_allowed in (True, False):
        capsys.readouterr()
        verdicts.append(module.create_env(model, _status_for_create(model, spec), dry_run=True,
                                          config=config, network_allowed=network_allowed))
        out, err = capsys.readouterr()
        assert "requires network authorization" not in out + err, (out, err)
        assert installs.argv == [], installs.argv
    assert verdicts[0] == verdicts[1] is False, verdicts
    assert assignment.read_text(encoding="utf-8") == before
