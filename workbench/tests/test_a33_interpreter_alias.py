"""A33-01 regression: `status` must keep the SELECTED invocation path for same-prefix aliases.

A venv exposes the same interpreter under several names (`bin/python`, `bin/python3.10`). With an
existing operator selection (`bin/python3.10`) and a persisted assignment naming the sibling alias
(`bin/python`), the two paths share a prefix, so `status`'s precedence branch did not fire and
`elif assigned` overwrote `res["interpreter"]` with the assigned path — reporting the DELETED
`bin/python` as `ready: true` while the launch resolver correctly used the selected alias.

Contract: whenever an explicit selection EXISTS, the reported interpreter is the SELECTED invocation
path carrying its OWN verdict; the assignment stays visible as `shared_interpreter` metadata. With no
selection the assignment fallback is unchanged (A30 semantics). `create_env` must not reinstall an
environment whose selected alias already exists just because the assigned alias string is stale.

These tests drive the production entry points against real CPython 3.10 venvs and assert only on the
observable status dict.
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

#: A real CPython 3.10 — the adapted TME contract's interpreter.
def _discover_python310() -> Path | None:
    """A real CPython 3.10 on this host, discovered rather than assumed at one path."""
    import shutil as _shutil

    candidates = [Path("/usr/local/bin/python3.10"), Path("/usr/bin/python3.10"),
                  Path("/tmp/a30-verify/venv310/bin/python"), Path("/tmp/a30-verify/venv310/bin/python3.10")]
    which = _shutil.which("python3.10")
    if which:
        candidates.append(Path(which))
    for directory in ("/usr/local/bin", "/usr/bin", "/opt/python/bin"):
        candidates.extend(sorted(Path(directory).glob("python3.10*")))
    for candidate in candidates:
        try:
            if (candidate.exists() and os.access(candidate, os.X_OK)
                    and not candidate.is_dir()
                    and subprocess.run([str(candidate), "-c", "import sys;print(sys.version_info[:2])"],
                                       capture_output=True, text=True).stdout.strip() == "(3, 10)"):
                return candidate
        except OSError:
            continue
    return None

_PY310 = _discover_python310()

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

def _python310(root: Path, name: str) -> Path:
    """A real venv whose interpreter IS CPython 3.10 — its `bin/python` and `bin/python3.10` aliases.

    Both aliases are materialized as REAL interpreter files copied from `_PY310`, because what `venv`
    leaves behind is host-dependent: with a versioned base (``python3.10``) it makes
    ``bin/python -> bin/python3.10 -> <base>``, but with an UNVERSIONED base it makes
    ``bin/python3.10 -> python``. The tests delete ``bin/python`` (the "assigned alias"), which in
    that second layout also destroyed the selected ``bin/python3.10`` — a dangling symlink, so
    ``status`` correctly reported CONFIGURED_MISSING and the suite passed or failed on which base
    interpreter the host happened to have. Each alias is now independently viable, so deleting one
    never removes the other's target.
    """
    if _PY310 is None:
        pytest.skip("no CPython 3.10 available on this host")
    import shutil

    env_dir = root / name
    result = subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(env_dir)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not build a CPython 3.10 venv: {result.stderr.strip()[-200:]}")
    bin_dir = env_dir / "bin"
    for alias in ("python", "python3.10"):
        target = bin_dir / alias
        staged = bin_dir / f".{alias}.materialized"
        shutil.copy2(_PY310, staged)
        staged.chmod(0o755)
        target.unlink(missing_ok=True)
        staged.rename(target)
    return bin_dir / "python"

def _config(tmp_path: Path):
    from workbench.backend.operator_config import resolve_config

    third_party = tmp_path / "third_party"
    (third_party / "TME").mkdir(parents=True, exist_ok=True)
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

def _stateful_model(*, packages: list[str] | None = None) -> dict:
    """TME's model_id and variable with a contract small enough to satisfy in a test venv."""
    from workbench.backend.registry import model_by_id

    model = dict(model_by_id("tme"))
    model["environment"] = {"python_env_var": "WORKBENCH_TME_PYTHON", "python": "3.10",
                            "packages": list(packages or ["stubdep==1.2.3"]),
                            "confidence": "TEST_CONTRACT"}
    model["environment_required"] = False
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

@_SKIP
def test_a33_01_deleted_assigned_alias_reports_the_selected_alias(tmp_path, monkeypatch):
    """Same prefix, assigned `bin/python` DELETED, selected `bin/python3.10` exists."""
    module = _manage("manage_environment_a33_deleted_alias")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, "env")            # .../bin/python
    selected = assigned.parent / "python3.10"         # .../bin/python3.10 (same prefix)
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    os.remove(assigned)                               # the assigned alias is stale
    # The FIXTURE contract: the selected alias is independently viable, so deleting the assigned one
    # cannot be what makes this test pass.
    assert selected.is_file() and os.access(selected, os.X_OK), selected
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)

    assert status["interpreter"] == str(selected), status
    assert Path(status["interpreter"]).is_file(), status          # never a dangling path
    assert status["ready"] is True, status
    assert status["interpreter_status"] == "RUNTIME_READY", status
    assert status["shared_interpreter"] == str(assigned), status  # metadata only

@_SKIP
def test_a33_01_existing_assigned_alias_still_reports_the_selected_alias(tmp_path, monkeypatch):
    """Both aliases exist; the operator chose `bin/python3.10`, so that path is reported."""
    module = _manage("manage_environment_a33_both_alias")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, "env")
    selected = assigned.parent / "python3.10"
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    assert assigned.is_file() and selected.is_file()
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)

    assert status["interpreter"] == str(selected), status
    assert Path(status["interpreter"]).is_file(), status
    assert status["ready"] is True and status["interpreter_status"] == "RUNTIME_READY", status

@_SKIP
def test_a33_01_no_override_reports_the_assignment(tmp_path, monkeypatch):
    """Unchanged behaviour: with no selection the assignment path decides and carries its verdict."""
    module = _manage("manage_environment_a33_no_override")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, "env")
    _install_stub(assigned, "stubdep", "1.2.3")
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    monkeypatch.delenv("WORKBENCH_TME_PYTHON", raising=False)

    status = module.status(model, config)

    assert status["interpreter"] == str(assigned), status
    assert Path(status["interpreter"]).is_file(), status
    assert status["ready"] is True and status["runtime_verified"] is True, status

@_SKIP
def test_a33_01_different_prefix_selection_still_wins(tmp_path, monkeypatch):
    """The already-fixed case stays green: a selection in a DIFFERENT environment wins."""
    module = _manage("manage_environment_a33_other_prefix")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, "old_env")
    selected = _python310(tmp_path, "new_env")
    _install_stub(selected, "stubdep", "1.2.3")
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    assert assigned.parent != selected.parent
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))

    status = module.status(model, config)

    assert status["interpreter"] == str(selected), status
    assert Path(status["interpreter"]).is_file(), status
    assert status["ready"] is True, status
    assert status["shared_interpreter"] == str(assigned), status

@_SKIP
def test_a33_01_stale_assigned_alias_does_not_reinstall_the_environment(
        tmp_path, monkeypatch, capsys):
    """A real run with a stale same-prefix alias installs into the SELECTED venv, nothing else."""
    from workbench.backend import environment_planner as planner

    module = _manage("manage_environment_a33_no_reinstall")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, "env")
    selected = assigned.parent / "python3.10"
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    os.remove(assigned)
    assert selected.is_file() and os.access(selected, os.X_OK), selected
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))
    spec = tmp_path / "requirements.txt"
    spec.write_text("stubdep==1.2.3\n", encoding="utf-8")

    installed: list[list[str]] = []
    real_run = module.subprocess.run

    def fake_run(argv, **kwargs):
        if "-m" in argv and "pip" in argv:            # fake only the install; probes stay real
            installed.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("the group must not be provisioned for a usable selected alias")

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(planner, "provision_environment_plan", forbidden)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status = {"model_id": model["model_id"], "source_synced": True, "ready": False,
              "env_var": "WORKBENCH_TME_PYTHON", "env_spec_path": str(spec),
              "env_spec_type": "requirements.txt"}
    for network_allowed in (True, False):
        capsys.readouterr()
        assert module.create_env(model, status, dry_run=True, config=config,
                                 network_allowed=network_allowed) is True
        out = capsys.readouterr().out
        assert "would provision or reuse the selected interpreter" in out, out
        assert "requires network authorization" not in out, out
    assert installed == [], installed                 # the dry run installed nothing

    assert module.create_env(model, status, dry_run=False, config=config,
                             network_allowed=True) is True
    out = capsys.readouterr().out
    assert installed and installed[0][0] == str(selected), (installed, out)
    assert "[GROUP]" not in out, out

@_SKIP
def test_a33_01_interrupted_staging_prefix_is_never_installed_into(
        tmp_path, monkeypatch, capsys):
    """An INTERRUPTED STAGING assignment must be refused, not repointed at its selected alias.

    HEAD refuses this shape -- the assignment branch runs and the group cannot be adopted or
    provisioned -- and nothing is installed. The alias precedence must not change that: letting it
    fire here pointed pip at a `.staging-*` interpreter, adopting exactly the interrupted staging
    prefix every other adoption path refuses (`_provision_group`: "Interrupted staging prefixes
    cannot be adopted as environments").
    """
    from workbench.backend import environment_planner as planner

    module = _manage("manage_environment_a33_staging")
    config = _config(tmp_path)
    assigned = _python310(tmp_path, ".staging-groups/cir-abc")   # interrupted staging prefix
    selected = assigned.parent / "python3.10"
    model = _stateful_model()
    _write_assignment(config.WORKBENCH_ARTIFACTS_ROOT, model["model_id"], assigned)
    os.remove(assigned)                                          # the assigned alias is stale
    monkeypatch.setenv("WORKBENCH_TME_PYTHON", str(selected))
    assert module._is_staging_path(str(selected)), selected

    installed: list[list[str]] = []
    real_run = module.subprocess.run

    def fake_run(argv, **kwargs):
        if "-m" in argv and "pip" in argv:                       # fake only the install
            installed.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(planner, "_solver_command", lambda: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status = {"model_id": model["model_id"], "source_synced": True, "ready": False,
              "env_var": "WORKBENCH_TME_PYTHON",
              "env_spec_path": str(tmp_path / "third_party" / "TME" / "requirements.txt"),
              "env_spec_type": "requirements.txt"}
    # HEAD's verdict for this shape: no stage is adopted, and the run does not succeed.
    assert module.create_env(model, status, dry_run=False, config=config,
                             network_allowed=True) is False
    out, err = capsys.readouterr()
    assert installed == [], (installed, out, err)                # never pip into the staging prefix
    assert "[GROUP]" in out, out                                 # the assignment branch handled it
    assert str(selected) not in out, (out, err)
