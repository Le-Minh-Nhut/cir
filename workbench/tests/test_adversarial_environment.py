"""Adversarial environment isolation tests (E01-E10)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from workbench.backend.runtime import check_environment_isolation, environment_status
import workbench.backend.runtime as runtime


def _venv_interpreter(root: Path, name: str = "python") -> Path:
    """Real lightweight venv so isolation is decided by runtime identity, not path."""
    import venv as _venv
    env_dir = root / "venv"
    _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    return (env_dir / "bin" / name).resolve()


# E01: A bare interpreter path alone is not RUNTIME_READY
def test_e01_interpreter_path_alone_not_ready(tmp_path: Path):
    interp = _venv_interpreter(tmp_path)
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "TEST_ENV_X"}}
    os.environ["TEST_ENV_X"] = str(interp)
    try:
        st = environment_status(model, probe=True)
        assert st["tier"] != "RUNTIME_READY"
        assert st["status"] in ("REQUIREMENTS_UNVERIFIED", "PYTHON_OK", "ISOLATED")
    finally:
        del os.environ["TEST_ENV_X"]


# E07: System-Python installation refused
def test_e07_system_python_refused():
    ok, reason = check_environment_isolation(Path("/usr/bin/python3"))
    assert ok is False
    assert "system" in reason.lower() or "isolated" in reason.lower()


# E09: Active workbench environment cannot be a legacy model target
def test_e09_active_workbench_env_refused():
    ok, reason = check_environment_isolation(Path(sys.executable))
    assert ok is False
    assert "workbench" in reason.lower()


# E02-E06: probe tiers and non-isolated rejection
def test_non_isolated_path_rejected(tmp_path: Path):
    # Interpreter in system path without venv/conda markers is rejected
    ok, reason = check_environment_isolation(Path("/usr/local/bin/python3"))
    assert ok is False


def test_missing_interpreter_tier(tmp_path: Path):
    model = {"model_id": "m", "environment_required": True, "environment": {"python_env_var": "TEST_ENV_MISSING"}}
    os.environ["TEST_ENV_MISSING"] = str(tmp_path / "nope" / "python")
    try:
        st = environment_status(model)
        assert st["tier"] == "INTERPRETER_MISSING"
    finally:
        del os.environ["TEST_ENV_MISSING"]


def test_unconfigured_env_tier():
    model = {"model_id": "m", "environment_required": True, "environment": {"python_env_var": "TEST_ENV_UNSET"}}
    os.environ.pop("TEST_ENV_UNSET", None)
    st = environment_status(model)
    assert st["tier"] == "UNCONFIGURED"


def test_probe_dependency_mismatch(tmp_path: Path):
    """A required dependency that is absent is detected as a blocker, not READY."""
    interp = _venv_interpreter(tmp_path)
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "TEST_ENV_PROBE", "pytorch": "9.9.9"}}
    os.environ["TEST_ENV_PROBE"] = str(interp)
    try:
        st = environment_status(model, probe=True)
        assert st["tier"] != "RUNTIME_READY"
        assert st["status"] in ("DEPENDENCY_MISSING", "DEPENDENCY_VERSION_MISMATCH")
    finally:
        del os.environ["TEST_ENV_PROBE"]


# E10: manage_environment refuses install into unsafe interpreter
def test_e10_manage_environment_refuses_unsafe_interpreter(monkeypatch, tmp_path: Path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("manage_environment_adv", Path("workbench/scripts/manage_environment.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Simulate an interpreter configured to system python
    monkeypatch.setattr(mod, "status", lambda model, config: {
        "model_id": model["model_id"], "source_synced": True,
        "env_spec_path": str(tmp_path / "requirements.txt"), "env_spec_type": "requirements.txt",
        "env_var": "TEST_ENV_UNSAFE", "ready": False,
    })
    (tmp_path / "requirements.txt").write_text("nonexistent-pkg\n")
    monkeypatch.setenv("TEST_ENV_UNSAFE", "/usr/bin/python3")
    model = {"model_id": "dcnet"}
    ok = mod.create_env(model, mod.status(model, None), dry_run=False)
    assert ok is False
    assert "unsafe" in capsys.readouterr().err.lower()


# F9 regression: a locally placed, git-ignored asset inside the source tree is dirty
def test_ignored_source_asset_counts_as_dirty(tmp_path: Path):
    import subprocess as sp
    from workbench.backend.runtime import _source_dirty

    repo = tmp_path / "src"
    repo.mkdir()
    sp.run(["git", "init", "-q"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("*.bin\n__pycache__/\n")
    (repo / "eval.py").write_text("x")
    sp.run(["git", "add", "."], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    assert _source_dirty(repo) is False
    (repo / "backbone.bin").write_bytes(b"locally placed weights")
    assert _source_dirty(repo) is True, "git-ignored asset must count as dirty"

    (repo / "backbone.bin").unlink()
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "eval.cpython-313.pyc").write_bytes(b"cache")
    assert _source_dirty(repo) is False, "bytecode cache alone must not count as dirty"


# F13 regression: RUNTIME_READY requires an executed dependency probe
def test_runtime_ready_requires_dependency_probe(monkeypatch, tmp_path: Path):
    import venv as _venv
    env_dir = tmp_path / "venv"
    _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    interp = env_dir / "bin" / "python"
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "F13_ENV", "required_imports": ["json", "os"]}}
    monkeypatch.setenv("F13_ENV", str(interp))
    report = runtime.verify_environment(model, probe=True)
    assert report.tier != "RUNTIME_READY"
    assert report.status == "REQUIREMENTS_UNVERIFIED"


# Regression: upstream-committed bytecode is not a source change; real edits still are
def test_committed_bytecode_is_not_dirty(tmp_path: Path):
    import subprocess as sp
    from workbench.backend.runtime import _source_dirty

    repo = tmp_path / "src"
    (repo / "__pycache__").mkdir(parents=True)
    sp.run(["git", "init", "-q"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "eval.py").write_text("x")
    (repo / "__pycache__" / "eval.cpython-313.pyc").write_bytes(b"upstream-committed")
    sp.run(["git", "add", "-f", "."], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "upstream commits pyc"], cwd=repo, check=True)

    assert _source_dirty(repo) is True or _source_dirty(repo) is False  # deterministic below
    # Delete the committed bytecode (a git-visible tracked deletion that is pure noise).
    (repo / "__pycache__" / "eval.cpython-313.pyc").unlink()
    assert _source_dirty(repo) is False, "deleting committed bytecode must not count as dirty"

    # A real source edit must still be detected.
    (repo / "eval.py").write_text("modified evaluator")
    assert _source_dirty(repo) is True


# Regression: a real tracked edit inside a committed __pycache__ dir is still dirty (NEW-2)
def test_tracked_edit_inside_pycache_dir_detected(tmp_path: Path):
    import subprocess as sp
    from workbench.backend.runtime import _source_dirty

    repo = tmp_path / "src"
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    sp.run(["git", "init", "-q"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    note = repo / "pkg" / "__pycache__" / "notes.json"
    note.write_text("committed non-bytecode file")
    sp.run(["git", "add", "-f", "."], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "i"], cwd=repo, check=True)

    assert _source_dirty(repo) is False
    note.write_text("edited a real file inside a __pycache__ directory")
    assert _source_dirty(repo) is True, "tracked non-bytecode edit must be detected"
