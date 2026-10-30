"""Execution identity must bind the interpreter the runtime resolver will actually launch.

A provisioning assignment (`artifacts/environment_assignment.json`) is normally the only place a
manager subprocess leaves the venv it created: it does not export `WORKBENCH_*_PYTHON` back to the
parent. `runtime.resolve_model_interpreter` therefore resolves process environment > local
`workbench.env` > assignment, while `_execution_identity` used to read only the setting and record
`UNCONFIGURED`. The evaluator then ran under the assigned venv but the fingerprint described no
environment at all, so a dependency rewrite inside that venv resumed a result produced by different
package content.

Each test drives the real `pipeline.run_stages` with a real venv, a real stage command and a real
stage log; the calls file is the evaluator's own output, not a mocked digest.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from workbench.backend import model_settings, runtime
from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import pipeline
from workbench.tests.test_execution_identity import _workspace

ENV_VAR = "WORKBENCH_ASSIGNMENT_IDENTITY_PYTHON"
MODEL_ID = "assignment_identity"
CODE = (
    "import demo, sys\n"
    "from pathlib import Path\n"
    "Path(sys.argv[1]).write_text(str(demo.value))\n"
    "with Path(sys.argv[2]).open('a') as handle:\n"
    "    handle.write(str(demo.value) + '\\n')\n"
)


class _Assigned:
    """A workspace whose model's interpreter is reachable only through the assignment file."""

    def __init__(self, cfg: WorkbenchConfig, stage: pipeline.Stage, model: dict, prefix: Path,
                 output: Path, calls: Path, state: Path) -> None:
        self.cfg = cfg
        self.stage = stage
        self.model = model
        self.prefix = prefix
        self.output = output
        self.calls = calls
        self.state = state

    @property
    def interpreter(self) -> Path:
        return self.prefix / "bin" / "python"

    @property
    def module(self) -> Path:
        return next(self.prefix.glob("lib/python*/site-packages")) / "demo.py"


def _model(environment_required: bool = True) -> dict:
    return {"model_id": MODEL_ID, "source_dir": None, "environment_required": environment_required,
            "environment": {"python_env_var": ENV_VAR, "packages": ["demo==1.0"], "cuda": "UNKNOWN"}}


def _venv(prefix: Path, value: str) -> Path:
    """A real venv without pip, carrying one importable distribution."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prefix)], check=True)
    site_packages = next(prefix.glob("lib/python*/site-packages"))
    (site_packages / "demo.py").write_text(f"value = {value}\n")
    dist = site_packages / "demo-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: demo\nVersion: 1.0\n")
    (dist / "RECORD").write_text("demo.py,sha256=unused,10\n")
    return prefix / "bin" / "python"


def _write_assignment(path: Path, interpreter: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "models": {MODEL_ID: {"state": "RUNTIME_VERIFIED", "interpreter": str(interpreter),
                              "environment_id": "test"}},
        "groups": []}), encoding="utf-8")


@pytest.fixture
def assigned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Assigned:
    model = _model()
    cfg, stage = _workspace(tmp_path, monkeypatch, model)
    cfg = replace(cfg, WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    prefix = cfg.WORKBENCH_ARTIFACTS_ROOT / "environments" / "model"
    interpreter = _venv(prefix, "1")
    _write_assignment(cfg.WORKBENCH_ARTIFACTS_ROOT / "environment_assignment.json", interpreter)
    output, calls = tmp_path / "rank.txt", tmp_path / "calls.txt"
    stage = replace(stage, commands=([str(interpreter), "-B", "-c", CODE, str(output), str(calls)],),
                    output_paths=(output,))
    # Only the persisted assignment may resolve: no process variable, no workbench.env value.
    monkeypatch.setattr(model_settings, "file_settings", lambda: {})
    monkeypatch.delenv(ENV_VAR, raising=False)
    return _Assigned(cfg, stage, model, prefix, output, calls, tmp_path / "pipeline-state.json")


def _run(work: _Assigned, *, resume: bool, state: Path | None = None) -> tuple[int, str]:
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        code = pipeline.run_stages(
            [work.stage], apply=True, authorized_capabilities={"allow_gpu_eval"}, resume=resume,
            config=work.cfg, state_path=state or work.state)
    return code, log.getvalue()


def _lines(log: str, marker: str) -> int:
    return sum(1 for line in log.splitlines() if marker in line)


def _identity_digest(model: dict, cfg: WorkbenchConfig) -> str:
    return pipeline._execution_identity(model, cfg)["digest"]


def _identity_stream(model: dict, cfg: WorkbenchConfig) -> str:
    """The digest stream `_execution_identity` feeds, captured instead of reimplemented."""
    import hashlib as real

    chunks: list[bytes] = []

    class _Recorder:
        def __init__(self, inner) -> None:
            self._inner = inner

        def update(self, data, *args, **kwargs):
            chunks.append(bytes(data))
            return self._inner.update(data, *args, **kwargs)

        def hexdigest(self) -> str:
            return self._inner.hexdigest()

    class _Wrapper:
        def sha256(self, *args, **kwargs):
            return _Recorder(real.sha256(*args, **kwargs))

        def __getattr__(self, name):
            return getattr(real, name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "hashlib", _Wrapper())
        pipeline._execution_identity(model, cfg)
    return b"".join(chunks).decode("utf-8", "replace")


def test_identity_agrees_with_resolver_and_binds_assignment_target(
        assigned: _Assigned, monkeypatch: pytest.MonkeyPatch) -> None:
    model, cfg = assigned.model, assigned.cfg
    interpreter = assigned.interpreter
    assert runtime.resolve_model_interpreter(model, cfg) == str(interpreter), "fixture must resolve"
    assert f"INTERPRETER:{interpreter}\0" in _identity_stream(model, cfg)

    # Agreement: the assignment path and an explicit setting naming the same venv describe the same
    # interpreter, so the identity must be identical. Before the fix the assignment-only digest fell
    # to UNCONFIGURED and did not.
    via_assignment = _identity_digest(model, cfg)
    monkeypatch.setenv(ENV_VAR, str(interpreter))
    assert _identity_digest(model, cfg) == via_assignment
    assert runtime.resolve_model_interpreter(model, cfg) == str(interpreter)

    # And the resolved environment's content is what the identity folds: a rewrite inside the venv
    # moves it in both selection modes.
    assigned.module.write_text("value = 7\n")
    changed = _identity_digest(model, cfg)
    assert changed != via_assignment
    monkeypatch.delenv(ENV_VAR)
    assert _identity_digest(model, cfg) == changed


def test_assignment_resolved_dependency_rewrite_invalidates_resume(assigned: _Assigned) -> None:
    code, log = _run(assigned, resume=False)
    assert code == 0, log
    assert assigned.output.read_text() == "1"
    assert assigned.calls.read_text().splitlines() == ["1"]
    assert _lines(log, "[RUN]") == 1 and _lines(log, "[SKIP] resume") == 0

    code, log = _run(assigned, resume=True)
    assert code == 0, log
    assert _lines(log, "[SKIP] resume") == 1 and _lines(log, "[RUN]") == 0
    assert assigned.output.read_text() == "1"
    assert assigned.calls.read_text().splitlines() == ["1"]

    # Only the installed bytes move: same dist METADATA/RECORD, same interpreter, same paths.
    assigned.module.write_text("value = 2\n")
    code, log = _run(assigned, resume=True)
    assert code == 0, log
    assert _lines(log, "[RUN]") == 1 and _lines(log, "[SKIP] resume") == 0
    assert assigned.output.read_text() == "2"
    assert assigned.calls.read_text().splitlines() == ["1", "2"]

    # A fresh run under the rewritten environment produces the same answer as the re-run did.
    code, log = _run(assigned, resume=False, state=assigned.state.parent / "fresh-state.json")
    assert code == 0, log
    assert assigned.output.read_text() == "2"
    assert assigned.calls.read_text().splitlines() == ["1", "2", "2"]
    assert _lines(log, "[RUN]") == 1


def test_explicit_selection_still_binds_dependency_content(
        assigned: _Assigned, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(assigned.interpreter))
    before = _identity_digest(assigned.model, assigned.cfg)
    assert f"INTERPRETER:{assigned.interpreter}\0" in _identity_stream(assigned.model, assigned.cfg)
    assigned.module.write_text("value = 3\n")
    assert _identity_digest(assigned.model, assigned.cfg) != before


def test_assignment_target_change_invalidates_identity(assigned: _Assigned) -> None:
    before = _identity_digest(assigned.model, assigned.cfg)
    second = assigned.cfg.WORKBENCH_ARTIFACTS_ROOT / "environments" / "model-two"
    second_interpreter = _venv(second, "1")  # identical dependency content
    _write_assignment(assigned.cfg.WORKBENCH_ARTIFACTS_ROOT / "environment_assignment.json",
                      second_interpreter)
    assert runtime.resolve_model_interpreter(assigned.model, assigned.cfg) == str(second_interpreter)
    assert f"INTERPRETER:{second_interpreter}\0" in _identity_stream(assigned.model, assigned.cfg)
    assert _identity_digest(assigned.model, assigned.cfg) != before


def test_without_assignment_identity_stays_unconfigured(assigned: _Assigned) -> None:
    assignment = assigned.cfg.WORKBENCH_ARTIFACTS_ROOT / "environment_assignment.json"
    assert "ENVIRONMENT_CONTENT" in _identity_stream(assigned.model, assigned.cfg)
    assignment.unlink()
    assert runtime.resolve_model_interpreter(assigned.model, assigned.cfg) is None
    stream = _identity_stream(assigned.model, assigned.cfg)
    assert "INTERPRETER:UNCONFIGURED\0" in stream
    assert "ENVIRONMENT_CONTENT" not in stream
    # The declared variable is still set, so the missing interpreter is not silently replaced.
    unconfigured = _identity_digest(assigned.model, assigned.cfg)
    assigned.module.write_text("value = 4\n")
    assert _identity_digest(assigned.model, assigned.cfg) == unconfigured


def test_model_without_variable_falls_back_to_controller_interpreter(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = {"model_id": MODEL_ID, "source_dir": None, "environment_required": False}
    cfg, _stage = _workspace(tmp_path, monkeypatch, model)
    cfg = replace(cfg, WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    monkeypatch.setattr(model_settings, "file_settings", lambda: {})
    assert runtime.resolve_model_interpreter(model, cfg) is None
    assert f"INTERPRETER:{sys.executable}\0" in _identity_stream(model, cfg)
