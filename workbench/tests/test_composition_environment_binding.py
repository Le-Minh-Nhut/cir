"""A35-C: the composition run identity must bind the EXECUTION environment's content.

The interpreter PATH, its Python/Torch versions and the verification metadata were all recorded,
but nothing in the run identity changed when the installed dependency CONTENT changed. The
authoritative environment fingerprint (`runtime._interpreter_fingerprint(..., authoritative=True)`)
does change, so the fix is to bind it into the semantic identity (run_id, rank cache, feature
caches) before the reuse decision.

Every check here runs through the REAL composition CLI and the REAL `model_interpreter_identity`;
the adapter executes the SAME interpreter the identity fingerprints (a real CPython 3.10 venv with
a real importable stub dependency), so nothing about the seam is mocked.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from workbench.composition import runner
from workbench.tests.test_circus_cli import _fixture, _log

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: A real CPython 3.10 with a real, isolated, importable dependency (same discovery as A31).
_PY310 = next((candidate for candidate in (
    Path("/usr/local/bin/python3.10"),
    Path("/tmp/a30-verify/venv310/bin/python"),
) if candidate.exists() and os.access(candidate, os.X_OK)
    and subprocess.run([str(candidate), "-c", "import sys;print(sys.version_info[:2])"],
                       capture_output=True, text=True).stdout.strip() == "(3, 10)"), None)

pytestmark = pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")

#: The adapter is a subprocess client of its model interpreter, like every real adapter: the rank
#: it reports IS the dependency value read from that interpreter's own environment.
ADAPTER_SOURCE = '''from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from workbench.composition.adapters import AdapterStatus, CompositionAdapter

MODEL_ID = "envbinding"
WIDTH = 8


def dependency_value(interpreter: str) -> int:
    probe = subprocess.run([interpreter, "-c", "import envdep; print(envdep.VALUE)"],
                           capture_output=True, text=True, check=True)
    return int(probe.stdout.strip())


def record(kind, value):
    path = os.environ.get("CIRCUS_TEST_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"model": MODEL_ID, "kind": kind, "value": value}) + "\\n")


class Synthetic(CompositionAdapter):
    model_id = MODEL_ID

    def __init__(self, config, checkpoint_id=None):
        super().__init__(config, checkpoint_id)
        # The interpreter this adapter actually spawns: the identity must fingerprint THIS path,
        # not a resolved base interpreter.
        self.interpreter = os.environ["WORKBENCH_TEST_MODEL_PYTHON"]

    @classmethod
    def status(cls, config=None):
        return AdapterStatus(cls.model_id, True, None, ("fixture",), {}, "synthetic env binding")

    def mode_supported(self, condition):
        return True, None

    def encode_targets(self, image_paths, categories):
        record("targets", dependency_value(self.interpreter))
        return np.zeros((len(image_paths), WIDTH), dtype=np.float32)

    def encode_queries(self, images, texts, categories):
        record("queries", dependency_value(self.interpreter))
        return np.zeros((len(images), WIDTH), dtype=np.float32)

    def rank(self, queries, targets, target_positions=None, categories=None):
        # Rank == the installed dependency's behaviour, so the cached rank is stale exactly when
        # the environment content it came from changed.
        value = dependency_value(self.interpreter)
        return [value for _ in target_positions]


ADAPTER = Synthetic
'''


def _venv310(root: Path) -> Path:
    """A real isolated CPython 3.10 environment."""
    assert _PY310 is not None
    env_dir = root / "model-env"
    subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(env_dir)], check=True)
    return env_dir / "bin" / "python"


def _site_packages(interpreter: Path) -> Path:
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        return site_packages
    raise AssertionError("no site-packages in the model environment")


def _install_dependency(interpreter: Path, value: int) -> Path:
    """A real importable distribution; its INSTALLED BYTES are the only thing that changes.

    The metadata version, the module path, and the byte LENGTH of the module are held constant, so
    an identity built from size/mtime/version cannot see the 1 -> 4 behaviour change; only the
    file's CONTENT differs.
    """
    site_packages = _site_packages(interpreter)
    module = site_packages / "envdep.py"
    module.write_text(f"VALUE = {value}\n", encoding="utf-8")
    # A stale ``.pyc`` would otherwise serve the OLD behaviour to a subprocess import, which is
    # not the change under test.
    shutil.rmtree(site_packages / "__pycache__", ignore_errors=True)
    dist = site_packages / "envdep-1.0.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: envdep\nVersion: 1.0\n")
    (dist / "RECORD").write_text(
        f"envdep.py,sha256=test,{len('VALUE = 0\\n')}\n"
        "envdep-1.0.dist-info/METADATA,sha256=test,49\n"
        "envdep-1.0.dist-info/RECORD,sha256=,0\n")
    return module


def _workspace(tmp_path: Path, model_python: Path):
    """The real CLI fixture, with the model contract pointed at the real 3.10 environment."""
    config, release, adapters, artifacts = _fixture(tmp_path, ("envbinding",))
    (adapters / "synthetic_envbinding.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
    registry_path = config.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["models"][0]["environment"] = {
        "python_env_var": "WORKBENCH_TEST_MODEL_PYTHON", "python": "3.10",
        "packages": ["envdep==1.0"], "cuda": "UNKNOWN"}
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return config, release, adapters, artifacts


def _run(config: Path, release: Path, adapters: Path, monkeypatch: pytest.MonkeyPatch,
         capsys: pytest.CaptureFixture[str], model_python: Path, *args: str) -> tuple[int, dict | None, str]:
    """The real CLI path; `runner.model_interpreter_identity` is deliberately NOT patched.

    Only the controller-side GPU probe is stubbed (it imports Torch and is unrelated to the
    environment identity under test).
    """
    monkeypatch.setattr(runner, "_gpu_identity", lambda: None)
    monkeypatch.syspath_prepend(str(adapters))
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(config.parent / "registry.yaml"))
    monkeypatch.setenv("WORKBENCH_CIRCUS_RELEASE_ROOT", str(release))
    monkeypatch.setenv("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE", "1")
    monkeypatch.setenv("WORKBENCH_TEST_MODEL_PYTHON", str(model_python))
    monkeypatch.setenv("CIRCUS_TEST_LOG", str(config.parent / "encodes.jsonl"))
    from workbench.scripts import composition_eval
    rc = composition_eval.main(["--config", str(config), *args])
    captured = capsys.readouterr()
    try:
        return rc, json.loads(captured.out), captured.err
    except json.JSONDecodeError:
        return rc, None, captured.err + captured.out


def _encodes(tmp_path: Path) -> int:
    return len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])


def _ranks(artifacts: Path, attempt: str) -> list[int | None]:
    matches = [path for path in (artifacts / "composition" / "results").rglob("multimodal.jsonl")
               if path.parent.parent.name == attempt and path.parent.name == "full"]
    assert len(matches) == 1, matches
    return [json.loads(line)["rank"] for line in matches[0].read_text(encoding="utf-8").splitlines()]


def _interpreter_identity(row: dict) -> dict:
    summary = json.loads(Path(row["summaries"]["full"]).read_text(encoding="utf-8"))
    return summary["runtime"]["model_interpreter"]


def test_dependency_content_change_invalidates_composition_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    model_python = _venv310(tmp_path)
    module = _install_dependency(model_python, 1)
    config, release, adapters, artifacts = _workspace(tmp_path, model_python)
    original_stat = module.stat()

    # (1) first run computes from dependency behaviour 1.
    rc, first, error = _run(config, release, adapters, monkeypatch, capsys, model_python,
                            "--models", "envbinding")
    assert rc == 0, error
    row = first["models"][0]
    assert row["status"] == "COMPLETED" and row["reused"] is False, row
    assert _ranks(artifacts, row["attempt"]) == [1, 1, 1, 1]
    first_count = _encodes(tmp_path)
    assert first_count == 4, _log(tmp_path / "encodes.jsonl")
    first_run_id = row["run_id"]
    first_identity = _interpreter_identity(row)
    assert isinstance(first_identity.get("environment_fingerprint"), str), first_identity
    assert len(first_identity["environment_fingerprint"]) == 64, first_identity

    # (2) unchanged resume: same identity, reused, nothing re-encoded.
    rc, resumed, error = _run(config, release, adapters, monkeypatch, capsys, model_python,
                              "--models", "envbinding", "--resume")
    assert rc == 0, error
    assert resumed["models"][0]["status"] == "REUSED", resumed["models"][0]
    assert resumed["models"][0]["run_id"] == first_run_id
    assert _ranks(artifacts, resumed["models"][0]["attempt"]) == [1, 1, 1, 1]
    assert _encodes(tmp_path) == first_count

    # (3) dependency CONTENT changes 1 -> 4: same version, same path, same byte LENGTH, and the
    #     original mtime restored, so only the installed bytes differ. The stale bytecode is
    #     dropped the way a real reinstall's fresh source read would be, so the interpreter really
    #     does return the new behaviour.
    module.write_text("VALUE = 4\n", encoding="utf-8")
    shutil.rmtree(module.parent / "__pycache__", ignore_errors=True)
    os.utime(module, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    rc, changed, error = _run(config, release, adapters, monkeypatch, capsys, model_python,
                              "--models", "envbinding", "--resume")
    assert rc == 0, error
    changed_row = changed["models"][0]
    assert changed_row["reused"] is False, changed_row
    assert changed_row["run_id"] != first_run_id, changed_row
    assert _ranks(artifacts, changed_row["attempt"]) == [4, 4, 4, 4]
    assert _encodes(tmp_path) == first_count + 4
    changed_identity = _interpreter_identity(changed_row)
    assert changed_identity["environment_fingerprint"] != first_identity["environment_fingerprint"]
    assert changed_identity["path"] == first_identity["path"]
    assert changed_identity["version"] == first_identity["version"]

    # (4) a fresh run under the SAME changed environment agrees with the resumed one.
    fresh_count = _encodes(tmp_path)
    rc, fresh, error = _run(config, release, adapters, monkeypatch, capsys, model_python,
                            "--models", "envbinding")
    assert rc == 0, error
    assert fresh["models"][0]["status"] == "COMPLETED", fresh["models"][0]
    assert fresh["models"][0]["run_id"] == changed_row["run_id"]
    assert _ranks(artifacts, fresh["models"][0]["attempt"]) == [4, 4, 4, 4]
    assert _encodes(tmp_path) == fresh_count + 4
