"""A36-01: a memo-warm verification must not hand out a stale capability payload as identity.

`runner.model_interpreter_identity` used to read the environment digest out of
`report.gpu_capability["interpreter_fingerprint"]` -- the payload of a *signed capability record*.
When `runtime._reuse` answered from its in-process memo, that payload came from a report produced
before the environment changed, and the cheap advisory fingerprint check could not tell: a
same-length in-place rewrite of an installed dependency with its mtime restored passes it. The
composition run id (and every rank/feature cache keyed by it) therefore stayed fixed while the
environment's content changed.

SYNTHETIC, TEST-ONLY EVIDENCE: the GPU capability record here is written by the PRODUCTION writer
`runtime.record_gpu_capability` with `probe_source="override"` under the documented test opt-in
`WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1`. No GPU is probed and nothing about a real CUDA host is
claimed; the records are signed but their capability claim is synthetic and MUST NEVER be treated
as research evidence. The interpreter, its installed dependency and the subprocess import are all
real.

Everything runs through the real composition CLI and the real `model_interpreter_identity`;
`verify_environment`, `_reuse` and `model_interpreter_identity` are never patched.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from workbench.backend import model_settings, runtime
from workbench.composition import runner
from workbench.tests.test_a33_interpreter_alias import _install_stub
from workbench.tests.test_circus_cli import _fixture, _log

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: A real CPython 3.10 -- the interpreter the adapted contract is verified against.
_PY310 = next((candidate for candidate in (
    Path("/usr/local/bin/python3.10"),
    Path("/usr/bin/python3.10"),
) if candidate.exists() and os.access(candidate, os.X_OK)
    and subprocess.run([str(candidate), "-c", "import sys;print(sys.version_info[:2])"],
                       capture_output=True, text=True).stdout.strip() == "(3, 10)"), None)

pytestmark = pytest.mark.skipif(_PY310 is None, reason="no CPython 3.10 available on this host")

#: The module name `_fixture(root, ("flat",))` writes and `adapters.adapter_class` imports.
_ADAPTER_MODULE = "synthetic_flat"

@pytest.fixture(autouse=True)
def _fresh_adapter_module():
    """Drop the dynamically-written adapter module around each test, both directions.

    `adapters.adapter_class` resolves the registry's `composition_adapter` with
    `importlib.import_module`, which caches the module in `sys.modules` for the whole pytest
    process. The fixture writes a DIFFERENT `synthetic_flat.py` per test directory, so without
    this the first file to run in a process decides which one every later file gets: this test
    would inherit `test_circus_cli`'s module (ranks [1,2,3,4], no `stubdep` import) and
    `test_circus_cli` would inherit this one's (`ModuleNotFoundError: No module named 'stubdep'`).
    """
    sys.modules.pop(_ADAPTER_MODULE, None)
    yield
    sys.modules.pop(_ADAPTER_MODULE, None)

#: The adapter reads its rank behaviour from the real installed dependency, through a REAL
#: subprocess into the interpreter the identity fingerprints.
_ADAPTER_SUBPROCESS = (
    "        import subprocess\n"
    "        runtime_rank = int(subprocess.check_output("
    "[os.environ[\"WORKBENCH_TEST_MODEL_PYTHON\"], \"-c\", "
    "\"import stubdep; print(stubdep.RANK)\"], text=True).strip())\n"
    "        for index, text in enumerate(texts):\n")

def _venv(root: Path) -> Path:
    assert _PY310 is not None
    prefix = root / "model-env"
    subprocess.run([str(_PY310), "-m", "venv", "--without-pip", str(prefix)], check=True)
    return prefix / "bin" / "python"

def _dependency(interpreter: Path) -> Path:
    for lib in (interpreter.parent.parent / "lib").glob("python*"):
        return lib / "site-packages" / "stubdep.py"
    raise AssertionError("no site-packages in the test environment")

def _workspace(root: Path, interpreter: Path):
    """The real CLI fixture, with the model contract pointed at the real 3.10 environment."""
    config, release, adapters, artifacts = _fixture(root, ("flat",))
    registry_path = config.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    model = registry["models"][0]
    model["environment"].update(python="3.10", packages=["stubdep==1.2.3"], cuda="11.8")
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    adapter = adapters / "synthetic_flat.py"
    source = adapter.read_text(encoding="utf-8")
    source = source.replace("        for index, text in enumerate(texts):", _ADAPTER_SUBPROCESS)
    source = source.replace("[int(match.group(1)), int(match.group(2))]",
                            "[int(match.group(1)), runtime_rank]")
    assert "runtime_rank" in source, "the adapter rewrite did not apply"
    adapter.write_text(source, encoding="utf-8")
    return model, config, release, adapters, artifacts

def _invoke(config: Path, resume: bool) -> dict:
    from workbench.scripts import composition_eval

    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = composition_eval.main(
            ["--config", str(config), "--models", "flat", *(["--resume"] if resume else [])])
    if rc:
        raise AssertionError({"rc": rc, "stdout": stdout.getvalue()[-4000:],
                              "stderr": stderr.getvalue()[-2000:]})
    return json.loads(stdout.getvalue())

def _observe(config: Path, artifacts: Path, root: Path, resume: bool) -> dict:
    report = _invoke(config, resume)
    row = report["models"][0]
    ranks = json.loads((artifacts / "composition" / "runs" / row["run_id"] / "ranks" /
                        "flat_fixture.json").read_text(encoding="utf-8"))
    return {"run_id": row["run_id"], "reused": row["reused"], "status": row["status"],
            "ranks": [ranks["ranks"]["multimodal"][f"q{index}"] for index in range(4)],
            "environment_fingerprint": ranks["identity"]["runtime_identity"]["environment_fingerprint"],
            "encode_calls": len([entry for entry in _log(root / "encodes.jsonl")
                                 if entry["kind"] in {"targets", "queries"}])}

def test_warm_memo_must_not_bind_a_stale_capability_payload(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKBENCH_NO_ENV_CACHE", raising=False)
    prefix = _venv(tmp_path)
    _install_stub(prefix, "torch", "2.7.1")
    _install_stub(prefix, "stubdep", "1.2.3")
    dependency = _dependency(prefix)
    dependency.write_text(dependency.read_text(encoding="utf-8") + "RANK = 1\n", encoding="utf-8")

    model, config, release, adapters, artifacts = _workspace(tmp_path, prefix)
    log_path = tmp_path / "encodes.jsonl"
    monkeypatch.syspath_prepend(str(adapters))
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(config.parent / "registry.yaml"))
    monkeypatch.setenv("WORKBENCH_CIRCUS_RELEASE_ROOT", str(release))
    monkeypatch.setenv("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE", "1")
    monkeypatch.setenv("WORKBENCH_TEST_MODEL_PYTHON", str(prefix))
    monkeypatch.setenv("CIRCUS_TEST_LOG", str(log_path))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    # The documented test opt-in: it lets a synthetic record satisfy the GPU gate. Never set in
    # production, and the record written below is labelled `probe_source="override"`.
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")
    monkeypatch.setattr(model_settings, "file_settings", lambda: {})
    monkeypatch.setattr(runner, "_gpu_identity", lambda: None)
    runtime.forget_environment()
    runtime.forget_gpu_capability()

    from workbench.scripts import composition_eval

    cfg = composition_eval._load_configuration(config)[2]
    seeded_fingerprint = runtime._record_environment_fingerprint(model, str(prefix))
    record = runtime.record_gpu_capability(model, {
        "probe_source": "override", "interpreter": str(prefix),
        "interpreter_fingerprint": seeded_fingerprint,
        "dependency_fingerprint": runtime._environment_contract_digest(model),
        "capability_verified": True, "cuda_available": True, "torch_cuda_version": "11.8",
    }, cfg)
    assert record.get("signature"), "the synthetic capability record must be signed"

    # The warm path really is exercised: the FIRST call performs the full inspection and memoizes
    # it, the SECOND is served by `_reuse`'s advisory branch -- the branch that used to return the
    # cached report without validating the payload it carries.
    first_report = runtime.verify_environment(model, probe=True, advisory=True, config=cfg)
    warm_report = runtime.verify_environment(model, probe=True, advisory=True, config=cfg)
    assert warm_report is first_report, "the second verification must be served from the memo"
    assert warm_report.gpu_capability == record, "the warm report carries the seeded payload"
    assert getattr(warm_report, "validated_environment_fingerprint", None) == seeded_fingerprint

    # The AUTHORITATIVE branch of `_reuse` re-derives the byte-exact digest from the current
    # content and proves the cached one unchanged, so a report served from there must expose that
    # digest as validated for this call rather than None.
    auth_first = runtime.verify_environment(model, probe=True, authoritative=True, config=cfg)
    assert auth_first.validated_environment_fingerprint == seeded_fingerprint
    auth_warm = runtime.verify_environment(model, probe=True, authoritative=True, config=cfg)
    assert auth_warm is auth_first, "the second authoritative verification must come from the memo"
    assert auth_warm.validated_environment_fingerprint is not None
    assert auth_warm.validated_environment_fingerprint == auth_first.environment_fingerprint

    first = _observe(config, artifacts, tmp_path, resume=False)
    assert first["ranks"] == [1, 1, 1, 1], first
    assert first["encode_calls"] == 4, first
    assert first["environment_fingerprint"] == seeded_fingerprint, first

    unchanged = _observe(config, artifacts, tmp_path, resume=True)
    assert unchanged["run_id"] == first["run_id"] and unchanged["reused"] is True, unchanged
    assert unchanged["encode_calls"] == first["encode_calls"], unchanged
    assert unchanged["ranks"] == [1, 1, 1, 1], unchanged

    # Same path, same version, same byte LENGTH, mtime moved forward: the cheap advisory check
    # (name fold, RECORD text, member mtimes) cannot see the rewrite, the authoritative digest can.
    dependency.write_text(dependency.read_text(encoding="utf-8").replace("RANK = 1", "RANK = 4"),
                          encoding="utf-8")
    stat = dependency.stat()
    os.utime(dependency, (stat.st_atime, stat.st_mtime + 2))
    assert subprocess.check_output(
        [str(prefix), "-c", "import stubdep; print(stubdep.RANK)"], text=True).strip() == "4"

    changed = _observe(config, artifacts, tmp_path, resume=True)
    assert changed["run_id"] != first["run_id"], changed
    assert changed["reused"] is False, changed
    assert changed["ranks"] == [4, 4, 4, 4], changed
    assert changed["encode_calls"] == 8, changed
    assert changed["environment_fingerprint"] != seeded_fingerprint, changed

    fresh = _observe(config, artifacts, tmp_path, resume=False)
    assert fresh["ranks"] == [4, 4, 4, 4], fresh
    assert fresh["run_id"] == changed["run_id"], (fresh, changed)

    # The cold control: with both memos dropped the resumed answer is still correct, so the checks
    # above are not passing merely because everything recomputes.
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    cold = _observe(config, artifacts, tmp_path, resume=True)
    assert cold["ranks"] == [4, 4, 4, 4], cold
    assert cold["environment_fingerprint"] != seeded_fingerprint, cold
