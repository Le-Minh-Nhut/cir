"""The launch gate must READ GPU capability records from the root it WRITES them to.

`verify_gpu_runtime(model, config, ...)` persists the capability through `record_gpu_capability(...,
config)`, but its reuse check called `_cached_gpu_capability(...)` with no configuration, so the read
resolved the PROCESS default while the write went to the caller's workspace. An operator whose
artifacts root is not the process default (a second workspace, a test, a container with the tree
mounted somewhere else) therefore never found the record it had just written: every launch re-probed
the GPU, and a record from a different workspace could be consumed in its place.

These tests use a real interpreter created with `venv` and the runtime's own test probe hook, so they
need no GPU and no checkpoint.
"""
from __future__ import annotations

import os
import venv
from dataclasses import replace
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.operator_config import resolve_config

PROBE_VARIABLE = "WORKBENCH_R15_GPU_PYTHON"


def _model(interpreter: Path) -> dict:
    return {
        "model_id": "r15-gpu-model",
        "environment_required": True,
        # `required_imports` decides whether the environment is usable at all: an empty contract
        # leaves the probe blocked with DEPENDENCY_MISSING before the capability logic runs, which
        # would make these tests measure the wrong thing.
        "environment": {"python_env_var": PROBE_VARIABLE, "pytorch": "1.12.1", "cuda": "11.7",
                        "required_imports": ["json"]},
    }


@pytest.fixture
def gpu_model(tmp_path: Path, monkeypatch) -> dict:
    """A real interpreter plus a deterministic probe, so a record can be signed for envs that exist."""
    from workbench.tests._core_fixture import build_core_module

    environment = tmp_path / "envs" / "probe"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = environment / "bin" / "python"
    site = next((environment / "lib").glob("python*")) / "site-packages"
    # The dependency contract is checked against the environment's real contents, so the fixture
    # environment needs an importable `torch` with distribution metadata and a `torch._C` the
    # orchestrator can really load -- a placeholder file is refused, which is the point of the
    # corroboration these tests sit downstream of.
    for name, version in (("torch", "1.12.1"), ("torchvision", "0.13.1")):
        (site / f"{name}.py").write_text(f"__version__ = '{version}'\n")
        dist = site / f"{name}-{version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        (dist / "RECORD").write_text(f"{name}.py,sha256=t,{(site / f'{name}.py').stat().st_size}\n")
    build_core_module(site / "_C_fixture.so")
    monkeypatch.setenv(PROBE_VARIABLE, str(interpreter))
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")

    def probe(model, report):  # noqa: ANN001 - the hook's own signature
        # The corroboration loads `torch_core_origin` itself and requires it to live inside the
        # environment whose dependencies were verified, so the synthetic probe reports the fixture's
        # real module and extension paths rather than claiming success outright.
        return {
            "probe": "gpu_capability", "cuda_available": True, "device_count": 1,
            "device_name": "Synthetic probe device", "capability_verified": True,
            "allocated_tensor": True, "computed": True, "error": None,
            "torch_has_compiled_core": True,
            "torch_module_file": str(site / "torch.py"),
            "torch_core_origin": str(site / "_C_fixture.so"),
        }

    runtime.set_gpu_probe_for_tests(probe)
    runtime.forget_gpu_capability()
    try:
        yield _model(interpreter)
    finally:
        runtime.clear_gpu_probe_for_tests()
        runtime.forget_gpu_capability()


def _workspace(tmp_path: Path, name: str) -> object:
    """A configuration whose artifacts root is NOT the process default."""
    artifacts = tmp_path / name / "artifacts"
    artifacts.mkdir(parents=True)
    return replace(resolve_config(), WORKBENCH_ARTIFACTS_ROOT=artifacts,
                   CIR_REPO_ROOT=tmp_path / name)


def test_the_reuse_check_reads_the_root_the_record_was_written_to(tmp_path, gpu_model):
    """The discriminating case: the record exists in the caller's workspace only.

    The process default is left WITHOUT a record. A read that resolves the process default finds
    nothing and re-probes; the assertion on the probe count is what makes this load-bearing.
    """
    config = _workspace(tmp_path, "operator")
    calls = 0
    original = runtime._GPU_PROBE_OVERRIDE

    def counted(model, report):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return original(model, report)

    runtime.set_gpu_probe_for_tests(counted)
    key = runtime._gpu_capability_key(gpu_model)
    record = runtime.gpu_capability_path(config, key)
    if record.exists():
        record.unlink()

    first = runtime.verify_gpu_runtime(gpu_model, config, authorized=True, require_cuda=True)
    assert first.get("capability_verified") is True
    assert record.is_file(), "the record must be persisted into the caller's workspace"
    assert calls == 1

    # A fresh process would have an empty in-memory cache: only the disk record can answer.
    runtime.forget_gpu_capability()
    second = runtime.verify_gpu_runtime(gpu_model, config, authorized=True, require_cuda=True,
                                        force=False)
    assert second.get("capability_verified") is True
    assert calls == 1, (
        "the reuse check re-probed the GPU: it read the capability back from the process default "
        f"instead of {config.WORKBENCH_ARTIFACTS_ROOT}")


def test_the_launch_gate_reads_the_root_the_record_was_written_to(tmp_path, gpu_model):
    """The gate that actually decides a launch, not just `verify_gpu_runtime`'s own reuse check.

    `require_launch_ready` is what `POST /api/jobs`, the job queue's launch guard and the evaluator
    call, and it reaches the capability through `verify_environment`. That path dropped the
    configuration, so a record written into the caller's workspace was missed and the launch was
    refused as unverified -- the reviewer's P1, reproduced here without `--verify-gpu`.
    """
    config = _workspace(tmp_path, "gate")
    runtime.verify_gpu_runtime(gpu_model, config, authorized=True, require_cuda=True)
    key = runtime._gpu_capability_key(gpu_model)
    assert runtime.gpu_capability_path(config, key).is_file()

    # A fresh process holds no in-memory verification: only the disk record can answer.
    runtime.forget_gpu_capability()
    runtime.forget_environment()
    interpreter = runtime.require_launch_ready(gpu_model, config=config)
    assert interpreter == os.environ[gpu_model["environment"]["python_env_var"]]


def test_the_launch_gate_refuses_when_the_workspace_holds_no_record(tmp_path, gpu_model):
    """The control: with the record absent from THIS workspace the gate must still refuse.

    Without this, a gate that simply stopped checking would pass the test above.
    """
    mine = _workspace(tmp_path, "empty")
    theirs = _workspace(tmp_path, "recorded")
    runtime.verify_gpu_runtime(gpu_model, theirs, authorized=True, require_cuda=True)
    runtime.forget_gpu_capability()
    runtime.forget_environment()

    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(gpu_model, config=mine)
    # The positive control: the workspace that holds the record is authorized.
    assert runtime.require_launch_ready(gpu_model, config=theirs)


def test_the_environment_memo_is_scoped_by_workspace(tmp_path, gpu_model, monkeypatch):
    """A verification performed for one workspace must not answer for another.

    The memo key carried the host but not the artifacts root, so whichever workspace was verified
    first in a process handed its GPU capability to a launch in a different one.
    """
    recorded = _workspace(tmp_path, "memo-recorded")
    empty = _workspace(tmp_path, "memo-empty")
    runtime.verify_gpu_runtime(gpu_model, recorded, authorized=True, require_cuda=True)

    # No `forget_environment()`: the memo is exactly what is under test.
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(gpu_model, config=empty)
    assert runtime.require_launch_ready(gpu_model, config=recorded)


def test_a_record_from_another_workspace_is_not_reused(tmp_path, gpu_model):
    """The inverse: a record in a DIFFERENT workspace must not certify this one.

    Reads resolve the caller's workspace, so a record that a different workspace holds for the same
    host is not evidence about this launch.
    """
    mine = _workspace(tmp_path, "mine")
    theirs = _workspace(tmp_path, "theirs")
    assert mine.WORKBENCH_ARTIFACTS_ROOT != theirs.WORKBENCH_ARTIFACTS_ROOT

    runtime.verify_gpu_runtime(gpu_model, theirs, authorized=True, require_cuda=True)
    foreign = runtime.gpu_capability_path(theirs, runtime._gpu_capability_key(gpu_model))
    assert foreign.is_file(), "the other workspace must actually hold a signed record"
    mine_path = runtime.gpu_capability_path(mine, runtime._gpu_capability_key(gpu_model))
    assert not mine_path.exists()

    runtime.forget_gpu_capability()
    report = runtime.verify_environment(gpu_model, probe=True, advisory=True)
    assert runtime._cached_gpu_capability(gpu_model, report, config=mine) is None, \
        "a capability recorded in another workspace must not be reused here"
    assert runtime._cached_gpu_capability(gpu_model, report, config=theirs) is not None, \
        "the positive control: the workspace that holds the record does reuse it"


def test_moving_the_implicit_default_workspace_revokes_a_borrowed_capability(
        tmp_path, gpu_model, monkeypatch):
    """The reported defect, driven exactly as an operator hits it: the implicit default moves.

    `require_launch_ready(model)` with no configuration resolves the process default itself. When
    that resolution happened lazily at each use, the environment memo and the capability memo were
    keyed by whatever root was current, so recording a capability with workspace A as the default
    and then moving the default to workspace B -- which holds no record -- still authorized the
    launch from A's memo. Scoping the key by the root is not enough on its own: the root has to be
    resolved ONCE at the entrance and used for the key, the read and the write.
    """
    config_a = _workspace(tmp_path, "implicit-a")
    config_b = _workspace(tmp_path, "implicit-b")

    def point_default_at(config):
        """Make `config`'s workspace the one the process resolves by default."""
        monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(config.WORKBENCH_ARTIFACTS_ROOT))
        monkeypatch.setenv("CIR_REPO_ROOT", str(config.CIR_REPO_ROOT))
        monkeypatch.setenv("WORKBENCH_RESULTS_ROOT", str(config.WORKBENCH_RESULTS_ROOT))

    point_default_at(config_a)
    runtime.verify_gpu_runtime(gpu_model, config_a, authorized=True, require_cuda=True)

    # 1. A is the default and holds the record: the gate authorizes without an explicit config.
    assert runtime.require_launch_ready(gpu_model) == os.environ[PROBE_VARIABLE]

    # 2. The default moves to B, which holds nothing. The same call must now refuse: a memo entry
    #    read for A must never answer for B.
    point_default_at(config_b)
    assert not list(runtime.gpu_capability_directory(config_b).glob("*.json")), \
        "workspace B must genuinely hold no capability record"
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(gpu_model)

    # 3. Moving the default back to A re-authorizes, so the refusal above is about the workspace
    #    and not about the process having lost the ability to reuse a record at all.
    point_default_at(config_a)
    assert runtime.require_launch_ready(gpu_model) == os.environ[PROBE_VARIABLE]

    # 4. An explicit configuration keeps its own meaning regardless of the default.
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(gpu_model, config=config_b)


def test_advisory_readiness_does_not_inherit_another_workspaces_gpu_status(
        tmp_path, gpu_model, monkeypatch):
    """A read-only readiness answer must not report another workspace's verified GPU.

    The advisory path is what a dashboard renders, so a borrowed `GPU_RUNTIME_READY` would tell an
    operator a workspace is ready when it is the neighbouring one that was probed.
    """
    recorded = _workspace(tmp_path, "advisory-recorded")
    empty = _workspace(tmp_path, "advisory-empty")
    runtime.verify_gpu_runtime(gpu_model, recorded, authorized=True, require_cuda=True)

    mine = runtime.verify_environment(gpu_model, probe=True, advisory=True, config=empty)
    theirs = runtime.verify_environment(gpu_model, probe=True, advisory=True, config=recorded)

    assert mine.gpu_capability is None, "the empty workspace must not report a verified GPU"
    assert mine.tier != "GPU_RUNTIME_READY"
    assert theirs.gpu_capability is not None, "the positive control: the recorded workspace reports it"


def test_the_process_default_still_works_when_no_config_is_supplied(tmp_path, gpu_model, monkeypatch):
    """Compatibility: callers that hold no configuration keep resolving the process default."""
    config = _workspace(tmp_path, "default")
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(config.WORKBENCH_ARTIFACTS_ROOT))
    monkeypatch.setenv("CIR_REPO_ROOT", str(config.CIR_REPO_ROOT))

    runtime.verify_gpu_runtime(gpu_model, config, authorized=True, require_cuda=True)
    runtime.forget_gpu_capability()
    report = runtime.verify_environment(gpu_model, probe=True, advisory=True)
    cached = runtime._cached_gpu_capability(gpu_model, report)
    assert cached is not None and cached.get("capability_verified") is True
