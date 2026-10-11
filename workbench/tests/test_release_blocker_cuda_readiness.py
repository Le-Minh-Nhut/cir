"""CUDA readiness state machine (CUDA-01..CUDA-14).

The laptop has no GPU. These tests therefore exercise the *production* state machine with a
deterministic probe injected through the test-only override, whose activation requires an
explicit environment opt-in so production execution can never select it. The real probe
implementation (``runtime.GPU_PROBE_CODE``) is asserted to exist, and CUDA-12 asserts that
the ordinary acceptance suite never runs it.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

import workbench.backend.runtime as runtime
from workbench.backend.operator_config import resolve_config


@pytest.fixture(autouse=True)
def _clean_gpu_state():
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    runtime.clear_gpu_probe_for_tests()
    yield
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    runtime.clear_gpu_probe_for_tests()


def _install_torch(directory: Path) -> None:
    for lib in (directory / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        module = site_packages / "torch.py"
        module.write_text("__version__ = '1.12.1'\n")
        dist = site_packages / "torch-1.12.1.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 1.12.1\n")
        (dist / "RECORD").write_text(f"torch.py,sha256=t,{module.stat().st_size}\n")
        return


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    """A real isolated environment whose contract declares a CUDA requirement."""
    import venv as _venv

    repo = tmp_path / "repo"
    repo.mkdir()
    interpreter_dir = repo / "limn_env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(interpreter_dir)
    _install_torch(interpreter_dir)

    monkeypatch.setenv("GPU_TEST_PYTHON", str(interpreter_dir / "bin" / "python"))
    config = replace(resolve_config(), CIR_REPO_ROOT=repo)
    model = {
        "model_id": "limn",
        "environment_required": True,
        "environment": {"python_env_var": "GPU_TEST_PYTHON", "pytorch": "1.12.1", "cuda": "12.4"},
    }
    return {"config": config, "model": model, "repo": repo,
            "interpreter": interpreter_dir / "bin" / "python"}


SUCCESS = {
    "cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
    "torch_cuda_version": "12.4", "device_name": "Synthetic GPU",
    "compute_capability": "8.9", "allocated_tensor": True, "computed": True, "error": None,
}
UNAVAILABLE = {
    "cuda_available": False, "device_count": 0, "torch_version": "1.12.1",
    "torch_cuda_version": None, "device_name": None, "compute_capability": None,
    "allocated_tensor": False, "computed": False, "error": "torch reports CUDA unavailable",
}
ALLOCATION_FAILED = {**SUCCESS, "allocated_tensor": False, "computed": False,
                     "error": "CUDA out of memory"}


def install_probe(payload: dict, monkeypatch) -> None:
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")
    runtime.set_gpu_probe_for_tests(lambda model, report: dict(payload))


# ------------------------------------------------------------------------------ CUDA-01

def test_cuda01_cpu_only_inspection_reports_pending(workspace):
    report = runtime.verify_environment(workspace["model"], probe=True)
    assert report.tier == "GPU_VERIFICATION_PENDING"
    assert report.status == "GPU_VERIFICATION_PENDING"
    assert report.cuda_available is None


# ------------------------------------------------------------------------------ CUDA-02

def test_cuda02_gpu_model_cannot_launch_without_verification(workspace):
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(workspace["model"])
    assert runtime.environment_status(workspace["model"], probe=True)["launch_ready"] is False


# ------------------------------------------------------------------------------ CUDA-03

def test_cuda03_authorized_successful_probe_verifies_capability(workspace, monkeypatch):
    install_probe(SUCCESS, monkeypatch)
    result = runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)

    assert result["capability_verified"] is True
    assert result["device_name"] == "Synthetic GPU"
    assert result["compute_capability"] == "8.9"
    assert result["probe_source"] == "override"
    assert result["probed_at"]

    report = runtime.verify_environment(workspace["model"], probe=True)
    assert report.tier == "GPU_RUNTIME_READY"
    assert report.cuda_available is True


# ------------------------------------------------------------------------ CUDA-04, CUDA-13

def test_cuda04_failed_probe_blocks_execution(workspace, monkeypatch):
    install_probe(UNAVAILABLE, monkeypatch)
    with pytest.raises(runtime.GpuVerificationError, match="not verified"):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(workspace["model"])


def test_cuda13_verified_result_has_a_production_path_to_launch(workspace, monkeypatch):
    """The transition PENDING -> GPU_RUNTIME_READY is reachable through production code."""
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"

    install_probe(SUCCESS, monkeypatch)
    runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)

    assert runtime.require_launch_ready(workspace["model"]) == str(workspace["interpreter"])
    assert runtime.environment_blockers(workspace["model"]) == []
    assert runtime.environment_status(workspace["model"], probe=True)["gpu_verified"] is True


# ------------------------------------------------------------------------ CUDA-05, CUDA-06

def test_cuda05_incorrect_interpreter_blocks_the_probe(workspace, monkeypatch):
    monkeypatch.setenv("GPU_TEST_PYTHON", str(workspace["repo"] / "missing" / "python"))
    with pytest.raises(runtime.GpuVerificationError):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)


def test_cuda06_missing_torch_blocks_the_probe(workspace):
    """A dependency contract that is not satisfied cannot reach a GPU probe."""
    broken = {**workspace["model"],
              "environment": {**workspace["model"]["environment"], "pytorch": "9.9.9"}}
    with pytest.raises(runtime.GpuVerificationError, match="blocked|not verified"):
        runtime.verify_gpu_runtime(broken, workspace["config"], authorized=True)


# ------------------------------------------------------------------------ CUDA-07, CUDA-08

def test_cuda07_cuda_unavailable_blocks(workspace, monkeypatch):
    install_probe(UNAVAILABLE, monkeypatch)
    with pytest.raises(runtime.GpuVerificationError, match="CUDA"):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)


def test_cuda08_device_allocation_failure_blocks(workspace, monkeypatch):
    install_probe(ALLOCATION_FAILED, monkeypatch)
    with pytest.raises(runtime.GpuVerificationError):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    # A CUDA build reported but an unusable device is NOT capability-verified.
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"


# ------------------------------------------------------------------------ CUDA-09, CUDA-10

def test_cuda09_changed_interpreter_invalidates_cached_gpu_verification(workspace, monkeypatch):
    import venv as _venv

    install_probe(SUCCESS, monkeypatch)
    runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_RUNTIME_READY"

    other = workspace["repo"] / "other_env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(other)
    _install_torch(other)
    monkeypatch.setenv("GPU_TEST_PYTHON", str(other / "bin" / "python"))
    runtime.forget_environment()

    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING", \
        "a different interpreter must not reuse another environment's GPU verification"


def test_cuda10_changed_dependency_fingerprint_invalidates_verification(workspace, monkeypatch):
    install_probe(SUCCESS, monkeypatch)
    runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_RUNTIME_READY"

    # Widen the declared contract: the verification described the old contract only.
    changed = {**workspace["model"],
               "environment": {**workspace["model"]["environment"], "pytorch": "1.12.2"}}
    runtime.forget_environment()
    assert runtime.verify_environment(changed, probe=True).tier != "GPU_RUNTIME_READY"


# ------------------------------------------------------------------------------ CUDA-11

def test_cuda11_plan_construction_and_launch_gate_disagree_deliberately(workspace):
    """A plan may be built while the launch gate still refuses — explicitly, not silently."""
    assert runtime.require_verified_model_interpreter(workspace["model"]) == str(workspace["interpreter"])
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(workspace["model"])

    # The plan-level view deliberately omits the environment blocker, which is what lets a
    # command be constructed; the launch gate still refuses.
    assert runtime.runtime_plan_blockers.__doc__
    assert runtime.environment_blockers(workspace["model"]), "the launch gate must still report the GPU"
    assert runtime.require_launch_ready is not runtime.require_verified_model_interpreter


# ------------------------------------------------------------------------------ CUDA-12

def test_cuda12_no_real_gpu_probe_runs_in_the_laptop_suite(workspace, monkeypatch):
    """Inspection must never run a CUDA probe; the real one exists but is opt-in only."""
    monkeypatch.delenv(runtime.GPU_PROBE_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runtime.clear_gpu_probe_for_tests()

    assert "torch.cuda.is_available" in runtime.GPU_PROBE_CODE, "the real probe must exist"

    called: list[str] = []
    original = runtime._run_probe

    def spy(interpreter, code, timeout=60, args=(), isolated=False):
        called.append(code)
        return original(interpreter, code, timeout, args, isolated)

    monkeypatch.setattr(runtime, "_run_probe", spy)
    runtime.forget_environment()
    report = runtime.verify_environment(workspace["model"], probe=True)

    assert report.tier == "GPU_VERIFICATION_PENDING"
    assert not any("cuda" in code.lower() for code in called), \
        "inspection must not run any CUDA probe"


# ------------------------------------------------------------------------------ CUDA-14

def test_cuda14_no_evaluator_launches_while_gpu_is_unverified(workspace, monkeypatch):
    """The launch gate refuses an unverified GPU model, so no evaluator can be started."""
    started: list[object] = []
    # Only an *evaluator* launch matters here; the identity probe is a different call.
    original = runtime.subprocess.Popen

    def spy(command, *args, **kwargs):
        if not any("-c" == part for part in command):
            started.append(command)
            pytest.fail("evaluator started")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "Popen", spy)
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(workspace["model"])
    assert started == []


def test_the_test_only_probe_is_unreachable_without_the_optin(workspace, monkeypatch):
    """Without the opt-in variable the override is ignored entirely.

    The verification path then runs its *own* probes (never the injected payload), so the
    capability can never be obtained from a test-only shortcut in production.
    """
    monkeypatch.delenv(runtime.GPU_PROBE_OVERRIDE_ENV, raising=False)
    used_override: list[dict] = []
    runtime.set_gpu_probe_for_tests(lambda model, report: used_override.append(True) or dict(SUCCESS))

    calls: list[str] = []
    original = runtime._run_probe

    def spy(interpreter, code, timeout=60, args=(), isolated=False):
        calls.append(code)
        return original(interpreter, code, timeout, args, isolated)

    monkeypatch.setattr(runtime, "_run_probe", spy)
    with pytest.raises(runtime.GpuVerificationError):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)

    assert used_override == [], "the test-only override must never be consulted in production"
    assert calls, "the real verification path must have run its own probes"
    # The GPU probe itself is only reached after the environment verifies; assert that the
    # capability was not granted rather than that a specific call happened.
    assert runtime.verify_environment(workspace["model"], probe=True).tier != "GPU_RUNTIME_READY"
