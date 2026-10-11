"""CUDA readiness state machine (CUDA-01..CUDA-14).

The laptop has no GPU. These tests therefore exercise the *production* state machine with a
deterministic probe injected through the test-only override, whose activation requires an
explicit environment opt-in so production execution can never select it. The real probe
implementation (``runtime.GPU_PROBE_CODE``) is asserted to exist, and CUDA-12 asserts that
the ordinary acceptance suite never runs it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import workbench.backend.runtime as runtime
from workbench.backend.operator_config import resolve_config
from workbench.tests._core_fixture import build_core_module


@pytest.fixture(autouse=True)
def _ledger_key(monkeypatch):
    """A GPU capability record is authenticated with the operator key, so tests set one."""
    import secrets as _secrets

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", _secrets.token_hex(32))
    yield


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
    # Corroboration that the probe executed a real installation. The fixture installs the
    # module it names, so the digest is computed from a file that exists.
    "torch_has_compiled_core": True,
}
UNAVAILABLE = {
    "cuda_available": False, "device_count": 0, "torch_version": "1.12.1",
    "torch_cuda_version": None, "device_name": None, "compute_capability": None,
    "allocated_tensor": False, "computed": False, "error": "torch reports CUDA unavailable",
}
ALLOCATION_FAILED = {**SUCCESS, "allocated_tensor": False, "computed": False,
                     "error": "CUDA out of memory"}


def install_probe(payload: dict, monkeypatch, install_prefix: Path | None = None) -> None:
    """Install a deterministic probe result for tests.

    The payload names an importable ``torch`` inside the environment under test, so the
    orchestrator's corroboration (module digest + compiled core) is satisfied by a real file.
    """
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")
    resolved = dict(payload)
    if resolved.get("torch_has_compiled_core") and install_prefix is not None:
        site_packages = next((lib / "site-packages"
                              for lib in (install_prefix / "lib").glob("python*")), None)
        if site_packages is not None:
            module = site_packages / "torch.py"
            module.write_text("__version__ = '1.12.1'\n")
            resolved["torch_module_file"] = str(module)
            # A real environment's core is a loadable extension exporting PyInit__C; the
            # fixture compiles one so the orchestrator's child-process load succeeds.
            resolved["torch_core_origin"] = str(build_core_module(site_packages / "_C_fixture.so"))
    runtime.set_gpu_probe_for_tests(lambda model, report: dict(resolved))


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
    install_probe(SUCCESS, monkeypatch, workspace['interpreter'].parent.parent)
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

    install_probe(SUCCESS, monkeypatch, workspace['interpreter'].parent.parent)
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

    install_probe(SUCCESS, monkeypatch, workspace['interpreter'].parent.parent)
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
    install_probe(SUCCESS, monkeypatch, workspace['interpreter'].parent.parent)
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

    assert "import torch" in runtime.GPU_PROBE_CODE, "the real probe must exist"
    assert "is_available" in runtime.GPU_PROBE_CODE

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


def test_a_forged_capability_record_cannot_grant_readiness(workspace, monkeypatch):
    """The recorded capability lives in the writable artifact tree, so it is authenticated.

    Forging it must require the operator key: a record written without a signature, or with a
    signature that does not verify, is ignored entirely.
    """
    capability_path = workspace["config"].CIR_REPO_ROOT / "workbench" / "artifacts" / "gpu_capability.json"
    capability_path.parent.mkdir(parents=True, exist_ok=True)

    forged = {
        "capability_key": runtime._gpu_capability_key(workspace["model"]),
        "capability_verified": True,
        "cuda_available": True,
        "allocated_tensor": True,
        "computed": True,
        "probe_source": "forged",
    }
    capability_path.write_text(json.dumps(forged))
    runtime.forget_environment()
    runtime.forget_gpu_capability()

    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"
    with pytest.raises(runtime.GpuVerificationError):
        runtime.require_launch_ready(workspace["model"])

    # A signature made with a different key is rejected too.
    forged["signature"] = "0" * 64
    capability_path.write_text(json.dumps(forged))
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"


def test_the_api_queue_cannot_start_an_unverified_gpu_evaluation(workspace, monkeypatch):
    """The queued-job path must enforce the same launch gate as the CLI.

    The UI plans through `guarded_plan`, which deliberately permits a GPU-pending model; the
    queue consumer must therefore refuse to start it.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("workbench_main_gpu",
                                                  Path("workbench/backend/main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    started: list[object] = []
    monkeypatch.setattr(module, "JOBS",
                        type("Q", (), {"enqueue": lambda self, *a, **k: started.append(a) or {"id": "x"}})())

    checkpoint = {"checkpoint_id": "ckpt", "filename": "f.pt",
                  "checkpoint_mapping_status": "VERIFIED_METADATA"}
    model = {**workspace["model"], "checkpoint_variants": [checkpoint],
             "native_protocol": "p", "supported_protocols": ["p"],
             "upstream_commit_sha": "a" * 40, "source_dir": "SyntheticModel"}
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model], "protocols": {}})
    monkeypatch.setattr(module, "guarded_plan",
                        lambda *a, **k: (type("P", (), {
                            "model_id": model["model_id"], "checkpoint_id": "ckpt",
                            "protocol": "p", "pin": "a" * 40, "actual_source_commit": "a" * 40,
                            "checkpoint": workspace["repo"] / "ckpt.pt",
                            "dataset_root": workspace["repo"] / "F", "command": ["x"],
                            "cwd": workspace["repo"]})(), []))
    monkeypatch.setattr(module, "resolve_config", lambda: workspace["config"])
    (workspace["repo"] / "ckpt.pt").write_bytes(b"x")

    payload = type("Job", (), {"model_id": model["model_id"], "checkpoint_id": "ckpt",
                               "protocol_id": "p", "top_k": 200})()
    with pytest.raises(module.HTTPException) as error:
        module.queue_job(payload)
    assert error.value.status_code == 409
    assert error.value.detail["error"] == "gpu_verification_required"
    assert started == [], "no job may be queued while GPU readiness is unverified"
    assert checkpoint["checkpoint_id"] == "ckpt"


def test_the_queue_rechecks_readiness_before_dispatch(workspace, monkeypatch):
    """A capability invalidated while a job waits must stop the dispatch.

    The request handler's check is not enough: the environment can change between enqueue and
    dispatch, so the consumer re-checks immediately before starting the process.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("jobs_gpu_probe",
                                                  Path("workbench/backend/jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dispatched: list[list[str]] = []
    monkeypatch.setattr(module, "subprocess", type("S", (), {
        "Popen": staticmethod(lambda *a, **k: dispatched.append(list(a)) or pytest.fail("dispatched"))})())

    queue = module.SingleGpuQueue()

    def refusing_guard(job):
        raise RuntimeError("execution environment not verified [GPU_VERIFICATION_PENDING]")

    queue.launch_guard = refusing_guard
    queue.enqueue("limn", "ckpt", "p", ["gpu-evaluator"], workspace["repo"],
                  workspace["repo"] / "logs", {"model_id": "limn"})
    import time as _time

    deadline = _time.monotonic() + 5
    job = None
    while _time.monotonic() < deadline:
        jobs = queue.list()
        if jobs and jobs[0]["status"] != "queued":
            job = jobs[0]
            break
        _time.sleep(0.02)
    assert job is not None, "the worker never processed the job"
    assert dispatched == [], "a job whose runtime is no longer ready must not dispatch"
    assert job["status"] == "blocked", job["status"]


def test_cancellation_wins_even_after_a_slow_launch_guard(workspace, monkeypatch):
    """A job cancelled while the dispatch guard ran must not be started."""
    import importlib.util
    import time as _time

    spec = importlib.util.spec_from_file_location("jobs_cancel_probe",
                                                  Path("workbench/backend/jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dispatched: list[list[str]] = []
    monkeypatch.setattr(module, "subprocess", type("S", (), {
        "Popen": staticmethod(lambda *a, **k: dispatched.append(list(a)) or pytest.fail("dispatched"))})())

    queue = module.SingleGpuQueue()
    seen: list[str] = []

    def slow_guard(job):
        seen.append(job.job_id)
        queue.cancel(job.job_id)          # cancellation arrives while the guard runs
        _time.sleep(0.05)

    queue.launch_guard = slow_guard
    queued = queue.enqueue("limn", "ckpt", "p", ["gpu-evaluator"], workspace["repo"],
                           workspace["repo"] / "logs", {"model_id": "limn"})
    job_id = queued.get("job_id") or queued.get("id")

    deadline = _time.monotonic() + 5
    while _time.monotonic() < deadline and job_id not in seen:
        _time.sleep(0.02)
    _time.sleep(0.2)
    assert seen, "the guard never ran"
    assert dispatched == [], "a cancelled job must never be dispatched"


def test_the_gpu_probe_can_import_the_environment_dependencies(workspace):
    """The authorized probe must see the environment's real packages.

    `-S` prevents site from applying the environment's own configuration, so the probe is
    handed the environment's distribution directories explicitly. Without this the probe
    could never import torch and the production GPU path would be unusable.
    """
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"

    # The fixture already installs a stub torch module; the probe must be able to import it.
    import json as _json

    probe_paths = tuple(str(path) for path in
                        runtime._site_packages_dirs(workspace["interpreter"].parent.parent))
    ok, out, err = runtime._run_probe(workspace["interpreter"], runtime.GPU_PROBE_CODE,
                                      isolated=True, timeout=120, args=probe_paths)
    assert ok, err
    payload = _json.loads(out.splitlines()[-1])
    assert payload["error"] is None, payload["error"]
    assert payload["torch_version"] == "1.12.1"
    assert payload["cuda_available"] is False


def test_a_same_size_dependency_rewrite_invalidates_the_capability(workspace, monkeypatch):
    """A same-size, same-mtime package edit must not keep a stale capability valid."""
    interpreter_dir = workspace["interpreter"].parent.parent
    site_packages = next((lib / "site-packages" for lib in (interpreter_dir / "lib").glob("python*")), None)
    assert site_packages is not None
    module = site_packages / "torch.py"
    module.write_text("__version__ = '1.12.1'\n")

    install_probe(SUCCESS, monkeypatch, workspace['interpreter'].parent.parent)
    runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_RUNTIME_READY"

    before = module.stat()
    module.write_text("__version__ = '9.9.9'\n")          # same size
    __import__("os").utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    runtime.forget_environment()

    rechecked = runtime.verify_environment(workspace["model"], probe=True)
    assert rechecked.tier != "GPU_RUNTIME_READY", \
        "a same-size package rewrite must invalidate the recorded capability"


def test_a_stub_module_cannot_supply_gpu_evidence(workspace, monkeypatch):
    """A single importable `torch.py` is not evidence of a real PyTorch installation.

    The probe reports whatever the imported module says, so a capability additionally
    requires corroboration: the module must be a real installation with its compiled core
    extension, and the orchestrator records the module's digest. A stub answers the probe's
    questions but cannot corroborate itself.
    """
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "stub-probe-key")

    for lib in (workspace["interpreter"].parent.parent / "lib").glob("python*"):
        site_packages = lib / "site-packages"
        stub = site_packages / "torch.py"
        stub.write_text(
            "__version__ = '1.12.1'\n"
            "class _Cuda:\n"
            "    def is_available(self): return True\n"
            "    def device_count(self): return 1\n"
            "    def get_device_name(self, i): return 'fake GPU'\n"
            "    def get_device_capability(self, i): return (8, 9)\n"
            "cuda = _Cuda()\n"
            "class _T:\n"
            "    def __init__(self, *a, **k): pass\n"
            "    def __mul__(self, o): return self\n"
            "    def sum(self): return self\n"
            "    def item(self): return 2.0\n"
            "def ones(*a, **k): return _T()\n")
        break

    # The stub claims everything the probe asks; it has no compiled core extension.
    runtime.set_gpu_probe_for_tests(lambda model, report: {
        "cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
        "torch_cuda_version": "12.4", "device_name": "fake GPU",
        "compute_capability": "8.9", "allocated_tensor": True, "computed": True,
        "torch_module_file": str(stub), "torch_has_compiled_core": False, "error": None})

    with pytest.raises(runtime.GpuVerificationError, match="compiled core"):
        runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    assert runtime.verify_environment(workspace["model"], probe=True).tier == "GPU_VERIFICATION_PENDING"


def test_a_python_file_named_under_torch_cannot_claim_a_compiled_core(workspace):
    """`torch._C` must be a compiled extension, not merely an importable Python module.

    `importlib.util.find_spec("torch._C")` succeeds for a plain `_C.py`, so checking only for
    its presence would let a pure-Python directory named `torch` mint a signed capability
    without any PyTorch build. The probe records the resolved origin and only a shared library
    counts as a compiled core.
    """
    import json as _json

    for lib in (workspace["interpreter"].parent.parent / "lib").glob("python*"):
        package = lib / "site-packages" / "torch"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("__version__ = '1.12.1'\n")
        (package / "_C.py").write_text("nothing = True\n")
        break

    ok, out, err = runtime._run_probe(
        workspace["interpreter"], runtime.GPU_PROBE_CODE, isolated=True, timeout=60,
        args=tuple(runtime._site_packages_dirs(workspace["interpreter"].parent.parent)))
    assert ok, err
    result = _json.loads(out.splitlines()[-1])
    assert result["torch_has_compiled_core"] is False
    assert str(result["torch_core_origin"]).endswith("_C.py")


def test_a_fake_shared_library_cannot_claim_a_compiled_core(workspace):
    """A file named `_C.so` must actually be a native binary.

    Following only the suffix still let a pure-Python `torch` package place arbitrary bytes
    in `_C.so` and satisfy corroboration. The probe inspects the header, so a text file named
    like a library is rejected.
    """
    import json as _json

    for lib in (workspace["interpreter"].parent.parent / "lib").glob("python*"):
        package = lib / "site-packages" / "torch"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("__version__ = '1.12.1'\n")
        (package / "_C.so").write_bytes(b"not a shared library")
        break

    ok, out, err = runtime._run_probe(
        workspace["interpreter"], runtime.GPU_PROBE_CODE, isolated=True, timeout=60,
        args=tuple(runtime._site_packages_dirs(workspace["interpreter"].parent.parent)))
    assert ok, err
    result = _json.loads(out.splitlines()[-1])
    assert result["torch_has_compiled_core"] is False
    assert str(result["torch_core_origin"]).endswith("_C.so")


def test_a_torch_module_outside_the_environment_cannot_corroborate(workspace, tmp_path):
    """The corroborating module must live inside the verified environment.

    Otherwise an identical copy of torch elsewhere on the machine satisfies the recorded
    digest without the verified environment containing the installation it claims.
    """
    outside = tmp_path / "elsewhere" / "torch.py"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("__version__ = '1.12.1'\n")
    runtime.set_gpu_probe_for_tests(lambda model, report: {
        "cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
        "torch_cuda_version": "12.4", "device_name": "Synthetic GPU",
        "compute_capability": "8.9", "allocated_tensor": True, "computed": True,
        "torch_module_file": str(outside), "torch_has_compiled_core": True, "error": None})
    try:
        with pytest.raises(runtime.GpuVerificationError, match="compiled|corroborated|verified"):
            runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    finally:
        runtime.clear_gpu_probe_for_tests()


def test_an_unrelated_binary_named_C_so_cannot_corroborate(workspace):
    """The core must export CPython's PyInit__C, not merely be a native binary.

    A copy of any ELF executable passed the header and suffix checks, and `find_spec` metadata
    is mutable model code, so the orchestrator loads the reported file itself in a child
    process and requires the module init symbol. `/bin/true` is a real native binary that does
    not export it.
    """
    for lib in (workspace["interpreter"].parent.parent / "lib").glob("python*"):
        package = lib / "site-packages" / "torch"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("__version__ = '1.12.1'\n")
        (package / "_C.so").write_bytes(Path("/bin/true").read_bytes())
        break

    assert runtime._exports_python_init(
        next((workspace["interpreter"].parent.parent / "lib").glob("python*")
             ) / "site-packages" / "torch" / "_C.so") is False
    runtime.set_gpu_probe_for_tests(lambda model, report: {
        "cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
        "torch_cuda_version": "12.4", "device_name": "fake",
        "compute_capability": "8.9", "allocated_tensor": True, "computed": True,
        "torch_has_compiled_core": True,
        "torch_core_origin": str(next((workspace["interpreter"].parent.parent / "lib")
                                      .glob("python*"))/"site-packages"/"torch"/"_C.so"),
        "torch_module_file": str(next((workspace["interpreter"].parent.parent / "lib")
                                      .glob("python*"))/"site-packages"/"torch"/"__init__.py"),
        "error": None})
    try:
        with pytest.raises(runtime.GpuVerificationError):
            runtime.verify_gpu_runtime(workspace["model"], workspace["config"], authorized=True)
    finally:
        runtime.clear_gpu_probe_for_tests()


def test_cancellation_cannot_slip_between_creation_and_registration(workspace, monkeypatch):
    """A cancel that arrives exactly while the process is being created must still kill it.

    Process creation and registration used to sit in two separate critical sections, so a
    cancellation arriving in the gap saw `preparing` with no registered process, marked the job
    cancelled, and the worker then published `running` over it — the child ran to completion
    uncancelled.
    """
    import importlib.util
    import subprocess as real_subprocess
    import threading
    import time as _time

    spec = importlib.util.spec_from_file_location("jobs_cancel_window",
                                                  Path("workbench/backend/jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    sentinel = workspace["repo"] / "child_ran_to_completion.txt"
    script = f"import time\ntime.sleep(2)\nopen({str(sentinel)!r}, 'w').write('ran')\n"
    real_popen = real_subprocess.Popen

    class SlowPopen:
        """Makes the creation-to-registration window wide enough to lose the race in."""

        def __init__(self, *args, **kwargs):
            self._process = real_popen(*args, **kwargs)
            _time.sleep(0.3)

        def __getattr__(self, name):
            return getattr(self._process, name)

    monkeypatch.setattr(module, "subprocess", type("S", (), {
        "Popen": staticmethod(SlowPopen), "PIPE": real_subprocess.PIPE})())

    queue = module.SingleGpuQueue()
    queue.enqueue("limn", "ckpt", "p", [sys.executable, "-c", script], workspace["repo"],
                  workspace["repo"] / "logs", {"model_id": "limn"})
    job_id = queue.list()[0]["job_id"]

    def cancel_once_running() -> None:
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            if queue.list()[0]["status"] == "running":
                queue.cancel(job_id)
                return
            _time.sleep(0.001)

    canceller = threading.Thread(target=cancel_once_running)
    canceller.start()
    deadline = _time.monotonic() + 10
    while _time.monotonic() < deadline and queue.list()[0]["status"] != "cancelled":
        _time.sleep(0.02)
    canceller.join()
    _time.sleep(2.5)   # long enough for an uncancelled child to finish and write its file

    assert queue.list()[0]["status"] == "cancelled"
    assert not sentinel.exists(), "a cancelled job's process ran to completion"


def test_a_same_mtime_rewrite_cannot_keep_an_old_member_digest(tmp_path):
    """The member digest must describe content, not a stat triple.

    A memo keyed on (path, size, mtime) is defeated by an in-place rewrite that restores
    mtime — `os.utime` does it — so a stale GPU capability could stay valid through a change
    the fingerprint was supposed to catch.
    """
    import os as _os

    module = tmp_path / "member.py"
    module.write_text("content-v1\n")
    before = module.stat()
    first = runtime._member_digest(module)

    module.write_text("content-v2\n")            # same size, same name
    _os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = module.stat()
    assert after.st_size == before.st_size and after.st_mtime_ns == before.st_mtime_ns

    assert runtime._member_digest(module) != first


def test_the_fallback_key_is_created_so_a_capability_can_be_signed(tmp_path, monkeypatch):
    """An operator who relies on the documented fallback key must still get a signed record.

    The ledger creates `<repo>/.workbench_ledger_key` on first use. Capability signing read the
    same file but never created it, so a fallback-key-only operator produced an unsigned
    capability and `require_launch_ready` refused it forever.
    """
    from dataclasses import replace

    from workbench.backend import operator_config

    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setattr(operator_config, "resolve_config",
                        lambda: replace(resolve_config(), CIR_REPO_ROOT=tmp_path))

    key = runtime._gpu_cache_key()
    assert key and len(key) == 32, "the fallback key must be created, not silently empty"
    assert (tmp_path / ".workbench_ledger_key").is_file()

    # The same key signs and verifies a record, which is what launch readiness consumes.
    record = {"capability_verified": True, "interpreter": "/x/python"}
    signature = runtime._gpu_capability_signature(record)
    assert signature
    assert runtime._gpu_capability_signature_valid({**record, "signature": signature})


def test_queued_job_children_get_no_operator_secrets(workspace, monkeypatch):
    """The queue dispatch path must filter secrets exactly as the CLI path does.

    The CLI evaluator child had `WORKBENCH_LEDGER_KEY` and `WORKBENCH_SECRET_*` stripped while
    the API/queued path handed them straight to the model environment, which could then forge
    ledger entries.
    """
    import importlib.util
    import time as _time

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "queued-operator-secret")
    monkeypatch.setenv("WORKBENCH_SECRET_BETA", "queued-other-secret")

    spec = importlib.util.spec_from_file_location("jobs_secret_filter",
                                                  Path("workbench/backend/jobs.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    probe = workspace["repo"] / "report_env.py"
    probe.write_text("import os, json\n"
                     "print(json.dumps(sorted(k for k in os.environ if 'WORKBENCH' in k)))\n")
    queue = module.SingleGpuQueue()
    queue.enqueue("limn", "ckpt", "p", [sys.executable, str(probe)], workspace["repo"],
                  workspace["repo"] / "logs", {"model_id": "limn"})

    deadline = _time.monotonic() + 15
    while _time.monotonic() < deadline and queue.list()[0]["status"] != "completed":
        _time.sleep(0.02)
    assert queue.list()[0]["status"] == "completed"
    leaked = json.loads((workspace["repo"] / "logs" / "stdout.log").read_text().strip().splitlines()[-1])
    assert leaked == [], leaked
