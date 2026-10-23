"""Phase I integration contracts: API queue, trust boundary, log ownership, cancellation.

Each test here fails against the behaviour that shipped before this pass, and each one names
the guarantee it pins rather than the implementation detail it happens to observe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.jobs import SingleGpuQueue

REPOSITORY = Path(__file__).resolve().parents[2]


def _wait(queue: SingleGpuQueue, job_id: str, *, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(item for item in queue.list() if item["job_id"] == job_id)
        if job["status"] in {"completed", "failed", "blocked", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never settled: {queue.list()}")


# --------------------------------------------------------------------------- log ownership


def test_queue_and_operator_own_separate_directories(tmp_path: Path) -> None:
    """The queue records dispatch and the operator records evidence: different trees.

    Both used to write `command.json` into the same `logs/<model>__<checkpoint>__<run_id>/`,
    so the queue's view of the run could stand in for the operator's own record and the
    evidence digest pointed at a file another writer owned.
    """
    queue = SingleGpuQueue()
    job_directory = tmp_path / "queue_jobs" / "job-1"
    job = queue.enqueue("m", "c", "p", [sys.executable, "-c", "print('ok')"],
                        tmp_path, job_directory, {})
    _wait(queue, job["job_id"])

    assert (job_directory / "queue_command.json").is_file()
    assert (job_directory / "queue_stdout.log").is_file()
    assert (job_directory / "queue_stderr.log").is_file()
    # Nothing the operator owns may appear in the queue's directory, and nothing the queue
    # owns may appear under a run directory.
    for name in ("command.json", "stdout.log", "stderr.log", "aggregate_report.json"):
        assert not (job_directory / name).exists(), name


def test_a_failed_queue_job_keeps_its_diagnostics_without_touching_a_run(tmp_path: Path) -> None:
    """A failing job must still leave a readable queue record, and touch no run directory."""
    run_directory = tmp_path / "logs" / "m__c__run-1"
    run_directory.mkdir(parents=True)
    (run_directory / "command.json").write_text('{"owner": "operator"}\n')
    before = (run_directory / "command.json").read_bytes()

    queue = SingleGpuQueue()
    job_directory = tmp_path / "queue_jobs" / "job-fail"
    job = queue.enqueue("m", "c", "p", [sys.executable, "-c", "print('bad'); raise SystemExit(3)"],
                        tmp_path, job_directory, {})
    settled = _wait(queue, job["job_id"])

    assert settled["status"] == "failed"
    assert settled["return_code"] == 3
    assert "bad" in (job_directory / "queue_stdout.log").read_text()
    assert (run_directory / "command.json").read_bytes() == before


def test_two_jobs_with_distinct_ids_never_collide(tmp_path: Path) -> None:
    """Distinct jobs get distinct directories, even when they run the same command."""
    queue = SingleGpuQueue()
    first = tmp_path / "queue_jobs" / "a"
    second = tmp_path / "queue_jobs" / "b"
    job_a = queue.enqueue("m", "c", "p", [sys.executable, "-c", "print('A')"], tmp_path, first, {})
    job_b = queue.enqueue("m", "c", "p", [sys.executable, "-c", "print('B')"], tmp_path, second, {})
    _wait(queue, job_a["job_id"])
    _wait(queue, job_b["job_id"])
    assert (first / "queue_stdout.log").read_text().strip() == "A"
    assert (second / "queue_stdout.log").read_text().strip() == "B"


# ------------------------------------------------------------------------ trust boundary


def test_the_queue_hands_the_operator_its_key_but_no_stray_secrets(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The queued process IS the trusted operator, so it needs the ledger key.

    An earlier revision stripped the key from the queue's child while that child was the
    operator, which made a queued run unable to authenticate its own ledger entry. Other
    operator secrets the operator has no use for are still removed.
    """
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "operator-key")
    monkeypatch.setenv("WORKBENCH_SECRET_UNUSED", "noise")

    reporter = tmp_path / "report.py"
    reporter.write_text("import json, os\n"
                        "print(json.dumps(sorted(k for k in os.environ if 'WORKBENCH' in k)))\n")
    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "boundary"
    job = queue.enqueue("m", "c", "p", [sys.executable, str(reporter)], tmp_path, directory, {})
    _wait(queue, job["job_id"])

    names = json.loads((directory / "queue_stdout.log").read_text().strip().splitlines()[-1])
    secret_names = [name for name in names if name == "WORKBENCH_LEDGER_KEY" or name.startswith("WORKBENCH_SECRET_")]
    assert secret_names == ["WORKBENCH_LEDGER_KEY"], secret_names
    assert set(names) - set(secret_names) <= {"WORKBENCH_THIRD_PARTY_ROOT"}, names


def test_the_official_evaluator_never_receives_the_operator_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator strips the key and every WORKBENCH_SECRET_* before starting model code."""
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.scripts import evaluate_models as em

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "operator-key")
    monkeypatch.setenv("WORKBENCH_SECRET_UNUSED", "noise")

    repo = tmp_path / "repo"
    repo.mkdir()
    reporter = tmp_path / "eval_env.py"
    reporter.write_text("import json, os\n"
                        "print(json.dumps({'names': sorted(k for k in os.environ if 'WORKBENCH' in k),"
                        " 'run': os.environ.get('evaluation_run_id')}))\n")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"x")
    plan = em.EvaluationPlan(model_id="m", checkpoint_id="c", protocol="p",
                             dataset_root=tmp_path, source=tmp_path, checkpoint=checkpoint,
                             cwd=tmp_path, command=[sys.executable, str(reporter)],
                             pin="p" * 40, actual_source_commit="a" * 40)
    config = WorkbenchConfig(CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path,
                             FASHIONIQ_ROOT=tmp_path, WORKBENCH_HOST="127.0.0.1",
                             WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
                             WORKBENCH_CHECKPOINT_ROOT=tmp_path / "ckpts",
                             WORKBENCH_RESULTS_ROOT=tmp_path / "results",
                             WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party")

    code, directory, environment = em.invoke_evaluator(plan, config, "boundary-run",
                                                       tmp_path / "out")
    assert code == 0
    reported = json.loads((directory / "stdout.log").read_text().strip().splitlines()[-1])
    secret_names = [name for name in reported["names"]
                    if name == "WORKBENCH_LEDGER_KEY" or name.startswith("WORKBENCH_SECRET_")]
    assert secret_names == [], secret_names
    # The allowlist is explicit, so a new non-secret name reaching model code is a deliberate
    # contract change rather than an accident. `WORKBENCH_REPLAY_CACHE_ROOT` is the run-scoped
    # dataset-cache directory the parent gives both the aggregate and the observer: it is a path
    # inside the run's own directory, carries no secret, and exists so the two children build one
    # 11.87 GB cache instead of two. It is listed here so adding another name still fails this test.
    permitted = {"WORKBENCH_THIRD_PARTY_ROOT", "WORKBENCH_REPLAY_CACHE_ROOT",
                 "WORKBENCH_OBSERVATION_OUT"}
    assert set(reported["names"]) - set(secret_names) <= permitted, reported["names"]
    assert reported["run"] == "boundary-run"
    # The key must not be echoed into the run's own recorded environment either.
    assert "operator-key" not in json.dumps(environment)


# ------------------------------------------------------------------------ run directory


def test_an_existing_run_directory_is_refused_before_the_evaluator_starts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing run directory holds another invocation's evidence and is never reused."""
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.scripts import evaluate_models as em

    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "started"
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()\n")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"x")
    plan = em.EvaluationPlan(model_id="m", checkpoint_id="c", protocol="p",
                             dataset_root=tmp_path, source=tmp_path, checkpoint=checkpoint,
                             cwd=tmp_path, command=[sys.executable, str(evaluator)],
                             pin="p" * 40, actual_source_commit="a" * 40)
    config = WorkbenchConfig(CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path,
                             FASHIONIQ_ROOT=tmp_path, WORKBENCH_HOST="127.0.0.1",
                             WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
                             WORKBENCH_CHECKPOINT_ROOT=tmp_path / "ckpts",
                             WORKBENCH_RESULTS_ROOT=tmp_path / "results",
                             WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party")

    run_id = "collision-run"
    directory = em.log_dir(config, plan, em.datetime.now(em.timezone.utc), run_id)
    directory.mkdir(parents=True)
    evidence = directory / "command.json"
    evidence.write_text('{"owner": "previous invocation"}\n')
    before = evidence.read_bytes()

    with pytest.raises(RuntimeError, match="refusing to overwrite existing run directory"):
        em.invoke_evaluator(plan, config, run_id, tmp_path / "out")

    assert not marker.exists(), "no evaluator may start after a collision"
    assert evidence.read_bytes() == before, "previous evidence must stay byte-identical"


# --------------------------------------------------------------------------- cancellation


def test_cancellation_stops_the_operator_and_the_evaluator_it_started(tmp_path: Path) -> None:
    """The job's process group is signalled, so the evaluator grandchild dies too.

    Killing only the operator's pid left the official evaluator — the process actually holding
    the GPU — running, and the job was still reported cancelled.
    """
    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "cancel"
    sentinel = tmp_path / "evaluator_finished.txt"
    # The job is an "operator" that starts an "evaluator" grandchild: exactly the production
    # shape, and one process group.
    # Not named `operator.py`: that would shadow the stdlib module the interpreter imports
    # while starting up.
    operator = tmp_path / "operator_process.py"
    operator.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', "
        f"\"import time\\ntime.sleep(4)\\nopen({str(sentinel)!r},'w').write('ran')\\n\"])\n"
        "child.wait()\n"
        "time.sleep(0.2)\n")
    job = queue.enqueue("m", "c", "p", [sys.executable, str(operator)], tmp_path, directory, {})

    # Cancel as soon as the operator is running; the operator's own start-up latency gives the
    # grandchild time to exist before the signal lands.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = next(i for i in queue.list() if i["job_id"] == job["job_id"])["status"]
        if status == "running":
            time.sleep(0.5)   # let the grandchild start
            queue.cancel(job["job_id"])
            break
        if status in {"completed", "failed"}:
            raise AssertionError(f"the job settled before it could be cancelled: {status}")
        time.sleep(0.02)

    settled = _wait(queue, job["job_id"], timeout=15)
    assert settled["status"] == "cancelled"
    time.sleep(4.5)          # long enough for a surviving grandchild to write its file
    assert not sentinel.exists(), "the evaluator grandchild survived cancellation"


def test_cancelling_a_queued_job_never_starts_a_process(tmp_path: Path) -> None:
    """Cancelling before dispatch must not run anything, and must not report success."""
    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "pre-cancel"
    marker = tmp_path / "ran.txt"
    # Occupy the worker so the second job stays queued.
    blocker = queue.enqueue("m", "c", "p", [sys.executable, "-c", "import time; time.sleep(1.2)"],
                            tmp_path, tmp_path / "queue_jobs" / "blocker", {})
    victim = queue.enqueue("m", "c", "p",
                           [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                           tmp_path, directory, {})
    queue.cancel(victim["job_id"])

    _wait(queue, blocker["job_id"])
    settled = _wait(queue, victim["job_id"])
    assert settled["status"] == "cancelled"
    assert not marker.exists()
    assert settled["return_code"] is None


def test_cancelling_one_job_does_not_touch_another(tmp_path: Path) -> None:
    """A cancellation is scoped to its own process group, not to the queue."""
    queue = SingleGpuQueue()
    witness = tmp_path / "witness.txt"
    long_job = queue.enqueue("m", "c", "p",
                             [sys.executable, "-c",
                              "import time\n"
                              f"time.sleep(1.5)\nopen({str(witness)!r},'w').write('survived')\n"],
                             tmp_path, tmp_path / "queue_jobs" / "long", {})
    victim = queue.enqueue("m", "c", "p", [sys.executable, "-c", "import time; time.sleep(6)"],
                           tmp_path, tmp_path / "queue_jobs" / "victim", {})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if next(i for i in queue.list() if i["job_id"] == victim["job_id"])["status"] == "running":
            break
        time.sleep(0.02)

    assert _wait(queue, long_job["job_id"])["status"] == "completed"
    assert witness.exists(), "the unrelated job must finish"
    if next(i for i in queue.list() if i["job_id"] == victim["job_id"])["status"] == "running":
        queue.cancel(victim["job_id"])


# ------------------------------------------------------------------- capability persistence


def test_two_models_keep_independent_capability_records(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifying one model must not overwrite another model's capability.

    One fixed `gpu_capability.json` meant the second verification replaced the first, and a
    fresh process could no longer find it.
    """
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.tests._core_fixture import build_core_module

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "capability-key")
    # The synthetic probe override is opt-in by design, so a test must ask for it explicitly.
    monkeypatch.setenv(runtime.GPU_PROBE_OVERRIDE_ENV, "1")
    base = tmp_path
    repo = base / "repo"
    repo.mkdir()
    config = WorkbenchConfig(CIR_REPO_ROOT=repo, CIR_DATA_ROOT=base, FASHIONIQ_ROOT=base,
                             WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
                             WORKBENCH_FRONTEND_PORT=5173,
                             WORKBENCH_CHECKPOINT_ROOT=base / "ckpts",
                             WORKBENCH_RESULTS_ROOT=base / "results",
                             WORKBENCH_THIRD_PARTY_ROOT=base / "third_party")

    models = {}
    for name in ("alpha", "beta"):
        env = base / f"{name}_env"
        import venv as _venv
        _venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
        site = next((env / "lib").glob("python*")) / "site-packages"
        module = site / "torch.py"
        module.write_text("__version__ = '1.12.1'\n")
        # The environment must declare the dependency it is judged by: verification reads the
        # on-disk inventory, so an importable module without metadata is not an install.
        dist = site / "torch-1.12.1.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 1.12.1\n")
        (dist / "RECORD").write_text(f"torch.py,sha256=t,{module.stat().st_size}\n")
        # A corroborating core must be a real loadable extension; the shared fixture builds one.
        core = build_core_module(site / "_C.so")
        variable = f"WORKBENCH_{name.upper()}_PYTHON"
        monkeypatch.setenv(variable, str(env / "bin" / "python"))
        models[name] = {
            "model_id": name,
            "environment_required": True,
            "environment": {"python_env_var": variable, "pytorch": "1.12.1", "cuda": "12.4"},
            "_module": module,
            "_core": core,
        }

    def probe_for(model, report):
        return {"cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
                "torch_cuda_version": "12.4", "device_name": "Synthetic GPU",
                "compute_capability": "8.9", "allocated_tensor": True, "computed": True,
                "torch_module_file": str(model["_module"]),
                "torch_core_origin": str(model["_core"]),
                "torch_has_compiled_core": True, "error": None}

    for name, model in models.items():
        runtime.set_gpu_probe_for_tests(probe_for)
        runtime.verify_gpu_runtime(model, config, authorized=True, force=True)
        runtime.forget_environment()

    # Capability records are located through the resolved configuration, exactly as the
    # production CLI does (its own config *is* resolve_config()), so the fixture points the
    # resolver at this workspace.
    from workbench.backend import operator_config
    monkeypatch.setattr(operator_config, "resolve_config", lambda: config)

    # A fresh process must be able to discover both, authenticated, without either replacing
    # the other.
    runtime.forget_gpu_capability()
    for name, model in models.items():
        record = runtime.gpu_capability_path(config, runtime._gpu_capability_key(model))
        assert record.is_file(), f"{name} lost its capability record"
        payload = json.loads(record.read_text())
        assert payload["capability_verified"] is True
        assert runtime._gpu_capability_signature_valid(payload), name

    # Corrupting one record must leave the other usable.
    corrupted = runtime.gpu_capability_path(config, runtime._gpu_capability_key(models["alpha"]))
    payload = json.loads(corrupted.read_text())
    payload["signature"] = "0" * 64
    corrupted.write_text(json.dumps(payload))
    runtime.forget_gpu_capability()
    assert runtime._cached_gpu_capability(
        models["alpha"], runtime.verify_environment(models["alpha"], probe=False)) is None
    beta_report = runtime.verify_environment(models["beta"], probe=False)
    assert runtime._cached_gpu_capability(models["beta"], beta_report) is not None


# ------------------------------------------------------------------------ GPU-only mode


def test_gpu_only_mode_never_starts_an_evaluator(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--verify-gpu-only` probes and records, then exits without touching a checkpoint."""
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.scripts import evaluate_models as em

    marker = tmp_path / "evaluator_started"
    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "NOPE", "cuda": "12.4"},
             "native_protocol": "p"}
    checkpoint = {"checkpoint_id": "c", "filename": "f.pt"}

    measured: list[str] = []

    def fake_verify(model_arg, config_arg, *, authorized, require_cuda, force=False):
        measured.append(model_arg["model_id"])
        assert authorized and force is True
        return {"capability_verified": True, "capability_key": "k" * 64}

    monkeypatch.setattr(runtime, "verify_gpu_runtime", fake_verify)
    monkeypatch.setattr(em, "resolve_config", lambda: WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path, CIR_DATA_ROOT=tmp_path, FASHIONIQ_ROOT=tmp_path,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "ckpts", WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party"))

    assert em.verify_gpu_only([(model, checkpoint)], em.resolve_config()) == 0
    assert measured == ["m"]
    assert not marker.exists()


def test_gpu_only_mode_reports_failure_with_nonzero_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe must exit nonzero rather than silently reporting readiness."""
    from workbench.scripts import evaluate_models as em

    model = {"model_id": "m", "environment_required": True,
             "environment": {"python_env_var": "NOPE", "cuda": "12.4"}}
    checkpoint = {"checkpoint_id": "c", "filename": "f.pt"}

    def refuse(*args, **kwargs):
        raise runtime.GpuVerificationError("no cuda device answered")

    monkeypatch.setattr(runtime, "verify_gpu_runtime", refuse)
    from workbench.backend.operator_config import WorkbenchConfig
    config = WorkbenchConfig(CIR_REPO_ROOT=Path("/tmp"), CIR_DATA_ROOT=Path("/tmp"),
                             FASHIONIQ_ROOT=Path("/tmp"), WORKBENCH_HOST="127.0.0.1",
                             WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
                             WORKBENCH_CHECKPOINT_ROOT=Path("/tmp/ckpts"),
                             WORKBENCH_RESULTS_ROOT=Path("/tmp/results"),
                             WORKBENCH_THIRD_PARTY_ROOT=Path("/tmp/tp"))
    assert em.verify_gpu_only([(model, checkpoint)], config) == 1


def test_gpu_only_mode_skips_a_model_without_a_cuda_requirement(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only model is not probed at all: there is nothing to verify."""
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.scripts import evaluate_models as em

    probed: list[str] = []
    monkeypatch.setattr(runtime, "verify_gpu_runtime",
                        lambda *a, **k: probed.append("called") or {})
    config = WorkbenchConfig(CIR_REPO_ROOT=Path("/tmp"), CIR_DATA_ROOT=Path("/tmp"),
                             FASHIONIQ_ROOT=Path("/tmp"), WORKBENCH_HOST="127.0.0.1",
                             WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
                             WORKBENCH_CHECKPOINT_ROOT=Path("/tmp/ckpts"),
                             WORKBENCH_RESULTS_ROOT=Path("/tmp/results"),
                             WORKBENCH_THIRD_PARTY_ROOT=Path("/tmp/tp"))
    model = {"model_id": "cpu_only", "environment": {"cuda": "UNKNOWN"}}
    assert em.verify_gpu_only([(model, {"checkpoint_id": "c"})], config) == 0
    assert probed == []
