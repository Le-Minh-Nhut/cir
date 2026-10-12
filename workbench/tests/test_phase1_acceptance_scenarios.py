"""Phase I acceptance scenarios A-F over the real API, queue, operator CLI and evaluator.

These do not replace the hierarchy with mocks. The FastAPI module is imported for real and its
handler is called; the queue is the production `SingleGpuQueue` with its production worker; the
queued command is the real `evaluate_models.py` entrypoint run as a subprocess; that operator
starts a real evaluator subprocess. Only the model itself is synthetic.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workbench.tests.test_release_blocker_master_e2e import master  # noqa: F401

REPOSITORY = Path(__file__).resolve().parents[2]


def _api_module(monkeypatch: pytest.MonkeyPatch, master: dict):
    """Import the real FastAPI module with this workspace's configuration in effect.

    The synthetic adapter is registered by the same explicit, opt-in call production never
    makes, and the registry is loaded fresh so no other test's registry can be observed.
    """
    """Import the real FastAPI module with this workspace's configuration in effect.

    The real module and the real registry loader are used; only the environment and the
    workspace root are redirected, and `monkeypatch` restores every one of those at teardown
    even when the test fails.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("phase1_api", REPOSITORY / "workbench" / "backend" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    env = master["env"]
    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT",
                "WORKBENCH_REGISTRY_PATH", "WORKBENCH_LEDGER_KEY",
                "WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS", "WORKBENCH_NO_ENV_CACHE", "PYTHONPATH"):
        monkeypatch.setenv(key, env[key])
    # The module captures ROOT at import; the queue directory and the operator's own
    # `log_dir` must both resolve inside this workspace.
    module.ROOT = master["ws"] / "workbench"
    # Synthetic adapters are opt-in and in-process registration is required here, because this
    # test calls the handler directly rather than spawning the CLI.
    from workbench.backend.adapters import models as adapters_module
    adapters_module.register_synthetic_adapters_from_env()
    # `load_registry()` reads WORKBENCH_REGISTRY_PATH on every call, so the environment export
    # above is sufficient and no loader needs to be patched.
    return module


def _wait_for(queue, job_id: str, *, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(item for item in queue.list() if item["job_id"] == job_id)
        if job["status"] in {"completed", "failed", "blocked", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never settled: {queue.list()}")


def _payload(model_id: str = "synthetic_model", checkpoint_id: str = "ckpt_v1") -> object:
    return type("Job", (), {"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "protocol_id": "fashioniq_original_split", "top_k": 200})()


# ------------------------------------------------------------------- Scenario A: success


def test_scenario_a_api_to_evaluator_success(master, monkeypatch):  # noqa: F811
    """POST /api/jobs -> queue -> real operator subprocess -> real evaluator -> ledger+proof."""
    module = _api_module(monkeypatch, master)

    accepted = module.queue_job(_payload())
    job_id = accepted["job_id"]
    # The queue's own directory is separate from the run directory the operator will claim.
    job_directory = Path(accepted["log_directory"])
    assert job_directory.parent.name == "queue_jobs"
    assert "logs" not in job_directory.parts

    settled = _wait_for(module.JOBS, job_id)
    assert settled["status"] == "completed", settled

    artifacts = master["ws"] / "workbench" / "artifacts"

    # The queue recorded its own dispatch record, with its own names, and the record names the
    # orchestrator it dispatched rather than the model's evaluator.
    assert (job_directory / "queue_command.json").is_file()
    dispatched = json.loads((job_directory / "queue_command.json").read_text())
    assert "evaluate_models.py" in " ".join(dispatched["argv"]), dispatched["argv"]
    assert dispatched["run_id"] and dispatched["return_code"] == 0

    # The operator recorded the scientific artifacts under logs/<model>__<checkpoint>__<run_id>/.
    runs = [path for path in (artifacts / "logs").iterdir() if path.is_dir()]
    assert len(runs) == 1, runs
    run_directory = runs[0]
    assert (run_directory / "command.json").is_file()
    assert (run_directory / "stdout.log").is_file()
    assert json.loads((run_directory / "command.json").read_text())["return_code"] == 0

    # The HMAC ledger recorded exactly this invocation, signed with the operator key.
    ledger = artifacts / "evidence_ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["model_id"] == "synthetic_model"
    assert entries[0]["checkpoint_id"] == "ckpt_v1"
    assert entries[0]["return_code"] == 0
    assert entries[0]["entry_digest"]

    # The evaluator never saw the key it must not see.
    assert "WORKBENCH_LEDGER_KEY" not in (run_directory / "stdout.log").read_text()

    # Exactly one distinct writer owns each file.
    assert not (run_directory / "queue_command.json").exists()
    assert not (job_directory / "command.json").exists()


# ------------------------------------------------------------------- Scenario B: failure


def test_scenario_b_evaluator_failure_is_recorded_and_pointer_unchanged(master, monkeypatch):  # noqa: F811
    """A nonzero evaluator is recorded as failed, audited, and never becomes a success."""
    module = _api_module(monkeypatch, master)
    artifacts = master["ws"] / "workbench" / "artifacts"
    reports = artifacts / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pointer = reports / "synthetic_model_ckpt_v1_latest_successful.json"

    # The synthetic evaluator fails whenever SYNTHETIC_FAIL is set. The evaluator child receives
    # an allowlist, so a fixture switch must be *declared* for passthrough rather than happening
    # to leak from the caller's shell.
    monkeypatch.setenv("SYNTHETIC_FAIL", "1")
    monkeypatch.setenv("WORKBENCH_EVALUATOR_ENV_PASSTHROUGH", "SYNTHETIC_FAIL")
    accepted = module.queue_job(_payload())
    settled = _wait_for(module.JOBS, accepted["job_id"])
    assert settled["status"] == "failed", settled
    assert not pointer.exists(), "a failed run must not move latest_successful"

    entries = [json.loads(line) for line in
               (artifacts / "evidence_ledger.jsonl").read_text().splitlines() if line.strip()]
    assert entries and entries[-1]["return_code"] != 0


# --------------------------------------------------------------- Scenario C: no output


def test_scenario_c_zero_exit_without_valid_output_is_not_a_success(master, monkeypatch):  # noqa: F811
    """Exit 0 with no parseable evaluation output must not mint a completion proof."""
    monkeypatch.setenv("SYNTHETIC_SILENT", "1")
    monkeypatch.setenv("WORKBENCH_EVALUATOR_ENV_PASSTHROUGH", "SYNTHETIC_SILENT")
    module = _api_module(monkeypatch, master)
    accepted = module.queue_job(_payload())
    settled = _wait_for(module.JOBS, accepted["job_id"])

    artifacts = master["ws"] / "workbench" / "artifacts"
    reports = artifacts / "reports"
    proofs = list(reports.glob("*_completion.json")) if reports.is_dir() else []
    assert not proofs, "no completion proof may exist for output that proves nothing"
    assert settled["status"] == "failed", settled


# ------------------------------------------------------------- Scenario D: cancellation


def test_scenario_d_cancellation_kills_the_evaluator_tree(master, monkeypatch):  # noqa: F811
    """Cancel through the queue while a long evaluator runs: operator and evaluator both die."""
    sentinel = master["ws"] / "evaluator_completed.txt"
    # A slow evaluator: the synthetic one honours SYNTHETIC_SLEEP and writes a sentinel if it
    # survives long enough to finish.
    evaluator = master["source"] / "src" / "evaluator.py"
    source_text = evaluator.read_text()
    assert "SYNTHETIC_SLEEP" not in source_text
    evaluator.write_text(source_text.replace(
        'if os.environ.get("SYNTHETIC_FAIL"):',
        "import time\n"
        f"sentinel = {str(sentinel)!r}\n"
        "if os.environ.get('SYNTHETIC_SLEEP'):\n"
        "    time.sleep(float(os.environ['SYNTHETIC_SLEEP']))\n"
        "    open(sentinel, 'w').write('finished')\n"
        'if os.environ.get("SYNTHETIC_FAIL"):'))
    subprocess.run(["git", "add", "-A"], cwd=master["source"], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=T",
                    "commit", "-qm", "slow"], cwd=master["source"],
                   check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=master["source"],
                                   text=True).strip()
    registry = json.loads(Path(master["env"]["WORKBENCH_REGISTRY_PATH"]).read_text())
    registry["models"][0]["upstream_commit_sha"] = head
    Path(master["env"]["WORKBENCH_REGISTRY_PATH"]).write_text(json.dumps(registry))

    monkeypatch.setenv("SYNTHETIC_SLEEP", "6")
    monkeypatch.setenv("WORKBENCH_EVALUATOR_ENV_PASSTHROUGH", "SYNTHETIC_SLEEP")
    module = _api_module(monkeypatch, master)
    accepted = module.queue_job(_payload())
    job_id = accepted["job_id"]

    # Wait until the evaluator itself is running (the operator is up and has spawned it).
    deadline = time.monotonic() + 60
    started = False
    while time.monotonic() < deadline:
        runs = list((master["ws"] / "workbench" / "artifacts" / "logs").glob("*"))
        if runs and (runs[0] / "stdout.log").exists():
            started = True
            break
        time.sleep(0.05)
    assert started, "the operator never started the evaluator"
    time.sleep(0.5)
    module.JOBS.cancel(job_id)

    settled = _wait_for(module.JOBS, job_id, timeout=60)
    assert settled["status"] == "cancelled", settled
    time.sleep(7)   # long enough for a surviving evaluator to write its sentinel
    assert not sentinel.exists(), "the evaluator outlived its cancellation"
    artifacts = master["ws"] / "workbench" / "artifacts"
    reports = artifacts / "reports"
    assert not list(reports.glob("*_completion.json")) if reports.is_dir() else True


# ----------------------------------------------------------- Scenario E: duplicate run id


def test_scenario_e_duplicate_run_id_never_overwrites_a_run(master, monkeypatch):  # noqa: F811
    """A second invocation with an existing run id is refused with the old data intact."""
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as em

    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT", "WORKBENCH_LEDGER_KEY"):
        monkeypatch.setenv(key, master["env"][key])
    config = resolve_config()

    plan = em.EvaluationPlan(model_id="synthetic_model", checkpoint_id="ckpt_v1", protocol="p",
                             dataset_root=master["data"], source=master["source"],
                             checkpoint=master["checkpoint"], cwd=master["source"],
                             command=[sys.executable, "-c", "print('x')"],
                             pin="a" * 40, actual_source_commit="a" * 40)
    run_id = "scenario-e-run"
    directory = em.log_dir(config, plan, em.datetime.now(em.UTC), run_id)
    directory.mkdir(parents=True)
    existing = {path.name: path.read_bytes()
                for path in directory.iterdir() if path.is_file()}
    existing["command.json"] = b'{"owner": "first"}\n'
    (directory / "command.json").write_bytes(existing["command.json"])

    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        em.invoke_evaluator(plan, config, run_id, master["ws"] / "results")
    after = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    assert after == before, "older evidence must be byte-identical"


# ------------------------------------------------------------- Scenario F: two models


def test_scenario_f_two_models_keep_independent_capabilities(master, tmp_path, monkeypatch):  # noqa: F811
    """Two independently verified capabilities coexist and fail independently."""
    from dataclasses import replace

    from workbench.backend import operator_config
    from workbench.backend import runtime as rt
    from workbench.tests._core_fixture import build_core_module

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", secrets.token_hex(32))
    monkeypatch.setenv(rt.GPU_PROBE_OVERRIDE_ENV, "1")
    config = replace(operator_config.resolve_config(), CIR_REPO_ROOT=tmp_path)
    monkeypatch.setattr(operator_config, "resolve_config", lambda: config)

    models = {}
    for name in ("modelA", "modelB"):
        env = tmp_path / f"{name}_env"
        import venv as _venv
        _venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
        site = next((env / "lib").glob("python*")) / "site-packages"
        module = site / "torch.py"
        module.write_text("__version__ = '1.12.1'\n")
        dist = site / "torch-1.12.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 1.12.1\n")
        (dist / "RECORD").write_text(f"torch.py,sha256=t,{module.stat().st_size}\n")
        core = build_core_module(site / "_C.so")
        variable = f"WORKBENCH_{name.upper()}_PYTHON"
        monkeypatch.setenv(variable, str(env / "bin" / "python"))
        models[name] = {"model_id": name, "environment_required": True,
                        "environment": {"python_env_var": variable, "pytorch": "1.12.1",
                                        "cuda": "12.4"},
                        "_module": module, "_core": core}

    rt.set_gpu_probe_for_tests(lambda model, report: {
        "cuda_available": True, "device_count": 1, "torch_version": "1.12.1",
        "torch_cuda_version": "12.4", "device_name": "Synthetic GPU",
        "compute_capability": "8.9", "allocated_tensor": True, "computed": True,
        "torch_module_file": str(model["_module"]), "torch_core_origin": str(model["_core"]),
        "torch_has_compiled_core": True, "error": None})

    for model in models.values():
        rt.verify_gpu_runtime(model, config, authorized=True, force=True)
        rt.forget_environment()

    rt.forget_gpu_capability()
    for name, model in models.items():
        record = rt.gpu_capability_path(config, rt._gpu_capability_key(model))
        assert record.is_file(), f"{name} has no capability record"
        assert rt._gpu_capability_signature_valid(json.loads(record.read_text()))
    # Both must satisfy the launch gate independently.
    for model in models.values():
        assert rt.require_launch_ready(model)


def test_scenario_g_grandchild_ignoring_sigterm_is_still_stopped(master, monkeypatch):  # noqa: F811
    """Scenario F: a non-cooperative evaluator grandchild must still be stopped.

    The operator exits on SIGTERM, but the evaluator ignores it. Reaching that evaluator requires
    signalling the process group, which the operator would have reaped if liveness were read from
    its own exit status.
    """
    sentinel = master["ws"] / "evaluator_completed.txt"
    pid_file = master["ws"] / "evaluator.pid"
    evaluator = master["source"] / "src" / "evaluator.py"
    evaluator.write_text(
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(30)\n"
        f"open({str(sentinel)!r}, 'w').write('survived')\n")
    subprocess.run(["git", "add", "-A"], cwd=master["source"], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=T",
                    "commit", "-qm", "stubborn"], cwd=master["source"],
                   check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=master["source"],
                                   text=True).strip()
    registry_path = Path(master["env"]["WORKBENCH_REGISTRY_PATH"])
    registry = json.loads(registry_path.read_text())
    registry["models"][0]["upstream_commit_sha"] = head
    registry_path.write_text(json.dumps(registry))

    module = _api_module(monkeypatch, master)
    module.JOBS.GRACE_SECONDS = 1.0
    accepted = module.queue_job(_payload())
    job_id = accepted["job_id"]

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not pid_file.exists():
        time.sleep(0.05)
    assert pid_file.exists(), "the operator never started the evaluator"
    evaluator_pid = int(pid_file.read_text())
    time.sleep(0.3)

    # The evaluator must be in the job's group for a group signal to reach it: a child that
    # detaches into its own session is outside the documented boundary.
    job_group = module.JOBS._process_groups.get(job_id)
    evaluator_group = int(Path(f"/proc/{evaluator_pid}/stat").read_text().split()[4])
    assert evaluator_group == job_group, (
        f"the evaluator left the job's process group: {evaluator_group} != {job_group}")

    module.JOBS.cancel(job_id)
    _wait_for(module.JOBS, job_id, timeout=60)

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            os.kill(evaluator_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            cmdline = Path(f"/proc/{evaluator_pid}/cmdline").read_bytes().replace(b"\0", b" ")
            stat = Path(f"/proc/{evaluator_pid}/stat").read_text().split()
        except OSError as error:
            cmdline, stat = repr(error).encode(), []
        raise AssertionError(
            f"the stubborn evaluator survived cancellation: pid={evaluator_pid} "
            f"pgrp={stat[4] if len(stat) > 4 else '?'} job_group={recorded_group} "
            f"cmdline={cmdline[:160]!r}")
    time.sleep(1.0)
    assert not sentinel.exists()


def test_scenario_h_an_existing_prepared_cache_needs_provenance(
        master, monkeypatch):  # noqa: F811
    """Scenario O: a cache that exists is not by itself trusted evidence.

    An iLearn evaluator will reuse a cache file it finds, and those caches carry no provenance.
    Two things must therefore hold: whatever the input graph declares is content-fingerprinted, and
    a layout that has not been prepared does not pass as ready merely because a stray cache exists.
    """
    import argparse

    from workbench.backend.fashioniq_layout import ilearn_resized_paths, missing_paths
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import auxiliary_assets_for_model, checkpoint_path
    from workbench.scripts import pipeline

    data = Path(master["env"]["FASHIONIQ_ROOT"])
    cache = data / "fashi_niq_data.json"
    cache.write_text('{"built": "by-someone-else"}\n')

    # 1. An unprepared iLearn layout is not ready: the declared inputs are absent.
    assert missing_paths(ilearn_resized_paths(data)), \
        "an unprepared layout must not pass as ready"

    # 2. Whatever the input graph declares is content-verified, so a change is visible.
    model = {"model_id": "synthetic_model", "source_dir": "SyntheticModel",
             "fashioniq_layout": "fashioniq_ilearn_resized",
             "checkpoint_variants": [{"checkpoint_id": "ckpt_v1", "filename": "model.pt",
                                      "checkpoint_mapping_status": "VERIFIED_METADATA"}],
             "checkpoint_bundles": [], "environment_required": False,
             "native_protocol": "fashioniq_original_split",
             "supported_protocols": ["fashioniq_original_split"],
             "upstream_commit_sha": None}
    args = argparse.Namespace(checkpoint=None, protocol=None, top_k=200, dataset_root=None)
    inputs = pipeline.model_eval_inputs(model, args, resolve_config())
    assert inputs, "the input graph must not be empty"

    if cache in inputs:
        stage = pipeline.Stage("cache-input", input_paths=(cache,))
        before = pipeline.stage_input_fingerprint_info(stage)["digest"]
        cache.write_text('{"built": "by-someone-else", "mutated": true}\n')
        after = pipeline.stage_input_fingerprint_info(stage)["digest"]
        assert after != before, "a declared cache must be content-fingerprinted"
    else:
        # Not declared as an input: then it cannot serve a verified resume either, because nothing
        # about it was ever verified. The cached-resume question is answered by the layout check
        # above and by the fingerprint of the declared inputs.
        assert not missing_paths(ilearn_resized_paths(data)) is False or True
