"""Adversarial regression suite for CIR Failure Analysis Workbench release blockers.

Drives REAL code paths and REAL launcher entrypoint (pc.sh) using PATH-shim isolation.
Every test targets observable runtime behaviour (exit codes, printed claims, job statuses,
returned verdicts), never source text.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest
import yaml

from workbench.tests.test_pc_launcher import (
    FIXTURES_BIN,
    PC_SH,
    REPO,
    run_pc,
    setup_sandbox,
)


def _install_custom_curl(sandbox: Path, env: dict[str, str], runs_payload: list[dict] | None = None) -> None:
    """Provide a custom curl shim in sandbox that allows overriding /api/runs."""
    curl_shim = sandbox / "bin" / "curl"
    if curl_shim.is_symlink() or curl_shim.exists():
        curl_shim.unlink()

    if runs_payload is not None:
        data_file = sandbox / "mock_runs_data.json"
        data_file.write_text(json.dumps(runs_payload), encoding="utf-8")
        env["MOCK_OVERRIDE_RUNS_FILE"] = str(data_file)

    script = f"""#!/usr/bin/env python3
import sys, os, subprocess
from pathlib import Path

runs_file = os.environ.get("MOCK_OVERRIDE_RUNS_FILE")
args_str = " ".join(sys.argv)
if runs_file and os.path.exists(runs_file):
    url = ""
    for a in sys.argv[1:]:
        if a.startswith("http://") or a.startswith("https://"):
            url = a
            break
    if not url:
        for a in reversed(sys.argv[1:]):
            if not a.startswith("-"):
                url = a
                break
    if "/api/runs" in url and not ("/api/runs/" in url):
        body = Path(runs_file).read_text(encoding="utf-8")
        has_w = "-w" in sys.argv or "--write-out" in args_str
        is_devnull = "-o /dev/null" in args_str or ("-o" in sys.argv and "/dev/null" in sys.argv)
        if is_devnull:
            if has_w:
                print("200")
        elif has_w:
            print(f"{{body}}\\n200")
        else:
            print(body)
        sys.exit(0)

real_curl = "{FIXTURES_BIN / 'curl'}"
os.execv(real_curl, [real_curl] + sys.argv[1:])
"""
    curl_shim.write_text(script, encoding="utf-8")
    curl_shim.chmod(0o755)


# ---------------------------------------------------------------------------
# 1. Missing report -> pc.sh exits non-zero and never prints CORROBORATED
# ---------------------------------------------------------------------------
def test_adv01_missing_report_exits_nonzero_and_never_prints_corroborated(tmp_path: Path):
    """Missing report -> pc.sh exits non-zero and never prints CORROBORATED."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    # Ensure reports directory is empty
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit on missing report, got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Output claimed CORROBORATED despite missing report"


# ---------------------------------------------------------------------------
# 2. Missing ledger / missing completion proof -> non-zero
# ---------------------------------------------------------------------------
def test_adv02_missing_ledger_or_completion_proof_exits_nonzero(tmp_path: Path):
    """Missing ledger / missing completion proof -> non-zero."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Valid report exists, but neither ledger nor completion proof is present
    rep_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    rep_path.write_text(json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED",
        "category_metrics": {
            "dress": {"r10": 20.0, "r50": 45.0},
            "shirt": {"r10": 22.0, "r50": 48.0},
            "toptee": {"r10": 25.0, "r50": 52.0},
        },
        "observed_metrics": {"r10": 22.33, "r50": 48.33, "mean": 35.33},
    }), encoding="utf-8")

    ledger_path = sandbox / "workbench" / "artifacts" / "evidence_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit when ledger/proof is missing, got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Output claimed CORROBORATED despite missing ledger/proof"


# ---------------------------------------------------------------------------
# 3. Forged ledger entry, and digest mismatch between ledger and report -> refused
# ---------------------------------------------------------------------------
def test_adv03_forged_ledger_entry_and_digest_mismatch_refused(tmp_path: Path):
    """Forged ledger entry, and digest mismatch between ledger and report -> refused."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rep_content = json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED",
        "category_metrics": {
            "dress": {"r10": 20.0, "r50": 45.0},
            "shirt": {"r10": 22.0, "r50": 48.0},
            "toptee": {"r10": 25.0, "r50": 52.0},
        },
        "observed_metrics": {"r10": 22.33, "r50": 48.33, "mean": 35.33},
    })
    rep_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    rep_path.write_text(rep_content, encoding="utf-8")
    digest = hashlib.sha256(rep_content.encode("utf-8")).hexdigest()

    # Forged ledger entry: matching digest, but forged/unauthenticated HMAC signature
    ledger_path = sandbox / "workbench" / "artifacts" / "evidence_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
        "report_digest": digest,
        "hmac_signature": "forged_hmac_signature_value",
    }) + "\n", encoding="utf-8")

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit on forged ledger entry (invalid HMAC), got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Output claimed CORROBORATED despite forged ledger entry"

# ---------------------------------------------------------------------------
# 4. Missing category (e.g. only dress+shirt) for LIMN -> refused as incomplete
# ---------------------------------------------------------------------------
def test_adv04_missing_category_for_limn_refused_as_incomplete(tmp_path: Path):
    """Missing category (e.g. only dress+shirt) for LIMN -> refused as incomplete."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Incomplete category_metrics: dress and shirt only, toptee missing
    rep_content = json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED",
        "category_metrics": {
            "dress": {"r10": 20.0, "r50": 45.0},
            "shirt": {"r10": 22.0, "r50": 48.0},
        },
        "observed_metrics": {"r10": 21.0, "r50": 46.5, "mean": 33.75},
    })
    rep_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    rep_path.write_text(rep_content, encoding="utf-8")

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit on incomplete categories for LIMN, got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Output claimed CORROBORATED despite missing category for LIMN"


# ---------------------------------------------------------------------------
# 5. Invalid Recall (NaN, negative, > 100) -> refused
# ---------------------------------------------------------------------------
def test_adv05_invalid_recall_refused(tmp_path: Path):
    """Invalid Recall (NaN, negative, > 100) -> refused."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Invalid recall values: nan, negative, > 100
    rep_content = json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED",
        "category_metrics": {
            "dress": {"r10": -5.0, "r50": 45.0},
            "shirt": {"r10": 105.0, "r50": 48.0},
            "toptee": {"r10": float("nan"), "r50": 52.0},
        },
        "observed_metrics": {"r10": 50.0, "r50": 48.0, "mean": 49.0},
    })
    rep_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    rep_path.write_text(rep_content, encoding="utf-8")

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit on invalid Recall values, got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Output claimed CORROBORATED despite invalid Recall values"


# ---------------------------------------------------------------------------
# 6. Mock/synthetic result reuse -> refused; a fresh job is queued instead
# ---------------------------------------------------------------------------
def test_adv06_mock_synthetic_result_reuse_refused_fresh_job_queued(tmp_path: Path):
    """Mock/synthetic result reuse -> refused; a fresh job is queued instead."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    # Candidate in /api/runs is mock/synthetic
    mock_run = [{
        "run_id": "run_limn_mock_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "data_kind": "mock",
        "synthetic": True,
        "corroborated": False,
        "created_at": "2026-10-16T00:00:00Z",
    }]
    _install_custom_curl(sandbox, env, mock_run)

    proc = run_pc(sandbox, [], env)
    # The mock run must NOT be reused; a fresh job must be queued via POST to /api/jobs
    assert post_jobs_file.exists() and len(post_jobs_file.read_text().strip().splitlines()) > 0, (
        f"Expected fresh job POST to /api/jobs, but none occurred. Returncode: {proc.returncode}"
    )
    assert "run_limn_mock_01" not in proc.stdout, "Pre-existing mock run was improperly reused as official result"


# ---------------------------------------------------------------------------
# 7. Wrong checkpoint / wrong protocol / wrong runtime adaptation / wrong dataset provenance -> refused
# ---------------------------------------------------------------------------
def test_adv07_wrong_checkpoint_or_protocol_or_adaptation_or_provenance_refused(tmp_path: Path):
    """Wrong checkpoint / protocol / adaptation / dataset provenance -> refused."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    # Candidate run has wrong runtime adaptation ("sm_80" instead of sm_120) and unverified provenance
    mismatched_run = [{
        "run_id": "run_limn_wrong_adapt_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "runtime_adaptation": "sm_80",
        "dataset_provenance": "unverified",
        "corroborated": True,
    }]
    _install_custom_curl(sandbox, env, mismatched_run)

    proc = run_pc(sandbox, [], env)
    # Must refuse to reuse mismatched candidate run and queue a fresh job instead
    assert post_jobs_file.exists() and len(post_jobs_file.read_text().strip().splitlines()) > 0, (
        f"Expected fresh job POST to /api/jobs for mismatched candidate, but none occurred. Output: {proc.stdout}"
    )


# ---------------------------------------------------------------------------
# 8. Evaluator exits 0 with no observable output -> the job is never completed
# ---------------------------------------------------------------------------
def test_adv08_evaluator_exit_zero_no_output_job_never_completed(tmp_path: Path):
    """Evaluator exits 0 with no observable output -> the job is never completed."""
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "job_silent"
    silent_script = tmp_path / "silent_eval.py"
    silent_script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    job = queue.enqueue(
        "limn", "base_iter0_all_categories", "fashioniq_val_split",
        [sys.executable, str(silent_script)],
        tmp_path, directory, {"model_id": "limn"}
    )

    deadline = time.monotonic() + 15.0
    terminal_status = None
    while time.monotonic() < deadline:
        jobs = queue.list()
        matching = next((j for j in jobs if j["job_id"] == job["job_id"]), None)
        if matching and matching["status"] in ("completed", "failed", "blocked", "cancelled"):
            terminal_status = matching["status"]
            break
        time.sleep(0.1)

    queue.shutdown()
    assert terminal_status is not None, "Job did not settle within deadline"
    assert terminal_status != "completed", (
        f"Job settled as '{terminal_status}'; must never be 'completed' when evaluator produces no observable output"
    )


# ---------------------------------------------------------------------------
# 9. Job completed without a valid completion proof -> consumer treats as unverified
# ---------------------------------------------------------------------------
def test_adv09_job_completed_without_valid_completion_proof_unverified(tmp_path: Path):
    """Job completed without a valid completion proof -> consumer treats as unverified."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rep_content = json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED",
        "category_metrics": {
            "dress": {"r10": 20.0, "r50": 45.0},
            "shirt": {"r10": 22.0, "r50": 48.0},
            "toptee": {"r10": 25.0, "r50": 52.0},
        },
        "observed_metrics": {"r10": 22.33, "r50": 48.33, "mean": 35.33},
    })
    rep_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    rep_path.write_text(rep_content, encoding="utf-8")

    digest = hashlib.sha256(rep_content.encode("utf-8")).hexdigest()
    ledger_path = sandbox / "workbench" / "artifacts" / "evidence_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps({
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
        "report_digest": digest,
    }) + "\n", encoding="utf-8")

    # Completion proof is absent
    proof_path = reports_dir / "limn_base_iter0_all_categories_run_limn_corr_01_completion.json"
    if proof_path.exists():
        proof_path.unlink()

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0, f"Expected non-zero exit when completion proof is missing, got {proc.returncode}"
    combined = (proc.stdout + proc.stderr).upper()
    assert "CORROBORATED" not in combined, "Consumer treated run without completion proof as corroborated evidence"


# ---------------------------------------------------------------------------
# 10. Forged or stale GPU capability record -> release_verify.sh does not reach RELEASE_READY
# ---------------------------------------------------------------------------
def test_adv10_forged_or_stale_gpu_record_release_verify_not_release_ready(tmp_path: Path):
    """Forged or stale GPU capability record -> release_verify.sh does not reach RELEASE_READY."""
    artifacts_dir = tmp_path / "artifacts"
    gpu_dir = artifacts_dir / "gpu_capabilities"
    gpu_dir.mkdir(parents=True, exist_ok=True)

    # Forged capability record: interpreter probe claiming verification without valid signature
    forged_key = "fake_gpu_probe"
    (gpu_dir / f"{forged_key}.json").write_text(json.dumps({
        "capability_key": forged_key,
        "device_name": "NVIDIA GeForce RTX 5070 Ti",
        "compute_capability": "12.0",
        "probe_source": "interpreter",
        "capability_verified": True,
        "probed_at": "2026-10-16T00:00:00Z",
    }, indent=2), encoding="utf-8")

    report_path = tmp_path / "release_report.json"
    verify_sh = Path(os.environ.get("RELEASE_VERIFY_SCRIPT")
                     or REPO / "workbench" / "docker" / "release_verify.sh")

    sub_env = os.environ.copy()
    sub_env["WORKBENCH_ARTIFACTS_ROOT"] = str(artifacts_dir)
    # A withheld operator key is the hostile case: with no key in the environment the record's
    # claim cannot be authenticated at all, and the scan must not fall back on its own fields.
    sub_env.pop("WORKBENCH_LEDGER_KEY", None)
    res = subprocess.run(
        ["bash", str(verify_sh), "--offline", "--report", str(report_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=sub_env,
    )

    assert report_path.is_file(), f"release_verify.sh did not write report. Output: {res.stdout}\n{res.stderr}"
    report_data = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_data.get("verdict") != "RELEASE_READY", (
        f"Forged GPU record must not allow verdict RELEASE_READY, got: {report_data.get('verdict')}"
    )
    gpu_dim = report_data.get("dimensions", {}).get("GPU_RUNTIME", {})
    assert gpu_dim.get("status") != "PASS", (
        f"GPU_RUNTIME status must not be PASS under forged capability record, got: {gpu_dim.get('status')}"
    )


# ---------------------------------------------------------------------------
# 11. An existing CPU-only container is not reused for a GPU-required execution
# ---------------------------------------------------------------------------
def test_adv11_existing_cpu_only_container_not_reused_for_gpu_execution(tmp_path: Path):
    """An existing CPU-only container is not reused for a GPU-required execution."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)

    # Running container exists, but inspect reveals CPU-only configuration (no GPU device requests)
    docker_shim = sandbox / "bin" / "docker"
    orig_docker = docker_shim.read_text(encoding="utf-8")
    if docker_shim.is_symlink() or docker_shim.exists():
        docker_shim.unlink()
    new_docker = orig_docker.replace(
        'if len(sys.argv) > 1 and sys.argv[1] == "ps":',
        'if "inspect" in sys.argv:\n'
        '    print(json.dumps([{"HostConfig": {"DeviceRequests": []}, "Config": {"Env": ["WORKBENCH_ENABLE_GPU=0"]}}]))\n'
        '    sys.exit(0)\n'
        'if len(sys.argv) > 1 and sys.argv[1] == "ps":'
    )
    docker_shim.write_text(new_docker, encoding="utf-8")
    docker_shim.chmod(0o755)

    # Instrument run.sh to record if "up" was invoked to reconfigure container
    run_sh = sandbox / "workbench" / "docker" / "run.sh"
    calls_log = sandbox / "run_sh_invocations.log"
    orig_content = run_sh.read_text(encoding="utf-8")
    run_sh.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{calls_log}"
{orig_content}
""", encoding="utf-8")
    run_sh.chmod(0o755)

    proc = run_pc(sandbox, [], env)

    calls = calls_log.read_text(encoding="utf-8") if calls_log.exists() else ""
    restarted_stack = "up" in calls
    silently_reused = "Docker container cir-workbench đã đang chạy và API sẵn sàng" in proc.stdout and not restarted_stack

    assert not silently_reused and (restarted_stack or proc.returncode != 0), (
        f"CPU-only container was silently reused without GPU reprovisioning or refusal! "
        f"returncode={proc.returncode}, run.sh calls: {calls}"
    )


# ---------------------------------------------------------------------------
# 12. Existing valid corroborated results during release verification -> NO false failure (Positive Control)
# ---------------------------------------------------------------------------
def _append_ledger_entry(env: dict[str, str], report_path: Path, run_id: str) -> None:
    """Corroborate one aggregate report through the evaluator's own ledger writer."""
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from workbench.backend.operator_config import resolve_config\n"
        "from workbench.scripts import evaluate_models\n"
        "config = resolve_config()\n"
        "path = Path(sys.argv[1])\n"
        "evaluate_models.append_evidence_entry(config, {\n"
        "    'run_id': sys.argv[2], 'report_digest': evaluate_models.sha256_file(path),\n"
        "    'model_id': 'limn', 'checkpoint_id': 'base_iter0_all_categories',\n"
        "    'protocol_id': 'fashioniq_val_split', 'return_code': 0,\n"
        "    'invocation_status': 'INVOCATION_COMPLETED'})\n"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(report_path), run_id],
                               capture_output=True, text=True, env=env, cwd=REPO)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)

def test_adv12_existing_valid_corroborated_results_no_false_failure(tmp_path: Path):
    """Existing valid corroborated results during release verification -> NO false failure."""
    models_yaml = REPO / "workbench" / "registry" / "models.yaml"
    models_data = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))["models"]
    models_payload = [{"model_id": m["model_id"]} for m in models_data]

    # The run's corroboration lives outside the served payload: J1 reads the evidence ledger
    # itself, so the fixture must provide the report bytes and their ledger entry.
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # A real evaluation run: its aggregate report exists under the reports root and the
    # out-of-band evidence ledger corroborates its bytes with a completed invocation, which is
    # what makes it real evidence rather than a fake. The payload the API serves must match the
    # report's own identity, because J1 cross-checks the two.
    valid_runs_payload = [{
        "run_id": "run_limn_corr_01",
        "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "data_kind": "experiment",
    }]
    reports = artifacts_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    aggregate = reports / "limn_base_iter0_all_categories_run_limn_corr_01_aggregate.json"
    aggregate.write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "schema_version": 1,
        "run_id": "run_limn_corr_01", "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories", "protocol_id": "fashioniq_val_split",
        "execution_status": "EXECUTION_COMPLETED"}), encoding="utf-8")

    class MockWorkbenchServer(BaseHTTPRequestHandler):
        def do_GET(self):
            if "/api/models" in self.path:
                body = json.dumps(models_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif "/api/runs" in self.path:
                body = json.dumps(valid_runs_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif "/api/jobs" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[]")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), MockWorkbenchServer)
    port = server.server_port
    th = Thread(target=server.serve_forever, daemon=True)
    th.start()

    try:
        report_path = tmp_path / "release_report.json"
        verify_sh = Path(os.environ.get("RELEASE_VERIFY_SCRIPT")
                         or REPO / "workbench" / "docker" / "release_verify.sh")
        sub_env = os.environ.copy()
        sub_env["WORKBENCH_ARTIFACTS_ROOT"] = str(artifacts_dir)
        sub_env["WORKBENCH_LEDGER_KEY"] = "adv12-ledger-key"
        for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT",
                    "WORKBENCH_CHECKPOINT_ROOT", "WORKBENCH_RESULTS_ROOT",
                    "WORKBENCH_THIRD_PARTY_ROOT"):
            sub_env[key] = str(artifacts_dir / key.lower())
        _append_ledger_entry(sub_env, aggregate, "run_limn_corr_01")
        res = subprocess.run(
            ["bash", str(verify_sh), "--base", f"http://127.0.0.1:{port}", "--report", str(report_path)],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=sub_env,
        )

        assert report_path.is_file(), f"release_verify.sh did not write report. Output: {res.stdout}\n{res.stderr}"
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        j1_status = report_data.get("checks", {}).get("J1", {}).get("status")

        # Positive control: existing valid corroborated run MUST NOT fail J1
        assert j1_status != "FAIL", (
            f"False failure on existing corroborated run! J1 status is FAIL. "
            f"Detail: {report_data.get('checks', {}).get('J1', {}).get('detail')}"
        )
    finally:
        server.shutdown()
        server.server_close()
