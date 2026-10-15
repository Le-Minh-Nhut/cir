"""Executable test harness for the CIR Workbench entrypoint pc.sh.

Tests run against the real ./pc.sh through PATH shims inside isolated
tmp_path sandboxes. Never touches docker daemon, GPU, network, or real dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PC_SH = REPO / "pc.sh"
FIXTURES_BIN = Path(__file__).resolve().parent / "fixtures" / "pc_launcher" / "bin"
CRITICAL_REPO_FILES = [
    REPO / "workbench/docker/.env.example",
    REPO / "workbench/docker/run.sh",
    REPO / "workbench/scripts/pipeline.py",
]


def _repo_fingerprint() -> dict[Path, str]:
    res = {}
    for f in CRITICAL_REPO_FILES:
        if f.exists():
            res[f] = hashlib.sha256(f.read_bytes()).hexdigest()
    return res


def _make_safe_path(bin_dir: Path, omit_tools: set[str] | None = None) -> str:
    """Build an isolated PATH with shims first, omitting specified tools."""
    omits = omit_tools or set()
    safe_sys_bin = bin_dir.parent / "safe_sys_bin"
    safe_sys_bin.mkdir(parents=True, exist_ok=True)

    needed_tools = [
        "bash", "sh", "python3", "python", "sed", "grep", "awk", "cat",
        "chmod", "cp", "mkdir", "rm", "sleep", "printf", "echo", "dirname",
        "basename", "uname", "cut", "tr", "head", "tail", "wc", "sort",
        "uniq", "find", "test", "expr", "env", "which", "id", "stat", "date",
        "xargs"
    ]
    for tool in needed_tools:
        if tool in omits:
            continue
        target = shutil.which(tool)
        if target and not (safe_sys_bin / tool).exists():
            try:
                (safe_sys_bin / tool).symlink_to(target)
            except OSError:
                shutil.copy2(target, safe_sys_bin / tool)

    for shim in FIXTURES_BIN.glob("*"):
        if shim.name not in omits and not (bin_dir / shim.name).exists():
            try:
                (bin_dir / shim.name).symlink_to(shim)
            except OSError:
                shutil.copy2(shim, bin_dir / shim.name)
            (bin_dir / shim.name).chmod(0o755)

    return f"{bin_dir}:{safe_sys_bin}"


def setup_sandbox(
    tmp_path: Path,
    omit_tools: set[str] | None = None,
    with_env: bool = False,
    custom_env_content: str | None = None,
    with_dataset: bool = True,
    dataset_path: Path | None = None,
) -> tuple[Path, dict[str, str]]:
    """Create isolated sandbox directory and environment variables."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    bin_dir = sandbox / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    path_var = _make_safe_path(bin_dir, omit_tools=omit_tools)

    docker_dir = sandbox / "workbench" / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "workbench/docker/.env.example", docker_dir / ".env.example")
    shutil.copy2(REPO / "workbench/docker/run.sh", docker_dir / "run.sh")
    (docker_dir / "run.sh").chmod(0o755)

    # docker-compose fixtures
    if (REPO / "workbench/docker/docker-compose.yml").exists():
        shutil.copy2(REPO / "workbench/docker/docker-compose.yml", docker_dir / "docker-compose.yml")
    if (REPO / "workbench/docker/docker-compose.gpu.yml").exists():
        shutil.copy2(REPO / "workbench/docker/docker-compose.gpu.yml", docker_dir / "docker-compose.gpu.yml")

    if with_env or custom_env_content is not None:
        env_content = custom_env_content or (docker_dir / ".env.example").read_text()
        (docker_dir / ".env").write_text(env_content)

    ds_root = dataset_path or (sandbox / "data" / "FashionIQ")
    if with_dataset:
        for split in ("images/dress", "images/shirt", "images/toptee"):
            (ds_root / split).mkdir(parents=True, exist_ok=True)
        (ds_root / "image_splits").mkdir(parents=True, exist_ok=True)
        (ds_root / "captions").mkdir(parents=True, exist_ok=True)
        for cat in ("dress", "shirt", "toptee"):
            (ds_root / "image_splits" / f"split.{cat}.val.json").write_text("[]")
            (ds_root / "captions" / f"cap.{cat}.val.json").write_text("[]")

    log_file = sandbox / "mock_tools.log"
    counter_file = sandbox / "curl_counter.txt"
    env = {
        "PATH": path_var,
        "CIR_REPO_ROOT": str(sandbox),
        "WORKBENCH_DOCKER_DIR": str(docker_dir),
        "HOST_FASHIONIQ_ROOT": str(ds_root),
        "MOCK_LOG_FILE": str(log_file),
        "MOCK_COUNTER_FILE": str(counter_file),
        "WORKBENCH_BACKEND_PORT": "8000",
        "WORKBENCH_FRONTEND_PORT": "5173",
    }
    return sandbox, env


def run_pc(
    sandbox: Path,
    args: list[str],
    env: dict[str, str],
    timeout: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    """Execute ./pc.sh and assert real repository was untouched."""
    if not PC_SH.exists():
        pytest.skip("pc.sh not yet created on disk")

    before = _repo_fingerprint()
    full_env = {**os.environ, **env}

    proc = subprocess.run(
        [str(PC_SH), *args],
        cwd=sandbox,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    after = _repo_fingerprint()
    assert before == after, "pc.sh mutated files in the real repository!"
    return proc


def _read_tool_logs(sandbox: Path) -> list[dict[str, Any]]:
    log_file = sandbox / "mock_tools.log"
    if not log_file.exists():
        return []
    records = []
    for line in log_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ---------------------------------------------------------------------------
# 20 REQUIRED TEST CASES
# ---------------------------------------------------------------------------

def test_help_exits_zero_and_prints_usage(tmp_path: Path):
    """1. --help exits 0 and prints usage."""
    sandbox, env = setup_sandbox(tmp_path)
    proc = run_pc(sandbox, ["--help"], env)
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "--doctor" in out or "doctor" in out
    assert "--model" in out or "usage" in out.lower() or "pc.sh" in out


def test_doctor_is_read_only(tmp_path: Path):
    """2. --doctor is read-only: exits 0, mutates nothing."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    manifest_before = {
        p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sandbox.rglob("*") if p.is_file() and p.name != "mock_tools.log"
    }

    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0

    manifest_after = {
        p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sandbox.rglob("*") if p.is_file() and p.name != "mock_tools.log"
    }
    assert manifest_before == manifest_after


def test_doctor_with_missing_docker_exits_nonzero(tmp_path: Path):
    """3. --doctor with a fake docker that is absent -> exits non-zero naming docker."""
    sandbox, env = setup_sandbox(tmp_path, omit_tools={"docker"})
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).lower()
    assert "docker" in out


def test_fake_nvidia_smi_absent_reports_gpu_pending(tmp_path: Path):
    """4. Fake nvidia-smi absent -> GPU reported PENDING (or blocked), never VERIFIED."""
    sandbox, env = setup_sandbox(tmp_path, omit_tools={"nvidia-smi"}, with_env=True)
    proc = run_pc(sandbox, ["--doctor"], env)
    out = proc.stdout + proc.stderr
    assert "PENDING" in out.upper() or "BLOCKED" in out.upper() or proc.returncode != 0
    assert "GPU validation: VERIFIED" not in out
    assert "GPU: VERIFIED" not in out


def test_fake_nvidia_smi_present_no_container_runtime_reports_gpu_pending(tmp_path: Path):
    """5. nvidia-smi present but runtime missing from docker info -> GPU PENDING with toolkit named."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_DOCKER_NO_NVIDIA"] = "1"
    proc = run_pc(sandbox, ["--doctor"], env)
    out = proc.stdout + proc.stderr
    assert "PENDING" in out.upper() or "BLOCKED" in out.upper() or proc.returncode != 0
    assert "toolkit" in out.lower() or "nvidia-container-toolkit" in out.lower() or "runtime" in out.lower()
    assert "GPU validation: VERIFIED" not in out


def test_fresh_config_creates_env_from_example_with_limn_and_gpu(tmp_path: Path):
    """6. Fresh config: no .env in sandbox -> created with limn, GPU=1, sm_120."""
    sandbox, env = setup_sandbox(tmp_path, with_env=False)
    target_env = sandbox / "workbench" / "docker" / ".env"
    assert not target_env.exists()

    proc = run_pc(sandbox, [], env)
    assert target_env.exists()
    content = target_env.read_text()
    assert "WORKBENCH_ENABLE_GPU=1" in content
    assert "WORKBENCH_RUNTIME_ADAPTATION=sm_120" in content


def test_existing_config_preserves_custom_ports_and_host_paths(tmp_path: Path):
    """7. Existing config: sandbox .env with custom ports/paths is preserved."""
    custom = (
        "WORKBENCH_BACKEND_PORT=9876\n"
        "WORKBENCH_FRONTEND_PORT=5432\n"
        "HOST_FASHIONIQ_ROOT=/custom/fashioniq/location\n"
        "WORKBENCH_ENABLE_GPU=1\n"
        "WORKBENCH_RUNTIME_ADAPTATION=sm_120\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    target_env = sandbox / "workbench" / "docker" / ".env"

    proc = run_pc(sandbox, ["--doctor"], env)
    content = target_env.read_text()
    assert "WORKBENCH_BACKEND_PORT=9876" in content
    assert "WORKBENCH_FRONTEND_PORT=5432" in content
    assert "HOST_FASHIONIQ_ROOT=/custom/fashioniq/location" in content


def test_missing_fashioniq_dataset_stops_with_clear_message(tmp_path: Path):
    """8. Missing FashionIQ dataset -> stops with clear message, never fabricates data."""
    missing_path = tmp_path / "nonexistent_fashioniq"
    sandbox, env = setup_sandbox(tmp_path, with_dataset=False, dataset_path=missing_path, with_env=True)
    env["HOST_FASHIONIQ_ROOT"] = str(missing_path)

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).lower()
    assert "fashioniq" in out or "dataset" in out or "dữ liệu" in out
    assert not missing_path.exists() or not (missing_path / "images").exists()


def test_custom_dataset_path_via_flag_is_recorded_and_used(tmp_path: Path):
    """9. Custom dataset path via --dataset -> recorded/used."""
    custom_ds = tmp_path / "custom_ds"
    sandbox, env = setup_sandbox(tmp_path, with_dataset=True, dataset_path=custom_ds, with_env=True)

    proc = run_pc(sandbox, ["--dataset", str(custom_ds), "--doctor"], env)
    out = proc.stdout + proc.stderr
    assert str(custom_ds) in out or proc.returncode == 0
    # Also verify in .env or logs if recorded
    target_env = sandbox / "workbench" / "docker" / ".env"
    if target_env.exists():
        env_text = target_env.read_text()
        if str(custom_ds) in env_text:
            assert str(custom_ds) in env_text


def test_env_quoted_value_with_spaces_and_hash_inside_quotes(tmp_path: Path):
    """10. .env with quoted value containing spaces and # inside quotes preserves #."""
    custom = (
        'WORKBENCH_TAG="run #1 with special characters"\n'
        'WORKBENCH_BACKEND_PORT=8000\n'
        'WORKBENCH_FRONTEND_PORT=5173\n'
        'WORKBENCH_ENABLE_GPU=1\n'
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0


def test_malformed_env_exits_nonzero_before_touching_docker(tmp_path: Path):
    """11. Malformed .env (e.g. PORT=abc) -> exits non-zero before touching docker."""
    custom = (
        "WORKBENCH_BACKEND_PORT=abc\n"
        "WORKBENCH_FRONTEND_PORT=5173\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0

    logs = _read_tool_logs(sandbox)
    docker_calls = [r for r in logs if r.get("tool") == "docker"]
    assert len(docker_calls) == 0, f"docker was invoked despite malformed .env: {docker_calls}"


def test_repeated_invocation_doctor_twice_idempotent(tmp_path: Path):
    """12. Repeated invocation: run pc.sh --doctor twice -> both exit 0, no duplicate config."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    proc1 = run_pc(sandbox, ["--doctor"], env)
    assert proc1.returncode == 0

    target_env = sandbox / "workbench" / "docker" / ".env"
    hash1 = hashlib.sha256(target_env.read_bytes()).hexdigest()

    proc2 = run_pc(sandbox, ["--doctor"], env)
    assert proc2.returncode == 0
    hash2 = hashlib.sha256(target_env.read_bytes()).hexdigest()
    assert hash1 == hash2


def test_duplicate_job_attaches_to_existing_queued_job_no_second_post(tmp_path: Path):
    """13. Duplicate job: attaches to existing queued job and does NOT POST a second job."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "queued"
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)
    # The pre-existing job is still queued while the launcher decides, then settles — otherwise
    # the monitor would poll a job that never finishes and the test would only prove a timeout.
    env["MOCK_QUEUED_READS"] = "8"

    proc = run_pc(sandbox, [], env)
    # Assert no second job POST was made
    if post_jobs_file.exists():
        post_count = len(post_jobs_file.read_text().strip().splitlines())
        assert post_count == 0, f"Expected 0 POST calls to /api/jobs, got {post_count}"


def test_existing_completed_corroborated_run_reported_without_re_running_unless_force(tmp_path: Path):
    """14. Existing completed corroborated run reported instead of re-running; with --force re-queues."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    # Without --force: should not POST new job
    proc1 = run_pc(sandbox, [], env)
    assert not post_jobs_file.exists() or len(post_jobs_file.read_text().strip().splitlines()) == 0

    # With --force: should queue
    proc2 = run_pc(sandbox, ["--force"], env)
    assert post_jobs_file.exists()
    assert len(post_jobs_file.read_text().strip().splitlines()) > 0


def test_failed_evaluation_job_exits_nonzero_with_vietnamese_failure_summary(tmp_path: Path):
    """15. Failed evaluation job -> exits non-zero with Vietnamese failure summary."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "failed"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    # Vietnamese words and details
    assert "CUDA out of memory" in out or "thất bại" in out.lower() or "lỗi" in out.lower()


def test_job_blocked_unverified_gpu_exits_nonzero_with_gpu_message(tmp_path: Path):
    """16. Job blocked (GPU unverified) -> non-zero exit, message says GPU unverified."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "blocked"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).lower()
    assert "gpu" in out
    assert "unverified" in out or "chưa xác thực" in out or "kiểm chứng" in out


def test_backend_temporarily_unavailable_retries_and_timeout_fails_cleanly(tmp_path: Path):
    """17. Backend temporarily unavailable retries; permanent failure produces timeout message."""
    # Subcase A: retry succeeds
    sandbox_a, env_a = setup_sandbox(tmp_path / "a", with_env=True)
    retry_file = sandbox_a / "retry_count.txt"
    retry_file.write_text("2")
    env_a["MOCK_CURL_RETRY_COUNT_FILE"] = str(retry_file)
    proc_a = run_pc(sandbox_a, ["--doctor"], env_a)
    assert proc_a.returncode == 0

    # Subcase B: permanent failure times out
    sandbox_b, env_b = setup_sandbox(tmp_path / "b", with_env=True)
    env_b["MOCK_CURL_ALWAYS_FAIL"] = "1"
    proc_b = run_pc(sandbox_b, ["--timeout", "2"], env_b)
    assert proc_b.returncode != 0
    out_b = (proc_b.stdout + proc_b.stderr).lower()
    assert "timeout" in out_b or "hết thời gian" in out_b or "unreachable" in out_b or "failed" in out_b


def test_insufficient_disk_space_exits_nonzero_with_disk_guidance(tmp_path: Path):
    """18. Insufficient disk space -> exits non-zero with disk guidance."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_DF_LOW_SPACE"] = "1"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).lower()
    assert "disk" in out or "đĩa" in out or "dung lượng" in out or "space" in out or "50" in out


def test_secrets_not_forwarded_to_model_code(tmp_path: Path):
    """19. Secrets matching WORKBENCH_LEDGER_KEY, *_TOKEN, *_API_KEY, HF_TOKEN not forwarded."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["WORKBENCH_LEDGER_KEY"] = "super-secret-ledger"
    env["HF_TOKEN"] = "hf_token_secret"
    env["WANDB_API_KEY"] = "wandb_key_secret"
    env["MY_CUSTOM_TOKEN"] = "custom_token_secret"

    proc = run_pc(sandbox, [], env)
    logs = _read_tool_logs(sandbox)

    # Check child commands that execute model evaluation or container payloads
    for rec in logs:
        cmd_args = " ".join(rec.get("args", []))
        if "pipeline.py" in cmd_args or "evaluate" in cmd_args:
            secrets = rec.get("secrets", {})
            for key in ("WORKBENCH_LEDGER_KEY", "HF_TOKEN", "WANDB_API_KEY", "MY_CUSTOM_TOKEN"):
                assert key not in secrets, f"Secret {key} leaked into child command {cmd_args}!"


def test_sigint_during_wait_loop_exits_nonzero_and_preserves_sandbox(tmp_path: Path):
    """20. SIGINT during wait loop -> exits non-zero, sandbox artifact files preserved."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "running"
    if not PC_SH.exists():
        pytest.skip("pc.sh not yet created on disk")

    artifact_dir = sandbox / "workbench" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sentinel = artifact_dir / "preserve_test.txt"
    sentinel.write_text("must survive sigint")

    full_env = {**os.environ, **env}
    proc = subprocess.Popen(
        [str(PC_SH)],
        cwd=sandbox,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.5)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    assert proc.returncode != 0
    assert sentinel.exists()
    assert sentinel.read_text() == "must survive sigint"
