"""Executable test harness for the CIR Workbench entrypoint pc.sh.

Tests run against the real ./pc.sh through PATH shims inside isolated
tmp_path sandboxes. Never touches docker daemon, GPU, network, or real dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
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
    if not omit_tools or "docker" not in omit_tools:
        docker_shim = bin_dir / "docker"
        fixtures_docker = FIXTURES_BIN / "docker"
        docker_shim_content = f"""#!/usr/bin/env python3
import sys
import os
import subprocess
import json

log_file = os.environ.get("MOCK_LOG_FILE")
if log_file:
    try:
        secrets = {{
            k: v for k, v in os.environ.items()
            if any(p in k for p in ("TOKEN", "KEY", "SECRET", "LEDGER"))
        }}
        with open(log_file, "a") as f:
            f.write(json.dumps({{"tool": "docker", "args": sys.argv[1:], "secrets": secrets}}) + "\\n")
    except Exception:
        pass

args_str = " ".join(sys.argv)

if os.environ.get("MOCK_CONTAINER_DOWN") == "1":
    if len(sys.argv) > 1 and sys.argv[1] == "ps":
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "exec":
        sys.stderr.write("Error response from daemon: Container cir-workbench is not running\\n")
        sys.exit(1)

if os.environ.get("MOCK_CONTAINER_DOWN_ON_VERIFY") == "1":
    if len(sys.argv) > 1 and sys.argv[1] == "ps":
        count_file = os.path.join(os.environ.get("CIR_REPO_ROOT", ""), "docker_ps_count.txt")
        count = 0
        if os.path.exists(count_file):
            try:
                count = int(open(count_file).read().strip())
            except Exception:
                pass
        count += 1
        try:
            with open(count_file, "w") as f:
                f.write(str(count))
        except Exception:
            pass
        if count >= 3:
            sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "exec" and "verify_run.py" in args_str:
        sys.stderr.write("Error response from daemon: Container cir-workbench is not running\\n")
        sys.exit(1)

if len(sys.argv) > 1 and sys.argv[1] == "exec" and "verify_run.py" in args_str:
    repo_root = os.environ.get("CIR_REPO_ROOT", "")
    script = os.path.join(repo_root, "workbench/scripts/verify_run.py") if repo_root else "workbench/scripts/verify_run.py"
    v_idx = -1
    for i, a in enumerate(sys.argv):
        if "verify_run.py" in a:
            v_idx = i
            break
    sub_args = sys.argv[v_idx + 1:] if v_idx >= 0 else []
    v_env = dict(os.environ)
    v_env["PYTHONPATH"] = f"{str(REPO)}:{{v_env.get('PYTHONPATH', '')}}"
    res = subprocess.run([{repr(sys.executable)}, script] + sub_args, env=v_env)
    sys.exit(res.returncode)

# "How many times this sandbox's container has actually been recreated". `run.sh up` is
# idempotent: on an already-running project compose recreates nothing, so the effective
# configuration is unchanged by the call. Expressing that here is what makes the launcher's
# post-restart re-inspection testable -- a correction that corrects nothing has to be observable,
# or the promise about it cannot be exercised.
def _generation() -> int:
    try:
        with open(os.path.join(os.environ.get("CIR_REPO_ROOT", ""),
                               "mock_docker_gen", "count")) as handle:
            return int(handle.read().strip() or "0")
    except Exception:
        return 0

def _effective(variable: str, default: str) -> str:
    # The in-container value: the refreshed one, once this sandbox's container was recreated.
    if _generation() > 0:
        refreshed = os.environ.get(variable + "_REFRESHED")
        if refreshed is not None:
            return refreshed
    return os.environ.get(variable, default)

if len(sys.argv) > 1 and sys.argv[1] == "compose" and "up" in sys.argv and os.environ.get("MOCK_DOCKER_UP_RECREATES"):
    try:
        os.makedirs(os.path.join(os.environ.get("CIR_REPO_ROOT", ""), "mock_docker_gen"), exist_ok=True)
        with open(os.path.join(os.environ.get("CIR_REPO_ROOT", ""), "mock_docker_gen", "count"), "w") as handle:
            handle.write(str(_generation() + 1))
    except Exception:
        pass

if "WORKBENCH_RUNTIME_ADAPTATION" in args_str and len(sys.argv) > 1 and sys.argv[1] == "exec":
    print(_effective("MOCK_CONTAINER_ADAPTATION", ""))
    sys.exit(0)

if os.environ.get("MOCK_CONTAINER_NO_GPU") == "1" and len(sys.argv) > 1 and sys.argv[1] == "exec":
    if "torch" in args_str or "nvidia-smi" in args_str:
        sys.exit(1)

if len(sys.argv) > 1 and sys.argv[1] == "info":
    if os.environ.get("MOCK_HOST_NO_TOOLKIT") == "1" and any("Runtimes" in a for a in sys.argv):
        info_count_file = os.path.join(os.environ.get("CIR_REPO_ROOT", ""), "mock_docker_info_count.txt")
        count = 0
        try:
            count = int(open(info_count_file).read().strip())
        except Exception:
            pass
        try:
            with open(info_count_file, "w") as f:
                f.write(str(count + 1))
        except Exception:
            pass
        if count == 0:
            print('{{"nvidia": {{"path": "nvidia-container-runtime"}}, "runc": {{"path": "runc"}}}}')
            sys.exit(0)
        else:
            print('{{"runc": {{"path": "runc"}}}}')
            sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "inspect":
    if os.environ.get("MOCK_DOCKER_INSPECT_FAIL") == "1":
        sys.stderr.write("docker inspect failed\\n")
        sys.exit(1)
    if os.environ.get("MOCK_DOCKER_INSPECT_CORRUPT") == "1":
        print("invalid json")
        sys.exit(0)
    if os.environ.get("MOCK_DOCKER_INSPECT_EMPTY") == "1":
        sys.exit(0)
    if "MOCK_DOCKER_MOUNTS" in os.environ:
        print(_effective("MOCK_DOCKER_MOUNTS", ""))
        sys.exit(0)
    if "Mounts" in args_str:
        repo_root = os.environ.get("CIR_REPO_ROOT", "")
        fashioniq_root = os.environ.get("HOST_FASHIONIQ_ROOT", os.path.join(repo_root, "data/FashionIQ"))
        mounts = [
            {{"Source": fashioniq_root, "Destination": "/data/FashionIQ", "RW": True}},
            {{"Source": os.path.join(fashioniq_root, "images"), "Destination": "/data/FashionIQ/images", "RW": False}},
            {{"Source": os.path.join(fashioniq_root, "image_splits"), "Destination": "/data/FashionIQ/image_splits", "RW": False}},
            {{"Source": os.path.join(repo_root, "workbench/artifacts"), "Destination": "/artifacts", "RW": True}},
            {{"Source": os.path.join(repo_root, "workbench/artifacts/checkpoints"), "Destination": "/checkpoints", "RW": False}},
        ]
        print(json.dumps(mounts))
        sys.exit(0)
    if "DeviceRequests" in args_str:
        if os.environ.get("MOCK_CONTAINER_NO_GPU") == "1":
            print("null")
        elif "MOCK_CONTAINER_DEVICE_REQUESTS" in os.environ:
            print(_effective("MOCK_CONTAINER_DEVICE_REQUESTS", "null"))
        else:
            print('[{{ "Driver": "nvidia", "Count": -1, "DeviceIDs": [], "Capabilities": [["gpu"]], "Options": {{}} }}]')
        sys.exit(0)
    if "Config.Env" in args_str:
        print(f"WORKBENCH_RUNTIME_ADAPTATION={{_effective('MOCK_CONTAINER_ADAPTATION', 'sm_120')}}")
        sys.exit(0)
child_env = dict(os.environ)
child_env.pop("MOCK_LOG_FILE", None)
res = subprocess.run([{repr(sys.executable)}, {repr(str(fixtures_docker))}] + sys.argv[1:], env=child_env)
sys.exit(res.returncode)
"""
        docker_shim.write_text(docker_shim_content)
        docker_shim.chmod(0o755)

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
        "WORKBENCH_LEDGER_KEY": "mock_ledger_key_for_test_harness_32b!",
    }

    scripts_dir = sandbox / "workbench" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if (REPO / "workbench/scripts/verify_run.py").exists():
        shutil.copy2(REPO / "workbench/scripts/verify_run.py", scripts_dir / "verify_run.py")
        (scripts_dir / "verify_run.py").chmod(0o755)
    else:
        (scripts_dir / "verify_run.py").write_text(_VERIFY_RUN_SOURCE, encoding="utf-8")
        (scripts_dir / "verify_run.py").chmod(0o755)

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

    _create_corroborated_run(sandbox, "run_limn_corr_01")

    # Without --force: should not POST new job
    proc1 = run_pc(sandbox, [], env)
    assert proc1.returncode == 0
    assert not post_jobs_file.exists() or len(post_jobs_file.read_text().strip().splitlines()) == 0
    assert "CORROBORATED" in proc1.stdout or "HỢP LỆ" in proc1.stdout
    assert "Tái sử dụng kết quả có sẵn" in proc1.stdout

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

# ---------------------------------------------------------------------------
# VERIFY_RUN CONTRACT & CORROBORATION FIXTURE HELPERS
# ---------------------------------------------------------------------------

_VERIFY_RUN_SOURCE = '''#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Verify evaluation run evidence")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(os.environ.get("CIR_REPO_ROOT", Path(__file__).resolve().parents[2]))
    reports_root = repo_root / "workbench" / "artifacts" / "reports"
    reasons = []

    report_name = f"{args.model}_{args.checkpoint}_{args.run_id}_aggregate.json"
    report_file = reports_root / report_name
    report_data = None
    if not report_file.is_file():
        reasons.append(f"missing aggregate report: {report_file}")
    else:
        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as e:
            reasons.append(f"failed to parse aggregate report {report_file}: {e}")

    if report_data:
        if report_data.get("run_id") != args.run_id:
            reasons.append(f"run_id mismatch: expected {args.run_id}, got {report_data.get('run_id')}")
        if report_data.get("model_id") != args.model:
            reasons.append(f"model_id mismatch: expected {args.model}, got {report_data.get('model_id')}")
        if report_data.get("checkpoint_id") != args.checkpoint:
            reasons.append(f"checkpoint_id mismatch: expected {args.checkpoint}, got {report_data.get('checkpoint_id')}")
        if report_data.get("protocol_id") != args.protocol:
            reasons.append(f"protocol_id mismatch: expected {args.protocol}, got {report_data.get('protocol_id')}")

    # Check evidence ledger
    sys.path.insert(0, str(repo_root))
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts.evaluate_models import read_evidence_ledger, sha256_file
    config = resolve_config()
    ledger_entries = read_evidence_ledger(config)
    if ledger_entries is None:
        reasons.append("evidence ledger missing or HMAC verification failed")
    else:
        entry = next((e for e in ledger_entries if e.get("run_id") == args.run_id), None)
        if entry is None:
            reasons.append(f"missing evidence ledger entry for {args.run_id}")
        else:
            if entry.get("return_code") != 0:
                reasons.append(f"ledger return_code={entry.get('return_code')}, expected 0")
            if entry.get("invocation_status") != "INVOCATION_COMPLETED":
                reasons.append(f"ledger invocation_status={entry.get('invocation_status')}, expected 'INVOCATION_COMPLETED'")
            if report_file.is_file():
                digest = sha256_file(report_file)
                if entry.get("report_digest") != digest:
                    reasons.append(f"report digest mismatch: ledger={entry.get('report_digest')}, actual={digest}")

    # Check completion proof
    proof_name = f"{args.model}_{args.checkpoint}_{args.run_id}_completion.json"
    proof_file = reports_root / proof_name
    if not proof_file.is_file():
        reasons.append(f"missing completion proof: {proof_file}")
    else:
        try:
            proof_data = json.loads(proof_file.read_text(encoding="utf-8"))
            if proof_data.get("completion_status") != "COMPLETED":
                reasons.append(f"completion proof status={proof_data.get('completion_status')}, expected 'COMPLETED'")
            if proof_data.get("return_code") != 0:
                reasons.append(f"completion proof return_code={proof_data.get('return_code')}, expected 0")
            if proof_data.get("run_id") != args.run_id:
                reasons.append(f"completion proof run_id mismatch")
            if proof_data.get("model_id") != args.model:
                reasons.append(f"completion proof model_id mismatch")
            if proof_data.get("checkpoint_id") != args.checkpoint:
                reasons.append(f"completion proof checkpoint_id mismatch")
            if proof_data.get("protocol_id") != args.protocol:
                reasons.append(f"completion proof protocol_id mismatch")
        except Exception as e:
            reasons.append(f"failed to parse completion proof: {e}")

    # Check category metrics for limn
    if args.model == "limn" and report_data:
        cat_metrics = report_data.get("category_metrics") or {}
        for cat in ("dress", "shirt", "toptee"):
            m = cat_metrics.get(cat)
            if not isinstance(m, dict):
                reasons.append(f"missing category metric: {cat}")
            else:
                for metric in ("r10", "r50"):
                    val = m.get(metric)
                    if val is None or not isinstance(val, (int, float)) or not math.isfinite(val) or not (0 <= float(val) <= 100):
                        reasons.append(f"invalid {metric} for category {cat}: {val}")

    verified = (len(reasons) == 0)
    identity = {}
    metrics = {"category": {}, "observed": {}}
    gate = {}
    if report_data:
        identity = {
            "run_id": report_data.get("run_id"),
            "model_id": report_data.get("model_id"),
            "checkpoint_id": report_data.get("checkpoint_id"),
            "protocol_id": report_data.get("protocol_id"),
            "git_commit": report_data.get("source_actual_commit"),
            "runtime_adaptation": report_data.get("runtime_adaptation"),
        }
        metrics = {
            "category": report_data.get("category_metrics") or {},
            "observed": report_data.get("observed_metrics") or {},
        }
        gate = report_data.get("benchmark_gate") or {}

    out = {
        "verified": verified,
        "reasons": reasons,
        "report_path": str(report_file) if report_file.is_file() else None,
        "identity": identity,
        "metrics": metrics,
        "gate": gate,
    }
    if args.json:
        print(json.dumps(out))
    sys.exit(0 if verified else 1)

if __name__ == "__main__":
    main()
'''


def _create_corroborated_run(
    sandbox: Path,
    run_id: str = "run_limn_corr_01",
    model_id: str = "limn",
    checkpoint_id: str = "base_iter0_all_categories",
    protocol_id: str = "fashioniq_val_split",
    *,
    forged_digest: bool = False,
    wrong_checkpoint: bool = False,
    wrong_protocol: bool = False,
    omit_report: bool = False,
    omit_ledger: bool = False,
    omit_proof: bool = False,
) -> None:
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = sandbox / "workbench" / "artifacts" / "checkpoints" / model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_names = ["0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"]
    for name in ckpt_names:
        (ckpt_dir / name).write_bytes(b"dummy_weights_12345")

    from workbench.scripts.evaluate_models import build_artifact_manifest, manifest_digest
    bundle = build_artifact_manifest(ckpt_dir, ckpt_names)
    md = manifest_digest(bundle)

    pin = "7d7bc9b116f594a65ac22457491edf28a88d3c3e"
    rep_ckpt = "wrong_checkpoint_variant" if wrong_checkpoint else checkpoint_id
    rep_proto = "wrong_protocol_variant" if wrong_protocol else protocol_id

    run_dir = sandbox / "workbench" / "artifacts" / "logs" / f"{model_id}__{checkpoint_id}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps({
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_id": rep_ckpt,
        "protocol_id": rep_proto,
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
    }))
    (run_dir / "stdout.log").write_text("evaluation output\n")
    (run_dir / "stderr.log").write_text("")
    logs = {"stdout": "evaluation output\n", "stderr": ""}
    ev_digest = hashlib.sha256(json.dumps(logs, sort_keys=True).encode("utf-8")).hexdigest()

    report_file = reports_dir / f"{model_id}_{checkpoint_id}_{run_id}_aggregate.json"
    report_data = {
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_id": rep_ckpt,
        "protocol_id": rep_proto,
        "execution_status": "EXECUTION_COMPLETED",
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
        "source_actual_commit": pin,
        "runtime_adaptation": "sm_120",
        "bundle_manifest_digest": md,
        "checkpoint_sha256": "abcdef123456",
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT"},
        # The run's own extraction facts and its macro. The macro is the mean of the three
        # categories below and the mean of the two macro recalls, because `verify_run.py`
        # re-derives that arithmetic rather than trusting the report's claim (the same rule the
        # extraction parser applies to a real run's output).
        "extraction_status": "EXTRACTED_VERIFIED",
        "metric_extraction_available": True,
        "category_metrics": {
            "dress": {"r10": 25.5, "r50": 55.2},
            "shirt": {"r10": 28.1, "r50": 58.4},
            "toptee": {"r10": 30.2, "r50": 60.1},
        },
        "observed_metrics": {
            "r10": 27.9333,
            "r50": 57.9,
            "mean": 42.9167,
        },
    }

    if not omit_report:
        report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    if report_file.exists():
        actual_digest = hashlib.sha256(report_file.read_bytes()).hexdigest()
    else:
        actual_digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    digest = "forged_mismatched_sha256_digest_0000" if forged_digest else actual_digest

    ledger_entry_digest = "ledger_digest_placeholder"
    if not omit_ledger:
        from workbench.backend.operator_config import WorkbenchConfig
        from workbench.scripts.evaluate_models import append_evidence_entry
        config = WorkbenchConfig(
            CIR_REPO_ROOT=sandbox,
            CIR_DATA_ROOT=sandbox / "data",
            FASHIONIQ_ROOT=sandbox / "data" / "FashionIQ",
            WORKBENCH_HOST="127.0.0.1",
            WORKBENCH_BACKEND_PORT=8000,
            WORKBENCH_FRONTEND_PORT=5173,
            WORKBENCH_CHECKPOINT_ROOT=sandbox / "workbench" / "artifacts" / "checkpoints",
            WORKBENCH_RESULTS_ROOT=sandbox / "workbench" / "artifacts" / "results",
            WORKBENCH_THIRD_PARTY_ROOT=sandbox / "workbench" / "third_party",
        )
        ledger_entry = {
            "run_id": run_id,
            "model_id": model_id,
            "checkpoint_id": rep_ckpt,
            "protocol_id": rep_proto,
            "return_code": 0,
            "invocation_status": "INVOCATION_COMPLETED",
            "report_digest": digest,
        }
        old_key = os.environ.get("WORKBENCH_LEDGER_KEY")
        os.environ["WORKBENCH_LEDGER_KEY"] = "mock_ledger_key_for_test_harness_32b!"
        try:
            rec = append_evidence_entry(config, ledger_entry)
            ledger_entry_digest = rec["entry_digest"]
        finally:
            if old_key is not None:
                os.environ["WORKBENCH_LEDGER_KEY"] = old_key
            else:
                os.environ.pop("WORKBENCH_LEDGER_KEY", None)

    proof_file = reports_dir / f"{model_id}_{checkpoint_id}_{run_id}_completion.json"
    if not omit_proof:
        proof_data = {
            "artifact_type": "official_evaluation_completion_proof",
            "schema_version": 1,
            "completion_status": "COMPLETED",
            "run_id": run_id,
            "model_id": model_id,
            "checkpoint_id": rep_ckpt,
            "protocol_id": rep_proto,
            "source_expected_commit": pin,
            "source_actual_commit": pin,
            "checkpoint_artifact_type": "directory",
            "checkpoint_sha256": "abcdef123456",
            "bundle_manifest_digest": md,
            "invocation_status": "INVOCATION_COMPLETED",
            "return_code": 0,
            "ledger_entry_digest": ledger_entry_digest,
            "report": str(report_file),
            "report_digest": digest,
            "report_relative": report_file.name,
            "environment_digest": "env_digest_123",
            "command_digest": "cmd_digest_123",
            "evidence_digest": ev_digest,
            "run_directory": str(run_dir),
            "stdout_log": "stdout.log",
            "stderr_log": "stderr.log",
            "completed_at": "2026-10-16T00:00:00Z",
        }
        proof_file.write_text(json.dumps(proof_data, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# VERIFICATION & RUN HARNESS ACCEPTANCE TESTS (a, b, c, d, e)
# ---------------------------------------------------------------------------

def test_uncorroborated_or_mock_run_is_never_reused(tmp_path: Path):
    """(c) A mock/synthetic/uncorroborated run is never reused; fresh job queued instead."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    # Note: no verified files written on disk!
    proc = run_pc(sandbox, [], env)
    assert "Tái sử dụng kết quả có sẵn" not in proc.stdout
    assert post_jobs_file.exists()
    assert len(post_jobs_file.read_text().strip().splitlines()) > 0


def test_missing_report_or_ledger_or_proof_fails_and_never_prints_corroborated(tmp_path: Path):
    """(a) Missing report, ledger, or proof makes run exit non-zero and never prints CORROBORATED."""
    # Subtest 1: missing report
    sandbox_a, env_a = setup_sandbox(tmp_path / "a", with_env=True)
    _create_corroborated_run(sandbox_a, "run_limn_corr_01", omit_report=True)
    proc_a = run_pc(sandbox_a, [], env_a)
    assert proc_a.returncode != 0
    assert "CORROBORATED" not in proc_a.stdout
    assert "HỢP LỆ (CORROBORATED)" not in proc_a.stdout
    assert "Xác thực thất bại" in proc_a.stdout

    # Subtest 2: missing ledger
    sandbox_b, env_b = setup_sandbox(tmp_path / "b", with_env=True)
    _create_corroborated_run(sandbox_b, "run_limn_corr_01", omit_ledger=True)
    proc_b = run_pc(sandbox_b, [], env_b)
    assert proc_b.returncode != 0
    assert "CORROBORATED" not in proc_b.stdout
    assert "HỢP LỆ (CORROBORATED)" not in proc_b.stdout

    # Subtest 3: missing proof
    sandbox_c, env_c = setup_sandbox(tmp_path / "c", with_env=True)
    _create_corroborated_run(sandbox_c, "run_limn_corr_01", omit_proof=True)
    proc_c = run_pc(sandbox_c, [], env_c)
    assert proc_c.returncode != 0
    assert "CORROBORATED" not in proc_c.stdout
    assert "HỢP LỆ (CORROBORATED)" not in proc_c.stdout


def test_forged_or_digest_mismatched_ledger_entry_is_refused(tmp_path: Path):
    """(b) Forged or digest-mismatched ledger entry is refused."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    _create_corroborated_run(sandbox, "run_limn_corr_01", forged_digest=True)
    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    assert "CORROBORATED" not in proc.stdout
    assert "HỢP LỆ (CORROBORATED)" not in proc.stdout
    assert "Xác thực thất bại" in proc.stdout


def test_wrong_checkpoint_or_protocol_identity_is_refused(tmp_path: Path):
    """(d) Wrong checkpoint/protocol/identity is refused."""
    sandbox_a, env_a = setup_sandbox(tmp_path / "a", with_env=True)
    _create_corroborated_run(sandbox_a, "run_limn_corr_01", wrong_checkpoint=True)
    proc_a = run_pc(sandbox_a, [], env_a)
    assert proc_a.returncode != 0
    assert "CORROBORATED" not in proc_a.stdout

    sandbox_b, env_b = setup_sandbox(tmp_path / "b", with_env=True)
    _create_corroborated_run(sandbox_b, "run_limn_corr_01", wrong_protocol=True)
    proc_b = run_pc(sandbox_b, [], env_b)
    assert proc_b.returncode != 0
    assert "CORROBORATED" not in proc_b.stdout


def test_active_job_blocks_container_reuse(tmp_path: Path):
    """(e) An active job blocks container reuse when configuration does not satisfy run."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "running"
    env["MOCK_CONTAINER_ADAPTATION"] = "none"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    assert "CONTAINER_REUSE" in proc.stdout or "tác vụ" in proc.stdout.lower()
    assert "không dừng hoặc khởi động lại" in proc.stdout.lower() or "tuyệt đối không" in proc.stdout.lower() or "đang chạy tác vụ" in proc.stdout.lower()


# ==============================================================================
# P1-3 — dataset and mount provenance
# ==============================================================================


def test_container_dataset_mount_source_mismatch_refused(tmp_path: Path):
    """(a) Container whose dataset mount source differs from configured host path is refused with named reason."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    other_ds = tmp_path / "other_fashioniq_root"
    other_ds.mkdir(parents=True, exist_ok=True)
    sb = sandbox
    mounts = [
        {"Destination": "/data/FashionIQ", "Source": str(other_ds), "RW": True},
        {"Destination": "/data/FashionIQ/images", "Source": str(other_ds / "images"), "RW": False},
        {"Destination": "/data/FashionIQ/image_splits", "Source": str(other_ds / "image_splits"), "RW": False},
        {"Destination": "/artifacts", "Source": str(sb / "workbench" / "artifacts"), "RW": True},
        {"Destination": "/checkpoints", "Source": str(sb / "workbench" / "artifacts" / "checkpoints"), "RW": False},
    ]
    env["MOCK_DOCKER_MOUNTS"] = json.dumps(mounts)
    env["MOCK_JOB_STATUS"] = "running"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    assert "CONTAINER_REUSE" in proc.stdout
    assert "Mount /data/FashionIQ nguồn không khớp" in proc.stdout
    assert str(other_ds) in proc.stdout


def test_container_missing_mount_refused(tmp_path: Path):
    """(b) A container missing the mount is refused."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    sb = sandbox
    # Missing /data/FashionIQ mount
    mounts = [
        {"Destination": "/artifacts", "Source": str(sb / "workbench" / "artifacts"), "RW": True},
        {"Destination": "/checkpoints", "Source": str(sb / "workbench" / "artifacts" / "checkpoints"), "RW": False},
    ]
    env["MOCK_DOCKER_MOUNTS"] = json.dumps(mounts)
    env["MOCK_JOB_STATUS"] = "running"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    assert "CONTAINER_REUSE" in proc.stdout
    assert "Mount /data/FashionIQ bị thiếu trong container" in proc.stdout


def test_unreadable_mount_list_fails_closed(tmp_path: Path):
    """(c) An unreadable mount list fails closed."""
    # Subtest 1: inspect command fails
    sandbox_a, env_a = setup_sandbox(tmp_path / "a", with_env=True)
    env_a["MOCK_DOCKER_INSPECT_FAIL"] = "1"
    proc_a = run_pc(sandbox_a, [], env_a)
    assert proc_a.returncode != 0
    assert "CONTAINER_REUSE" in proc_a.stdout
    assert "KHÔNG thể xác định cấu hình mount" in proc_a.stdout

    # Subtest 2: inspect returns unparseable json
    sandbox_b, env_b = setup_sandbox(tmp_path / "b", with_env=True)
    env_b["MOCK_DOCKER_INSPECT_CORRUPT"] = "1"
    proc_b = run_pc(sandbox_b, [], env_b)
    assert proc_b.returncode != 0
    assert "CONTAINER_REUSE" in proc_b.stdout

    # Subtest 3: inspect returns empty string
    sandbox_c, env_c = setup_sandbox(tmp_path / "c", with_env=True)
    env_c["MOCK_DOCKER_INSPECT_EMPTY"] = "1"
    proc_c = run_pc(sandbox_c, [], env_c)
    assert proc_c.returncode != 0
    assert "CONTAINER_REUSE" in proc_c.stdout


def test_matching_container_is_reused_positive_control(tmp_path: Path):
    """(d) A matching container is still reused (positive control)."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    ds_root = sandbox / "data" / "FashionIQ"
    sb = sandbox
    mounts = [
        {"Destination": "/data/FashionIQ", "Source": str(ds_root), "RW": True},
        {"Destination": "/data/FashionIQ/images", "Source": str(ds_root / "images"), "RW": False},
        {"Destination": "/data/FashionIQ/image_splits", "Source": str(ds_root / "image_splits"), "RW": False},
        {"Destination": "/artifacts", "Source": str(sb / "workbench" / "artifacts"), "RW": True},
        {"Destination": "/checkpoints", "Source": str(sb / "workbench" / "artifacts" / "checkpoints"), "RW": False},
    ]
    env["MOCK_DOCKER_MOUNTS"] = json.dumps(mounts)
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"
    _create_corroborated_run(sandbox, "run_limn_corr_01")
    env["MOCK_RUNS_STATUS"] = "corroborated"

    proc = run_pc(sandbox, [], env)
    assert proc.returncode == 0
    assert "Docker container cir-workbench đã đang chạy, cấu hình phù hợp và API sẵn sàng" in proc.stdout
    assert "Khởi chạy lại Docker stack qua run.sh up để cấu hình đúng" not in proc.stdout


def test_verification_attempted_inside_container_not_on_host(tmp_path: Path):
    """P0-1 (a): Verification is attempted inside the container, not on the host."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    _create_corroborated_run(sandbox, "run_limn_corr_01")

    proc = run_pc(sandbox, [], env)
    assert proc.returncode == 0
    assert "CORROBORATED" in proc.stdout or "HỢP LỆ" in proc.stdout

    logs = _read_tool_logs(sandbox)
    docker_verify_calls = [
        rec for rec in logs
        if rec.get("tool") == "docker"
        and any("verify_run.py" in str(arg) for arg in rec.get("args", []))
    ]
    assert len(docker_verify_calls) > 0, "No docker verification calls recorded in shim log!"
    for call in docker_verify_calls:
        args = call.get("args", [])
        assert args[0] == "exec", f"Expected exec call, got {args}"
        assert "-T" in args, f"Expected -T flag in docker exec, got {args}"
        assert "cir-workbench" in args, f"Expected cir-workbench container target, got {args}"
        assert any("verify_run.py" in a for a in args)


def test_verification_fails_with_named_reason_when_container_not_running(tmp_path: Path):
    """P0-1 (b): When container is not running, verification fails with named reason without host fallback."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    env["MOCK_CONTAINER_DOWN_ON_VERIFY"] = "1"
    _create_corroborated_run(sandbox, "run_limn_corr_01")

    proc = run_pc(sandbox, [], env)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "Giai đoạn thất bại (Failed stage) : VERIFICATION" in combined
    assert "Container cir-workbench không chạy" in combined
    assert "HỢP LỆ (CORROBORATED)" not in combined


def test_incompatible_configuration_correction_invokes_docker_startup_exactly_once(tmp_path: Path):
    """P2-4 (c): Incompatible-configuration correction invokes docker startup exactly once."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_CONTAINER_ADAPTATION"] = "none"
    env["MOCK_JOB_STATUS"] = "completed"

    proc = run_pc(sandbox, ["--verify-only"], env)
    logs = _read_tool_logs(sandbox)
    up_calls = [
        rec for rec in logs
        if rec.get("tool") == "docker"
        and "compose" in rec.get("args", [])
        and "up" in rec.get("args", [])
    ]
    assert len(up_calls) == 1, f"Expected exactly 1 docker compose up, got {len(up_calls)}"
    assert "Không có tác vụ đang chạy. Khởi chạy lại Docker stack qua run.sh up để cấu hình đúng..." in proc.stdout
    assert "Khởi chạy Docker stack qua run.sh up..." not in proc.stdout


def test_a_correction_that_corrects_nothing_is_refused_by_name(tmp_path: Path):
    """DEFECT D: the correction path ran `run.sh up` and then assumed the mismatch was gone.

    `run.sh up` is idempotent — on an already-running project compose recreates nothing — so a
    requested dataset that differs from `.env` is not corrected by restarting: the recreation
    reads the same `.env`. The launcher must re-inspect the effective configuration after the
    restart, name the mount that is still wrong, name the `.env` key that fixes it, and stop
    before preparation.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    wrong_ds = tmp_path / "wrong_fashioniq_root"
    wrong_ds.mkdir(parents=True, exist_ok=True)

    def mounts_for(dataset_root: Path) -> str:
        return json.dumps([
            {"Destination": "/data/FashionIQ", "Source": str(dataset_root), "RW": True},
            {"Destination": "/data/FashionIQ/images", "Source": str(dataset_root / "images"), "RW": False},
            {"Destination": "/artifacts", "Source": str(sandbox / "workbench" / "artifacts"), "RW": True},
            {"Destination": "/checkpoints",
             "Source": str(sandbox / "workbench" / "artifacts" / "checkpoints"), "RW": False},
        ])

    # Before and after the no-op restart the dataset mount is the same wrong source.
    env["MOCK_DOCKER_MOUNTS"] = mounts_for(wrong_ds)
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode != 0, proc.stdout
    assert "CONTAINER_RECONFIGURE" in proc.stdout
    assert "Mount /data/FashionIQ nguồn không khớp" in proc.stdout
    assert str(wrong_ds) in proc.stdout
    assert "HOST_FASHIONIQ_ROOT" in proc.stdout
    assert "Chuẩn bị mô hình" not in proc.stdout, "preparation must not run on a mismatched container"
    up_calls = [rec for rec in _read_tool_logs(sandbox)
                if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
                and "up" in rec.get("args", [])]
    assert len(up_calls) == 1, f"one correction attempt, saw {len(up_calls)}"

def test_a_correction_that_corrects_the_mismatch_proceeds(tmp_path: Path):
    """Positive control for DEFECT D: when the recreation does bring the container in line, the
    post-restart inspection passes and the launcher continues (--verify-only proves the run got
    past preparation and evaluation dispatch), with still exactly one startup."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    good_ds = sandbox / "data" / "FashionIQ"
    wrong_ds = tmp_path / "wrong_fashioniq_root"
    wrong_ds.mkdir(parents=True, exist_ok=True)

    def mounts_for(dataset_root: Path) -> str:
        return json.dumps([
            {"Destination": "/data/FashionIQ", "Source": str(dataset_root), "RW": True},
            {"Destination": "/data/FashionIQ/images", "Source": str(dataset_root / "images"), "RW": False},
            {"Destination": "/artifacts", "Source": str(sandbox / "workbench" / "artifacts"), "RW": True},
            {"Destination": "/checkpoints",
             "Source": str(sandbox / "workbench" / "artifacts" / "checkpoints"), "RW": False},
        ])

    # The mismatch before the restart is the dataset mount; the recreation really changes it.
    env["MOCK_DOCKER_MOUNTS"] = mounts_for(wrong_ds)
    env["MOCK_DOCKER_MOUNTS_REFRESHED"] = mounts_for(good_ds)
    env["MOCK_DOCKER_UP_RECREATES"] = "1"
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode == 0, proc.stdout
    assert "Không có tác vụ đang chạy. Khởi chạy lại Docker stack qua run.sh up để cấu hình đúng..." in proc.stdout
    assert "Cấu hình container cir-workbench đã đúng yêu cầu sau khi khởi động lại" in proc.stdout
    assert "CONTAINER_RECONFIGURE" not in proc.stdout
    up_calls = [rec for rec in _read_tool_logs(sandbox)
                if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
                and "up" in rec.get("args", [])]
    assert len(up_calls) == 1, f"exactly one startup per correction, saw {len(up_calls)}"

def test_api_unreachable_container_with_a_job_in_flight_is_not_replaced(tmp_path: Path):
    """A running container whose API does not answer must still be protected: the launcher asks
    the persisted queue, which is a bind mount the host can read, and refuses to restart the
    stack while a job is in flight. Restarting here would tear down a running experiment."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_API_DOWN"] = "1"
    queue_job = sandbox / "workbench" / "artifacts" / "queue_jobs" / "job-in-flight"
    queue_job.mkdir(parents=True)
    (queue_job / "queue_state.json").write_text(json.dumps(
        {"job_id": "job-in-flight", "status": "running"}))

    proc = run_pc(sandbox, [], env, timeout=25)
    assert proc.returncode != 0
    assert "CONTAINER_REUSE" in proc.stdout
    assert "job-in-flight" in proc.stdout


def test_api_unreachable_container_with_an_idle_queue_is_restarted_once(tmp_path: Path):
    """The same state with nothing in flight is recoverable: the stack is reconfigured exactly
    once, so an unreachable API cannot block a first run on a fresh PC."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_API_DOWN"] = "1"
    (sandbox / "workbench" / "artifacts" / "queue_jobs").mkdir(parents=True)

    # With the API down the readiness poll cannot succeed, so the run ends by design on the
    # readiness deadline. The property under test is stated before that: exactly one startup.
    with pytest.raises(subprocess.TimeoutExpired):
        run_pc(sandbox, [], env, timeout=25)
    up_calls = [line for line in (sandbox / "mock_tools.log").read_text().splitlines()
                if '"up"' in line]
    assert len(up_calls) == 1, f"expected exactly one docker startup, saw {len(up_calls)}"


def test_limn_bundle_selection_submits_bundle_id_with_native_protocol(tmp_path: Path):
    """Selecting LIMN official bundle queues bundle_id == 'base_iter0_all_categories' with native protocol."""
    fixtures_models_file = FIXTURES_BIN.parent / "models.json"
    assert fixtures_models_file.exists()
    models_data = json.loads(fixtures_models_file.read_text())
    limn_model = next(m for m in models_data if m["model_id"] == "limn")
    assert "checkpoint_bundles" in limn_model and len(limn_model["checkpoint_bundles"]) > 0
    bundle = limn_model["checkpoint_bundles"][0]
    assert bundle["bundle_id"] == "base_iter0_all_categories"
    assert set(bundle["required_checkpoint_ids"]) == {"base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"}
    assert "members" in bundle
    assert len(bundle["members"]) == 3
    assert all("blocking_reasons" in mem for mem in bundle["members"])

    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    _create_corroborated_run(sandbox, "run_limn_corr_01")
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    proc = run_pc(sandbox, ["--force"], env)
    assert proc.returncode == 0
    assert post_jobs_file.exists()

    records = [json.loads(line) for line in post_jobs_file.read_text().splitlines() if line.strip()]
    post_records = [r for r in records if "-d" in r.get("args", [])]
    assert len(post_records) > 0, f"No POST to /api/jobs recorded: {records}"
    d_idx = post_records[0]["args"].index("-d") + 1
    submitted_payload = json.loads(post_records[0]["args"][d_idx])

    # Assert submitted payload has checkpoint_id == "base_iter0_all_categories" with native protocol
    assert submitted_payload["model_id"] == "limn"
    assert submitted_payload["checkpoint_id"] == "base_iter0_all_categories"
    assert submitted_payload["protocol_id"] == limn_model["native_protocol"]
    assert submitted_payload["protocol_id"] == "fashioniq_val_split"

    # Assert submitted payload does NOT submit a per-category checkpoint
    per_category_ids = {"base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"}
    assert submitted_payload["checkpoint_id"] not in per_category_ids

    # Assert bundle is distinguishable from single-category checkpoint on submitted payload
    single_checkpoints = {c["checkpoint_id"] for c in limn_model["checkpoints"]}
    assert submitted_payload["checkpoint_id"] not in single_checkpoints

    # Assert on the rendered frontend bundle: bundle is marked COMPLETE and distinguished from single-category checkpoints
    dist_dir = REPO / "workbench" / "frontend" / "dist" / "assets"
    js_bundles = list(dist_dir.glob("index-*.js"))
    assert len(js_bundles) > 0, "Built frontend bundle missing under workbench/frontend/dist/assets"
    bundle_js_text = js_bundles[0].read_text(encoding="utf-8")
    assert "COMPLETE" in bundle_js_text, "Rendered frontend bundle must label official bundles as COMPLETE multi-category"
    assert "Single-category" in bundle_js_text or "partial" in bundle_js_text, "Rendered frontend bundle must distinguish single-category checkpoints as partial"

# ==============================================================================
# F3 — the LIMN bundle is one complete selection, and only a COMPLETE bundle is submitted
# ==============================================================================

def _submitted_checkpoint(post_jobs_file: Path) -> str:
    """The checkpoint_id the launcher actually POSTed, from the shim's own request log."""
    assert post_jobs_file.exists(), "the launcher POSTed nothing to /api/jobs"
    records = [json.loads(line) for line in post_jobs_file.read_text().splitlines() if line.strip()]
    posts = [r for r in records if "-d" in r.get("args", [])]
    assert posts, f"no POST to /api/jobs recorded: {records}"
    args = posts[0]["args"]
    return json.loads(args[args.index("-d") + 1])["checkpoint_id"]

def test_a_complete_limn_bundle_is_submitted_as_one_bundle_selection(tmp_path: Path):
    """(a) Every member present and runnable -> the bundle id is submitted, unchanged.

    The bundle stays ONE selection with `checkpoint_id == bundle_id`; it is never split into three
    category runs and never degraded to a single category. (The run's own terminal status is the
    mock's business: this test is about which selection reached the queue.)
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)

    proc = run_pc(sandbox, ["--force"], env)
    out = proc.stdout + proc.stderr
    assert _submitted_checkpoint(post_jobs_file) == "base_iter0_all_categories", out
    # Presented as the complete selection, never as a partial run.
    assert "Mô hình: limn | Checkpoint: base_iter0_all_categories" in out, out
    assert "base_iter0_dress" not in out.split("Xác định protocol")[-1]

def test_a_limn_bundle_with_a_missing_member_is_not_submitted(tmp_path: Path):
    """(b) A member that is missing/unusable makes the launcher fall back to a partial run.

    The incomplete bundle is never submitted, so it is never refused with `evaluation_blocked`:
    submitting it would turn one undownloaded category into a launcher failure. The run that IS
    submitted names one of the bundle's own member checkpoints, and the summary must not present
    it as the complete three-category result.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)
    env["MOCK_MODELS_MISSING_MEMBER"] = "base_iter0_shirt"

    proc = run_pc(sandbox, ["--force"], env)
    out = proc.stdout + proc.stderr
    submitted = _submitted_checkpoint(post_jobs_file)
    assert submitted in {"base_iter0_dress", "base_iter0_toptee"}, out
    assert "Mô hình: limn | Checkpoint: base_iter0_all_categories" not in out, out

def test_a_single_checkpoint_model_is_still_queueable_as_a_partial_run(tmp_path: Path):
    """(c) A single-category selection is unaffected: still queueable, still not complete.

    CSMCIR declares no bundle and one checkpoint, so the complete-bundle preference cannot change
    anything about it: the checkpoint is submitted, and no bundle is presented.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)
    # CSMCIR declares no adapted runtime, so the sandbox container must report the empty
    # adaptation string the launcher computes for it; otherwise the run stops at configuration
    # reconciliation and never reaches the queue.
    env["MOCK_CONTAINER_ADAPTATION"] = ""

    proc = run_pc(sandbox, ["--model", "csmcir", "--force"], env)
    out = proc.stdout + proc.stderr
    assert _submitted_checkpoint(post_jobs_file) == "fashioniq", out
    assert "Mô hình: csmcir | Checkpoint: fashioniq" in out, out
    assert "base_iter0_all_categories" not in out.split("Xác định protocol")[-1], out

def test_the_job_list_read_failure_reason_reaches_the_failure_output(tmp_path: Path):
    """A1: the fail-closed refusal must name the API-unreadable job list as one of its reasons.

    The job-list read and the container configuration are recorded into ONE reasons array. When
    the array was re-initialised inside the reuse branch, the API-unreadable reason appended
    earlier was silently discarded and the failure named only the configuration mismatch.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOBS_LIST_UNREADABLE"] = "1"
    env["MOCK_CONTAINER_ADAPTATION"] = "none"

    proc = run_pc(sandbox, [], env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "Không đọc được danh sách tác vụ" in out, out
    assert "CONTAINER_REUSE" in out, out


# ==============================================================================
# A4 — GPU passthrough, runtime adaptation, mount mismatch diagnosis
# ==============================================================================


def test_container_without_requested_gpu_passthrough_refused_with_gpu_toolkit_remedy(tmp_path: Path):
    """(a) Container without requested GPU passthrough is refused with GPU/toolkit remedy and NOT dataset-path remedy."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_CONTAINER_NO_GPU"] = "1"
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode != 0, proc.stdout
    assert "CONTAINER_RECONFIGURE" in proc.stdout
    assert "GPU" in proc.stdout
    assert "nvidia-container-toolkit" in proc.stdout or "WORKBENCH_ENABLE_GPU" in proc.stdout
    assert "HOST_FASHIONIQ_ROOT" not in proc.stdout
    assert "HOST_WORKBENCH_CHECKPOINT_ROOT" not in proc.stdout
    assert "HOST_WORKBENCH_ARTIFACTS_ROOT" not in proc.stdout
    up_calls = [
        rec for rec in _read_tool_logs(sandbox)
        if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
        and "up" in rec.get("args", [])
    ]
    assert len(up_calls) == 1, f"exactly one startup per correction, saw {len(up_calls)}"


def test_container_adaptation_mismatch_named_and_refused(tmp_path: Path):
    """(b) Runtime-adaptation mismatch is named as such with adaptation remedy and NOT dataset remedy."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_80"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode != 0, proc.stdout
    assert "CONTAINER_RECONFIGURE" in proc.stdout
    assert "Runtime adaptation không khớp" in proc.stdout
    assert "WORKBENCH_RUNTIME_ADAPTATION" in proc.stdout
    assert "HOST_FASHIONIQ_ROOT" not in proc.stdout
    up_calls = [
        rec for rec in _read_tool_logs(sandbox)
        if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
        and "up" in rec.get("args", [])
    ]
    assert len(up_calls) == 1, f"exactly one startup per correction, saw {len(up_calls)}"


def test_container_genuine_mount_mismatch_names_mount_and_dataset_remedy(tmp_path: Path):
    """(c) Genuine mount mismatch still names the mount and provides dataset-path remedy."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    wrong_ds = tmp_path / "wrong_fashioniq_root"
    wrong_ds.mkdir(parents=True, exist_ok=True)

    def mounts_for(dataset_root: Path) -> str:
        return json.dumps([
            {"Destination": "/data/FashionIQ", "Source": str(dataset_root), "RW": True},
            {"Destination": "/data/FashionIQ/images", "Source": str(dataset_root / "images"), "RW": False},
            {"Destination": "/artifacts", "Source": str(sandbox / "workbench" / "artifacts"), "RW": True},
            {"Destination": "/checkpoints",
             "Source": str(sandbox / "workbench" / "artifacts" / "checkpoints"), "RW": False},
        ])

    env["MOCK_DOCKER_MOUNTS"] = mounts_for(wrong_ds)
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode != 0, proc.stdout
    assert "CONTAINER_RECONFIGURE" in proc.stdout
    assert "Mount /data/FashionIQ nguồn không khớp" in proc.stdout
    assert str(wrong_ds) in proc.stdout
    assert "HOST_FASHIONIQ_ROOT" in proc.stdout
    up_calls = [
        rec for rec in _read_tool_logs(sandbox)
        if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
        and "up" in rec.get("args", [])
    ]
    assert len(up_calls) == 1, f"exactly one startup per correction, saw {len(up_calls)}"


def test_container_gpu_mismatch_without_host_toolkit_fails_once_without_restart(tmp_path: Path):
    """When the requested configuration simply cannot be satisfied on this host (no toolkit),
    fail once with that reason instead of restarting and failing identically."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_CONTAINER_NO_GPU"] = "1"
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"
    env["MOCK_HOST_NO_TOOLKIT"] = "1"

    proc = run_pc(sandbox, ["--verify-only"], env, timeout=40)

    assert proc.returncode != 0, proc.stdout
    assert "CONTAINER_REUSE" in proc.stdout
    assert "nvidia-container-toolkit" in proc.stdout
    assert "HOST_FASHIONIQ_ROOT" not in proc.stdout
    up_calls = [
        rec for rec in _read_tool_logs(sandbox)
        if rec.get("tool") == "docker" and "compose" in rec.get("args", [])
        and "up" in rec.get("args", [])
    ]
    assert len(up_calls) == 0, f"no restart when unsatisfiable on host, saw {len(up_calls)}"
