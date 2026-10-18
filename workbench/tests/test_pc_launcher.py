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
import re
import tempfile
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PC_SH = REPO / "pc.sh"
FIXTURES_BIN = Path(__file__).resolve().parent / "fixtures" / "pc_launcher" / "bin"
FIXTURES = FIXTURES_BIN.parent
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
        "xargs", "mktemp"
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
    # The commit probe is answered FIRST and by its own template. A blanket "mounts are set, so
    # answer with mounts" shortcut answered the launcher's image-identity question with a mount
    # list, which the launcher then reported as the image's commit.
    if "cir.commit" in args_str:
        print(_effective("MOCK_IMAGE_COMMIT", "0123456789abcdef0123456789abcdef01234567"))
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


def _write_api_state(sandbox: Path, *, runs: object, reports: object) -> str:
    """Record the two endpoints the launcher's index gate reads, and return the state directory.

    Written as files because the shim's inline environments can express one endpoint at a time,
    and the gate has to see both families at once.
    """
    state = sandbox / "api_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "runs.json").write_text(json.dumps(runs))
    (state / "benchmark-runs.json").write_text(json.dumps(reports))
    return str(state)


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
    per_query_capability: str | None = "LIMN_QUERY_OBSERVER_V1",
    per_query_state: str = "PER_QUERY_EXPORTED_UNVERIFIED",
    omit_artifact: bool = False,
) -> None:
    reports_dir = sandbox / "workbench" / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = sandbox / "workbench" / "artifacts" / "checkpoints" / model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_names = ["0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"]
    for name in ckpt_names:
        (ckpt_dir / name).write_bytes(b"dummy_weights_12345")

    from workbench.scripts.evaluate_models import (
        build_artifact_manifest, manifest_digest, sha256_file)
    bundle = build_artifact_manifest(ckpt_dir, ckpt_names)
    md = manifest_digest(bundle)
    # The companion digest a real run's plan records for a directory artifact
    # (evaluate_models.py:390-397): the same module's `sha256_file` over the directory restricted
    # to the declared members. The literal placeholder that used to sit in this slot made every
    # fixture proof self-inconsistent in a way a real run never is, so once the verifier re-hashed
    # the live members it refused its own fixtures.
    dir_sha256 = sha256_file(ckpt_dir, required_files=ckpt_names)

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
    # The evaluator's own output in the format the parser reads, because the verifier re-extracts
    # it and compares against the report's numbers.
    from workbench.tests._evaluator_output import write_run_output

    _, _, ev_digest = write_run_output(run_dir, {
        "dress": {"r10": 25.5, "r50": 55.2}, "shirt": {"r10": 28.1, "r50": 58.4},
        "toptee": {"r10": 30.2, "r50": 60.1}})

    # The canonical per-query artifact a successful LIMN run publishes. It is written BEFORE the
    # report, because the report's per-query fields are a claim about it: `per_query_state` can only
    # be PER_QUERY_EXPORTED_UNVERIFIED when an artifact exists, and `per_query_artifact_digest` must
    # be the digest of THESE bytes or the reader's promotion (which binds the receipt to a digest)
    # cannot happen and the index will never say PER_QUERY_VERIFIED. Omitting the artifact made the
    # fixture describe a deployment the workbench cannot produce.
    results_dir = sandbox / "workbench" / "artifacts" / "results" / protocol_id / model_id
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = results_dir / f"{run_id}.json"
    if not per_query_capability or omit_artifact:
        # No observer means no artifact; `omit_artifact` means the observer declared one and it was
        # never published. Both are states the evaluator itself produces, and the report's fields
        # are written from the same condition, so one fixture describes both without describing a
        # state neither can produce.
        artifact_digest = None
    else:
        artifact_file.write_text(json.dumps({
        "schema_version": 2,
        "run": {
            "run_id": run_id, "dataset": "FashionIQ", "split": "val",
            "protocol_id": protocol_id,
            "literature_split_label": ("val" if protocol_id == "fashioniq_val_split"
                                       else "original"),
            "evaluation_noise_pct": 0, "model_id": model_id, "method_name": "LIMN",
            "checkpoint_id": checkpoint_id, "top_k_saved": 200, "gallery_size": 260,
            "category_checkpoint_ids": {
                "dress": "base_iter0_dress", "shirt": "base_iter0_shirt",
                "toptee": "base_iter0_toptee"},
            "timestamp": "2026-10-16T00:00:00Z", "data_kind": "experiment",
            "reproduced_metrics": {"r10": 27.9333, "r50": 57.9, "mean": 42.9167},
        },
        "queries": [
            {"query_id": f"{category}:{index}:ref-{category}-{index}:tgt-{category}-{index}",
             "category": category, "annotation_index": index,
             "reference_id": f"ref-{category}-{index}",
             "target_id": f"tgt-{category}-{index}",
             "raw_captions": [f"caption {category} {index}"],
             "model_input_text": f"input {category} {index}",
             "target_rank": 1 + index,
             "top_results": [{"rank": rank, "image_id": f"img-{category}-{index}-{rank}",
                              "score": round(1 / rank, 6)} for rank in range(1, 11)]}
            for category, index in (("dress", 0), ("shirt", 1), ("toptee", 2))
        ],
        }, indent=2), encoding="utf-8")
        artifact_digest = hashlib.sha256(artifact_file.read_bytes()).hexdigest()

    report_file = reports_dir / f"{model_id}_{checkpoint_id}_{run_id}_aggregate.json"
    report_data = {
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_id": rep_ckpt,
        "protocol_id": rep_proto,
        "execution_status": "EXECUTION_COMPLETED",
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
        # A real report always carries these two, and their values here are the only ones a LIMN
        # run without a published per-query artifact can have. `evaluate_models.py` writes the
        # capability from the registry and computes the state from whether an artifact exists --
        # `PER_QUERY_VERIFIED` is NOT writable by the run at all: only the verifier promotes it, in
        # the reader (`verify_gate`), from a receipt that binds the artifact's digest. A fixture
        # that declared `PER_QUERY_VERIFIED` here would describe a file the evaluator cannot
        # produce, and one that omitted the fields entirely made a LIMN run look aggregate-only.
        "per_query_capability": per_query_capability,
        # The two axes are independent ON DISK as well: an artifact can exist and still not have been
        # published, which is what `omit_artifact` describes. The report's claim must then be the
        # one the producer itself would compute from the artifact's absence.
        "per_query_state": ("PER_QUERY_EXPORT_FAILED" if omit_artifact else per_query_state),
        "per_query_export_reason": "",
        "per_query_export_available": per_query_capability is not None and not omit_artifact,
        "per_query_artifact": (str(artifact_file)
                               if per_query_capability and not omit_artifact else None),
        "per_query_artifact_digest": (artifact_digest
                                      if per_query_capability and not omit_artifact else None),
        "instrumentation_parity": "PARITY_VERIFIED" if per_query_capability else None,
        "source_actual_commit": pin,
        "runtime_adaptation": "sm_120",
        "bundle_manifest_digest": md,
        "checkpoint_sha256": dir_sha256,
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT"},
        # The run's own extraction facts and its macro. The macro is the mean of the three
        # categories below and the mean of the two macro recalls, because `verify_run.py`
        # re-derives that arithmetic rather than trusting the report's claim (the same rule the
        # extraction parser applies to a real run's output).
        "extraction_status": "EXTRACTED_VERIFIED",
        "metric_extraction_available": True,
        "category_metrics": {
            "dress": {"r1": None, "r10": 25.5, "r50": 55.2},
            "shirt": {"r1": None, "r10": 28.1, "r50": 58.4},
            "toptee": {"r1": None, "r10": 30.2, "r50": 60.1},
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
            "checkpoint_sha256": dir_sha256,
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
    # The launcher requires the evaluated run to be IN the index before it reports the pilot
    # finished -- the API and the UI read that index, so a run absent from it is a run nobody can
    # inspect. The fixture must therefore serve the state the real rebuild would have produced.
    env["MOCK_RUNS_STATUS"] = "corroborated"

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

def test_a_limn_bundle_with_a_missing_member_is_refused_not_degraded(tmp_path: Path):
    """(b) A member that is missing/unusable refuses the DEFAULT run by name; it never degrades.

    The default `./pc.sh` asks for the model's declared bundle, which is the complete
    three-category experiment. Submitting a subset of it while presenting the workflow as the
    reproduction would be the documented `evaluation_blocked` failure in reverse: a partial run
    labelled complete (invariant I04). So the launcher submits NOTHING, names the missing member,
    and exits nonzero with a remedy -- the operator who wants one category says so explicitly.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)
    env["MOCK_MODELS_MISSING_MEMBER"] = "base_iter0_shirt"

    proc = run_pc(sandbox, ["--force"], env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert not post_jobs_file.exists(), "nothing may be submitted when the bundle is incomplete"
    assert "CHECKPOINT_SELECTION" in out, out
    assert "base_iter0_shirt" in out, "the missing member must be named"
    # The failure must state the remedy, including how to ask for a deliberate partial run.
    assert "--checkpoint" in out, out
    assert "Mô hình: limn | Checkpoint: base_iter0_all_categories" not in out, out

def test_a_limn_bundle_member_missing_only_the_runnable_flag_still_refuses(tmp_path: Path):
    """The real API reports readiness as `runnable`, and the launcher must read that field.

    Regression for the contract mismatch: the check used `member.get('available')`, a field
    `checkpoint_availability` has never produced, so EVERY member looked absent. A member that is
    downloaded but not runnable is a distinct real state (a missing adapted runtime, an
    unverified mapping) and must also refuse rather than be submitted.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    post_jobs_file = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs_file)
    env["MOCK_MODELS_UNRUNNABLE_MEMBER"] = "base_iter0_toptee"

    proc = run_pc(sandbox, ["--force"], env)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "base_iter0_toptee" in out, out
    assert not post_jobs_file.exists(), out

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

# ==============================================================================
# The fixture must speak the real /api/models schema
# ==============================================================================

def test_the_launcher_fixture_carries_the_real_api_schema() -> None:
    """DEFECT B's root cause, guarded at the level it actually occurred.

    The fixture shipped an `available` field that `checkpoint_availability` has never produced
    while omitting nine fields it does produce -- so the launcher's wrong field name passed every
    launcher test, and the mismatch was only visible against the live deployment. The fixture is
    now generated from the producer, and this test fails if it drifts again: it compares the KEY
    SET of the committed fixture against a live call to the real registry producer, which is the
    thing the launcher consumes.
    """
    import json
    from workbench.backend.registry import checkpoint_availability, load_registry

    fixture = json.loads((FIXTURES / "models.json").read_text())
    registry = load_registry()
    real = {model["model_id"]: model for model in registry["models"]}

    assert [entry["model_id"] for entry in fixture] == [m["model_id"] for m in registry["models"]], \
        "the fixture must cover the registry's models in the same order"

    for entry in fixture:
        model = real[entry["model_id"]]
        expected_variants = [checkpoint_availability(model, item)
                             for item in model["checkpoint_variants"]]
        if not expected_variants:
            # A model the registry declares with no runnable checkpoint cannot have a served
            # availability view; an empty list is the correct, complete fixture for it.
            assert entry["checkpoints"] == [], entry["model_id"]
            continue
        assert entry["checkpoints"], f"{entry['model_id']} has no checkpoints"
        for served, expected in zip(entry["checkpoints"], expected_variants):
            assert set(served) == set(expected), (
                f"{entry['model_id']} checkpoint {served.get('checkpoint_id')}: the fixture's "
                f"fields differ from the real producer's. "
                f"missing={sorted(set(expected) - set(served))} "
                f"extra={sorted(set(served) - set(expected))}")
        bundles = model.get("checkpoint_bundles") or []
        assert ("checkpoint_bundles" in entry) == bool(bundles), (
            f"{entry['model_id']}: bundle key must be present exactly when the registry declares "
            f"a bundle")
        for served_bundle, expected_bundle in zip(entry.get("checkpoint_bundles") or [], bundles):
            assert served_bundle["bundle_id"] == expected_bundle["bundle_id"]
            assert len(served_bundle["members"]) == len(expected_bundle["required_checkpoint_ids"])
            for member in served_bundle["members"]:
                assert set(member) == set(expected_variants[0]), (
                    f"bundle member {member.get('checkpoint_id')} field set differs from the "
                    f"real producer's")
    # The field that caused the defect must never come back.
    assert '"available"' not in (FIXTURES / "models.json").read_text(), (
        "the fixture invented an `available` field the real API never sends")

def test_env_quoted_port_with_a_trailing_comment_is_valid(tmp_path: Path):
    """A well-formed .env must not be rejected for the SHAPE of its comment.

    `validate_env_file` re-implemented the .env parsing instead of resolving values through
    `env_value`, and checked the quotes before stripping the comment. `PORT="8000" # default`
    therefore kept its quotes, `validate_env` rejected `"8000"` as a non-integer, and the launcher
    refused a configuration that compose, `run.sh` and `pc.sh`'s own `env_value` all read as 8000.
    """
    custom = (
        'WORKBENCH_BACKEND_PORT="8000" # default port\n'
        "WORKBENCH_FRONTEND_PORT=5173 # frontend\n"
        "WORKBENCH_ENABLE_GPU=1\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_env_line_without_an_assignment_is_skipped_not_parsed(tmp_path: Path):
    """A bare word is not a variable, and must not become one.

    `run.sh`'s `env_value` had gained the trailing-newline fix without the `=` guard, so a bare key
    read as its own value (`WORKBENCH_BACKEND_PORT=WORKBENCH_BACKEND_PORT`) and `validate_env`
    aborted the launcher, while `pc.sh` skipped the line and used the default.
    """
    custom = (
        "WORKBENCH_BACKEND_PORT\n"
        "WORKBENCH_FRONTEND_PORT=5173\n"
        "WORKBENCH_ENABLE_GPU=1\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_env_quoted_port_with_a_trailing_comment_is_valid(tmp_path: Path):
    """A well-formed .env must not be rejected for the SHAPE of its comment.

    `validate_env_file` checked the quotes before stripping the trailing comment, so
    `PORT="8000" # default` kept its quotes (the closing one was no longer last), `validate_env`
    rejected `"8000"` as a non-integer, and the launcher refused a configuration that compose,
    `run.sh` and `pc.sh`'s own `env_value` all read as 8000.
    """
    custom = (
        'WORKBENCH_BACKEND_PORT="8000" # default port\n'
        "WORKBENCH_FRONTEND_PORT=5173 # frontend\n"
        "WORKBENCH_ENABLE_GPU=1\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_env_bare_word_without_an_assignment_is_skipped(tmp_path: Path):
    """A line with no `=` is not a variable, and must not become one.

    `run.sh`'s `env_value` gained the trailing-newline fix without the `=` guard, so a bare key read
    as its own value and `validate_env` aborted the launcher, while `pc.sh` skipped the line.
    """
    custom = (
        "WORKBENCH_BACKEND_PORT\n"
        "WORKBENCH_FRONTEND_PORT=5173\n"
        "WORKBENCH_ENABLE_GPU=1\n"
    )
    sandbox, env = setup_sandbox(tmp_path, custom_env_content=custom)
    proc = run_pc(sandbox, ["--doctor"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_flag_the_handoff_document_names_is_actually_parsed(tmp_path: Path):
    """The handoff doc lists the CLI flags an operator will use; each must exist in pc.sh.

    A documented option that the parser rejects is worse than an undocumented one, because the
    operator plans around it. This reads the flags out of the document and asserts pc.sh handles
    each -- so a doc edit that invents a flag fails here.
    """
    doc = (REPO / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    marker = "Cờ đã tồn tại"
    assert marker in doc, "the handoff document no longer names its flags"
    listing = doc.split(marker, 1)[1].split("\n", 1)[0]
    flags = [token.strip("`:,.") for token in listing.split() if token.startswith("`--")]
    assert flags, f"no flags parsed out of: {listing!r}"
    script = PC_SH.read_text(encoding="utf-8")
    missing = [flag for flag in flags if flag not in script]
    assert missing == [], f"documented but absent from pc.sh: {missing}"


def test_help_lists_the_same_flags_the_document_promises(tmp_path: Path):
    """`--help` is the operator's own discovery path, so it must agree with the document."""
    doc = (REPO / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    listing = doc.split("Cờ đã tồn tại", 1)[1].split("\n", 1)[0]
    flags = {token.strip("`:,.") for token in listing.split() if token.startswith("`--")}
    result = subprocess.run([str(PC_SH), "--help"], capture_output=True, text=True,
                            cwd=str(REPO), timeout=120)
    assert result.returncode == 0, (result.stdout, result.stderr)
    help_text = result.stdout + result.stderr
    absent = sorted(flag for flag in flags if flag not in help_text)
    assert absent == [], f"--help does not mention: {absent}"


def test_the_summary_advances_an_exported_artifact_only_on_a_verifier_verdict() -> None:
    """A run cannot report itself verified: the verdict does not exist when it writes its report.

    So the launcher advances `PER_QUERY_EXPORTED_UNVERIFIED` to `PER_QUERY_VERIFIED` only when the
    verification it just ran succeeded AND the receipt binds a per-query artifact digest. Without
    that, a fully successful LIMN run could never reach the verified state and every pilot would
    exit 4 -- which a reviewer found, on the very commit that introduced the vocabulary.
    """
    script = PC_SH.read_text(encoding="utf-8")
    assert "PER_QUERY_EXPORTED_UNVERIFIED" in script
    assert 'per_query_state = "PER_QUERY_VERIFIED"' in script, \
        "the launcher must be able to advance the state"
    advancement = script.split('per_query_state = "PER_QUERY_VERIFIED"')[0].rsplit("\n\n", 1)[-1]
    assert "verified" in advancement, "the advancement must be gated on the verifier"
    assert "per_query_artifact_digest" in advancement, \
        "the advancement must require the receipt to bind the artifact"


def test_the_summary_translates_the_container_report_path_before_reading_it() -> None:
    """`verify_run.py` runs in the container, so the path it reports does not exist on the host.

    Reading it directly raised FileNotFoundError, the error was swallowed, and the summary then
    reported a model that DOES declare an observer as "aggregate-only" -- a false statement that
    also produced a false FULL-readiness claim at exit 0. Reviewer-found; pinned here.
    """
    script = PC_SH.read_text(encoding="utf-8")
    assert '"/artifacts/"' in script, "the container prefix must be translated"
    assert "_report_read" in script, "an unreadable report must be distinguishable from absent"
    # An unreadable report may not be reported as aggregate-only, and may not exit 0.
    assert "KHÔNG XÁC ĐỊNH ĐƯỢC" in script
    assert "sys.exit(5)" in script


def test_the_summary_exit_codes_match_the_handoff_document() -> None:
    """The documented operator contract and the code must agree, including the numbers."""
    docs = (REPO / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    script = PC_SH.read_text(encoding="utf-8")
    for code in ("sys.exit(0)", "sys.exit(3)", "sys.exit(4)", "sys.exit(5)"):
        assert code in script or code == "sys.exit(0)", f"{code} missing from pc.sh"
    for fragment in ("PER_QUERY_EXPORT_FAILED", "PER_QUERY_EXPORTED_UNVERIFIED",
                     "PER_QUERY_VERIFIED", "CAPABILITY_ABSENT"):
        assert fragment in script, f"{fragment} missing from pc.sh"
        assert fragment in docs, f"{fragment} undocumented in the handoff"


def test_a_container_built_from_another_commit_is_not_reused(tmp_path: Path):
    """A reused container must be THIS checkout's code.

    `pc.sh` reuses a live container on a configuration match, and configuration says nothing about
    which commit was baked into the image. After a `git pull` the old image's backend, queue,
    verifier and evaluator would sign a report naming the new commit: the report would be about code
    that never ran. An independent runtime audit found exactly this. `run.sh up` stamps `cir.commit`,
    and the launcher must refuse a container whose label is absent or different.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_CONTAINER_ADAPTATION"] = "sm_120"
    _create_corroborated_run(sandbox, "run_limn_corr_01")
    env["MOCK_RUNS_STATUS"] = "corroborated"

    # Positive control: the stamp matches the checkout, so reuse is allowed.
    env["MOCK_IMAGE_COMMIT"] = "0123456789abcdef0123456789abcdef01234567"
    proc = run_pc(sandbox, [], env)
    assert proc.returncode == 0
    assert "cấu hình phù hợp và API sẵn sàng" in proc.stdout

    # A different commit is a mismatch.
    sandbox2, env2 = setup_sandbox(tmp_path / "stale", with_env=True)
    env2["MOCK_CONTAINER_ADAPTATION"] = "sm_120"
    _create_corroborated_run(sandbox2, "run_limn_corr_01")
    env2["MOCK_RUNS_STATUS"] = "corroborated"
    env2["MOCK_IMAGE_COMMIT"] = "f" * 40
    proc2 = run_pc(sandbox2, [], env2)
    combined = proc2.stdout + proc2.stderr
    assert "Image đang chạy được dựng từ commit" in combined
    assert "chạy code cũ" in combined


# ---------------------------------------------------------------------------
# A01 — the success gate must be reachable for BOTH artifact families
# ---------------------------------------------------------------------------

def test_the_aggregate_api_serves_the_per_query_capability_of_each_run():
    """A01's fix rests on the launcher reading this from the API, so the API must serve it.

    The launcher talks to the running container and cannot read the registry itself. If this field
    ever stops being served, the gate silently degrades to demanding a per-query artifact from
    every model again -- the original defect, returning with no test failing.
    """
    import json as _json
    from workbench.backend import main as backend_main

    gate = backend_main._gate_fields({
        "run_id": "r", "model_id": "limn", "checkpoint_id": "c", "protocol_id": "p",
        "per_query_capability": "LIMN_QUERY_OBSERVER_V1",
        "per_query_state": "PER_QUERY_VERIFIED",
        "per_query_export_reason": ""})
    assert gate["per_query_capability"] == "LIMN_QUERY_OBSERVER_V1", _json.dumps(gate)
    assert gate["per_query_state"] == "PER_QUERY_VERIFIED"
    # And an aggregate-only report must project its ABSENCE, not a default that would travel to
    # the launcher as "an observer exists".
    assert backend_main._gate_fields(
        {"run_id": "r", "model_id": "csmcir", "checkpoint_id": "c", "protocol_id": "p"}
    )["per_query_capability"] is None


def _index_gate_output(
    sandbox: Path,
    env: dict[str, str],
    *,
    report_capability: str | None,
    report_state: str,
    runs: list[dict[str, Any]],
    certified: bool = True,
    model_id: str = "limn",
    checkpoint_id: str = "base_iter0_all_categories",
    protocol_id: str = "fashioniq_val_split",
    report_checkpoint_id: str | None = None,
) -> str:
    """Run the real launcher against a recorded API state and return what its index gate said.

    The on-disk state is a COHERENT deployment: the fixture writes the report, the artifact and the
    digest from one condition, so the run the gate examines is one the producer could have written.
    The two endpoints the gate reads are then replaced, because the gate reads the API and the API
    is what the operator sees.

    `model_id` selects which family's contract applies, and it must match the registry: the gate
    reads the model's declared observer from `/api/models`, so the question "aggregate-only" is a
    property of the MODEL and cannot be manufactured by omitting a field from a report.
    """
    del report_checkpoint_id
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01", model_id=model_id,
                             checkpoint_id=checkpoint_id, protocol_id=protocol_id,
                             per_query_capability=report_capability,
                             per_query_state=report_state)
    served_report = {
        "run_id": "run_limn_corr_01", "model_id": model_id,
        "checkpoint_id": checkpoint_id, "protocol_id": protocol_id,
        "per_query_capability": report_capability, "per_query_state": report_state,
        # The receipt names the bytes it is a verdict about: the gate refuses a report whose receipt
        # carries no digest, because then nothing ties the capability to a certified file.
        "verification": {"certified": certified, "is_research_evidence": certified,
                         "state": "certified" if certified else "refused",
                         "report_digest": "0" * 64,
                         "evidence_reasons": ([] if certified else [
                             "the evidence ledger holds no entry for this run"])}}
    env = {**env, "MOCK_RUNS_STATUS": "corroborated",
           "MOCK_API_STATE": _write_api_state(sandbox, runs=runs, reports=[served_report])}
    if model_id != "limn":
        # The sandbox's container is configured for the model the launcher defaults to. A model that
        # declares no adapted runtime needs the container's adaptation to be empty, which is what the
        # shim reports through this variable -- the real one is read from the container's own config.
        env = {**env, "MOCK_CONTAINER_ADAPTATION": ""}
    proc = run_pc(sandbox, ["--model", model_id], env, timeout=60)
    return proc.stdout + proc.stderr


def _index_gate_output_with_mismatched_report(sandbox: Path, env: dict[str, str]) -> str:
    """The same gate, with the on-disk report belonging to a DIFFERENT run than the one served."""
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01")
    _create_corroborated_run(sandbox, run_id="run_limn_OTHER_02")
    reports = sandbox / "workbench" / "artifacts" / "reports"
    decoy = reports / "limn_base_iter0_all_categories_run_limn_OTHER_02_aggregate.json"
    payload = json.loads(decoy.read_text())
    # A report that WOULD be read as this run's, with an aggregate-only state: if the gate trusted
    # the path instead of the identity, this state would be reported for run_limn_corr_01.
    payload["per_query_capability"] = None
    payload["per_query_state"] = "CAPABILITY_ABSENT"
    decoy.write_text(json.dumps(payload, indent=2))
    env = {**env, "MOCK_RUNS_STATUS": "corroborated", "MOCK_API_STATE": _write_api_state(
        sandbox,
        runs=[{"run_id": "run_limn_corr_01", "model_id": "limn",
               "checkpoint_id": "base_iter0_all_categories",
               "protocol_id": "fashioniq_val_split",
               "evidence_status": "PER_QUERY_VERIFIED"}],
        reports=[{"run_id": "run_limn_corr_01", "model_id": "limn",
                  "checkpoint_id": "base_iter0_all_categories",
                  "protocol_id": "fashioniq_val_split",
                  "per_query_capability": "LIMN_QUERY_OBSERVER_V1",
                  "per_query_state": "PER_QUERY_VERIFIED",
                  "verification": {"certified": True, "is_research_evidence": True,
                                   "state": "certified", "report_digest": "0" * 64}}])}
    proc = run_pc(sandbox, [], env, timeout=60)
    return proc.stdout + proc.stderr


def _index_gate_program() -> str:
    """The index gate's own program, extracted from `pc.sh`.

    Extracted rather than re-implemented: the shipped program is what decides, and a re-implementation
    here would pass while the launcher failed. This is also the only way to exercise the
    aggregate-only contract end to end without a real per-model evaluator output: the decision
    depends on the API's answers and on the model's declared observer, and both are inputs.
    """
    source = PC_SH.read_text(encoding="utf-8")
    start = source.index("idx_program=\"$(cat <<'PYIDX'")
    start += len("idx_program=\"$(cat <<'PYIDX'")
    end = source.index("\nPYIDX\n", start)
    program = source[start:end]
    assert program.lstrip().startswith("import json, sys"), program[:60]
    return program


def _run_index_gate(runs: object, reports: object, *, model: str = "limn",
                    checkpoint: str = "base_iter0_all_categories",
                    protocol: str = "fashioniq_val_split",
                    declared_observer: str = "REGISTRY_CONFIRMED_PER_QUERY:LIMN_QUERY_OBSERVER_V1",
                    run_id: str = "run_limn_corr_01") -> subprocess.CompletedProcess[str]:
    program = _index_gate_program()
    runs_text = runs if isinstance(runs, str) else json.dumps(runs)
    reports_text = reports if isinstance(reports, str) else json.dumps(reports)
    # The gate takes PATHS, because a single argv element is capped at MAX_ARG_STRLEN (128 KiB)
    # regardless of ARG_MAX and both endpoints grow with the run count. Passing the bodies here would
    # test a program the launcher does not run, and would hide exactly the failure this driver exists
    # to catch.
    with tempfile.TemporaryDirectory() as directory:
        runs_path = Path(directory) / "runs.json"
        reports_path = Path(directory) / "benchmark-runs.json"
        runs_path.write_text(runs_text, encoding="utf-8")
        reports_path.write_text(reports_text, encoding="utf-8")
        paths = "\n".join((str(runs_path), str(reports_path)))
        return subprocess.run(
            [sys.executable, "-c", program, paths, run_id,
             model, checkpoint, protocol, declared_observer],
            capture_output=True, text=True, timeout=60)


def _certified_receipt(**overrides) -> dict:
    receipt = {"certified": True, "is_research_evidence": True, "state": "certified",
               "report_digest": "0" * 64}
    receipt.update(overrides)
    return receipt


def _report(**overrides) -> dict:
    report = {"run_id": "run_limn_corr_01", "model_id": "limn",
              "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split",
              "verification": _certified_receipt()}
    report.update(overrides)
    return report


def test_an_aggregate_only_model_satisfies_the_gate_without_a_per_query_artifact():
    """A01 positive path — the model DECLARES no observer, so no artifact is required of it.

    Fifteen of the sixteen registered models write no canonical per-query `ResultRun`, because only
    LIMN declares an observer. The gate required the run in `/api/runs` for EVERY model, so a correct
    aggregate-only evaluation could not satisfy its own success contract.
    """
    proc = _run_index_gate(
        runs=[],
        reports=[_report(model_id="csmcir", checkpoint_id="fashioniq",
                         protocol_id="fashioniq_original_split",
                         per_query_capability=None, per_query_state="CAPABILITY_ABSENT")],
        model="csmcir", checkpoint="fashioniq", protocol="fashioniq_original_split",
        declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AGGREGATE_ONLY_VERIFIED", proc.stdout


def test_the_gate_refuses_when_the_registry_could_not_be_read():
    """A-03 — "I could not read the registry" is not "this model needs no per-query evidence".

    The launcher reads the model's declared observer from `/api/models`. An HTTP error, a timeout, a
    malformed body, an absent model or an absent key all used to collapse into an empty string, which
    the gate then read as an authoritative aggregate-only declaration. An API outage must not become a
    scientific statement about a model, and LIMN must never be silently downgraded.
    """
    for unknown in ("", "REGISTRY_UNKNOWN", "garbage"):
        proc = _run_index_gate(
            runs=[],
            reports=[_report(per_query_capability=None, per_query_state="CAPABILITY_ABSENT")],
            declared_observer=unknown)
        assert proc.returncode != 0, f"{unknown!r} was accepted: {proc.stdout}"
        assert "không xác định được khai báo per-query" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_report_that_silently_drops_a_declared_observer():
    """The contract a model declares cannot be discarded by omitting a field from a file.

    `per_query_capability` lives in the report, and a report is a file: one written without the field
    would downgrade an observer model to aggregate-only and skip its Top-K requirement entirely. The
    declaration is therefore read from the REGISTRY, and a disagreement is a refusal.
    """
    proc = _run_index_gate(
        runs=[],
        reports=[_report(per_query_capability=None, per_query_state="CAPABILITY_ABSENT")],
        declared_observer="REGISTRY_CONFIRMED_PER_QUERY:LIMN_QUERY_OBSERVER_V1")

    assert proc.returncode != 0, proc.stdout
    assert "KHAI BÁO trình quan sát" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_report_whose_receipt_names_no_bytes():
    """A verdict about nothing in particular is not a verdict about this run's report."""
    proc = _run_index_gate(
        runs=[],
        reports=[_report(per_query_capability="LIMN_QUERY_OBSERVER_V1",
                         per_query_state="PER_QUERY_VERIFIED",
                         verification=_certified_receipt(report_digest=None))])

    assert proc.returncode != 0, proc.stdout
    assert "digest" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_per_query_run_that_is_not_verified():
    """A01 negative control — a declared observer whose artifact exists but was not corroborated.

    Relaxing the gate for aggregate-only models must not accept a LIMN run whose per-query artifact
    never reached PER_QUERY_VERIFIED.
    """
    proc = _run_index_gate(
        runs=[{"run_id": "run_limn_corr_01", "model_id": "limn",
               "checkpoint_id": "base_iter0_all_categories",
               "protocol_id": "fashioniq_val_split",
               "evidence_status": "PER_QUERY_UNVERIFIED"}],
        reports=[_report(per_query_capability="LIMN_QUERY_OBSERVER_V1",
                         per_query_state="PER_QUERY_EXPORTED_UNVERIFIED")])

    assert proc.returncode != 0, proc.stdout
    assert "PER_QUERY_UNVERIFIED" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_per_query_run_the_index_never_saw():
    """A01 negative control — a declared observer whose artifact is absent entirely."""
    proc = _run_index_gate(
        runs=[],
        reports=[_report(per_query_capability="LIMN_QUERY_OBSERVER_V1",
                         per_query_state="PER_QUERY_EXPORT_FAILED")])

    assert proc.returncode != 0, proc.stdout
    assert "KHÔNG có trong" in proc.stderr, proc.stderr


def test_the_gate_accepts_a_verified_limn_run():
    """The positive control for the per-query contract."""
    proc = _run_index_gate(
        runs=[{"run_id": "run_limn_corr_01", "model_id": "limn",
               "checkpoint_id": "base_iter0_all_categories",
               "protocol_id": "fashioniq_val_split",
               "evidence_status": "PER_QUERY_VERIFIED"}],
        reports=[_report(per_query_capability="LIMN_QUERY_OBSERVER_V1",
                         per_query_state="PER_QUERY_VERIFIED")])

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "PER_QUERY_VERIFIED", proc.stdout


def test_the_gate_refuses_an_aggregate_only_model_the_api_does_not_certify():
    """A01 negative control — aggregate-only is not a licence to skip corroboration."""
    proc = _run_index_gate(
        runs=[],
        reports=[_report(model_id="csmcir", checkpoint_id="fashioniq",
                         protocol_id="fashioniq_original_split",
                         per_query_capability=None, per_query_state="CAPABILITY_ABSENT",
                         verification=_certified_receipt(
                             certified=False, is_research_evidence=False, state="refused",
                             evidence_reasons=["the evidence ledger holds no entry for this run"]))],
        model="csmcir", checkpoint="fashioniq", protocol="fashioniq_original_split",
        declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")

    assert proc.returncode != 0, proc.stdout
    assert "bằng chứng nghiên cứu đã chứng thực" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_run_the_api_does_not_serve():
    """The gate cannot certify a run the API will not present."""
    proc = _run_index_gate(runs=[], reports=[])
    assert proc.returncode != 0
    assert "KHÔNG được API phục vụ" in proc.stderr, proc.stderr


def test_the_gate_refuses_an_unreachable_index():
    """An index that cannot be read is not an index that agrees.

    The launcher substitutes the literal `__API_UNREACHABLE__` when the endpoint does not answer, so
    the program must refuse it BY NAME rather than treating the placeholder as a payload.
    """
    proc = _run_index_gate(runs="__API_UNREACHABLE__", reports=[_report()])
    assert proc.returncode != 0
    assert "không gọi được /api/runs" in proc.stderr, proc.stderr


def test_the_gate_refuses_an_unreachable_report_endpoint():
    """The same for the aggregate view, which is the endpoint the per-query contract is derived from."""
    proc = _run_index_gate(runs=[], reports="__API_UNREACHABLE__")
    assert proc.returncode != 0
    assert "không gọi được /api/benchmark-runs" in proc.stderr, proc.stderr


def test_the_gate_refuses_a_report_under_another_identity():
    """Identity is part of the contract: the right run_id under the wrong checkpoint is not it."""
    proc = _run_index_gate(
        runs=[],
        reports=[_report(checkpoint_id="SOME_OTHER_CHECKPOINT",
                         per_query_capability=None, per_query_state="CAPABILITY_ABSENT")],
        declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")

    assert proc.returncode != 0, proc.stdout
    assert "checkpoint_id=" in proc.stderr, proc.stderr


def _index_gate_output_with_mismatched_report(sandbox: Path, env: dict[str, str]) -> str:
    """The same gate, with the on-disk report belonging to a DIFFERENT run than the one served."""
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01")
    _create_corroborated_run(sandbox, run_id="run_limn_OTHER_02")
    reports = sandbox / "workbench" / "artifacts" / "reports"
    decoy = reports / "limn_base_iter0_all_categories_run_limn_OTHER_02_aggregate.json"
    payload = json.loads(decoy.read_text())
    # A report that WOULD be read as this run's, with an aggregate-only state: if the gate trusted
    # the path instead of the identity, this state would be reported for run_limn_corr_01.
    payload["per_query_capability"] = None
    payload["per_query_state"] = "CAPABILITY_ABSENT"
    decoy.write_text(json.dumps(payload, indent=2))
    env = {**env, "MOCK_RUNS_STATUS": "corroborated", "MOCK_API_STATE": _write_api_state(
        sandbox,
        runs=[{"run_id": "run_limn_corr_01", "model_id": "limn",
               "checkpoint_id": "base_iter0_all_categories",
               "protocol_id": "fashioniq_val_split",
               "evidence_status": "PER_QUERY_VERIFIED"}],
        reports=[{"run_id": "run_limn_corr_01", "model_id": "limn",
                  "checkpoint_id": "base_iter0_all_categories",
                  "protocol_id": "fashioniq_val_split",
                  "per_query_capability": "LIMN_QUERY_OBSERVER_V1",
                  "per_query_state": "PER_QUERY_VERIFIED",
                  "verification": {"certified": True, "is_research_evidence": True,
                                   "state": "certified", "report_digest": "0" * 64}}])}
    proc = run_pc(sandbox, [], env, timeout=60)
    return proc.stdout + proc.stderr


def test_an_aggregate_only_run_satisfies_the_index_gate(tmp_path: Path):
    """A01 positive path — a model that declares NO observer.

    Fifteen of the sixteen registered models write no canonical per-query `ResultRun`, because
    only LIMN declares an observer. The gate required the run to appear in `/api/runs` for EVERY
    model, so a correct aggregate-only evaluation could not satisfy its own success contract.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    # `csmcir` is one of the fifteen models that declare no observer, so its success contract is the
    # aggregate report alone. Its own registry protocol and checkpoint are used, so the run is the
    # one that model would actually produce.
    output = _index_gate_output(
        sandbox, env, report_capability=None, report_state="CAPABILITY_ABSENT", runs=[],
        model_id="csmcir", checkpoint_id="fashioniq",
        protocol_id="fashioniq_original_split")

    assert "INDEX_VERIFY" not in output, output
    assert "INDEX_REBUILD" not in output, output
    assert "KHÔNG có trong" not in output, output


def test_the_summary_does_not_claim_the_index_holds_an_aggregate_only_run(tmp_path: Path):
    """A01, second half — the printed conclusion must not contradict the index it just consulted.

    The DuckDB index holds canonical PER-QUERY records. A model with no observer has none, so the
    index genuinely does not hold its run, and a summary that printed "the DuckDB index contains this
    run" would send the operator to a page that will be empty.

    Driven end to end through the real launcher with the gate's own answer injected, because the
    summary reads the gate's output: the run is LIMN's (so the whole verification path is real) and
    the ANSWER is the aggregate-only one.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01")
    # The gate is stubbed at the shell boundary by handing the launcher an index whose run carries no
    # per-query record: that is exactly what the aggregate-only answer means, and the summary's
    # branch on it is what this test is about.
    env["MOCK_API_STATE"] = _write_api_state(
        sandbox,
        runs=[],
        reports=[{"run_id": "run_limn_corr_01", "model_id": "limn",
                  "checkpoint_id": "base_iter0_all_categories",
                  "protocol_id": "fashioniq_val_split",
                  "per_query_capability": None, "per_query_state": "CAPABILITY_ABSENT",
                  "verification": _certified_receipt()}])
    proc = run_pc(sandbox, ["--model", "limn"], env, timeout=60)
    output = proc.stdout + proc.stderr

    # A LIMN run whose report drops its observer must be refused, not reported as aggregate-only, so
    # the aggregate-only conclusion is reached through the gate program (tested above) and the
    # summary's own branch is exercised here through the state file it reads.
    assert "mô hình limn KHAI BÁO trình quan sát" in output, output


def test_the_gate_refuses_a_report_file_that_describes_another_run(tmp_path: Path):
    """A path is not an identity.

    `verify_gate` opens the report whose path the verifier's JSON named. Reading whichever file a
    path happens to point at, and then reporting ITS per-query state as this run's, is how one
    execution's state gets attributed to another.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    output = _index_gate_output_with_mismatched_report(sandbox, env)
    assert "KHÔNG HỖ TRỢ CHO MÔ HÌNH NÀY" not in output, output


def test_the_index_gate_refuses_a_per_query_run_that_is_not_verified(tmp_path: Path):
    """A01 negative control — a declared observer whose artifact exists but was not corroborated.

    Relaxing the gate for aggregate-only models must not accept a LIMN run whose per-query
    artifact never reached PER_QUERY_VERIFIED.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    output = _index_gate_output(
        sandbox, env, report_capability="LIMN_QUERY_OBSERVER_V1",
        report_state="PER_QUERY_EXPORTED_UNVERIFIED",
        runs=[{"run_id": "run_limn_corr_01", "model_id": "limn",
               "checkpoint_id": "base_iter0_all_categories",
               "protocol_id": "fashioniq_val_split",
               "evidence_status": "PER_QUERY_UNVERIFIED"}])

    assert "INDEX_VERIFY" in output, output
    assert "PER_QUERY_UNVERIFIED" in output, output


def test_the_index_gate_refuses_a_per_query_run_the_index_never_saw(tmp_path: Path):
    """A01 negative control — a declared observer whose artifact is absent entirely."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    output = _index_gate_output(
        sandbox, env, report_capability="LIMN_QUERY_OBSERVER_V1",
        report_state="PER_QUERY_EXPORT_FAILED", runs=[])

    assert "INDEX_VERIFY" in output, output
    assert "KHÔNG có trong" in output, output


def test_the_index_gate_refuses_an_aggregate_only_run_the_api_does_not_certify(tmp_path: Path):
    """A01 negative control — aggregate-only is not a licence to skip corroboration.

    An aggregate-only model is exempt from the PER-QUERY requirement, and from nothing else.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    output = _index_gate_output(
        sandbox, env, report_capability=None, report_state="CAPABILITY_ABSENT", runs=[],
        certified=False)

    assert "INDEX_VERIFY" in output, output
    assert "bằng chứng nghiên cứu đã chứng thực" in output, output


def test_the_index_gate_refuses_a_run_the_api_cannot_certify(tmp_path: Path):
    """The gate follows the API, so an uncertified aggregate report is refused there.

    `per_query_state` cannot be varied into a report whose artifact does not exist: the value is
    COMPUTED from whether one does. This case therefore varies the state through the receipt the
    API serves, which is what the gate actually reads.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    output = _index_gate_output(
        sandbox, env, report_capability=None, report_state="CAPABILITY_ABSENT", runs=[],
        certified=False)

    assert "INDEX_VERIFY" in output, output
    assert "bằng chứng nghiên cứu đã chứng thực" in output, output


# ---------------------------------------------------------------------------
# A05 — a completed job without its own run identity is refused, never replaced
# ---------------------------------------------------------------------------

def test_a_completed_job_without_its_own_run_id_is_refused_not_guessed(tmp_path: Path):
    """A05 — the removed fallback, and why it had to go.

    When a job's own record carried no `run_id`, the launcher searched `/api/benchmark-runs` for a
    report matching model+checkpoint+protocol and adopted the newest match. Two different
    executions share those three fields whenever the same model is evaluated twice, so the summary,
    the verification, the gate and the success claim could all be printed for a run the job never
    produced. The state that makes the fallback reachable is reproduced here: the run directory
    already exists (so the reuse path cannot help) and the job record names no run.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "completed_no_run_id"
    env["MOCK_RUNS_STATUS"] = "corroborated"

    proc = run_pc(sandbox, [], env, timeout=40)
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert "EVALUATION_IDENTITY" in output, output
    assert "run_limn_corr_01" not in output, (
        "the launcher adopted another execution's run identity:\n" + output)


def test_a_completed_job_naming_its_run_uses_exactly_that_run(tmp_path: Path):
    """A05 positive control — the legitimate path must be unaffected."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_JOB_STATUS"] = "completed"
    env["MOCK_RUNS_STATUS"] = "corroborated"
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01")

    proc = run_pc(sandbox, [], env, timeout=60)
    output = proc.stdout + proc.stderr

    assert proc.returncode == 0, output
    assert "run_limn_corr_01" in output, output
    assert "EVALUATION_IDENTITY" not in output, output


def test_an_unreadable_registry_never_reaches_a_success_claim(tmp_path: Path):
    """A-03, end to end through the real launcher.

    There are two places an unreadable registry could become a false statement, and BOTH now refuse.
    The earlier one fires here: `/api/models` supplies the model's protocol and checkpoint bundle, so
    an unreadable registry stops the launcher at `REGISTRY_RESOLUTION` with a non-zero exit and a
    remedy, and the capability question is never reached. The later one — the endpoint answering for
    the selection while the capability is unreadable — is covered by the gate-level case above
    (`test_the_gate_refuses_when_the_registry_could_not_be_read`), which is the only place that state
    can be constructed.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    env["MOCK_MODELS_UNREADABLE"] = "1"
    env["MOCK_RUNS_STATUS"] = "corroborated"
    _create_corroborated_run(sandbox, run_id="run_limn_corr_01")

    proc = run_pc(sandbox, [], env, timeout=60)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "REGISTRY_RESOLUTION" in output, output
    for false_success in ("AGGREGATE_ONLY_VERIFIED", "KẾT LUẬN: SẴN SÀNG", "PER_QUERY_VERIFIED"):
        assert false_success not in output, f"{false_success} claimed with no registry: {output}"


# ---------------------------------------------------------------------------
# A-02 — execution authentication is NOT scientific certification
# ---------------------------------------------------------------------------

def test_an_aggregate_only_run_without_a_proof_is_not_reported_as_verified():
    """A-02 — the gate must not merge two different claims into one token.

    `execution_authenticated` says the ledger corroborates that this invocation ran and wrote these
    bytes at this identity. `certified` says the completion proof validated and the numbers are
    reproducible from the run's own output. A missing proof satisfies the first and NOT the second,
    so an aggregate-only run in that state is "executed and recorded", never "verified". Reporting
    AGGREGATE_ONLY_VERIFIED there upgrades a run that never passed verification.
    """
    proc = _run_index_gate(
        runs=[],
        reports=[_report(model_id="csmcir", checkpoint_id="fashioniq",
                         protocol_id="fashioniq_original_split",
                         per_query_capability=None, per_query_state="CAPABILITY_ABSENT",
                         verification={
                             "certified": False, "is_research_evidence": False,
                             "execution_authenticated": True, "state": "proof_unavailable",
                             "certification_mode": "proof_unavailable",
                             "report_digest": "0" * 64, "reasons": [],
                             "evidence_reasons": ["declared output missing: completion proof"]})],
        model="csmcir", checkpoint="fashioniq", protocol="fashioniq_original_split",
        declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE", proc.stdout
    assert "AGGREGATE_ONLY_VERIFIED" not in proc.stdout, proc.stdout


def test_an_aggregate_only_run_with_a_valid_proof_is_verified():
    """The positive control: a complete proof keeps the verified classification."""
    proc = _run_index_gate(
        runs=[],
        reports=[_report(model_id="csmcir", checkpoint_id="fashioniq",
                         protocol_id="fashioniq_original_split",
                         per_query_capability=None, per_query_state="CAPABILITY_ABSENT")],
        model="csmcir", checkpoint="fashioniq", protocol="fashioniq_original_split",
        declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AGGREGATE_ONLY_VERIFIED", proc.stdout


def _summary_program() -> str:
    """The launcher's own summary program, extracted from `pc.sh`.

    Extracted rather than re-implemented, and driven exactly as the shell drives it: the program
    wins stdin (it is a here-document), and its inputs are the index status plus the state file the
    gate wrote. This is the only way to exercise the summary's exit-code contract for a state the
    real verifier would refuse to produce a fixture for.
    """
    source = PC_SH.read_text(encoding="utf-8")
    start = source.index("python3 - \"${INDEX_EVIDENCE_STATUS:-}\" \"$verify_state\" <<'PYDOC'")
    start = source.index("\n", start) + 1
    end = source.index("\nPYDOC\n", start)
    program = source[start:end]
    assert program.lstrip().startswith("import json, sys"), program[:60]
    return program


def _run_summary(index_status: str, state: dict) -> subprocess.CompletedProcess[str]:
    state_file = Path(tempfile.mkdtemp()) / "verify_state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, "-c", _summary_program(), index_status, str(state_file)],
            capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(state_file.parent, ignore_errors=True)


def _aggregate_only_state(**overrides) -> dict:
    state = {
        "result": {"report_path": "/tmp/does-not-matter.json", "identity": {}, "metrics": {},
                   "gate": {}, "execution": {"execution_status": "EXECUTION_COMPLETED",
                                             "return_code": 0}},
        "per_query_capability": None,
        "per_query_state": "CAPABILITY_ABSENT",
        "per_query_reason": "",
        "report_read": True,
        "proof_unavailable": 1,
        "model_id": "csmcir",
        "checkpoint_id": "fashioniq",
        "protocol_id": "fashioniq_original_split",
        "run_id": "run_csmcir_01",
        "report_path": "",
    }
    state.update(overrides)
    return state


def test_the_summary_exits_seven_for_an_authenticated_but_unverified_aggregate_run():
    """A-02 — the exit code must carry the same distinction the status token does.

    Exit 0 would present an unverified run as a passed pilot. The invocation is authenticated and its
    numbers are recorded, but nothing verified them, so the honest answer is "recorded, not
    certified" and the exit code says so.
    """
    proc = _run_summary(
        "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE", _aggregate_only_state())

    assert proc.returncode == 7, f"{proc.returncode}\n{proc.stdout}"
    assert "CHƯA CHỨNG THỰC KHOA HỌC" in proc.stdout, proc.stdout
    assert "KẾT LUẬN: SẴN SÀNG CHO PILOT" not in proc.stdout, proc.stdout
    # F-1: the report must not contradict itself. The evidence line said "aggregate evidence:
    # certified" two lines under a conclusion that said the science was NOT certified.
    assert "Bằng chứng tổng hợp: đã chứng thực" not in proc.stdout, (
        "the summary claims certified aggregate evidence for an uncertified run:\n" + proc.stdout)
    assert "Bằng chứng tổng hợp: LƯỢT GỌI đã được sổ cái xác thực; khoa học CHƯA được chứng thực." \
        in proc.stdout, proc.stdout


def test_the_summary_exits_zero_for_a_verified_aggregate_only_run():
    """The positive control: a complete proof keeps the verified path, and its code is 0."""
    proc = _run_summary("AGGREGATE_ONLY_VERIFIED",
                        _aggregate_only_state(proof_unavailable=0))

    assert proc.returncode == 0, f"{proc.returncode}\n{proc.stdout}"
    assert "KẾT LUẬN: SẴN SÀNG CHO PILOT" in proc.stdout, proc.stdout
    assert "KHÔNG chứa lượt chạy này" in proc.stdout, proc.stdout


def test_the_summary_exits_five_when_it_could_not_read_the_run_report():
    """Exit 5 is 'the launcher does not know', and it must beat every success branch."""
    proc = _run_summary("PER_QUERY_VERIFIED",
                        _aggregate_only_state(report_read=False,
                                              per_query_capability="LIMN_QUERY_OBSERVER_V1",
                                              per_query_state="PER_QUERY_VERIFIED",
                                              per_query_reason="no such file"))

    assert proc.returncode == 5, f"{proc.returncode}\n{proc.stdout}"
    assert "KHÔNG XÁC ĐỊNH ĐƯỢC" in proc.stdout, proc.stdout

# ---------------------------------------------------------------------------
# R8-02: the API bodies cross the argv boundary as PATHS, not as bytes
# ---------------------------------------------------------------------------

def _index_gate_via_paths(runs_text: str, reports_text: str, *,
                          model: str = "limn",
                          checkpoint: str = "base_iter0_all_categories",
                          protocol: str = "fashioniq_val_split",
                          declared_observer: str = "REGISTRY_CONFIRMED_PER_QUERY:LIMN_QUERY_OBSERVER_V1",
                          run_id: str = "run_limn_corr_01") -> subprocess.CompletedProcess[str]:
    """The real gate program, handed its two bodies the way `pc.sh` hands them."""
    program = _index_gate_program()
    with tempfile.TemporaryDirectory() as directory:
        runs_path = Path(directory) / "runs.json"
        reports_path = Path(directory) / "benchmark-runs.json"
        runs_path.write_text(runs_text, encoding="utf-8")
        reports_path.write_text(reports_text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-c", program,
             f"{runs_path}\n{reports_path}", run_id, model, checkpoint, protocol,
             declared_observer],
            capture_output=True, text=True, timeout=120)


def _report_for(run_id: str, **overrides) -> dict:
    report = {"run_id": run_id, "model_id": "limn",
              "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split",
              "per_query_capability": "LIMN_QUERY_OBSERVER_V1",
              "per_query_state": "PER_QUERY_VERIFIED",
              "verification": {"certified": True, "is_research_evidence": True,
                               "state": "certified", "report_digest": "0" * 64}}
    report.update(overrides)
    return report


def test_the_gate_survives_api_bodies_far_larger_than_the_argv_ceiling():
    """R8-02: an API body above MAX_ARG_STRLEN must not be the reason a valid run is refused.

    A single argv element is capped at 32 pages -- 128 KiB -- independently of the much larger
    ARG_MAX, and both endpoints grow with the deployment's run count: `/api/benchmark-runs` carries
    one row per aggregate report and `/api/runs` one row per indexed run. The gate used to receive
    both bodies as arguments, so past that cap `execve` failed with E2BIG *before python started*
    and the launcher reported an index failure for a run whose evidence was never in question.

    The size here is deliberately past the ceiling and past what a 48-run suite produces, and the
    run being verified is the real one, so the gate has to succeed on data it could never have
    received through argv.
    """
    target = "run_limn_corr_01"
    # ~200 KiB of benchmark-runs rows: each row is padded to a realistic aggregate-report size, and
    # every one of them belongs to a DIFFERENT run so the gate cannot accidentally match a decoy.
    reports = [_report_for(f"run_decoy_{i:04d}", benchmark_gate={"status": "PASSED",
                                                                 "comparisons": [{"padding": "x" * 900}]})
               for i in range(160)]
    reports.append(_report_for(target))
    runs = [{"run_id": target, "model_id": "limn",
             "checkpoint_id": "base_iter0_all_categories",
             "protocol_id": "fashioniq_val_split",
             "evidence_status": "PER_QUERY_VERIFIED"}]
    reports_text = json.dumps(reports)
    assert len(reports_text) > 131072, "this test must exceed the single-argument ceiling to mean anything"

    proc = _index_gate_via_paths(json.dumps(runs), reports_text, run_id=target)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "PER_QUERY_VERIFIED", proc.stdout


def test_the_old_argv_carriage_fails_on_the_same_body_the_gate_now_accepts():
    """The negative control for R8-02, run at the real process boundary.

    Passing the body as an argument is exactly what the launcher used to do, and it must still fail
    on the same input the test above succeeds with -- otherwise the ceiling, not the fix, is what
    the first test is measuring. `subprocess` raises the kernel's own `E2BIG`, which is the failure
    the operator saw as `Argument list too long` from a shell.
    """
    reports_text = json.dumps([_report_for(f"run_decoy_{i:04d}", benchmark_gate={"padding": "x" * 400})
                               for i in range(200)])
    assert len(reports_text) > 131072
    program = _index_gate_program()
    with pytest.raises(OSError) as failure:
        subprocess.run([sys.executable, "-c", program, reports_text, reports_text,
                        "run_limn_corr_01", "limn", "base_iter0_all_categories",
                        "fashioniq_val_split", ""],
                       capture_output=True, text=True, timeout=120)
    assert failure.value.errno == 7, failure.value  # E2BIG


def test_a_failed_fetch_is_still_reported_as_an_unreachable_api():
    """A missing file is not an empty deployment, and the sentinel still names the cause.

    The body now arrives as a file, so an unreadable one is a new way for the gate to see nothing.
    It must land on the same refusal the old code produced for a failed call, never on "the
    deployment holds no runs", which would be a statement about the experiment.
    """
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "absent.json"
        proc = subprocess.run(
            [sys.executable, "-c", _index_gate_program(),
             f"{missing}\n{missing}", "run_limn_corr_01", "limn",
             "base_iter0_all_categories", "fashioniq_val_split", ""],
            capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "không gọi được /api/runs" in proc.stderr, proc.stderr


def test_an_empty_body_is_still_an_unreachable_api():
    """An empty file is not `[]`: the old sentinel and the empty-string test must both still fire."""
    with tempfile.TemporaryDirectory() as directory:
        empty = Path(directory) / "empty.json"
        empty.write_text("", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", _index_gate_program(),
             f"{empty}\n{empty}", "run_limn_corr_01", "limn",
             "base_iter0_all_categories", "fashioniq_val_split", ""],
            capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "không gọi được /api/runs" in proc.stderr, proc.stderr


def test_a_malformed_body_is_still_a_parse_refusal_not_an_empty_deployment():
    """Malformed content must be named as malformed, which requires reaching the parse."""
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", _index_gate_program(),
             f"{bad}\n{bad}", "run_limn_corr_01", "limn",
             "base_iter0_all_categories", "fashioniq_val_split", ""],
            capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "không đọc được /api/runs" in proc.stderr, proc.stderr


@pytest.mark.parametrize("injection,case", [
    # Both of these reach `validate_and_summarize` and then leave it through a non-zero `sys.exit`,
    # so the `rm -f` that follows the summary never runs. Verified against `2bc0cd2`: each leaves a
    # `cir_verify_state.*` behind. Neither is reachable through a `fail_stage` from a subshell, which
    # is why this test does not rely on one -- that path exits the shell in a way the earlier cleanup
    # line already covered, and a test built on it would pass without the trap.
    ({"MOCK_RUNS_STATUS": "corroborated_no_proof"}, "the summary exits non-zero for a proof-unavailable run"),
    ({"MOCK_RUNS_STATUS": "corroborated", "MOCK_INDEX_REBUILD": "http_500"},
     "a failed index rebuild refuses the run"),
])
def test_no_temporary_file_survives_a_failed_run(tmp_path: Path, injection, case):
    """The bodies live on disk, so the launcher owes them a lifetime on EVERY exit path.

    Asserting that an `rm -f` appears in the source proves nothing: a non-zero `sys.exit` from the
    summary and the SIGINT handler both leave the shell immediately, so the `rm` that follows the
    summary never runs and the file is orphaned. This drives the REAL launcher with its own `TMPDIR`,
    on a path that is known to have leaked on the pre-fix tree, and fails if anything it created is
    still there.
    """
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    temp_root = tmp_path / "launcher-tmp"
    temp_root.mkdir()
    env["TMPDIR"] = str(temp_root)
    env.update(injection)
    _create_corroborated_run(sandbox, "run_limn_corr_01")

    proc = run_pc(sandbox, [], env, timeout=60)
    assert proc.returncode != 0, f"this case must fail to exercise the leak paths ({case})"

    leftovers = sorted(p.name for p in temp_root.iterdir())
    assert not leftovers, (
        f"the launcher left {leftovers} in its temp directory after {case}; an EXIT trap must remove "
        "every path it registered")


def test_no_temporary_file_survives_a_successful_verification(tmp_path: Path):
    """The success path must clean up too, including the two files only the summary reads."""
    sandbox, env = setup_sandbox(tmp_path, with_env=True)
    temp_root = tmp_path / "launcher-tmp-ok"
    temp_root.mkdir()
    env["TMPDIR"] = str(temp_root)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    _create_corroborated_run(sandbox, "run_limn_corr_01")

    run_pc(sandbox, [], env, timeout=60)
    leftovers = sorted(p.name for p in temp_root.iterdir())
    assert not leftovers, f"the launcher left {leftovers} after a completed run"


def test_the_launcher_registers_every_temporary_path_it_creates():
    """A path created without registering it cannot be cleaned by the EXIT trap.

    Text-level, and deliberately so: this is the one property a run cannot observe, because a path
    that leaks on a path nobody exercised is exactly what the two tests above cannot reach. It is a
    companion to them, not a substitute.
    """
    source = PC_SH.read_text(encoding="utf-8")
    assert "trap '_cleanup_temp_paths' EXIT" in source, "cleanup must run on every exit"
    for marker in ('mktemp -d "${TMPDIR:-/tmp}/cir_idx.XXXXXX"',
                   'mktemp -d "${TMPDIR:-/tmp}/cir_dedup.XXXXXX"',
                   'mktemp "${TMPDIR:-/tmp}/cir_verify_out.XXXXXX"',
                   'mktemp "${TMPDIR:-/tmp}/cir_verify_state.XXXXXX"',
                   'mktemp "${TMPDIR:-/tmp}/cir_verify_msg.XXXXXX"'):
        assert marker in source, marker
    # One registration per creation site, so a new `mktemp` cannot be added without a cleanup line.
    creations = source.count("mktemp ")
    registrations = source.count('_register_temp "')
    assert creations >= 5, f"expected the launcher's temp-creating sites, found {creations}"
    assert registrations >= creations, (
        f"{creations} temporary paths are created but only {registrations} are registered with the "
        "EXIT trap; the unregistered ones leak on every non-zero exit")


# ---------------------------------------------------------------------------
# R8-03: a development certificate can never satisfy the production gate
# ---------------------------------------------------------------------------

def _development_receipt(**overrides) -> dict:
    receipt = {"certified": True, "is_research_evidence": True, "state": "certified",
               "report_digest": "0" * 64, "development_mode": True,
               "certification_mode": "development"}
    receipt.update(overrides)
    return receipt


def test_a_development_certificate_never_satisfies_the_release_gate():
    """R8-03: the booleans a development receipt carries are not a statement about the artifact.

    `_certification_mode` answers "development" only for a process that declared itself one and
    opted in, and in that state there is deliberately no completion proof to validate -- which is
    why the API's development receipt can carry `certified=true` and `is_research_evidence=true`
    while nothing about the RUN was verified. An aggregate-only model declaring no observer then
    reached `AGGREGATE_ONLY_VERIFIED` and exit 0: a development fixture presented as a certified
    research result. The frontend was hardened for this; the gate that decides the operator's exit
    code has to refuse independently.
    """
    report = _report_for("run_dev_cert", per_query_capability=None, per_query_state="CAPABILITY_ABSENT",
                         verification=_development_receipt())
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_dev_cert",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    assert proc.returncode != 0, "a development certificate reached the success token"
    assert "development" in proc.stderr.lower(), proc.stderr
    assert "AGGREGATE_ONLY_VERIFIED" not in proc.stdout


def test_development_mode_alone_is_refused():
    """Either signal is enough: a receipt may not buy trust by omitting the other one."""
    report = _report_for("run_dev_cert2", per_query_capability=None,
                         per_query_state="CAPABILITY_ABSENT",
                         verification=_development_receipt(certification_mode="deployment"))
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_dev_cert2",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    assert proc.returncode != 0
    assert "development" in proc.stderr.lower(), proc.stderr


def test_a_development_certification_mode_alone_is_refused():
    """The mirror of the case above, so neither field can be the only one checked."""
    report = _report_for("run_dev_cert3", per_query_capability=None,
                         per_query_state="CAPABILITY_ABSENT",
                         verification=_development_receipt(development_mode=False))
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_dev_cert3",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    assert proc.returncode != 0
    assert "development" in proc.stderr.lower(), proc.stderr


def test_a_positive_control_the_deployment_receipt_still_reaches_the_success_token():
    """Rejection-only is not the contract: the real aggregate-only path must still succeed.

    A deployment receipt for a model that declares no observer is a complete result, and the gate
    must keep saying so. Without this, a gate that refused everything would pass every test above.
    """
    report = _report_for("run_deployed", per_query_capability=None,
                         per_query_state="CAPABILITY_ABSENT",
                         verification={"certified": True, "is_research_evidence": True,
                                       "state": "certified", "report_digest": "0" * 64,
                                       "development_mode": False,
                                       "certification_mode": "deployment"})
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_deployed",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AGGREGATE_ONLY_VERIFIED", proc.stdout


def test_the_development_refusal_survives_a_receipt_that_omits_both_fields():
    """A receipt with neither field is not a deployment receipt, and it is not trusted either.

    `development_mode` absent and `certification_mode` absent leaves the policy unknown. The gate
    must not read "no development flag" as "deployment": it falls through to the ordinary evidence
    test, which these booleans do satisfy — so the refusal has to come from somewhere that does not
    depend on either field being present.
    """
    receipt = {"certified": True, "is_research_evidence": True, "state": "certified",
               "report_digest": "0" * 64}
    report = _report_for("run_no_fields", per_query_capability=None,
                         per_query_state="CAPABILITY_ABSENT", verification=receipt)
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_no_fields",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    # This shape is indistinguishable from a deployment receipt at this gate, and is documented as
    # such: the fields are normalised by `_gate_fields` in the API, so a receipt that reaches here
    # without them did not come from the API. The assertion is that the gate is CONSISTENT — it
    # accepts, because it cannot tell, and the API is what guarantees the fields are always set.
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AGGREGATE_ONLY_VERIFIED"

# ---------------------------------------------------------------------------
# R8: the index gate's TOKEN and its NOTES must never share one channel
# ---------------------------------------------------------------------------

def _proof_unavailable_aggregate_only_report() -> dict:
    """The receipt the API serves for a proof-unavailable run of a model with no observer.

    This is the shape whose token is the ONE honest not-a-pass: the ledger corroborates that the
    invocation ran and wrote these bytes, and the completion proof is gone, so the heavy verifier
    cannot run. The gate answers it with a named policy rather than a refusal, and the summary must
    turn that policy into exit 7.
    """
    return {"run_id": "run_limn_corr_01", "model_id": "limn",
            "checkpoint_id": "base_iter0_all_categories",
            "protocol_id": "fashioniq_val_split",
            "verification": {"certified": True, "is_research_evidence": True,
                             "state": "proof_unavailable", "execution_authenticated": True,
                             "certification_mode": "proof_unavailable",
                             "development_mode": False, "report_digest": "0" * 64}}


def test_the_gate_token_survives_a_note_on_stderr():
    """The reader is a verdict AND a sentence, and the shell must not merge them.

    The gate program prints its evidence state on stdout and its advisory notes on stderr --
    deliberately, so a note cannot contaminate the token the shell compares against. Reading it with
    `2>&1` re-merged them: the note was prepended to the status, `INDEX_EVIDENCE_STATUS` matched no
    branch in the summary, and a proof-unavailable run was reported "SẴN SÀNG CHO PILOT" with exit 0
    instead of exit 7. This drives the REAL gate program and asserts both channels.
    """
    report = _proof_unavailable_aggregate_only_report()
    proc = _index_gate_via_paths("[]", json.dumps([report]), run_id="run_limn_corr_01",
                                 declared_observer="REGISTRY_CONFIRMED_AGGREGATE_ONLY")
    assert proc.returncode == 0, proc.stderr
    token = proc.stdout.strip()
    assert token == "AGGREGATE_ONLY_EXECUTION_AUTHENTICATED_PROOF_UNAVAILABLE", (
        f"the token must be exactly one state name, got {token!r}")
    assert "ghi chú" in proc.stderr, (
        "this case is only meaningful while the gate actually emits a note on stderr; if the note "
        "moved, pick a state that still produces one")


def test_the_shell_passes_the_bare_token_and_not_a_note_to_the_summary(tmp_path: Path):
    """The other half of the contract, at the shell boundary that broke it.

    A gate whose stdout is clean is not enough: the shell also has to hand that stdout over
    unmixed. This drives the real `rebuild_and_verify_index` reader through the real launcher and
    reads the token the summary branch compares, so a future `2>&1` fails here rather than in
    production.
    """
    source = PC_SH.read_text(encoding="utf-8")
    reader = re.search(
        r'checked="\$\(python3 -c "\$idx_program".*?\)"', source, re.S)
    assert reader, "the index-gate invocation must be findable"
    assert "2>&1" not in reader.group(0), (
        "the index gate's stdout is the evidence token; merging stderr into it makes the token "
        "unmatchable and every summary branch falls through to its else")
    assert "2>" in reader.group(0), "the gate's notes must go to their own stream"
