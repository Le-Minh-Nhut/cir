"""Real concurrency: two OS processes contending for one workflow state file."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def _child_script(state_path: Path, ready: Path, hold: float) -> str:
    return textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(REPOSITORY)!r})
        from pathlib import Path
        import workbench.scripts.pipeline as pipeline

        state = Path({str(state_path)!r})
        ready = Path({str(ready)!r})
        try:
            handle = pipeline.acquire_workflow_lock(state)
        except pipeline.WorkflowLockError:
            print("LOCK_DENIED")
            sys.exit(3)
        data = pipeline.load_pipeline_state(state)
        data.setdefault("stages", {{}}).setdefault("child", {{}})
        data["stages"]["child"]["pid"] = __import__("os").getpid()
        pipeline.save_pipeline_state(data, state)
        ready.write_text("held")
        time.sleep({hold})
        pipeline.release_workflow_lock(handle)
        print("OK")
    """)


def test_two_processes_contend_for_same_state(tmp_path: Path):
    state = tmp_path / "state.json"
    ready = tmp_path / "ready"
    holder = subprocess.Popen([sys.executable, "-c", _child_script(state, ready, 2.0)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait until the holder owns the lock.
        for _ in range(100):
            if ready.exists():
                break
            import time
            time.sleep(0.05)
        assert ready.exists(), "holder never acquired the lock"

        # Second process must be refused, not silently overwrite.
        contender = subprocess.run([sys.executable, "-c", _child_script(state, tmp_path / "ready2", 0.0)],
                                   capture_output=True, text=True, timeout=30)
        assert contender.returncode == 3, f"contender was not refused: {contender.stdout} {contender.stderr}"
        assert "LOCK_DENIED" in contender.stdout
    finally:
        holder.wait(timeout=30)

    assert holder.returncode == 0
    data = json.loads(state.read_text())
    assert "child" in data["stages"]


def test_lock_released_after_process_exit(tmp_path: Path):
    state = tmp_path / "state.json"
    ready = tmp_path / "ready"
    first = subprocess.run([sys.executable, "-c", _child_script(state, ready, 0.0)],
                           capture_output=True, text=True, timeout=30)
    assert first.returncode == 0
    # After the holder exits, a new process must acquire the lock immediately.
    second = subprocess.run([sys.executable, "-c", _child_script(state, tmp_path / "ready3", 0.0)],
                            capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, f"lock not recovered after exit: {second.stdout} {second.stderr}"


def test_lock_released_after_crash(tmp_path: Path):
    """A killed process must not leave a permanently held lock (flock auto-release)."""
    state = tmp_path / "state.json"
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(REPOSITORY)!r})
        from pathlib import Path
        import workbench.scripts.pipeline as pipeline
        handle = pipeline.acquire_workflow_lock(Path({str(state)!r}))
        Path({str(tmp_path / 'crashed_ready')!r}).write_text("held")
        time.sleep(60)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import time
    for _ in range(100):
        if (tmp_path / "crashed_ready").exists():
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=30)

    follower = subprocess.run([sys.executable, "-c", _child_script(state, tmp_path / "ready4", 0.0)],
                              capture_output=True, text=True, timeout=30)
    assert follower.returncode == 0, f"stale lock survived a killed process: {follower.stdout} {follower.stderr}"


def test_independent_workspaces_do_not_block(tmp_path: Path):
    a = tmp_path / "a" / "state.json"
    b = tmp_path / "b" / "state.json"
    ra = tmp_path / "ra"
    rb = tmp_path / "rb"
    pa = subprocess.Popen([sys.executable, "-c", _child_script(a, ra, 1.0)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pb = subprocess.Popen([sys.executable, "-c", _child_script(b, rb, 0.0)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out_b, err_b = pb.communicate(timeout=30)
    assert pb.returncode == 0, f"independent workspace blocked: {out_b} {err_b}"
    pa.communicate(timeout=30)
    assert pa.returncode == 0

# ------------------------------------------------------------------------- shutdown ownership

def _alive(pid: int) -> bool:
    """Liveness from the kernel, so a reaped child cannot be mistaken for a running one."""
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def test_shutdown_terminates_every_live_job_process_group(tmp_path: Path):
    """A job leads its own session (`start_new_session=True`), so the SIGTERM that `docker stop`
    sends to the backend never reaches it: without a shutdown path the evaluator child survived
    the process and kept the GPU. This spawns a real operator with a real sleeping grandchild —
    neither ignores SIGTERM — and asserts the whole group is gone after shutdown() returns."""
    import os
    import sys
    import time

    from workbench.backend.jobs import SingleGpuQueue

    script = tmp_path / "operator_process.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(tmp_path / 'grandchild.pid')!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n")
    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 2.0
    job = queue.enqueue("m", "c", "p", [sys.executable, str(script)], tmp_path,
                        tmp_path / "queue_jobs" / "shutdown", {})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (tmp_path / "grandchild.pid").exists():
        time.sleep(0.02)
    assert (tmp_path / "grandchild.pid").is_file(), "fixture never started"
    # The pid file appears before the operator reports `running`; wait for the dispatch itself.
    while time.monotonic() < deadline:
        status = next(item["status"] for item in queue.list() if item["job_id"] == job["job_id"])
        if status == "running":
            break
        time.sleep(0.02)
    assert status == "running", status
    grandchild = int((tmp_path / "grandchild.pid").read_text())
    group = queue._process_groups[job["job_id"]]
    assert _alive(grandchild), "the fixture's grandchild is not running"

    survivors = queue.shutdown()

    assert survivors == [], f"a group survived shutdown: {survivors}"
    assert not _alive(grandchild), "the evaluator grandchild outlived the backend shutdown"
    try:
        os.killpg(group, 0)
        alive_group = True
    except ProcessLookupError:
        alive_group = False
    assert not alive_group, "the job's process group still has a member"
    record = next(item for item in queue.list() if item["job_id"] == job["job_id"])
    assert record["status"] == "cancelled", record

def test_shutdown_stops_the_worker_thread(tmp_path: Path):
    """The worker thread waits on the queue condition forever; shutdown must wake it, or the
    process it is joined from hangs instead of exiting."""
    import time

    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    assert queue._worker.is_alive()
    assert queue.shutdown() == []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and queue._worker.is_alive():
        time.sleep(0.05)
    assert not queue._worker.is_alive(), "the worker thread survived shutdown"
