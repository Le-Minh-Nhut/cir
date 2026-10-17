"""Real concurrency: two OS processes contending for one workflow state file."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
import threading


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

# ---------------------------------------------------------- shutdown vs dispatch race
#
# `_work` decides to spawn a process, and `shutdown()` sets `_stopping` and collects the
# process tables, under the same condition. These tests hold the worker inside a *blocking*
# launch guard — the window between "job is preparing" and "process exists" — and start
# shutdown in that window, so a spawn that happens afterwards is unambiguous.

def _job(queue, job_id: str) -> dict:
    return next(item for item in queue.list() if item["job_id"] == job_id)

def _wait_until(predicate, timeout: float = 15.0) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()

def _persisted_job(directory) -> dict:
    return json.loads((Path(directory) / "queue_state.json").read_text(encoding="utf-8"))

def _sentinel_command(path: Path) -> list[str]:
    """A command whose only effect is to prove it was started."""
    return [sys.executable, "-c",
            f"from pathlib import Path; Path({str(path)!r}).write_text('ran')"]

def _blocking_guard(entered, release):
    def guard(job):
        entered.set()
        assert release.wait(timeout=30), "the launch guard was never released"
    return guard

def _shutdown_while_the_guard_blocks(queue, release):
    """Run shutdown while the worker sits in the launch guard, then release the guard.

    Returns ``(survivors, elapsed)``. The guard is released as soon as shutdown has been
    observed to begin, so `elapsed` measures shutdown's own bound — its worker join — and not
    the guard's sleep.
    """
    import threading
    import time

    survivors: list[str] = []
    thread = threading.Thread(target=lambda: survivors.extend(queue.shutdown()))
    started = time.monotonic()
    thread.start()
    assert _wait_until(lambda: queue._stopping, timeout=10), "shutdown never began"
    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive(), "shutdown did not return"
    return survivors, time.monotonic() - started

def test_shutdown_during_a_blocking_launch_guard_never_starts_a_process(tmp_path: Path):
    """A job in `preparing` when shutdown begins must never be dispatched.

    The spawn decision is made under the queue lock, and `shutdown()` sets `_stopping` under
    that same lock, so a shutdown that has begun cannot be followed by a `Popen`. The command
    writes a sentinel, so a process that was started stays visible even though nobody would
    ever reap its group.
    """
    import threading
    import time

    from workbench.backend.jobs import SingleGpuQueue

    sentinel = tmp_path / "evaluator_started"
    queue = SingleGpuQueue()
    entered, release = threading.Event(), threading.Event()
    queue.launch_guard = _blocking_guard(entered, release)
    directory = tmp_path / "queue_jobs" / "preparing"
    queued = queue.enqueue("m", "c", "p", _sentinel_command(sentinel), tmp_path, directory, {})

    assert entered.wait(timeout=15), "the job never reached the launch guard"
    assert _job(queue, queued["job_id"])["status"] == "preparing"

    survivors, elapsed = _shutdown_while_the_guard_blocks(queue, release)
    # Give a mis-timed dispatch every chance to appear before asserting its absence.
    time.sleep(0.5)

    assert survivors == [], f"a group survived shutdown: {survivors}"
    assert elapsed < 10.0, f"shutdown exceeded its worker-join bound: {elapsed:.2f}s"
    assert not sentinel.exists(), "a process was started after shutdown had begun"
    assert queue._processes == {} and queue._process_groups == {}
    assert not queue._worker.is_alive(), "the worker outlived shutdown"

    settled = _job(queue, queued["job_id"])
    assert settled["status"] == "cancelled", settled
    assert settled["status"] != "completed"
    assert _persisted_job(directory)["status"] == "cancelled", _persisted_job(directory)

def test_shutdown_during_a_blocking_launch_guard_cannot_reach_popen(tmp_path: Path, monkeypatch):
    """The spawn itself is refused, not merely followed by a cleanup.

    `subprocess.Popen` is replaced with a recorder, so any call at all is a failure whether or
    not the process would have been reapable afterwards. This is the atomic form of the check
    above: `_stopping` and the spawn decision are read under one lock.
    """
    import threading
    import time

    import workbench.backend.jobs as jobs_module
    from workbench.backend.jobs import SingleGpuQueue

    dispatched: list[list[str]] = []

    def refusing_popen(*args, **kwargs):
        dispatched.append(list(args))
        raise RuntimeError("Popen was called despite the queue having begun shutdown")

    # The stub carries the attributes a real dispatch would touch (`PIPE` is read for the
    # child's streams), so a genuine spawn attempt reaches the recorder instead of dying on an
    # attribute error — which would leave `dispatched` empty and pass vacuously.
    monkeypatch.setattr(jobs_module, "subprocess",
                        type("S", (), {"PIPE": -1, "STDOUT": -2, "DEVNULL": -3,
                                       "Popen": staticmethod(refusing_popen)})())

    queue = SingleGpuQueue()
    entered, release = threading.Event(), threading.Event()
    queue.launch_guard = _blocking_guard(entered, release)
    queued = queue.enqueue("m", "c", "p", ["gpu-evaluator"], tmp_path,
                           tmp_path / "queue_jobs" / "popen", {})
    assert entered.wait(timeout=15), "the job never reached the launch guard"

    _shutdown_while_the_guard_blocks(queue, release)
    time.sleep(0.5)

    assert dispatched == [], "a process was created after shutdown had begun"
    settled = _job(queue, queued["job_id"])
    assert settled["status"] == "cancelled", settled
    assert settled["status"] != "completed"

def test_shutdown_settles_a_queued_job_it_never_started(tmp_path: Path):
    """A job still waiting behind another must settle honestly instead of staying `queued`."""
    import threading

    from workbench.backend.jobs import SingleGpuQueue

    running_sentinel = tmp_path / "first_started"
    queued_sentinel = tmp_path / "second_started"
    queue = SingleGpuQueue()
    entered, release = threading.Event(), threading.Event()
    queue.launch_guard = _blocking_guard(entered, release)
    first = queue.enqueue("m", "c", "p", _sentinel_command(running_sentinel), tmp_path,
                          tmp_path / "queue_jobs" / "first", {})
    assert entered.wait(timeout=15), "the first job never reached the launch guard"
    second_directory = tmp_path / "queue_jobs" / "second"
    second = queue.enqueue("m", "c", "p", _sentinel_command(queued_sentinel), tmp_path,
                           second_directory, {})
    assert _job(queue, second["job_id"])["status"] == "queued"

    _shutdown_while_the_guard_blocks(queue, release)
    assert _wait_until(lambda: _job(queue, second["job_id"])["status"] != "queued", timeout=5)

    for queued in (first, second):
        settled = _job(queue, queued["job_id"])
        assert settled["status"] == "cancelled", settled
        assert settled["status"] != "completed"
        assert _persisted_job(queued["log_directory"])["status"] == "cancelled"
    assert not running_sentinel.exists() and not queued_sentinel.exists()

def test_shutdown_cancels_a_running_job_and_never_reports_completion(tmp_path: Path):
    """A job mid-flight when shutdown begins: group killed, state terminal and not `completed`."""
    import time

    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 2.0
    directory = tmp_path / "queue_jobs" / "running"
    queued = queue.enqueue("m", "c", "p", [sys.executable, "-c", "import time; time.sleep(60)"],
                           tmp_path, directory, {})
    assert _wait_until(lambda: _job(queue, queued["job_id"])["status"] == "running"), queue.list()
    pid = queue._processes[queued["job_id"]].pid

    survivors = queue.shutdown()

    assert survivors == [], f"a group survived shutdown: {survivors}"
    assert _wait_until(lambda: not _alive(pid), timeout=5), "the operator outlived shutdown"
    settled = _job(queue, queued["job_id"])
    assert settled["status"] == "cancelled", settled
    assert settled["status"] != "completed"
    assert _persisted_job(directory)["status"] == "cancelled"
    assert queue._processes == {} and queue._process_groups == {}

def test_shutdown_leaves_a_completed_job_completed(tmp_path: Path):
    """A job that already settled must not be rewritten by a later shutdown."""
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "done"
    queued = queue.enqueue("m", "c", "p", [sys.executable, "-c", "print('evidence')"],
                           tmp_path, directory, {"model_id": "m"})
    assert _wait_until(lambda: _job(queue, queued["job_id"])["status"] == "completed"), queue.list()

    assert queue.shutdown() == []
    settled = _job(queue, queued["job_id"])
    assert settled["status"] == "completed", settled
    assert _persisted_job(directory)["status"] == "completed"
    assert "[STOP]" not in settled["stderr_tail"]


def test_enqueue_after_shutdown_refuses_explicitly_and_never_leaves_job_queued(tmp_path: Path):
    """F7: enqueue after shutdown raises QueueShutdownError and leaves no queued job."""
    from workbench.backend.jobs import SingleGpuQueue, QueueShutdownError
    import pytest

    queue = SingleGpuQueue()
    assert queue.shutdown() == []
    assert not queue._worker.is_alive()

    directory = tmp_path / "queue_jobs" / "post_shutdown"
    with pytest.raises(QueueShutdownError) as exc_info:
        queue.enqueue("m", "c", "p", [sys.executable, "-c", "pass"], tmp_path, directory, {})

    assert "shut down" in str(exc_info.value)
    assert not any(j["status"] == "queued" for j in queue.list())
    assert not (directory / "queue_state.json").exists()


def test_shutdown_settled_job_persists_shutdown_cause_distinct_from_operator_cancellation(tmp_path: Path):
    """F6: shutdown records the shutdown cause in persisted queue_state.json, distinct from cancel()."""
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    entered = threading.Event()
    release = threading.Event()
    queue.launch_guard = _blocking_guard(entered, release)

    blocker_dir = tmp_path / "queue_jobs" / "blocker"
    queue.enqueue("m", "c", "p", [sys.executable, "-c", "pass"], tmp_path, blocker_dir, {})
    assert entered.wait(timeout=15), "blocker never reached launch guard"

    op_cancel_dir = tmp_path / "queue_jobs" / "operator_cancel"
    op_job = queue.enqueue("m", "c", "p", [sys.executable, "-c", "pass"], tmp_path, op_cancel_dir, {})
    queue.cancel(op_job["job_id"])

    shutdown_job_dir = tmp_path / "queue_jobs" / "shutdown_settled"
    queue.enqueue("m", "c", "p", [sys.executable, "-c", "pass"], tmp_path, shutdown_job_dir, {})

    _shutdown_while_the_guard_blocks(queue, release)

    op_persisted = _persisted_job(op_cancel_dir)
    shutdown_persisted = _persisted_job(shutdown_job_dir)

    assert op_persisted["status"] == "cancelled"
    assert op_persisted.get("error") is None, f"operator cancellation must not claim shutdown: {op_persisted}"

    assert shutdown_persisted["status"] == "cancelled"
    assert shutdown_persisted.get("error") == "workbench shut down before dispatch"


def test_legitimate_queued_job_runs_to_completion_without_shutdown(tmp_path: Path):
    """F7 regression guard: normal queue processing continues when no shutdown is active."""
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "healthy_run"
    marker = tmp_path / "marker.txt"
    queued = queue.enqueue("m", "c", "p",
                           [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran'); print('evidence')"],
                           tmp_path, directory, {"model_id": "m"})
    assert _wait_until(lambda: _job(queue, queued["job_id"])["status"] == "completed"), queue.list()
    settled = _job(queue, queued["job_id"])
    assert settled["status"] == "completed"
    assert settled["return_code"] == 0
    assert settled["error"] is None
    assert marker.read_text() == "ran"
    assert _persisted_job(directory)["status"] == "completed"


def test_queue_command_json_is_durable_and_valid_the_moment_job_is_terminal(tmp_path: Path):
    """F8: observing terminal status guarantees queue_command.json is valid and has return_code."""
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "atomic_dispatch"
    queued = queue.enqueue("m", "c", "p",
                           [sys.executable, "-c", "print('dispatch record ready')"],
                           tmp_path, directory, {"model_id": "m"})
    assert _wait_until(lambda: _job(queue, queued["job_id"])["status"] == "completed"), queue.list()
    command_path = directory / "queue_command.json"
    assert command_path.is_file()
    data = json.loads(command_path.read_text(encoding="utf-8"))
    assert data["return_code"] == 0
    assert data["finished_at"] is not None
    assert data["job_id"] == queued["job_id"]
