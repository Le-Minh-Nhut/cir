"""Cancellation must stop the whole managed process group, not just the queue's direct child.

The queue's direct child is the trusted operator; the process that actually holds the GPU is the
official evaluator it starts. A cancellation that only reaps the operator leaves that evaluator
running, which is why these tests assert on real /proc liveness rather than on `job.status`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path



from workbench.backend.jobs import SingleGpuQueue


def _wait_status(queue: SingleGpuQueue, job_id: str, status: str, *, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(item for item in queue.list() if item["job_id"] == job_id)
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never reached {status}: {queue.list()}")


def _alive(pid: int) -> bool:
    """Liveness from the kernel, so a reaped child cannot be mistaken for a running one."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


_SLEEPING_GRANDCHILD = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, {handler})\n"
    "time.sleep({seconds})\n"
    "open({sentinel!r}, 'w').write('ran')\n"
)


def _operator(tmp_path: Path, *, ignore_sigterm: bool, hold: float,
              seconds: float, sentinel: Path) -> Path:
    """An 'operator' that starts a sleeping 'evaluator' grandchild and stays alive for `hold`.

    The grandchild is the process that matters: it is deliberately non-cooperative when
    `ignore_sigterm` is set, so only a group-wide SIGKILL can stop it. The operator stays alive
    long enough for the cancellation to arrive while the job is genuinely running.
    """
    handler = "signal.SIG_IGN" if ignore_sigterm else "signal.SIG_DFL"
    grandchild = _SLEEPING_GRANDCHILD.format(handler=handler, seconds=seconds,
                                             sentinel=str(sentinel))
    script = tmp_path / "operator_process.py"
    script.write_text(
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"open({str(tmp_path / 'grandchild.pid')!r}, 'w').write(str(child.pid))\n"
        f"time.sleep({hold})\n"
        "sys.exit(0)\n")
    return script


def _start(queue: SingleGpuQueue, tmp_path: Path, script: Path) -> str:
    job = queue.enqueue("m", "c", "p", [os.sys.executable, str(script)], tmp_path,
                        tmp_path / "queue_jobs" / "cancel", {})
    _wait_status(queue, job["job_id"], "running")
    # Let the grandchild exist and record its pid before the cancellation lands.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (tmp_path / "grandchild.pid").exists():
        time.sleep(0.02)
    time.sleep(0.4)
    return job["job_id"]


def test_cancellation_kills_a_grandchild_that_ignores_sigterm(tmp_path: Path) -> None:
    """SIGTERM is ignored by both processes, so the escalation must reach the group.

    This is the case the earlier implementation missed: the leader exits by itself, its handle is
    reaped, and a liveness check that consulted only the leader's `poll()` concluded the job was
    already gone — leaving the evaluator alive and the job reported cancelled.
    """
    sentinel = tmp_path / "evaluator_ran_to_completion"
    script = _operator(tmp_path, ignore_sigterm=True, hold=30, seconds=12, sentinel=sentinel)
    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 1.0        # keep the bounded wait short inside a test
    job_id = _start(queue, tmp_path, script)
    grandchild = int((tmp_path / "grandchild.pid").read_text())
    assert _alive(grandchild), "the fixture did not produce a running grandchild"

    queue.cancel(job_id)
    _wait_status(queue, job_id, "cancelled")

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _alive(grandchild):
        time.sleep(0.05)
    assert not _alive(grandchild), "the evaluator grandchild survived cancellation"
    # Long enough that a surviving child would have completed and written its sentinel.
    time.sleep(12.5)
    assert not sentinel.exists()


def test_cancellation_stops_a_grandchild_that_honours_sigterm(tmp_path: Path) -> None:
    """The ordinary case: SIGTERM is enough, and no escalation is needed."""
    sentinel = tmp_path / "evaluator_ran_to_completion"
    script = _operator(tmp_path, ignore_sigterm=False, hold=30, seconds=10, sentinel=sentinel)
    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 1.0
    job_id = _start(queue, tmp_path, script)

    queue.cancel(job_id)
    _wait_status(queue, job_id, "cancelled")
    time.sleep(1.0)
    assert not sentinel.exists()


def test_cancelling_one_job_does_not_touch_its_neighbour(tmp_path: Path, monkeypatch) -> None:
    """Cancellation is scoped to one process group, so another job's work continues."""
    killed: list[int] = []
    real_killpg = os.killpg

    def recording_killpg(group: int, signum: int) -> None:
        killed.append(group)
        real_killpg(group, signum)

    monkeypatch.setattr(os, "killpg", recording_killpg)

    # The queue runs one job at a time, so the witness must complete before the victim starts.
    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 1.0
    witness = queue.enqueue("m", "c", "p", [os.sys.executable, "-c", "print('witness')"],
                            tmp_path, tmp_path / "queue_jobs" / "witness", {})
    _wait_status(queue, witness["job_id"], "completed")
    # The witness's own process was long gone, so nothing about it may be signalled now.
    killed.clear()

    sentinel = tmp_path / "evaluator_ran_to_completion"
    script = _operator(tmp_path, ignore_sigterm=True, hold=30, seconds=10, sentinel=sentinel)
    victim = _start(queue, tmp_path, script)
    victim_group = queue._process_groups[victim]

    queue.cancel(victim)
    _wait_status(queue, victim, "cancelled")
    # Exactly the cancelled job's group was signalled: not the witness's, not the queue's own pid.
    assert killed and set(killed) == {victim_group}, killed
    assert next(item for item in queue.list()
                if item["job_id"] == witness["job_id"])["status"] == "completed"


def test_cancellation_while_queued_starts_nothing(tmp_path: Path) -> None:
    """A queued job cancelled before dispatch never creates a process at all."""
    marker = tmp_path / "ran"
    queue = SingleGpuQueue()
    blocker = queue.enqueue("m", "c", "p",
                            [os.sys.executable, "-c", "import time; time.sleep(1.5)"],
                            tmp_path, tmp_path / "queue_jobs" / "blocker", {})
    victim = queue.enqueue("m", "c", "p",
                           [os.sys.executable, "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                           tmp_path, tmp_path / "queue_jobs" / "victim", {})
    queue.cancel(victim["job_id"])
    _wait_status(queue, blocker["job_id"], "completed")
    assert _wait_status(queue, victim["job_id"], "cancelled")["return_code"] is None
    assert not marker.exists()


def test_a_cancelled_job_never_reports_completion(tmp_path: Path) -> None:
    """The worker must not overwrite `cancelled` with `completed` when the child exits 0."""
    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "race"
    job = queue.enqueue("m", "c", "p", [os.sys.executable, "-c", "import time; time.sleep(2)"],
                        tmp_path, directory, {})
    _wait_status(queue, job["job_id"], "running")
    queue.cancel(job["job_id"])
    # Give the worker ample time to mis-publish if the guard were missing.
    time.sleep(3)
    settled = next(item for item in queue.list() if item["job_id"] == job["job_id"])
    assert settled["status"] == "cancelled", settled


def test_cancellation_does_not_block_status_queries(tmp_path: Path) -> None:
    """The bounded shutdown runs outside the queue lock, so other calls still answer."""
    sentinel = tmp_path / "evaluator_ran_to_completion"
    script = _operator(tmp_path, ignore_sigterm=True, hold=60, seconds=20, sentinel=sentinel)
    queue = SingleGpuQueue()
    queue.GRACE_SECONDS = 3.0
    job_id = _start(queue, tmp_path, script)

    import threading
    started = time.monotonic()
    cancel_thread = threading.Thread(target=queue.cancel, args=(job_id,))
    cancel_thread.start()

    # Poll the list while the cancellation is working; these must return promptly.
    observed = 0
    while cancel_thread.is_alive() and time.monotonic() - started < 20:
        queue.list()
        observed += 1
        time.sleep(0.01)
    cancel_thread.join()
    assert observed > 0, "status queries were blocked by the cancellation"
    assert time.monotonic() - started < 20
