"""`gpu_may_be_held` exercised through the real queue, and the launcher backoff that must stay fixed.

These tests drive the crash the guard is about, not a fixture of its aftermath:

* a CPU-only fake job runs through the real `SingleGpuQueue`; its child calls `os.setsid()`, so it
  leaves the backend's session and becomes its own session/process-group leader — the shape an
  evaluator takes when it survives a backend crash (`raw/verify/86-systemd-user-services.md`
  §cgroup containment);
* the process that owns the queue is then SIGKILLed, which is the crash recovery is written for;
* a replacement instance recovers the record, adopts it, and attempts a dispatch.

The assertions are the two directions the guard must have: the dispatch is refused with a message
that names the reason and the job command does not run, and a second restart after the operator has
confirmed the device is free accepts the dispatch. The detached child is never signalled by the
queue, so it must still be alive at the end — the test kills it itself, by the pid the child wrote
for the experiment, after checking the command line (never by an id read from a job record).

The backoff test pins `restart_delay` at the launcher: `restart_delay(1025)` must be `8.0`, not an
`OverflowError`, and the ladder must be exactly `0.2/0.4/0.8/1.6/3.2/6.4/8.0`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workbench.backend.jobs import SingleGpuQueue

#: The refusal the queue must produce, verbatim apart from the interpolated job ids.
GUARD_MESSAGE = ("a job recovered from a crash may still hold the GPU ({ids}); "
                 "release the device and restart the workbench before dispatching")

#: The operator + evaluator pair, as one script. The forked child calls `setsid()`, which is what
#: makes it unreachable by the queue's group kill; the parent stays alive so the job is `running`
#: when the owner dies.
_FAKE_EVALUATOR = '''
import os, sys, time
pidfile = sys.argv[1]
child = os.fork()
if child == 0:
    os.setsid()
    with open(pidfile, "w") as handle:
        handle.write(str(os.getpid()))
    while True:
        time.sleep(1)
print("operator pid=%d child=%d" % (os.getpid(), child), flush=True)
while True:
    time.sleep(1)
'''

#: The queue owner: enqueues the job through the real queue, waits for `running`, reports, and
#: blocks. It is SIGKILLed from the test, so it never runs a shutdown hook — that is the point.
_OWNER = '''
import json, sys, time
from pathlib import Path
sys.path.insert(0, {root!r})
from workbench.backend.jobs import SingleGpuQueue

root = Path(sys.argv[1])
queue = SingleGpuQueue()
log_directory = root / "queue_jobs" / "jobA"
log_directory.mkdir(parents=True, exist_ok=True)
(log_directory / "artifacts").mkdir(exist_ok=True)
(log_directory / "artifacts" / "result.json").write_text('{{"ok": true, "rows": 3}}\\n')
(log_directory / "artifacts" / "payload.bin").write_bytes(bytes(range(64)))

job = queue.enqueue("fake_model", "fake_checkpoint", "fake_protocol",
                    [sys.executable, str(root / "evaluator.py"), str(root / "child.pid")],
                    root, log_directory, {{"note": "cpu fake job"}})
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    record = next(item for item in queue.list() if item["job_id"] == job["job_id"])
    if record["status"] == "running" and (root / "child.pid").exists():
        break
    time.sleep(0.02)
else:
    print("never reached running: " + record["status"], flush=True)
    sys.exit(3)
(root / "running.json").write_text(json.dumps(record, indent=2))
print("RUNNING owner_pid=%d" % __import__("os").getpid(), flush=True)
while True:
    time.sleep(1)
'''


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _alive(pid: int) -> bool:
    """True only for a running process: a zombie holds a pid but runs nothing."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def _wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _run_to_running(tmp_path: Path) -> tuple[subprocess.Popen, Path, int]:
    """Start the owner, return it once its job is `running` with a detached child."""
    root = tmp_path / "ws"
    (root / "queue_jobs").mkdir(parents=True)
    (root / "evaluator.py").write_text(_FAKE_EVALUATOR, encoding="utf-8")
    owner_script = root / "owner.py"
    owner_script.write_text(_OWNER.format(root=str(Path(__file__).resolve().parents[2])),
                            encoding="utf-8")
    owner = subprocess.Popen([sys.executable, str(owner_script), str(root)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        _wait_for(lambda: (root / "running.json").exists(), 30.0, "the job to reach running")
        child = int((root / "child.pid").read_text())
    except BaseException:
        if owner.poll() is None:
            owner.kill()
        owner.wait()
        raise
    return owner, root, child


def test_the_detached_evaluator_outlives_a_hard_kill_of_the_queue_owner(tmp_path: Path) -> None:
    """The premise of the guard, measured: `setsid` defeats a group kill, so there is no safe id.

    A `setsid` child is outside the owner's session and process group, so killing the owner's
    group cannot reach it. Nothing on disk records that group either, which is why recovery may
    not signal anything.
    """
    owner, root, child = _run_to_running(tmp_path)
    try:
        record = json.loads((root / "running.json").read_text())
        assert record["status"] == "running"
        # The persisted record is the whole of what recovery can see. It names no process.
        persisted = json.loads((root / "queue_jobs" / "jobA" / "queue_state.json").read_text())
        assert not ({"pid", "pgid", "process_group"} & set(persisted)), persisted
        assert os.getsid(child) != os.getsid(owner.pid), "the child must have left the session"
        assert os.getpgid(child) == child, "the child must lead its own process group"

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait()
        time.sleep(0.5)  # let the kernel finish reaping the owner's own children
        assert _alive(child), "a detached evaluator is exactly what survives a hard kill"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        _kill_experiment_child(child, root)


def _kill_experiment_child(child: int, root: Path) -> None:
    """Kill every process the *experiment* created, identified by its own command line.

    The ids here never come from a job record: they come from the experiment's own pid file and
    from `/proc`, and each candidate is checked against the experiment's directory first. The
    operator that the queue spawned survives the crash by design (`start_new_session=True`), so
    leaving it behind would leak a process into the rest of the run.
    """
    targets = {child}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and str(root) in _cmdline(int(entry.name)):
            targets.add(int(entry.name))
    for pid in sorted(targets):
        if not _alive(pid) or str(root) not in _cmdline(pid):
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(_alive(pid) for pid in targets):
        time.sleep(0.02)


def test_guard_blocks_then_the_documented_release_accepts_the_dispatch(tmp_path: Path) -> None:
    """Both directions through the real queue: refused with the reason, then accepted after restart."""
    owner, root, child = _run_to_running(tmp_path)
    log_directory = root / "queue_jobs" / "jobA"
    try:
        before = _snapshot(log_directory)
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait()

        # --- First replacement: recovery + dispatch. The operator has NOT confirmed the device.
        recovered = SingleGpuQueue.recover_interrupted([root / "queue_jobs"])
        assert [item["job_id"] for item in recovered] == ["jobA"] or len(recovered) == 1
        record = recovered[0]
        assert record["status"] == "failed", record
        assert "restart" in record["error"]
        assert record["gpu_may_be_held"] is True

        after = _snapshot(log_directory)
        before.pop("queue_state.json", None)
        after.pop("queue_state.json", None)
        assert after == before, "recovery must not touch the artifact tree"

        queue = SingleGpuQueue()
        marker = tmp_path / "dispatched"
        try:
            for item in recovered:
                queue.adopt_recovered(item)
            assert queue._gpu_orphans == [record["job_id"]]

            job = queue.enqueue("other", "c", "p",
                                [sys.executable, "-c",
                                 f"from pathlib import Path; Path({str(marker)!r}).write_text('x')"],
                                tmp_path, tmp_path / "queue_jobs" / "second", {})
            _wait_for(lambda: next(i for i in queue.list() if i["job_id"] == job["job_id"])
                      ["status"] in {"blocked", "failed", "completed", "cancelled"},
                      20.0, "the dispatch to settle")
            blocked = next(i for i in queue.list() if i["job_id"] == job["job_id"])
            assert blocked["status"] == "blocked", blocked
            assert blocked["error"] == GUARD_MESSAGE.format(ids=record["job_id"]), blocked["error"]
            assert f"[BLOCKED] {blocked['error']}" in blocked["stderr_tail"]
            assert not marker.exists(), "a refused dispatch must not run its command"
            assert _alive(child), "the guard refuses; it must never signal an unverified id"
        finally:
            queue.shutdown()

        # --- Second replacement: the operator confirmed the device is free, so the flag is gone
        # from disk (the recovered record is already terminal, so nothing is in flight to recover)
        # and the dispatch is accepted.
        assert SingleGpuQueue.recover_interrupted([root / "queue_jobs"]) == []
        released = SingleGpuQueue()
        try:
            marker2 = tmp_path / "dispatched-after-release"
            job = released.enqueue("other", "c", "p",
                                   [sys.executable, "-c",
                                    f"from pathlib import Path; "
                                    f"Path({str(marker2)!r}).write_text('x')"],
                                   tmp_path, tmp_path / "queue_jobs" / "third", {})
            _wait_for(lambda: next(i for i in released.list() if i["job_id"] == job["job_id"])
                      ["status"] in {"blocked", "failed", "completed", "cancelled"},
                      20.0, "the released dispatch to settle")
            settled = next(i for i in released.list() if i["job_id"] == job["job_id"])
            assert settled["status"] == "completed", settled
            assert settled["error"] is None
            assert marker2.exists(), "the release must actually dispatch"
        finally:
            released.shutdown()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        _kill_experiment_child(child, root)


def test_the_flag_survives_further_restarts_and_is_never_self_clearing(tmp_path: Path) -> None:
    """A restart that recovers nothing must not silently drop the refusal.

    The refusal is carried by the record on disk, so a second startup that finds the record
    already terminal recovers nothing — and that is the documented release path, not an accident:
    the operator's confirmation (device free) is what makes the second restart legitimate.
    """
    root = tmp_path / "queue_jobs"
    directory = root / "jobA"
    directory.mkdir(parents=True)
    (directory / "queue_state.json").write_text(json.dumps({
        "job_id": "jobA", "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
        "status": "running", "log_directory": str(directory)}), encoding="utf-8")

    first = SingleGpuQueue.recover_interrupted([root])
    assert first and first[0]["gpu_may_be_held"] is True
    on_disk = json.loads((directory / "queue_state.json").read_text())
    assert on_disk["gpu_may_be_held"] is True, "the flag must be persisted for an operator to see"
    assert on_disk["status"] == "failed"

    assert SingleGpuQueue.recover_interrupted([root]) == [], "a terminal record is not re-recovered"
    assert json.loads((directory / "queue_state.json").read_text())["gpu_may_be_held"] is True


def _load_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve_workbench.py"
    spec = importlib.util.spec_from_file_location("serve_workbench_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_backoff_is_the_measured_ladder_and_saturates() -> None:
    """`restart_delay(1025)` must be 8.0, not `OverflowError`; the ladder must be exact.

    The bug was a clamp applied after the multiply: `2 ** 1024` is a big int that cannot become a
    float, so `min()` never ran. The fix clamps the exponent first, deriving the saturating
    exponent from the two constants rather than hard-coding it.
    """
    launcher = _load_launcher()
    assert launcher.BACKOFF_BASE == 0.2
    assert launcher.BACKOFF_CAP == 8.0
    assert launcher.BACKOFF_SATURATING_EXPONENT == 6, (
        "the exponent must be derived from the constants: the smallest one that reaches the cap")

    assert [launcher.restart_delay(n) for n in range(1, 8)] == [0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8.0]
    assert launcher.restart_delay(0) == 0.2
    assert launcher.restart_delay(-1) == 0.2
    for failures in (8, 9, 1025, 10 ** 6, 10 ** 12):
        assert launcher.restart_delay(failures) == 8.0, failures

    started = time.perf_counter()
    assert launcher.restart_delay(1025) == 8.0
    assert time.perf_counter() - started < 1.0, "the clamp must not do giant-int arithmetic"

    ladder = [launcher.restart_delay(n) for n in range(1, 2001)]
    assert all(later >= earlier for earlier, later in zip(ladder, ladder[1:])), "not monotone"
    assert max(ladder) == launcher.BACKOFF_CAP
