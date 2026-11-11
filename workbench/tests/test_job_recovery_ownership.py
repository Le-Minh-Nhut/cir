"""Startup recovery distinguishes an interrupted job from a finished one, and never kills an id
it cannot prove it owns.

Recovery runs before any HTTP handler exists, so the evidence it can rely on is the persisted
record only: `queue_state.json` says a job was in flight, but nothing on disk records the process
group that was running it. These tests drive the real `SingleGpuQueue` against real filesystem
fixtures and assert the consequences — rewritten state, untouched artifacts, no subprocess, a
foreign process left alone, and a named refusal instead of a blind dispatch when the GPU may
still be held.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workbench.backend.jobs import SingleGpuQueue


def _write_state(root: Path, name: str, payload: object, *, malformed: bool = False) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "queue_state.json"
    path.write_text("{not json at all\n" if malformed else json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8")
    return path


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _record(name: str, status: str, log_directory: Path) -> dict:
    return {"job_id": f"job-{name}", "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
            "status": status, "started_at": "2026-01-01T00:00:00+00:00",
            "log_directory": str(log_directory)}


def test_in_flight_record_becomes_failed_and_artifacts_stay_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "queue_jobs"
    for index, status in enumerate(("queued", "preparing", "running")):
        log_directory = root / f"job{index}"
        state = _write_state(root, f"job{index}", _record(f"job{index}", status, log_directory))
        (log_directory / "artifacts").mkdir(parents=True, exist_ok=True)
        (log_directory / "queue_stdout.log").write_text("evaluator output\n", encoding="utf-8")
        (log_directory / "queue_stderr.log").write_text("noise\n", encoding="utf-8")
        (log_directory / "artifacts" / "result.json").write_text('{"ok": true}\n', encoding="utf-8")
        before = _snapshot(log_directory)

        recovered = SingleGpuQueue.recover_interrupted([root])

        assert [item["job_id"] for item in recovered] == [f"job-job{index}"]
        assert recovered[0]["status"] == "failed"
        assert "restart" in recovered[0]["error"]
        after = _snapshot(log_directory)
        after.pop("queue_state.json")
        before.pop("queue_state.json")
        assert after == before, "recovery must not touch anything but queue_state.json"
        on_disk = json.loads(state.read_text(encoding="utf-8"))
        assert on_disk["status"] == "failed" and on_disk["completed_at"]
        # queued never had a process; the other two may have left the evaluator's group behind.
        assert bool(on_disk.get("gpu_may_be_held")) is (status != "queued")


def test_finished_record_is_not_touched(tmp_path: Path) -> None:
    root = tmp_path / "queue_jobs"
    log_directory = root / "done"
    state = _write_state(root, "done", _record("done", "completed", log_directory))
    os.utime(state, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    before_stat, before_bytes = state.stat(), state.read_bytes()

    assert SingleGpuQueue.recover_interrupted([root]) == []

    after = state.stat()
    assert state.read_bytes() == before_bytes
    assert after.st_mtime_ns == before_stat.st_mtime_ns
    assert after.st_ino == before_stat.st_ino


@pytest.mark.parametrize("payload,malformed,expected", [
    ({"status": "running"}, False, ["job-good"]),                  # no job_id, no log_directory
    ({"model_id": "m", "status": "queued"}, False, ["job-good"]),  # no job_id
    # A job_id but no log_directory: a real job whose snapshot lost a field, so it is still
    # recovered (its state file is rewritten in place) and adopted with an empty log directory.
    ({"job_id": "j", "status": "running"}, False, ["j", "job-good"]),
    (None, True, ["job-good"]),                                    # unparseable JSON
    ([{"job_id": "j", "status": "running"}], False, ["job-good"]),  # JSON list, not a record
    ("hello", False, ["job-good"]),                                # JSON string, not a record
])
def test_incomplete_records_are_skipped_without_raising(tmp_path: Path,
                                                        payload: object,
                                                        malformed: bool,
                                                        expected: list[str]) -> None:
    root = tmp_path / "queue_jobs"
    broken = _write_state(root, "broken", payload, malformed=malformed)
    broken_before = broken.read_bytes()
    good_directory = root / "good"
    _write_state(root, "good", _record("good", "running", good_directory))

    recovered = SingleGpuQueue.recover_interrupted([root])

    # The valid record is still recovered: one bad file must not abort the whole scan.
    assert [item["job_id"] for item in recovered] == expected
    if expected == ["job-good"]:
        assert broken.read_bytes() == broken_before

    queue = SingleGpuQueue()
    for record in recovered:
        queue.adopt_recovered(record)
    assert [job["job_id"] for job in queue.list()] == expected
    queue.shutdown()


def test_recovery_never_spawns_a_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"recovery started a subprocess: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "Popen", explode)
    root = tmp_path / "queue_jobs"
    for index, status in enumerate(("queued", "preparing", "running", "completed")):
        _write_state(root, f"job{index}", _record(f"job{index}", status, root / f"job{index}"))

    recovered = SingleGpuQueue.recover_interrupted([root])

    assert len(recovered) == 3
    queue = SingleGpuQueue()
    try:
        for record in recovered:
            queue.adopt_recovered(record)
        assert len(queue.list()) == 3
    finally:
        queue.shutdown()


def test_foreign_processes_survive_recovery(tmp_path: Path) -> None:
    mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                               start_new_session=True)
    try:
        root = tmp_path / "queue_jobs"
        _write_state(root, "running", _record("running", "running", root / "running"))
        SingleGpuQueue.recover_interrupted([root])
        time.sleep(0.2)
        assert mine.poll() is None
        assert foreign.poll() is None
        queue = SingleGpuQueue()
        try:
            for record in SingleGpuQueue.recover_interrupted([root]):
                queue.adopt_recovered(record)
            queue.shutdown()
            assert mine.poll() is None, "shutdown must not signal a pid it did not start"
            assert foreign.poll() is None
        finally:
            for child in (mine, foreign):
                if child.poll() is None:
                    child.kill()
                child.wait()
    finally:
        for child in (mine, foreign):
            if child.poll() is None:
                child.kill()
            child.wait()


def test_dispatch_is_refused_while_a_recovered_orphan_may_hold_the_gpu(tmp_path: Path) -> None:
    root = tmp_path / "queue_jobs"
    _write_state(root, "orphan", _record("orphan", "running", root / "orphan"))
    queue = SingleGpuQueue()
    try:
        for record in SingleGpuQueue.recover_interrupted([root]):
            queue.adopt_recovered(record)
        marker = tmp_path / "ran"
        job = queue.enqueue("m2", "c", "p",
                            [sys.executable, "-c",
                             f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                            tmp_path, tmp_path / "queue_jobs" / "new", {})

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            current = next(item for item in queue.list() if item["job_id"] == job["job_id"])
            if current["status"] in {"blocked", "failed", "completed", "cancelled"}:
                break
            time.sleep(0.02)

        assert current["status"] == "blocked", current
        assert "job-orphan" in (current["error"] or "")
        assert "may still hold the GPU" in (current["error"] or "")
        assert not marker.exists(), "a refused dispatch must not run the job"
    finally:
        queue.shutdown()
