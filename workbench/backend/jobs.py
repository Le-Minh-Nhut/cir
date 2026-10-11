from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, Lock, Thread
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    job_id: str
    model_id: str
    checkpoint_id: str
    protocol_id: str
    command: list[str]
    cwd: str
    log_directory: str
    provenance: dict[str, object]
    status: str = "queued"
    created_at: str = field(default_factory=now)
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


# Operator configuration the queue must pass on, because the process it starts *is* the
# trusted operator and cannot do its job without it. Everything else that looks secret is
# removed. The operator itself strips these from the environment it hands to the official
# evaluator, which is the untrusted side of the boundary.
_OPERATOR_ENV_ALLOWLIST = ("WORKBENCH_LEDGER_KEY",)


def _child_environment(job: Job) -> dict[str, str]:
    """The environment the queued process runs with.

    What the queue starts is the trusted evaluation operator (``evaluate_models.py``), and the
    operator needs the ledger key to authenticate the entry it appends. Stripping it here — as
    an earlier revision did, while still calling the child "untrusted" — made a queued run
    unable to audit itself at all.

    The names on ``_OPERATOR_ENV_ALLOWLIST`` are therefore preserved, and every other
    ``WORKBENCH_SECRET_*`` is dropped: those belong to operator configuration the operator does
    not need. The operator in turn removes ``WORKBENCH_LEDGER_KEY`` and ``WORKBENCH_SECRET_*``
    from the environment it gives the official evaluator, so model code never sees them.

    This is not OS-level isolation: a same-user process can still read same-user files. It is
    the minimum privilege each role needs, nothing more.
    """
    child_env = dict(os.environ)
    for name in [key for key in child_env
                 if key.startswith("WORKBENCH_SECRET_")
                 and key not in _OPERATOR_ENV_ALLOWLIST]:
        child_env.pop(name, None)
    child_env["evaluation_run_id"] = job.job_id
    return child_env


class SingleGpuQueue:
    """Runs one preflighted upstream evaluator at a time."""

    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._condition = Condition(Lock())
        self._worker = Thread(target=self._work, daemon=True)
        self._worker.start()

    def enqueue(self, model_id: str, checkpoint_id: str, protocol_id: str, command: list[str], cwd: Path, log_directory: Path, provenance: dict[str, object]) -> dict:
        with self._condition:
            job = Job(str(uuid4()), model_id, checkpoint_id, protocol_id, command, str(cwd), str(log_directory), provenance)
            self._jobs.append(job)
            self._condition.notify()
            return asdict(job)

    def list(self) -> list[dict]:
        with self._condition:
            return [asdict(job) for job in self._jobs]

    def cancel(self, job_id: str) -> dict:
        with self._condition:
            job = next((item for item in self._jobs if item.job_id == job_id), None)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"queued", "preparing"}:
                job.status = "cancelled"
                job.completed_at = now()
            elif job.status == "running":
                process = self._processes.get(job_id)
                if process is None:
                    raise ValueError("running process is unavailable")
                job.status = "cancelled"
                self._terminate_group(process)
            else:
                raise ValueError(f"cannot cancel {job.status} job")
            return asdict(job)

    def _next_job(self) -> Job:
        with self._condition:
            while not any(job.status == "queued" for job in self._jobs):
                self._condition.wait()
            job = next(job for job in self._jobs if job.status == "queued")
            job.status = "preparing"
            job.started_at = now()
            return job

    @staticmethod
    def _terminate_group(process: "subprocess.Popen[str]", *, grace: float = 10.0) -> None:
        """Stop the job's whole process group: the operator and the evaluator it started.

        ``start_new_session`` made the job a session and group leader, so its pid is the group
        id. SIGTERM first so the operator can unwind, then SIGKILL for anything that ignores
        it. Both calls are guarded against a group that already exited (ESRCH) and against a
        recycled pid, which the group signal cannot hit because the group id exists only while
        one of its processes does.
        """
        import signal
        import time

        if process.poll() is not None:
            return
        try:
            group = os.getpgid(process.pid)
        except (OSError, ProcessLookupError):
            group = None
        for signum, deadline in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
            if process.poll() is not None:
                return
            try:
                if group is not None:
                    os.killpg(group, signum)
                else:
                    process.send_signal(signum)
            except (OSError, ProcessLookupError):
                return
            waited = 0.0
            while waited < deadline and process.poll() is None:
                time.sleep(0.05)
                waited += 0.05
        # Never report a cancelled job while a process it owns is still alive.
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    @staticmethod
    def _append(job: Job, field: str, line: str) -> None:
        value = (getattr(job, field) + line)[-12000:]
        setattr(job, field, value)

    def _work(self) -> None:
        while True:
            job = self._next_job()
            with self._condition:
                if job.status == "cancelled":
                    continue
            directory = Path(job.log_directory)
            try:
                directory.mkdir(parents=True, exist_ok=False)
                # Readiness is re-checked here, immediately before dispatch: a job may sit in
                # the queue while the environment changes (the GPU capability is invalidated
                # by an interpreter or dependency change), and a stale check in the request
                # handler must not authorise a launch.
                launch_guard = getattr(self, "launch_guard", None)
                if launch_guard is not None:
                    try:
                        launch_guard(job)
                    except Exception as error:
                        with self._condition:
                            # A cancellation that already happened must not be overwritten.
                            if job.status != "cancelled":
                                job.status = "blocked"
                                self._append(job, "stderr_tail", f"[BLOCKED] {error}\n")
                        continue
                # Queue bookkeeping lives under `queue_jobs/<job_id>/` and the run's
                # scientific artifacts under `logs/<model>__<checkpoint>__<run_id>/`. They are
                # different writers with different lifetimes: the queue records dispatch, the
                # operator records evidence. Sharing one directory let the queue's
                # `command.json` describe the run instead of the operator's own record, and the
                # evidence digest pointed at a file the queue could rewrite.
                metadata = {**job.provenance, "job_id": job.job_id, "argv": job.command, "cwd": job.cwd, "started_at": job.started_at, "finished_at": None, "return_code": None}
                (directory / "queue_command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                with (directory / "queue_stdout.log").open("w", encoding="utf-8") as stdout, (directory / "queue_stderr.log").open("w", encoding="utf-8") as stderr:
                    with self._condition:
                        # The guard above can be slow, so a cancellation that arrived while it
                        # ran must still win. Creation, registration (which is what cancel
                        # terminates) and the status transition happen under the same lock, so
                        # no cancellation can observe `preparing` with no process to kill and
                        # then be overwritten by `running`.
                        if job.status == "cancelled":
                            continue
                        # The job command is the evaluation operator, which starts the
                        # official evaluator as its own child. Killing only the operator's PID
                        # would leave that evaluator — the process actually holding the GPU —
                        # running. The job therefore leads its own process group and a
                        # cancellation signals the group. A child that deliberately detaches
                        # into a new session escapes this; containment of arbitrary hostile
                        # code needs an OS sandbox, which this workbench does not claim.
                        process = subprocess.Popen(job.command, cwd=job.cwd, text=True,
                                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                   env=_child_environment(job),
                                                   start_new_session=True)
                        job.status = "running"
                        self._processes[job.job_id] = process

                    def copy(stream, destination, field: str) -> None:
                        assert stream is not None
                        for line in stream:
                            destination.write(line)
                            destination.flush()
                            with self._condition:
                                self._append(job, field, line)

                    readers = [Thread(target=copy, args=(process.stdout, stdout, "stdout_tail")), Thread(target=copy, args=(process.stderr, stderr, "stderr_tail"))]
                    for reader in readers:
                        reader.start()
                    code = process.wait()
                    for reader in readers:
                        reader.join()
                with self._condition:
                    self._processes.pop(job.job_id, None)
                    job.return_code = code
                    if job.status != "cancelled":
                        job.status = "completed" if code == 0 else "failed"
                    job.completed_at = now()
                metadata.update(finished_at=job.completed_at, return_code=code)
                (directory / "queue_command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            except Exception as error:
                with self._condition:
                    self._processes.pop(job.job_id, None)
                    job.error = str(error)
                    if job.status != "cancelled":
                        job.status = "failed"
                    job.completed_at = now()
