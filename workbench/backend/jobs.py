from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, Lock, Thread
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

#: Credentials the operator never needs to run an evaluation. The operator's own allowlist above
#: names the single exception; everything else here is withheld from the queued process as well,
#: so the boundary is enforced at both hops rather than only at the inner one.
_OPERATOR_ENV_DENY_PREFIXES = (
    "WORKBENCH_SECRET_",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
    "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AZURE_", "GOOGLE_APPLICATION_CREDENTIALS", "GCP_",
    "DOCKER_", "KUBECONFIG", "SLACK_TOKEN", "STRIPE_", "NPM_TOKEN",
)


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
    # The operator needs its own configuration and the ledger key. Unrelated credentials are not
    # part of operating the workbench, so they are withheld here as well: a queued operator has no
    # reason to hold the operator's cloud or hub tokens.
    for name in list(child_env):
        if any(name == prefix or name.startswith(prefix)
               for prefix in _OPERATOR_ENV_DENY_PREFIXES
               if name not in _OPERATOR_ENV_ALLOWLIST):
            child_env.pop(name, None)
    child_env["evaluation_run_id"] = job.job_id
    return child_env


class DuplicateJobError(RuntimeError):
    """An identical model+checkpoint+protocol job is already queued or running."""

    def __init__(self, job_id: str, status: str):
        super().__init__(f"{job_id} ({status})")
        self.job_id = job_id
        self.status = status

class SingleGpuQueue:
    """Runs one preflighted upstream evaluator at a time."""

    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._processes: dict[str, subprocess.Popen[str]] = {}
        # Jobs whose cancellation is in progress: the worker must not publish a completion for
        # them while their group is still being torn down.
        self._cancelling: set[str] = set()
        self._process_groups: dict[str, int | None] = {}
        # Set by shutdown(): the worker leaves its loop instead of waiting for the next job.
        self._stopping = False
        self._condition = Condition(Lock())
        self._worker = Thread(target=self._work, daemon=True)
        self._worker.start()

    def enqueue(self, model_id: str, checkpoint_id: str, protocol_id: str, command: list[str], cwd: Path, log_directory: Path, provenance: dict[str, object], *, refuse_duplicate: bool = False) -> dict:
        with self._condition:
            if refuse_duplicate:
                # The duplicate check is atomic with the append: a handler that checks first
                # and enqueues later races a concurrent enqueue past the check (TOCTOU).
                for existing in self._jobs:
                    if (existing.status in {"queued", "preparing", "running"}
                            and existing.model_id == model_id
                            and existing.checkpoint_id == checkpoint_id
                            and existing.protocol_id == protocol_id):
                        raise DuplicateJobError(existing.job_id, existing.status)
            job = Job(str(uuid4()), model_id, checkpoint_id, protocol_id, command, str(cwd), str(log_directory), provenance)
            self._jobs.append(job)
            self._persist(job)
            self._condition.notify()
            return asdict(job)

    def list(self) -> list[dict]:
        with self._condition:
            return [asdict(job) for job in self._jobs]

    def cancel(self, job_id: str) -> dict:
        """Cancel a job and, for a running one, stop every process in its group.

        The signal and the bounded wait happen *outside* the queue lock: the shutdown can take
        seconds, and holding the condition for it would stall status queries and the worker. The
        handle is removed under the lock first, so the worker's finalization cannot race the
        cancellation into publishing `completed`, and the job is only reported as cancelled once
        nothing it owns is still alive.
        """
        with self._condition:
            job = next((item for item in self._jobs if item.job_id == job_id), None)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"queued", "preparing"}:
                job.status = "cancelled"
                job.completed_at = now()
                self._persist(job)
                return asdict(job)
            if job.status != "running":
                raise ValueError(f"cannot cancel {job.status} job")
            process = self._processes.pop(job_id, None)
            group = self._process_groups.pop(job_id, None)
            if process is None:
                raise ValueError("running process is unavailable")
            job.status = "cancelled"
            self._cancelling.add(job_id)

        survivor = self._terminate_group(process, group=group)

        with self._condition:
            self._cancelling.discard(job_id)
            live = next((item for item in self._jobs if item.job_id == job_id), None)
            if live is not None and survivor:
                # Truthful reporting: a process this job owns outlived the cancellation.
                live.error = "cancellation could not stop every process in the job's group"
                self._append(live, "stderr_tail",
                             "[WARN] a process in this job's group survived cancellation\n")
            return asdict(live) if live is not None else {"job_id": job_id, "status": "cancelled"}

    def shutdown(self) -> list[str]:
        """Stop the worker and every process group it still owns.

        A job leads its own session (`start_new_session=True`), so it outlives the backend by
        design: the operator's evaluator child — the process that actually holds the GPU —
        does not die with this process, and a container stop would leave it running. Each live
        job is torn down with exactly the group kill `cancel` uses; this only supplies the
        "every job" scope. The bounded wait runs outside the queue lock, as in `cancel`, so a
        slow teardown cannot stall the status API.

        Returns the ids of jobs whose group outlived the bounded kill, so the caller can report
        a real orphan instead of claiming a clean stop.
        """
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            live = [(job, self._processes.pop(job.job_id, None),
                     self._process_groups.pop(job.job_id, None))
                    for job in self._jobs if job.job_id in self._processes]
        survivors: list[str] = []
        for job, process, group in live:
            with self._condition:
                self._cancelling.add(job.job_id)
            survivor = self._terminate_group(process, group=group)
            with self._condition:
                self._cancelling.discard(job.job_id)
                if job.status != "cancelled":
                    job.status = "cancelled"
                    job.completed_at = now()
                    self._append(job, "stderr_tail", "[STOP] workbench is shutting down\n")
                    self._persist(job)
            if survivor:
                survivors.append(job.job_id)
        if self._worker.is_alive():
            # Outside the lock: the worker takes it to publish a job's final state.
            self._worker.join(timeout=10.0)
        return survivors

    def adopt_recovered(self, record: dict) -> None:
        """Load a startup-recovered record into the in-memory queue as a terminal job.

        The record's state on disk is already truthful (`failed`); this only makes it
        visible through the status API after a restart.
        """
        job = Job(
            record.get("job_id") or str(uuid4()),
            str(record.get("model_id")), str(record.get("checkpoint_id")),
            str(record.get("protocol_id")), [], "",
            str(record.get("log_directory") or ""),
            {}, status=str(record.get("status") or "failed"),
            created_at=str(record.get("created_at") or now()),
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            return_code=record.get("return_code"),
            error=record.get("error"),
        )
        with self._condition:
            if any(existing.job_id == job.job_id for existing in self._jobs):
                return
            self._jobs.append(job)

    def _next_job(self) -> Job | None:
        """The next queued job, or None once shutdown has been requested."""
        with self._condition:
            while not self._stopping and not any(job.status == "queued" for job in self._jobs):
                self._condition.wait()
            if self._stopping:
                return None
            job = next(job for job in self._jobs if job.status == "queued")
            job.status = "preparing"
            job.started_at = now()
            self._persist(job)
            return job

    @staticmethod
    def _group_alive(group: int | None, process: "subprocess.Popen[str]") -> bool:
        """True while the job's process group still has a member running.

        The operator exiting is not the same event as the group being gone: the evaluator it
        started is a separate process in the same group and may outlive it deliberately (it can
        ignore SIGTERM). Liveness is therefore taken from the *group*, using signal 0, which is a
        permission/absence probe and never delivers a signal.
        """
        import signal

        if group is not None:
            try:
                os.killpg(group, 0)
                return True
            except ProcessLookupError:
                return False         # nothing left in the group
            except PermissionError:
                return True          # exists, but not ours to signal
            except OSError:
                pass
        return process.poll() is None

    #: Seconds SIGTERM is given before SIGKILL escalates. A class-level constant so an operator
    #: (or a test) can shorten the bounded wait deliberately.
    GRACE_SECONDS = 10.0

    @staticmethod
    def _terminate_group(process: "subprocess.Popen[str]", *, group: int | None = None,
                         grace: float | None = None) -> bool:
        """Stop the job's whole process group and report whether anything still runs.

        ``start_new_session`` made the job a session and group leader, so its pid is the group
        id. SIGTERM first with a bounded grace period, then SIGKILL for members that ignore it —
        escalation must not be skipped merely because the *operator* exited, because the
        evaluator is the group member that actually holds the GPU.

        A group id is only meaningful while a member exists, so a recycled pid cannot be signalled
        accidentally: once ``killpg`` reports ESRCH there is nothing left in that group. A child
        that deliberately calls ``setsid`` leaves this group and is outside the boundary this
        workbench documents.
        """
        import signal
        import time

        if grace is None:
            grace = SingleGpuQueue.GRACE_SECONDS
        if group is None:
            # Fall back to the live lookup only when dispatch could not record the group.
            try:
                group = os.getpgid(process.pid)
            except (OSError, ProcessLookupError):
                group = None

        for signum, deadline in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
            if not SingleGpuQueue._group_alive(group, process):
                return False
            try:
                if group is not None:
                    os.killpg(group, signum)
                else:
                    process.send_signal(signum)
            except (OSError, ProcessLookupError):
                return False
            waited = 0.0
            while waited < deadline and SingleGpuQueue._group_alive(group, process):
                time.sleep(0.05)
                waited += 0.05

        alive = SingleGpuQueue._group_alive(group, process)
        if alive:
            # Last resort: the direct child, so at least the operator cannot linger.
            try:
                process.kill()
            except OSError:
                pass
            alive = SingleGpuQueue._group_alive(group, process)
        return alive

    @staticmethod
    def _append(job: Job, field: str, line: str) -> None:
        value = (getattr(job, field) + line)[-12000:]
        setattr(job, field, value)

    @staticmethod
    def _persist(job: "Job") -> None:
        """Write a status snapshot next to the queue's dispatch record.

        The queue lives in RAM; a container restart would otherwise make every job vanish
        from the status API while its evaluator artifacts still exist — or worse, leave a
        `running` job that no process backs. Callers hold the queue lock.
        """
        try:
            directory = Path(job.log_directory)
            # A queued job has no log directory yet: the worker creates it only when the job
            # is dispatched. Persist anyway — a restart must still recover queued jobs.
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "queue_state.json").write_text(
                    json.dumps({key: getattr(job, key) for key in (
                        "job_id", "model_id", "checkpoint_id", "protocol_id", "status",
                        "started_at", "completed_at", "return_code", "error")}, indent=2) + "\n",
                    encoding="utf-8")
        except OSError:
            # Persistence is best-effort bookkeeping; the dispatch record on disk remains
            # the durable trace even when this snapshot write fails.
            pass

    @classmethod
    def recover_interrupted(cls, roots: list[Path]) -> list[dict]:
        """Mark jobs recorded as queued/preparing/running when the process died as failed.

        Called once at backend startup: a `running` status after a restart is never true —
        no process backs it — so the record becomes an honest `failed` with the reason.
        Returns the recovered records for logging.
        """
        recovered: list[dict] = []
        for root in roots:
            if not root.is_dir():
                continue
            for state_path in sorted(root.glob("*/queue_state.json")):
                try:
                    record = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if record.get("status") in ("queued", "preparing", "running"):
                    record["status"] = "failed"
                    record["error"] = "workbench restarted while the job was in flight"
                    record["completed_at"] = datetime.now(timezone.utc).isoformat()
                    recovered.append(record)
                    try:
                        state_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                    except OSError:
                        pass
        return recovered

    def _work(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            with self._condition:
                if job.status == "cancelled":
                    continue
            directory = Path(job.log_directory)
            try:
                directory.mkdir(parents=True, exist_ok=True)
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
                                self._persist(job)
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
                        self._persist(job)
                        self._processes[job.job_id] = process
                        # Capture the group now: once the leader exits, `getpgid` on its pid
                        # fails and the evaluator it started would become unreachable.
                        try:
                            self._process_groups[job.job_id] = os.getpgid(process.pid)
                        except (OSError, ProcessLookupError):
                            self._process_groups[job.job_id] = None

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
                    # A cancellation in flight wins: the worker must not publish a completion for
                    # a job whose group is still being torn down, whatever the child's exit code.
                    if job.status != "cancelled" and job.job_id not in self._cancelling:
                        job.status = "completed" if code == 0 else "failed"
                    job.completed_at = now()
                    self._persist(job)
                metadata.update(finished_at=job.completed_at, return_code=code)
                (directory / "queue_command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            except Exception as error:
                with self._condition:
                    self._processes.pop(job.job_id, None)
                    job.error = str(error)
                    if job.status != "cancelled" and job.job_id not in self._cancelling:
                        job.status = "failed"
                    job.completed_at = now()
                    self._persist(job)
