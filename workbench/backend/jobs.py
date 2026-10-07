from __future__ import annotations

import json
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
                process.terminate()
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
                metadata = {**job.provenance, "job_id": job.job_id, "argv": job.command, "cwd": job.cwd, "started_at": job.started_at, "finished_at": None, "return_code": None}
                (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
                    process = subprocess.Popen(job.command, cwd=job.cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with self._condition:
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
                (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            except Exception as error:
                with self._condition:
                    self._processes.pop(job.job_id, None)
                    job.error = str(error)
                    if job.status != "cancelled":
                        job.status = "failed"
                    job.completed_at = now()
