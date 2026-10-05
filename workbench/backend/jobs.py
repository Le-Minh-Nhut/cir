from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4


@dataclass
class Job:
    job_id: str
    model_id: str
    checkpoint_id: str
    protocol_id: str
    command: list[str]
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


class SingleGpuQueue:
    """Serial evaluation queue. Execution is deliberately disabled in laptop mode."""

    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._lock = Lock()

    def enqueue(self, model_id: str, checkpoint_id: str, protocol_id: str, command: list[str]) -> Job:
        with self._lock:
            job = Job(str(uuid4()), model_id, checkpoint_id, protocol_id, command)
            self._jobs.append(job)
            return job

    def list(self) -> list[dict]:
        with self._lock:
            return [asdict(job) for job in self._jobs]

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = next((item for item in self._jobs if item.job_id == job_id), None)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"queued", "preparing"}:
                raise ValueError(f"cannot cancel {job.status} job")
            job.status = "cancelled"
            job.completed_at = datetime.now(UTC).isoformat()
            return asdict(job)
