"""CPU-only lifecycle checks for the TME resident worker's failure paths.

A failed TME worker used to `restart()` (== `close()`) on the failure path, which removed the job
directory -- and with it `worker.stderr.log` -- before the runner's `finally` could copy that log
out. These checks pin the split: stop the process, latch the failure, keep the directory.
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.composition import model_tme
from workbench.composition.adapters import AdapterWorkerError


def _adapter(tmp_path: Path) -> model_tme.Adapter:
    adapter = model_tme.Adapter()
    adapter.interpreter = sys.executable
    adapter.work_root = tmp_path / "work"
    adapter._verified_interpreter = sys.executable
    return adapter


def _image(tmp_path: Path) -> str:
    path = tmp_path / "image.png"
    path.write_bytes(b"")
    return str(path)


def _worker_logs(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "work").rglob("worker.stderr.log"))


class _OkWorker:
    """Resident stub: answers the argv job, then accepts stdin jobs until stop."""

    def __init__(self, argv: list[str]) -> None:
        self.returncode: int | None = None
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=lambda: None)
        self.stdout = SimpleNamespace(readline=self._readline)
        self._response = ""
        self._process(argv[argv.index("--worker-job") + 1])

    def _process(self, job_path: str) -> None:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        np.save(job["output_path"], np.zeros((len(job["image_paths"]), 256), dtype=np.float32),
                allow_pickle=False)
        self._response = 'CIRCUS_WORKER:{"ok": true}' + "\n"

    def _readline(self) -> str:
        response, self._response = self._response, ""
        return response

    def _write(self, request: str) -> int:
        payload = json.loads(request)
        if payload.get("stop"):
            self.returncode = 0
        elif "job_path" in payload:
            self._process(payload["job_path"])
        return len(request)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _NotOkWorker:
    _LINE = 'CIRCUS_WORKER:{"ok": false, "error_type": "RuntimeError", "error": "boom"}' + "\n"

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = SimpleNamespace(readline=lambda: self._LINE)
        self.stdin = SimpleNamespace(write=lambda request: len(request), flush=lambda: None,
                                     close=lambda: None)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_failed_child_latches_preserves_log_and_recovers(tmp_path, monkeypatch) -> None:
    marker = f"UNIQUE_TRACEBACK_{uuid.uuid4().hex}"
    adapter = _adapter(tmp_path)
    real_popen = subprocess.Popen
    spawned: list[list[str]] = []

    def start(argv, **kwargs):
        spawned.append(list(argv))
        child = (f"import sys;sys.stderr.write({marker!r} + chr(10));"
                 "sys.stderr.flush();sys.exit(9)")
        return real_popen([sys.executable, "-c", child], **kwargs)

    monkeypatch.setattr(model_tme.subprocess, "Popen", start)
    for _ in range(2):
        with pytest.raises(AdapterWorkerError, match=r"model=tme.*exit_status=9") as error:
            adapter.encode_queries([_image(tmp_path)], ["edit"], [None])
        assert type(error.value) is AdapterWorkerError
    assert len(spawned) == 1, "the failure latch must hold; a second call must not spawn"
    logs = _worker_logs(tmp_path)
    assert len(logs) == 1
    assert marker in logs[0].read_text(encoding="utf-8")
    adapter.close()
    assert marker in logs[0].read_text(encoding="utf-8"), "close() must not delete a failed job dir"

    # A fresh instance has no latch and spawns again: recovery is per-instance, not global.
    recovered = _adapter(tmp_path)
    workers: list[_OkWorker] = []

    def start_ok(argv, **kwargs):
        worker = _OkWorker(list(argv))
        workers.append(worker)
        return worker

    monkeypatch.setattr(model_tme.subprocess, "Popen", start_ok)
    result = recovered.encode_queries([_image(tmp_path)], ["edit"], [None])
    assert len(workers) == 1
    assert result.shape == (1, 256)
    recovered.close()


def test_non_ok_reply_latches_and_preserves_job_dir(tmp_path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    spawned: list[list[str]] = []

    def start(argv, **kwargs):
        spawned.append(list(argv))
        return _NotOkWorker()

    monkeypatch.setattr(model_tme.subprocess, "Popen", start)
    for _ in range(2):
        with pytest.raises(AdapterWorkerError,
                           match=r"adapter_worker_job_failed: model=tme.*RuntimeError: boom") as error:
            adapter.encode_queries([_image(tmp_path)], ["edit"], [None])
        assert type(error.value) is AdapterWorkerError
    assert len(spawned) == 1, "the failure latch must hold; a second call must not spawn"
    logs = _worker_logs(tmp_path)
    assert len(logs) == 1 and logs[0].is_file()
    adapter.close()
    assert logs[0].is_file(), "close() must not delete a failed job dir"


# ---------------------------------------------------------------------------
# ENCODER and LIMN carry the same resident-worker contract as TME above.
# ---------------------------------------------------------------------------

def _encoder_adapter(tmp_path: Path):
    from workbench.composition import model_encoder

    config = SimpleNamespace(
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ",
        CIR_REPO_ROOT=Path(__file__).resolve().parents[2],
    )
    adapter = model_encoder.Adapter(config=config)
    adapter.interpreter = sys.executable
    adapter.work_root = tmp_path / "work"
    return adapter


def _limn_adapter(tmp_path: Path):
    from workbench.composition import model_limn

    adapter = model_limn.Adapter(config=None)
    adapter.interpreter = sys.executable
    adapter.work_root = tmp_path / "work"
    return adapter


@pytest.mark.parametrize("kind", ["encoder", "limn"])
def test_encoder_and_limn_failed_child_latch_and_preserve_log(tmp_path, monkeypatch, kind) -> None:
    module = __import__(f"workbench.composition.model_{kind}", fromlist=["Adapter"])
    adapter = _encoder_adapter(tmp_path) if kind == "encoder" else _limn_adapter(tmp_path)
    marker = f"UNIQUE_TRACEBACK_{uuid.uuid4().hex}"
    real_popen = subprocess.Popen
    spawned: list[list[str]] = []

    def start(argv, **kwargs):
        spawned.append(list(argv))
        child = (f"import sys;sys.stderr.write({marker!r} + chr(10));"
                 "sys.stderr.flush();sys.exit(9)")
        return real_popen([sys.executable, "-c", child], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", start)
    for _ in range(2):
        with pytest.raises(AdapterWorkerError,
                           match=rf"model={kind}.*exit_status=9") as error:
            adapter.encode_queries([_image(tmp_path)], ["edit"], ["dress" if kind == "limn" else None])
        assert type(error.value) is AdapterWorkerError
    assert len(spawned) == 1, "the failure latch must hold; a second call must not spawn"
    logs = _worker_logs(tmp_path)
    assert len(logs) == 1
    assert marker in logs[0].read_text(encoding="utf-8")
    adapter.close()
    assert marker in logs[0].read_text(encoding="utf-8"), "close() must not delete a failed job dir"
