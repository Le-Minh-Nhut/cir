from __future__ import annotations

from importlib import import_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

from workbench.composition.inputs import adapt_query_inputs

MODELS = {
    "habit": "workbench.composition.model_habit",
    "intent": "workbench.composition.model_intent",
    "hint": "workbench.composition.model_hint",
    "tme": "workbench.composition.model_tme",
}
RESIDENT_MODELS = ("habit", "intent", "hint")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")




class _FakeWorker:
    def __init__(self, argv, jobs, fail=False):
        self.jobs = jobs
        self.returncode = 7 if fail else None
        self.stdout = SimpleNamespace(readline=lambda: self._reply())
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=lambda: None)
        self._response = ""
        if not fail:
            self._run(argv[-1])

    def _reply(self):
        response, self._response = self._response, ""
        return response

    def _run(self, job_path):
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        self.jobs.append(job)
        shape = (len(job["image_paths"]), 256) if job["operation"] == "queries" else (len(job["image_paths"]), 32, 256)
        np.save(job["output_path"], np.full(shape, len(self.jobs), dtype=np.float32), allow_pickle=False)
        self._response = 'CIRCUS_WORKER:{"ok":true}' + chr(10)

    def _write(self, request):
        payload = json.loads(request)
        if payload.get("stop"):
            self.returncode = 0
        elif "job_path" in payload:
            self._run(payload["job_path"])
        return len(request)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("model_id", RESIDENT_MODELS)
def test_resident_worker_reuses_process_and_cleans_jobs(model_id, monkeypatch, tmp_path) -> None:
    from workbench.composition import adapters
    import workbench.backend.runtime as runtime

    module = import_module(MODELS[model_id])
    adapter = module.Adapter()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.touch()
    adapter.interpreter = sys.executable
    adapter.checkpoint_path = checkpoint
    adapter.work_root = tmp_path / "work"
    adapter._interpreter_verified = False
    monkeypatch.setattr(runtime, "require_verified_model_interpreter", lambda model, config=None: sys.executable)
    jobs, workers = [], []

    def start(argv, **kwargs):
        worker = _FakeWorker(argv, jobs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", start)
    images = []
    for index in range(6):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(path)
        images.append(str(path))
    outputs = [adapter.encode_targets([images[index]], [None]) for index in range(3)]
    query_conditions = ("multimodal", "text_only_black_image", "image_only_empty_text")
    outputs.extend(adapter.encode_queries([images[index + 3]], [condition], [None])
                   for index, condition in enumerate(query_conditions))
    assert len(workers) == 1
    assert [job["image_paths"][0] for job in jobs] == [str(Path(path).resolve()) for path in images]
    assert [job["texts"] for job in jobs[3:]] == [[condition] for condition in query_conditions]
    assert [float(result.flat[0]) for result in outputs] == [1, 2, 3, 4, 5, 6]
    assert [result.shape for result in outputs] == [(1, 32, 256)] * 3 + [(1, 256)] * 3
    worker = workers[0]
    adapter.close()
    adapter.close()
    assert worker.returncode == 0
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.parametrize("model_id", RESIDENT_MODELS)
def test_resident_worker_failure_is_shared_and_not_restarted(model_id, monkeypatch, tmp_path) -> None:
    from workbench.composition import adapters
    import workbench.backend.runtime as runtime

    module = import_module(MODELS[model_id])
    adapter = module.Adapter()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.touch()
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    adapter.interpreter = sys.executable
    adapter.checkpoint_path = checkpoint
    adapter.work_root = tmp_path / "work"
    adapter._interpreter_verified = False
    monkeypatch.setattr(runtime, "require_verified_model_interpreter", lambda model, config=None: sys.executable)
    workers = []

    def start(argv, **kwargs):
        worker = _FakeWorker(argv, [], fail=True)
        workers.append(worker)
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", start)
    for _ in range(2):
        with pytest.raises(adapters.AdapterWorkerError, match=rf"model={model_id}.*exit_status=7") as error:
            adapter.encode_queries([str(image)], ["edit"], [None])
        assert type(error.value) is adapters.AdapterWorkerError
    assert len(workers) == 1
    adapter.close()
def test_adapters_import_and_declare_full_gallery_contract() -> None:
    from workbench.composition.adapters import CompositionAdapter

    for model_id, module_name in MODELS.items():
        module = import_module(module_name)
        assert module.ADAPTER is module.Adapter
        assert issubclass(module.Adapter, CompositionAdapter)
        adapter = module.Adapter()
        assert adapter.model_id == model_id
        assert adapter.reference_eligible is True
        assert adapter.category_scopes == (None,)
        adapter.close()


def test_status_is_honest_and_reports_every_mode() -> None:
    for module_name in MODELS.values():
        status = import_module(module_name).Adapter.status()
        assert status.ready is False
        assert status.reason
        assert set(status.per_mode) == set(CONDITIONS)


def test_unknown_mode_has_reason() -> None:
    for module_name in MODELS.values():
        supported, reason = import_module(module_name).Adapter().mode_supported("not-a-condition")
        assert supported is False
        assert reason


def test_cpu_score_exact_rank_and_stable_tie_order() -> None:
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    targets = np.array([
        [[1.0, 0.0], [-1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [0.8, 0.2]],
    ], dtype=np.float32)
    for module_name in MODELS.values():
        adapter = import_module(module_name).Adapter()
        np.testing.assert_allclose(adapter.score(queries, targets), [[1.0, 1.0, 0.8]])
        assert adapter.rank(queries, targets, [[1]]) == [2]
        assert adapter.rank(queries, targets, [[0]]) == [1]

def test_query_adaptation_stages_black_path_and_empty_text(monkeypatch, tmp_path) -> None:
    original = Image.new("RGB", (7, 5), (17, 29, 41))
    original_path = tmp_path / "reference.png"
    original.save(original_path)
    edit = "make sleeves shorter"
    for model_id, module_name in MODELS.items():
        module = import_module(module_name)
        adapter = module.Adapter()
        adapter.work_root = tmp_path / model_id / "work"
        jobs = []
        def fake_run_job(operation, image_paths, texts=(), categories=()):
            jobs.append((operation, list(image_paths), list(texts), list(categories)))
            return np.zeros((len(image_paths), 2), dtype=np.float32)

        monkeypatch.setattr(adapter, "_run_job", fake_run_job)
        black, text = adapt_query_inputs(original, edit, "text_only_black_image")
        black_path = tmp_path / model_id / "black.png"
        black_path.parent.mkdir(parents=True, exist_ok=True)
        black.save(black_path, format="PNG")
        adapter.encode_queries([str(black_path)], [text], [None])
        with Image.open(jobs[-1][1][0]) as staged:
            assert staged.size == original.size
            assert staged.getbbox() is None
        assert jobs[-1][2] == [edit]

        adapt_query_inputs(original, edit, "image_only_empty_text")
        adapter.encode_queries([str(original_path)], [""], [None])
        assert jobs[-1][2] == [""]


def test_tme_text_only_worker_reads_staged_black_image_not_cache(monkeypatch, tmp_path) -> None:
    """Resident-protocol transport; staged-black-image scientific assertion unchanged."""
    module = import_module(MODELS["tme"])
    adapter = module.Adapter()
    reference = Image.new("RGB", (6, 4), (100, 80, 60))
    black, text = adapt_query_inputs(reference, "turn it blue", "text_only_black_image")
    black_path = tmp_path / "black.png"
    black.save(black_path, format="PNG")
    adapter.interpreter = sys.executable
    adapter.work_root = tmp_path / "work"
    adapter.checkpoint_path = tmp_path / "tme-checkpoint.pth"
    adapter.checkpoint = {"checkpoint_id": "fashioniq_0"}
    adapter.model = {"model_id": "tme"}
    seen = []

    def inspect_job(job_path):
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        with Image.open(job["image_paths"][0]) as staged:
            assert staged.size == reference.size
            assert staged.getbbox() is None
        assert job["operation"] == "queries"
        assert job["texts"] == ["turn it blue"]
        seen.append(job["image_paths"][0])
        np.save(job["output_path"], np.zeros((1, 256), dtype=np.float32), allow_pickle=False)

    class FakeStdout:
        def __init__(self, response):
            self.response = response

        def readline(self):
            response, self.response = self.response, ""
            return response

    class FakeStdin:
        def write(self, request):
            payload = json.loads(request)
            if "job_path" in payload:
                inspect_job(payload["job_path"])
                self.process.stdout.response = 'CIRCUS_WORKER:{"ok":true}' + chr(10)
            return len(request)

        def flush(self):
            pass

        def close(self):
            pass

    class FakePopen:
        def __init__(self, argv, **kwargs):
            assert argv[:3] == [sys.executable, "-m", "workbench.composition.model_tme"]
            inspect_job(argv[-1])
            self.stdout = FakeStdout('CIRCUS_WORKER:{"ok":true}' + chr(10))
            self.stdin = FakeStdin()
            self.stdin.process = self
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def require_interpreter(model, config=None):
        return sys.executable

    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "require_verified_model_interpreter", require_interpreter)
    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)
    result = adapter.encode_queries([str(black_path)], [text], [None])
    assert result.shape == (1, 256)
    assert seen == [str(black_path.resolve())]
    adapter.close()
