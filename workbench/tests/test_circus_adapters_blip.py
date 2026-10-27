"""CPU-only contract checks for the BLIP composition adapter subprocess clients."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workbench.composition import model_csmcir, model_conesep, model_ptha_mtst
from workbench.composition import adapters, inputs, runner

MODULES = (model_csmcir, model_conesep, model_ptha_mtst)
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


@pytest.mark.parametrize(("module", "model_id"), [
    (model_csmcir, "csmcir"), (model_conesep, "conesep"), (model_ptha_mtst, "ptha_mtst")
])
def test_adapter_import_and_unready_status(module, model_id: str) -> None:
    assert module.ADAPTER is module.Adapter
    assert issubclass(module.Adapter, adapters.CompositionAdapter)
    assert module.Adapter.model_id == model_id
    assert module.Adapter.category_scopes == (None,)
    assert module.Adapter.reference_eligible is True
    status = module.Adapter.status()
    assert status.ready is False
    assert status.reason
    assert set(status.per_mode) == set(CONDITIONS)
    assert all(isinstance(supported, bool) for supported, _reason in status.per_mode.values())
    status = module.Adapter.status()
    for condition in CONDITIONS:
        supported, reason = module.Adapter.__new__(module.Adapter).mode_supported(condition)
        assert supported == status.per_mode[condition][0]
        assert reason == status.per_mode[condition][1]
        if not supported:
            assert reason and "COT_ours2" in reason and "utils_csmcir.py" in reason


@pytest.mark.parametrize("module", MODULES)
def test_worker_root_required(module) -> None:
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = None
    with pytest.raises(adapters.AdapterWorkRootError, match="work_root"):
        adapter.encode_queries([], [], [])


@pytest.mark.parametrize("module", MODULES)
def test_native_max_token_score_and_tie_rank(module) -> None:
    adapter = module.Adapter.__new__(module.Adapter)
    query = np.zeros((1, 8192), dtype=np.float32)
    query[0, 0] = 1
    targets = np.zeros((2, 8192), dtype=np.float32)
    targets[:, 0] = 1
    np.testing.assert_array_equal(adapter.score(query, targets), np.ones((1, 2), dtype=np.float32))
    assert adapter.rank(query, targets, [[0]]) == [1]
    assert adapter.rank(query, targets, [[1]]) == [2]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_runner_forwards_adapted_inputs_to_adapter(condition: str, tmp_path: Path) -> None:
    source = tmp_path / "reference.png"
    Image.new("RGB", (13, 7), (22, 99, 201)).save(source)
    query = SimpleNamespace(qid="q-1", query_txt="make this blue", category="dress")

    class Recorder:
        received: tuple[list[str], list[str], list[str | None]] | None = None

        def encode_queries(self, image_paths, texts, categories):
            self.received = (list(image_paths), list(texts), list(categories))
            return np.zeros((len(image_paths), 4), dtype=np.float32)

    recorder = Recorder()
    output = tmp_path / f"{condition}.npy"
    runner._encode_queries(recorder, [query], [source], [0], condition, tmp_path,
                           output, "fingerprint", 4, 1)
    assert recorder.received is not None
    paths, texts, categories = recorder.received
    assert texts == ["" if condition == "image_only_empty_text" else query.query_txt]
    assert categories == ["dress"]
    with Image.open(paths[0]) as forwarded:
        assert forwarded.size == (13, 7)
        if condition == "text_only_black_image":
            assert np.asarray(forwarded).sum() == 0
        else:
            assert np.asarray(forwarded)[0, 0].tolist() == [22, 99, 201]

@pytest.mark.parametrize("module", (model_csmcir,))
def test_nonzero_worker_exit_is_named_and_returns_no_matrix(module, tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.config = SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path)
    adapter.work_root = tmp_path / "runner-work"
    adapter._interpreter = sys.executable
    adapter._model = {"source_dir": "model-source"}
    adapter._checkpoint = {"resolved_path": str(checkpoint), "checkpoint_id": "variant",
                           "checkpoint_training_noise_pct": 50}
    adapter.checkpoint_id = "variant"
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))
    with pytest.raises(adapters.AdapterWorkerError, match=r"csmcir.*scope.*7"):
        adapter.encode_queries([str(checkpoint)], ["edit"], [None])


def _resident_adapter(module, model_id: str, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path / "work"
    adapter._interpreter = sys.executable
    adapter._worker = None
    adapter._worker_log = None
    adapter._worker_counter = 0
    adapter._worker_failed = None
    if model_id == "conesep":
        adapter.config = SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path)
        adapter._model = {"source_dir": "model-source"}
        adapter._checkpoint = {"resolved_path": str(checkpoint), "checkpoint_id": "variant",
                               "checkpoint_training_noise_pct": 50}
        adapter._job_root = None
        adapter._interpreter_verified = True
    else:
        adapter.config = SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path)
        adapter._model = {"source_dir": "model-source"}
        adapter._checkpoint = {"resolved_path": str(checkpoint)}
        adapter._job_root = None
    return adapter


class _ResidentFakeWorker:
    def __init__(self, argv, **kwargs):
        self.returncode = None
        self.jobs = []
        self.stdout = SimpleNamespace(readline=self._readline)
        self.stdin = SimpleNamespace(write=self._write, flush=lambda: None, close=lambda: None)
        self.response = ""
        self._process_job(argv[-1])

    def _readline(self):
        response, self.response = self.response, ""
        return response

    def _process_job(self, path):
        job = json.loads(Path(path).read_text(encoding="utf-8"))
        self.jobs.append(job)
        shape = (len(job["image_paths"]), 8192)
        output_key = "output_path"
        np.save(job[output_key], np.full(shape, len(self.jobs), dtype=np.float32), allow_pickle=False)
        self.response = 'CIRCUS_WORKER:{"ok":true}' + chr(10)

    def _write(self, request):
        payload = json.loads(request)
        if payload.get("stop"):
            self.returncode = 0
        elif "job_path" in payload:
            self._process_job(payload["job_path"])
        return len(request)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize(("module", "model_id"), [
    (model_conesep, "conesep"), (model_ptha_mtst, "ptha_mtst"),
])
def test_resident_worker_reuse_and_cleanup(module, model_id, monkeypatch, tmp_path) -> None:
    adapter = _resident_adapter(module, model_id, tmp_path)
    workers = []

    def start(argv, **kwargs):
        worker = _ResidentFakeWorker(argv, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", start)
    paths = []
    for index in range(6):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (2, 2), (index, 0, 0)).save(path)
        paths.append(str(path))
    results = [adapter.encode_targets([path], [None]) for path in paths[:3]]
    conditions = ("multimodal", "text_only_black_image", "image_only_empty_text")
    results.extend(adapter.encode_queries([path], [condition], [None])
                   for condition, path in zip(conditions, paths[3:]))
    assert len(workers) == 1
    assert [job["image_paths"][0] for job in workers[0].jobs] == [str(Path(path).resolve()) for path in paths]
    assert [job["texts"] for job in workers[0].jobs[3:]] == [[condition] for condition in conditions]
    assert [float(result.flat[0]) for result in results] == [1, 2, 3, 4, 5, 6]
    assert all(result.shape == (1, 8192) for result in results)
    worker = workers[0]
    adapter.close()
    adapter.close()
    assert worker.returncode == 0
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.parametrize(("module", "model_id"), [
    (model_conesep, "conesep"), (model_ptha_mtst, "ptha_mtst"),
])
def test_resident_worker_failure_uses_shared_error_and_stays_failed(module, model_id, monkeypatch, tmp_path) -> None:
    adapter = _resident_adapter(module, model_id, tmp_path)
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    workers = []

    class FailedWorker(_ResidentFakeWorker):
        def __init__(self, argv, **kwargs):
            self.returncode = 7
            self.jobs = []
            self.stdout = SimpleNamespace(readline=lambda: "")
            self.stdin = SimpleNamespace(write=lambda value: len(value), flush=lambda: None, close=lambda: None)
            workers.append(self)

    monkeypatch.setattr(module.subprocess, "Popen", FailedWorker)
    for _ in range(2):
        with pytest.raises(adapters.AdapterWorkerError, match=rf"model={model_id}.*7") as error:
            adapter.encode_queries([str(image)], ["edit"], [None])
        assert type(error.value) is adapters.AdapterWorkerError
    assert len(workers) == 1
    adapter.close()


def test_ptha_query_vision_autocast_source_structure() -> None:
    """Source-structure test: query vision call must stay inside native autocast context."""
    tree = ast.parse(Path(model_ptha_mtst.__file__).read_text(encoding="utf-8"))
    worker = next(node for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_worker_main")
    autocast_with = [node for node in ast.walk(worker) if isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "maybe_autocast" for item in node.items)]
    vision_calls = [node for context in autocast_with for node in ast.walk(context)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "ln_vision" and node.args
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Attribute)
                    and node.args[0].func.attr == "visual_encoder"]
    assert vision_calls
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "extract_target_features" for node in ast.walk(worker))
