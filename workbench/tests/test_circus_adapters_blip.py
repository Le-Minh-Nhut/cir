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


@pytest.mark.parametrize(("module", "model_id", "scopes"), [
    (model_csmcir, "csmcir", ("dress", "shirt", "toptee")),
    (model_conesep, "conesep", (None,)), (model_ptha_mtst, "ptha_mtst", (None,))
])
def test_adapter_import_and_unready_status(module, model_id: str, scopes) -> None:
    assert module.ADAPTER is module.Adapter
    assert issubclass(module.Adapter, adapters.CompositionAdapter)
    assert module.Adapter.model_id == model_id
    assert module.Adapter.category_scopes == scopes
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
            # The CSMCIR reason used to cite missing `COT_ours2` captions and `utils_csmcir.py`. That was
            # measured wrong: the evaluation path opens those files, they are published at the pinned
            # revision with matching sha256, and the paths an earlier report demanded belong to a dead
            # class. Pinning the old string would have preserved a refuted claim. Each adapter now only
            # has to state a specific reason.
            assert reason and len(reason) > 20


@pytest.mark.parametrize("module", MODULES)
def test_worker_root_required(module) -> None:
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = None
    with pytest.raises(adapters.AdapterWorkRootError, match="work_root"):
        adapter.encode_queries([], [], [])


@pytest.mark.parametrize("module", MODULES)
def test_native_max_token_score_and_tie_rank(module) -> None:
    adapter = module.Adapter.__new__(module.Adapter)
    query_width = getattr(adapter, "query_dimension", None) or adapter.score_dimension
    query = np.zeros((1, query_width), dtype=np.float32)
    query[0, 0] = 1
    targets = np.zeros((2, adapter.score_dimension), dtype=np.float32)
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
def test_nonzero_worker_exit_is_named_and_returns_no_matrix(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(model_csmcir, "_resolve", lambda _config, _checkpoint_id: (
        {"source_dir": "model-source"},
        {"resolved_path": str(checkpoint), "checkpoint_id": "variant"},
        sys.executable,
    ))
    adapter = model_csmcir.Adapter(SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path,
                                                   WORKBENCH_ARTIFACTS_ROOT=tmp_path), "variant")
    adapter.work_root = tmp_path / "runner-work"
    adapter._interpreter_verified = True
    workers = []

    class FailedWorker:
        stdin = SimpleNamespace(write=lambda _value: 0, flush=lambda: None)
        stdout = SimpleNamespace(readline=lambda: "")
        returncode = 7

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def start(*_args, **_kwargs):
        worker = FailedWorker()
        workers.append(worker)
        return worker

    monkeypatch.setattr(model_csmcir.subprocess, "Popen", start)
    for _ in range(2):
        with pytest.raises(adapters.AdapterWorkerError, match=r"CSMCIR worker exited with status 7"):
            adapter.encode_queries([str(checkpoint)], ["edit"], ["dress"])
    assert len(workers) == 1
    assert adapter._worker_failed == "CSMCIR worker exited with status 7"


def _resident_adapter(module, model_id: str, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    adapter = module.Adapter.__new__(module.Adapter)
    # Run the REAL base initialiser rather than assigning its fields by hand. `CompositionAdapter
    # .__init__` is what assigns `batch_size = resolved_batch_size()` (`adapters.py:273`), which
    # `_run` writes into the worker job JSON; a `__new__`-only fixture never had it, so every
    # `encode_*` raised `AttributeError: 'Adapter' object has no attribute 'batch_size'` before
    # reaching the resident-worker behaviour these tests are about.
    #
    # The base initialiser is called directly, not `module.Adapter(...)`: the subclass's own
    # `__init__` runs `_resolve`, which needs the registry and a staged checkpoint path. Nothing it
    # would set is read here -- this fixture already supplies `_model`, `_checkpoint` and
    # `_interpreter` -- while `batch_size` comes from the operator override exactly as in a real
    # run, so the fixture cannot drift to a value production would never use.
    adapters.CompositionAdapter.__init__(
        adapter, SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path), "variant")
    adapter.work_root = tmp_path / "work"
    adapter._interpreter = sys.executable
    adapter._worker = None
    adapter._worker_log = None
    adapter._worker_counter = 0
    adapter._worker_failed = None
    if model_id == "conesep":
        adapter._model = {"source_dir": "model-source"}
        adapter._checkpoint = {"resolved_path": str(checkpoint), "checkpoint_id": "variant",
                               "checkpoint_training_noise_pct": 50}
        adapter._job_root = None
        adapter._interpreter_verified = True
    else:
        adapter._model = {"source_dir": "model-source"}
        adapter._checkpoint = {"resolved_path": str(checkpoint)}
        adapter._job_root = None
        adapter._interpreter_verified = True
    return adapter


class _ResidentFakeWorker:
    def __init__(self, argv, *, query_dimension=8192, **kwargs):
        self.returncode = None
        self.query_dimension = query_dimension
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
        dimension = self.query_dimension if job["operation"] == "queries" else 8192
        shape = (len(job["image_paths"]), dimension)
        np.save(job["output_path"], np.full(shape, len(self.jobs), dtype=np.float32), allow_pickle=False)
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
        query_dimension = getattr(module.Adapter, "query_dimension", None) or module.Adapter.score_dimension
        worker = _ResidentFakeWorker(argv, query_dimension=query_dimension, **kwargs)
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
    query_dimension = getattr(module.Adapter, "query_dimension", None) or module.Adapter.score_dimension
    assert [result.shape for result in results] == [(1, 8192)] * 3 + [(1, query_dimension)] * 3
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


def test_ptha_native_query_layout_and_max_token_score() -> None:
    query = np.zeros((1, 256), dtype=np.float32)
    query[0, :3] = [0.5, -0.25, 0.75]
    targets = np.zeros((2, 32, 256), dtype=np.float32)
    targets[0, 4, :3] = [0.5, 1.0, 0.0]
    targets[0, 8, :3] = [-1.0, 0.0, 1.0]
    targets[1, 0, :3] = [0.25, -0.5, 1.0]
    expected = np.empty((len(query), len(targets)), dtype=np.float32)
    for qi, q in enumerate(query):
        for gi, gallery_item in enumerate(targets):
            expected[qi, gi] = max(float(np.dot(q, token)) for token in gallery_item)
    adapter = model_ptha_mtst.Adapter.__new__(model_ptha_mtst.Adapter)
    actual = adapter.score(query, targets.reshape(2, 8192))
    assert query.shape[1] == model_ptha_mtst.Adapter.query_dimension
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_ptha_upstream_save_memory_preserves_features_scores_and_ranks(tmp_path: Path) -> None:
    """Execute pinned consumer functions with real Q16/G64 batching and both storage flags."""
    import ast
    import typing

    import torch

    source_root = Path(__file__).resolve().parents[1] / "third_party/MTST_PTHA/MTST_PTHA/src"

    class FashionIQDataset:
        def __init__(self, mode: str) -> None:
            self.mode = mode
            self.dress_types = ["dress"]
            self.split = "val"
            self.rows = [("item0", "item20", ("red", "long")),
                         ("item1", "item21", ("blue", "short"))]

        def __len__(self) -> int:
            return 70 if self.mode == "classic" else 2

    class BatchLoader:
        def __init__(self, dataset, batch_size, **kwargs) -> None:
            assert batch_size in (16, 64)
            if dataset.mode == "relative":
                assert kwargs["shuffle"] is False
            self.dataset = dataset
            self.batch_size = batch_size

        def __iter__(self):
            if self.dataset.mode == "classic":
                names = [f"item{i}" for i in range(len(self.dataset))]
                for start in range(0, len(names), self.batch_size):
                    stop = start + self.batch_size
                    yield names[start:stop], torch.arange(start, min(stop, len(names))).reshape(-1, 1).float()
            else:
                for start in range(0, len(self.dataset.rows), self.batch_size):
                    rows = self.dataset.rows[start:start + self.batch_size]
                    yield ([row[0] for row in rows], [row[1] for row in rows],
                           [[row[2][0] for row in rows], [row[2][1] for row in rows]])

    def source_function(path: Path, name: str, namespace: dict) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), "exec"), namespace)

    class Model:
        device = torch.device("cpu")

        def extract_target_features(self, images, mode):
            assert mode == "mean"
            return images + 10, images + 20

        def inference(self, references, gallery, texts):
            return references[:, :1] + gallery[:, :1].T + len(texts)

    utils_ns = {
        "Union": typing.Union, "Tuple": typing.Tuple, "List": typing.List, "Path": Path,
        "CIRRDataset": type("CIRRDataset", (), {}),
        "FashionIQDataset": FashionIQDataset, "CIRCODataset": type("CIRCODataset", (), {}),
        "DataLoader": BatchLoader, "collate_fn": None, "device": torch.device("cpu"),
        "torch": torch, "tqdm": lambda values: values,
    }
    source_function(source_root / "utils.py", "extract_index_blip_features", utils_ns)
    model = Model()
    dataset = FashionIQDataset("classic")
    extracted = [utils_ns["extract_index_blip_features"](dataset, model, save_memory=flag)
                 for flag in (False, True)]
    (features, frozen), names = extracted[0]
    (features_memory, frozen_memory), names_memory = extracted[1]
    from workbench.replay.ptha_mtst import _bounded_native_gallery_store
    with _bounded_native_gallery_store(model, utils_ns, len(dataset), tmp_path / "gallery") as storage:
        disk_extracted = utils_ns["extract_index_blip_features"](dataset, model, save_memory=True)
    (features_disk, frozen_disk), names_disk = disk_extracted
    assert names == names_memory == names_disk
    torch.testing.assert_close(features.cpu(), features_memory.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(frozen.cpu(), frozen_memory.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(features.cpu(), features_disk.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(frozen.cpu(), frozen_disk.cpu(), rtol=0, atol=0)
    assert storage["batch_count"] == 70 and storage["stack_count"] == 2
    assert storage["batch_shapes"] == [[[64, 1], [64, 1]], [[6, 1], [6, 1]]]
    assert storage["batch_dtypes"] == [["torch.float32", "torch.float32"],
                                        ["torch.float32", "torch.float32"]]

    class Processor:
        def __getitem__(self, key):
            assert key == "eval"
            return str

    validate_ns = {
        "Union": typing.Union, "Tuple": typing.Tuple, "List": typing.List,
        "FashionIQDataset": FashionIQDataset, "DataLoader": BatchLoader,
        "collate_fn": None, "torch": torch, "np": np, "tqdm": lambda values: values,
        "itemgetter": __import__("operator").itemgetter,
    }
    source_function(source_root / "validate_blip.py", "generate_fiq_val_predictions", validate_ns)
    relative = FashionIQDataset("relative")
    outputs = [validate_ns["generate_fiq_val_predictions"](
        model, relative, extracted[index][1], extracted[index][0], Processor(), save_memory=flag)
        for index, flag in enumerate((False, True))]
    native_scores = outputs[0][0]
    adapter_scores = outputs[1][0]
    disk_scores = validate_ns["generate_fiq_val_predictions"](
        model, relative, names_disk, disk_extracted[0], Processor(), save_memory=True)[0]
    assert storage["feature_shapes"] == [[1], [1]]
    assert storage["feature_dtypes"] == ["torch.float32", "torch.float32"]
    assert storage["feature_nbytes"] == [280, 280]
    torch.testing.assert_close(native_scores, disk_scores, rtol=0, atol=0)
    assert torch.equal(torch.argsort(1 - native_scores, dim=-1),
                       torch.argsort(1 - disk_scores, dim=-1))
    assert native_scores.shape == disk_scores.shape == (2, 70)
    torch.testing.assert_close(native_scores, adapter_scores, rtol=0, atol=0)
    assert torch.equal(torch.argsort(1 - native_scores, dim=-1),
                       torch.argsort(1 - adapter_scores, dim=-1))
    del disk_extracted, features_disk, frozen_disk
    storage["cleanup"]()
    assert not (tmp_path / "gallery").exists()
    assert native_scores.shape == (2, 70)
