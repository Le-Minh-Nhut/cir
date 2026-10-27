"""CPU-only contract checks for the BLIP composition adapter subprocess clients."""
from __future__ import annotations

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

@pytest.mark.parametrize("module", MODULES)
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
    with pytest.raises(adapters.AdapterWorkerError, match=r"(csmcir|conesep|ptha_mtst).*scope.*7"):
        adapter.encode_queries([str(checkpoint)], ["edit"], [None])
