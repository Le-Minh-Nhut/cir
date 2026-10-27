from __future__ import annotations

from importlib import import_module
import json
import sys

import numpy as np
from PIL import Image

from workbench.composition.inputs import adapt_query_inputs


MODELS = {
    "habit": "workbench.composition.model_habit",
    "intent": "workbench.composition.model_intent",
    "hint": "workbench.composition.model_hint",
    "tme": "workbench.composition.model_tme",
}
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


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

    def require_interpreter(model, config=None):
        return sys.executable

    def fake_subprocess(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "workbench.composition.model_tme"]
        job = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        with Image.open(job["image_paths"][0]) as staged:
            assert staged.size == reference.size
            assert staged.getbbox() is None
        assert job["operation"] == "queries"
        assert job["texts"] == ["turn it blue"]
        seen.append(job["image_paths"][0])
        np.save(job["output_path"], np.zeros((1, 256), dtype=np.float32), allow_pickle=False)

    from pathlib import Path
    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "require_verified_model_interpreter", require_interpreter)
    monkeypatch.setattr(module.subprocess, "run", fake_subprocess)
    result = adapter.encode_queries([str(black_path)], [text], [None])
    assert result.shape == (1, 256)
    assert seen == [str(black_path.resolve())]
    adapter.close()
