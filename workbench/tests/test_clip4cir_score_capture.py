from __future__ import annotations

import numpy as np

from types import SimpleNamespace

import torch

from workbench.replay.clip4cir_score_capture import bind_fashioniq_rows
from workbench.replay.clip4cir_score_capture import _ArgsortCapture


def test_argsort_capture_returns_native_order_and_records_scores() -> None:
    seen: list[tuple[np.ndarray, np.ndarray, tuple[object, ...], dict[str, object]]] = []
    native_argsort = np.argsort
    observer = _ArgsortCapture(
        native_argsort,
        lambda values, order, args, kwargs: seen.append((values.copy(), order.copy(), args, kwargs)),
    )
    scores = np.array([[2.0, 1.0, 3.0]], dtype=np.float32)
    order = observer(scores, axis=-1)
    np.testing.assert_array_equal(order, native_argsort(scores, axis=-1))
    np.testing.assert_array_equal(seen[0][0], scores)
    np.testing.assert_array_equal(seen[0][1], order)
    assert seen[0][2] == ()
    assert seen[0][3] == {"axis": -1}

def test_bind_fashioniq_rows_preserves_source_scores_order_and_labels() -> None:
    dataset = SimpleNamespace(
        triplets=[{"candidate": "ref", "target": "target"}],
        dress_types=["dress"],
    )
    scores = torch.tensor([[0.2, 0.1, 0.4]], dtype=torch.float32)
    order = torch.tensor([[1, 0, 2]], dtype=torch.int64)
    rows = bind_fashioniq_rows(dataset, scores, order, ["ref", "target", "other"])
    assert rows[0]["query_id"] == "dress#row0:ref:target"
    assert rows[0]["target_id"] == "target"
    assert rows[0]["reference_id"] == "ref"
    np.testing.assert_array_equal(rows[0]["distances"], scores[0])
    np.testing.assert_array_equal(rows[0]["native_order"], order[0])

def test_comparison_uses_native_emitted_order_and_positive_rank(tmp_path, capsys) -> None:
    import argparse
    import json
    from workbench.replay.clip4cir_diagnostic import _compare

    native_dir, adapter_dir = tmp_path / "native", tmp_path / "adapter"
    native_dir.mkdir()
    adapter_dir.mkdir()
    gallery = np.asarray(["ref", "target", "other"])
    scores = np.asarray([[0.2, 0.1, 0.4]], dtype=np.float32)
    emitted_order = np.argsort(scores, axis=-1)
    for category in ("dress", "shirt", "toptee"):
        np.savez(native_dir / f"{category}.native.npz", query_ids=["q"], target_ids=["target"],
                 reference_ids=["ref"], gallery_ids=gallery, distances=scores, native_order=emitted_order)
        np.savez(adapter_dir / f"{category}.adapter.npz", query_ids=["q"], target_ids=["target"],
                 reference_ids=["ref"], gallery_ids=gallery, scores=1 - scores)
    _compare(argparse.Namespace(native_dir=native_dir, adapter_dir=adapter_dir, model_id="model"))
    result = json.loads(capsys.readouterr().out)
    for category in result["categories"]:
        assert category["native_positive_rank_r1"] == 1
        assert category["adapter_positive_rank_r1"] == 1
        assert category["rank_mismatch_queries"] == 0


def test_production_worker_receives_declared_job_schema(tmp_path, monkeypatch) -> None:
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import workbench.composition.model_clip4cir as module

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    observed = {}

    class Worker:
        def __init__(self, argv, **kwargs):
            job = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            observed.update(job)
            np.save(job["output_path"], np.ones((1, 640), dtype=np.float32))
            self.stdin = SimpleNamespace(write=lambda value: None, flush=lambda: None)
            self.stdout = SimpleNamespace(readline=lambda: 'CIRCUS_WORKER:{"ok":true}\n')

        def poll(self):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    monkeypatch.setattr(module, "worker_environment", lambda: {})
    adapter = module.Clip4CirBase.__new__(module.Clip4CirBase)
    adapter.model_id = "combiner_rn50x4_noft"
    adapter.work_root = tmp_path / "work"
    adapter._job_root = None
    adapter._worker_counter = 0
    adapter._worker_failed = None
    adapter._interpreter = "python"
    adapter._interpreter_verified = True
    adapter._checkpoint = {"checkpoint_id": "fixture", "checkpoint_mapping_status": "VERIFIED_METADATA"}
    adapter.members = {"combiner": checkpoint}
    adapter.source_root = tmp_path / "source"
    adapter.batch_size = 1
    adapter._worker = None
    adapter._worker_log = None
    features = module._run(adapter, "targets", [str(checkpoint)], (), [None])
    assert features.shape == (1, 640)
    assert {key: value for key, value in observed.items() if key != "output_path"} == {
        "operation": "targets", "model_id": "combiner_rn50x4_noft",
        "image_paths": [str(checkpoint.resolve())], "texts": [], "categories": [None],
        "source_root": str(tmp_path / "source"), "members": {"combiner": str(checkpoint)},
        "checkpoint_id": "fixture", "batch_size": 1,
    }
    assert Path(observed["output_path"]).parent == Path(adapter._job_root)
