"""Small checks for DCNet FashionIQ paired-score ranking contract."""
from __future__ import annotations

import numpy as np



def test_source_trace_captures_final_score_and_topk_only_in_native_frame():
    import sys
    import torch
    from workbench.replay.dcnet_fashioniq_pair import _source_trace

    source_file = "/pinned/DCNet/trainer/trainer_joint.py"
    namespace = {"torch": torch}
    code = compile(
        "def _valid_epoch():\n"
        "    category = 'dress'\n"
        "    j = 0\n"
        "    score_comb = torch.tensor([3.0, 1.0])\n"
        "    _, topk = score_comb.topk(k=2)\n"
        "    result = topk.tolist()\n"
        "    return result\n",
        source_file, "exec")
    exec(code, namespace)
    captured = {}
    trace = _source_trace(captured, source_file, capture_line=6)
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        namespace["_valid_epoch"]()
    finally:
        sys.settrace(previous)
    assert captured["dress"][0]["query_row"] == 0
    np.testing.assert_array_equal(captured["dress"][0]["scores"], [3.0, 1.0])
    np.testing.assert_array_equal(captured["dress"][0]["top50"], [0, 1])

def test_derived_order_is_stable_descending_and_complete():
    from workbench.replay.dcnet_fashioniq_pair import _derived_order

    scores = np.asarray([[2.0, 3.0, 3.0, -1.0]], dtype=np.float32)
    np.testing.assert_array_equal(_derived_order(scores), [[1, 2, 0, 3]])
from workbench.replay.dcnet_fashioniq_pair import _fused_native
from workbench.replay.dcnet_fashioniq_pair import _write_native_archive


def test_native_archive_uses_pickle_free_query_metadata(tmp_path):
    rows = {
        "query_rows": [
            {"candidate": "ref-a", "target": "target-a"},
            {"candidate": "ref-a", "target": "target-b"},
        ],
        "gallery_ids": ["ref-a", "target-a", "target-b"],
        "composition": np.arange(6, dtype=np.float32).reshape(2, 3),
        "correction": np.ones((2, 3), dtype=np.float32),
    }
    path = tmp_path / "native.npz"
    _write_native_archive(path, "dress", rows)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["gallery_ids"].tolist() == rows["gallery_ids"]
        metadata = __import__("json").loads(archive["query_metadata_json"].item())
    assert metadata == [
        {"category": "dress", "row": 0, "candidate": "ref-a", "target": "target-a", "reference": "ref-a"},
        {"category": "dress", "row": 1, "candidate": "ref-a", "target": "target-b", "reference": "ref-a"},
    ]

def test_native_fusion_matches_source_global_gallery_log_softmax():
    import torch
    import torch.nn.functional as F

    composition = torch.tensor([[3.0, 2.0, -1.0], [0.5, -0.5, 1.0]])
    correction = torch.tensor([[0.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    actual = _fused_native(composition, correction)
    expected = (F.log_softmax(composition, -1) + F.log_softmax(correction, -1)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_reference_mask_precedes_each_native_gallery_softmax():
    import torch
    import torch.nn.functional as F
    from workbench.composition.model_dcnet import _mask_reference_scores

    composition = torch.tensor([[2.0, 1.0, 0.0]])
    correction = torch.tensor([[0.0, 3.0, 1.0]])
    _mask_reference_scores(composition, correction, [1])
    fused = F.log_softmax(composition, dim=-1) + F.log_softmax(correction, dim=-1)
    torch.testing.assert_close(composition[0, 1], torch.tensor(-10e10))
    torch.testing.assert_close(correction[0, 1], torch.tensor(-10e10))
    assert fused.argmax(dim=-1).item() == 0
    post_fusion = (F.log_softmax(torch.tensor([[2.0, 1.0, 0.0]]), dim=-1) +
                   F.log_softmax(torch.tensor([[0.0, 3.0, 1.0]]), dim=-1))
    assert not torch.allclose(fused, post_fusion)


def test_recovery_rejects_active_model_controller_and_hashes_both_archives(tmp_path):
    import hashlib
    import json
    import pytest
    from workbench.replay.dcnet_fashioniq_pair_recover import (
        _load_native_category, _require_distinct_controller,
    )
    from workbench.replay.dcnet_fashioniq_pair import _write_native_archive

    _require_distinct_controller("/env/ctl312h/bin/python", "/env/dcnet310/bin/python")
    with pytest.raises(RuntimeError, match="must not run in the model interpreter"):
        _require_distinct_controller("/env/dcnet310/bin/python", "/env/dcnet310/bin/python")
    native_root = tmp_path / "native"
    native_root.mkdir()
    rows = {"query_rows": [{"candidate": "ref", "target": "target"}],
            "gallery_ids": ["ref", "target"],
            "composition": np.asarray([[2.0, 1.0]], dtype=np.float32),
            "correction": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    component = native_root / "native-dress-scores.npz"
    _write_native_archive(component, "dress", rows)
    final = native_root / "native-dress-final-scores.npz"
    np.savez_compressed(final, scores=np.asarray([[0.0, 1.0]], dtype=np.float32),
                        top50=np.asarray([[1, 0]], dtype=np.int64),
                        derived_order=np.asarray([[1, 0]], dtype=np.int64))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = [{"category": "dress", "row": 0, "candidate": "ref",
                 "target": "target", "reference": "ref"}]
    identity = {"native_final_score_file": final.name, "native_final_score_sha256": sha(final),
                "query_count": 1, "gallery_count": 2,
                "query_order_sha256": hashlib.sha256(json.dumps(metadata, separators=(",", ":")).encode()).hexdigest(),
                "gallery_order_sha256": hashlib.sha256(json.dumps(rows["gallery_ids"], separators=(",", ":")).encode()).hexdigest()}
    expected = {"final": sha(final), "components": sha(component)}
    scores, top50, order, gallery_ids, query_meta = _load_native_category(
        native_root, "dress", identity, expected)
    np.testing.assert_array_equal(scores, [[0.0, 1.0]])
    np.testing.assert_array_equal(top50, [[1, 0]])
    assert gallery_ids == rows["gallery_ids"] and query_meta == metadata
    expected["components"] = "0" * 64
    with pytest.raises(ValueError, match="components native archive differs"):
        _load_native_category(native_root, "dress", identity, expected)



def test_hit_flip_accounting_keeps_gained_and_lost_queries_separate():
    from workbench.replay.dcnet_fashioniq_pair_recover import _hit_flips

    native = np.asarray([[9, 8, 7], [9, 8, 7], [9, 8, 7]], dtype=np.float32)
    adapted = np.asarray([[8, 9, 7], [9, 7, 8], [9, 8, 7]], dtype=np.float32)
    targets = np.asarray([1, 1, 0])
    assert _hit_flips(native, adapted, targets, (1, 2)) == {
        1: {"gained": [0], "lost": []},
        2: {"gained": [], "lost": [1]},
    }


def test_native_component_replay_preserves_source_top50():
    import torch
    import torch.nn.functional as F

    composition = torch.tensor([[2.0, -1.0, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]])
    correction = torch.tensor([[0.3, 1.4, -0.7, 0.2], [1.0, -0.3, 0.6, 0.5]])
    source_final = (F.log_softmax(composition, dim=-1) +
                    F.log_softmax(correction, dim=-1))
    replayed = (F.log_softmax(composition.clone(), dim=-1) +
                F.log_softmax(correction.clone(), dim=-1))
    torch.testing.assert_close(replayed, source_final, rtol=0, atol=0)
    source_top50 = source_final.topk(k=4, dim=-1, largest=True, sorted=True).indices
    replayed_top50 = replayed.topk(k=4, dim=-1, largest=True, sorted=True).indices
    torch.testing.assert_close(replayed_top50, source_top50)


def test_score_capture_is_opt_in_and_preserves_packed_inputs(tmp_path, monkeypatch):
    import workbench.composition.model_dcnet as module

    adapter = object.__new__(module.Adapter)
    adapter._packed_layout = {"query_width": 6, "target_width": 2}
    query = np.arange(12, dtype=np.float32).reshape(2, 6)
    target = np.arange(8, dtype=np.float32).reshape(4, 2)
    component_path = tmp_path / "capture" / "components"

    def fake_score(self, queries, targets, reference_indices=None, capture_components_path=None):
        assert capture_components_path == component_path
        assert np.array_equal(queries, query)
        assert np.array_equal(targets, target)
        return np.zeros((len(queries), len(targets)), dtype=np.float32)

    monkeypatch.setattr(module, "_score", fake_score)
    result = adapter.score(query, target, capture_components_path=component_path)
    assert component_path.parent.is_dir()
    assert result.shape == (2, 4)


def test_native_g1_capture_scope_is_fixed_and_feature_pack_is_ordered():
    from workbench.replay.dcnet_native_g1_capture import QUERY_ROWS, _pack
    import torch

    assert QUERY_ROWS == (657, 933, 1362, 1980)
    packed = _pack({"a": torch.tensor([[1.0]]), "b": torch.tensor([[3.0]])}, ("a", "b"))
    torch.testing.assert_close(packed, torch.tensor([[1.0, 3.0]]))
def test_native_capture_rewrites_only_isolated_config_output(tmp_path):
    import json
    from workbench.replay.dcnet_native_g1_capture import _isolated_config

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = {"trainer": {"save_dir": "/must/not/write"}, "arch": {"type": "DCNet"}}
    (checkpoint / "config.json").write_text(json.dumps(source), encoding="utf-8")
    config_root = tmp_path / "isolated"
    config_root.mkdir()
    _isolated_config(checkpoint, config_root)
    configured = json.loads((config_root / "config.json").read_text(encoding="utf-8"))
    assert configured["trainer"]["save_dir"] == str(config_root / "native-work")
    assert configured["arch"] == source["arch"]
    assert json.loads((checkpoint / "config.json").read_text(encoding="utf-8")) == source


def test_request_surfaces_worker_traceback(monkeypatch, tmp_path):
    import json
    import pytest
    from io import StringIO
    import workbench.composition.model_dcnet as module

    adapter = object.__new__(module.Adapter)
    adapter._job_root = tmp_path
    adapter._worker_counter = 0
    adapter._worker_log = None
    adapter._interpreter = "/fake/python"
    source = tmp_path / "source"
    model = source / "model" / "model.py"
    model.parent.mkdir(parents=True)
    model.touch()
    response = {"executable": "/fake/python", "prefix": "/fake", "cwd": str(tmp_path),
                "argv": [], "model_module": str(model), "torch_module": "/fake/torch.py",
                "torchvision_module": "/fake/vision.py", "torch_version": "test",
                "torchvision_version": "test", "device": "cuda:0", "ok": False,
                "operation": "queries", "error_type": "RuntimeError", "error": "device mismatch",
                "traceback": "Traceback (most recent call last):\nRuntimeError: device mismatch"}

    class Worker:
        def __init__(self, stdout):
            self.stdout = stdout

        def poll(self):
            return None

    worker = Worker(StringIO("DCNET_WORKER:" + json.dumps(response) + "\n"))
    adapter._worker = None

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: worker)
    job = {"output_path": str(tmp_path / "out.npy"), "source_root": str(source)}
    with pytest.raises(module.Refused, match="operation='queries'.*RuntimeError: device mismatch") as caught:
        module._request(adapter, job, (1, 1))
    assert "Traceback (most recent call last)" in str(caught.value)


def test_adapter_score_rejects_invalid_native_reference_indices():
    import pytest
    import workbench.composition.model_dcnet as module

    adapter = object.__new__(module.Adapter)
    adapter._packed_layout = {"query_width": 6, "target_width": 2}
    with pytest.raises(ValueError, match="one in-range gallery index per query"):
        adapter.score(np.zeros((2, 6)), np.zeros((3, 2)), reference_indices=[2])

