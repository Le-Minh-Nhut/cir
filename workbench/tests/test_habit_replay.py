"""Tests for HABIT replay wrapper and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.habit import (
    CHECKPOINTS,
    DEFAULT_VARIANT,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)

import json
from types import SimpleNamespace

import numpy as np

from workbench.replay.habit import _capture_native_rankings, _verify_native_receipt


def test_habit_native_capture_checks_both_passes_and_native_recall(tmp_path: Path):
    split = tmp_path / "image_splits"
    split.mkdir()
    (split / "split.dress.val.json").write_text('["ref", "target", "other"]')
    query = {"source_img_id": 0, "target_img_id": 1, "mod": {"str": "make it red"}}
    dataset = SimpleNamespace(path=str(tmp_path), test_queries_dress=[query],
                              test_targets_dress=[{"target_img_id": i} for i in range(3)])
    module = SimpleNamespace(np=np)
    query_features = np.array([[1.0, 0.0]], dtype=np.float32)
    target_features = np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    output = tmp_path / "dress.json"
    with _capture_native_rankings(module, dataset, "dress", output):
        scores = module.np.matmul(query_features, target_features.T).squeeze()
        scores[0] = -10e10
        first = [module.np.argsort(-scores)[:50]]
        second = [module.np.argsort(-scores)[:50]]
    assert np.array_equal(first[0], second[0])
    assert module.np is np
    receipt = json.loads(output.read_text())
    assert receipt["verified_identical_passes"] == 1
    with np.load(output.with_suffix(".scores.npz"), allow_pickle=False) as scores:
        assert scores["scores"].shape == (1, 3)
        assert scores["orders"].shape == (1, 3)
        assert scores["query_ids"].tolist() == ["0:ref:target"]
        assert scores["reference_positions"].tolist() == [0]
        assert scores["target_positions"].tolist() == [1]
    assert receipt["queries"][0]["target_rank"] == 1
    assert _verify_native_receipt(output, "dress", {"dress_r1": 100.0,
                                                     "dress_r10": 100.0,
                                                     "dress_r50": 100.0})


def test_habit_native_capture_blocks_different_recall_passes(tmp_path: Path):
    split = tmp_path / "image_splits"
    split.mkdir()
    (split / "split.dress.val.json").write_text('["ref", "target", "other"]')
    query = {"source_img_id": 0, "target_img_id": 1, "mod": {"str": "make it red"}}
    dataset = SimpleNamespace(path=str(tmp_path), test_queries_dress=[query],
                              test_targets_dress=[{"target_img_id": i} for i in range(3)])
    module = SimpleNamespace(np=np)
    output = tmp_path / "dress.json"
    with _capture_native_rankings(module, dataset, "dress", output):
        scores = module.np.matmul(np.array([[1.0, 0.0]], dtype=np.float32),
                                  np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]], dtype=np.float32).T).squeeze()
        scores[0] = -10e10
        module.np.argsort(-scores)
        scores[1], scores[2] = scores[2], scores[1]
        module.np.argsort(-scores)
    receipt = json.loads(output.read_text())
    assert "native argsort passes differ" in receipt["capture_error"]
    assert not _verify_native_receipt(output, "dress", {})

def test_habit_protocol_and_model_id():
    assert MODEL["model_id"] == "habit"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert DEFAULT_VARIANT == "fiq_n02"


def test_habit_checkpoint_variants():
    assert "fiq_n02" in CHECKPOINTS
    assert "fiq_n05" in CHECKPOINTS
    assert "fiq_n08" in CHECKPOINTS
    for var in CHECKPOINTS.values():
        assert var["checkpoint_training_noise_pct"] in (20, 50, 80)
        assert var["evaluation_noise_pct"] == 0


def test_habit_preflight_fails_on_empty_paths(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(ckpt_dir, data_dir, source_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("checkpoint" in b.lower() for b in blockers)
    assert any("dataset" in b.lower() for b in blockers)


def test_habit_validate_source_pinned():
    habit_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "HABIT"
    blockers = validate_source(habit_source)
    assert blockers == []


def test_habit_native_signature():
    habit_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "HABIT"
    test_py = habit_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category, txt_processors):"
            break
    else:
        pytest.fail("def test(...) signature not found in HABIT test.py")

def test_habit_category_dataset_only_materializes_requested_category() -> None:
    from workbench.replay.habit import _habit_category_dataset

    events = []
    class Dataset:
        def __init__(self):
            for category in ("dress", "shirt", "toptee"):
                queries, targets = self.get_test_data(category)
                setattr(self, f"test_queries_{category}", queries)
                setattr(self, f"test_targets_{category}", targets)
                datasets.save_obj((queries, targets), f"{category}.pkl")

        def get_test_data(self, category):
            events.append(("category", category))
            return ([category], [category])

    datasets = SimpleNamespace(FashionIQ=Dataset, save_obj=lambda obj, path: events.append(("save", path)))
    with _habit_category_dataset(datasets, "dress"):
        instance = Dataset()

    assert getattr(instance, "test_queries_dress") == ["dress"]
    assert getattr(instance, "test_targets_dress") == ["dress"]
    assert getattr(instance, "test_queries_shirt") == getattr(instance, "test_targets_shirt") == []
    assert getattr(instance, "test_queries_toptee") == getattr(instance, "test_targets_toptee") == []
    assert events == [("category", "dress")]
    assert datasets.FashionIQ.get_test_data is Dataset.get_test_data

def test_habit_worker_publishes_result_before_success_reply(tmp_path: Path) -> None:
    from workbench.composition.model_habit import _publish_worker_result

    output = tmp_path / "features.npy"
    events = []

    def reply(ok: bool) -> None:
        events.append((ok, np.load(output, allow_pickle=False)))

    features = np.asarray([[1.0, 2.0]], dtype=np.float32)
    _publish_worker_result(output, features, reply)

    assert len(events) == 1 and events[0][0] is True
    assert np.array_equal(events[0][1], features)

def test_habit_control_failure_persists_deltas_before_strict_refusal(tmp_path: Path) -> None:
    from workbench.scripts.habit_native_adapter_pair import _record_adapter_controls

    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    actual = np.asarray([[1.0, 2.25]], dtype=np.float32)
    arrays = {"query_repeat_expected": expected, "query_repeat_actual": actual,
              "query_repeat_delta": np.abs(expected - actual)}
    controls = {"full_forward_repeat_equal": False, "b1_bn_equal": True,
                "query_repeat_max_abs_delta": 0.25,
                "tolerance": {"comparison": "exact array equality", "atol": 0, "rtol": 0}}
    with pytest.raises(ValueError, match="full_forward_repeat"):
        _record_adapter_controls(tmp_path, arrays, controls)
    with np.load(tmp_path / "adapter_control_deltas.npz", allow_pickle=False) as recorded:
        assert np.array_equal(recorded["query_repeat_delta"], np.asarray([[0.0, 0.25]], dtype=np.float32))
    recorded_controls = json.loads((tmp_path / "adapter_control_comparison.json").read_text())
    assert recorded_controls["full_forward_repeat_equal"] is False
    assert recorded_controls["b1_bn_equal"] is True
    assert recorded_controls["tolerance"]["atol"] == recorded_controls["tolerance"]["rtol"] == 0
