from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OBSERVER_PATH = REPO / "workbench/scripts/third_model_native_observer.py"


def load_observer():
    spec = importlib.util.spec_from_file_location("third_model_native_observer_test", OBSERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observes_native_post_mask_scores_and_repeat_sort(tmp_path):
    observer_module = load_observer()
    evaluator = SimpleNamespace(np=np)
    observer = observer_module.observe_test_numpy(evaluator, 2, "test")
    try:
        sims = evaluator.np.matmul(np.eye(2, dtype=np.float32),
                                   np.arange(6, dtype=np.float32).reshape(2, 3))
        sims[0, 0] = -10e10
        sims[1, 1] = -10e10
        orders = [evaluator.np.argsort(-sims[row]) for row in range(2)]
        for row in range(2):
            evaluator.np.argsort(-sims[row])
    finally:
        evaluator.np = np

    receipt = observer_module.write_native_scores(
        observer, tmp_path / "native.npz", model_id="test", category="dress",
        query_ids=["q0", "q1"], gallery_ids=["g0", "g1", "g2"],
        reference_positions=[0, 1], target_positions=[2, 2])
    with np.load(tmp_path / "native.npz") as data:
        np.testing.assert_array_equal(data["scores"], sims)
        np.testing.assert_array_equal(data["orders"], np.stack(orders))
    assert receipt["matmul_calls"] == 1
    assert receipt["identical_repeat_passes"] is True
    assert receipt["query_count"] == 2 and receipt["gallery_count"] == 3

def test_target_rank_uses_native_descending_score_order():
    scores = np.asarray([[-1e10, 0.8, 0.1, 0.4]], dtype=np.float32)
    order = np.argsort(-scores, axis=1)
    assert order[0, -1] == 0
    assert np.flatnonzero(order[0] == 1)[0] + 1 == 1
    assert order[0, :3].tolist() == [1, 3, 2]

def test_native_observer_requires_two_full_sort_passes(tmp_path):
    observer_module = load_observer()
    evaluator = SimpleNamespace(np=np)
    observer = observer_module.observe_test_numpy(evaluator, 1, "fixture")
    scores = evaluator.np.matmul(np.asarray([[2.0]], dtype=np.float32),
                                 np.asarray([[1.0, 0.0]], dtype=np.float32))
    scores[0, 0] = -10e10
    evaluator.np.argsort(-scores[0])
    try:
        observer_module.write_native_scores(
            observer, tmp_path / "incomplete.npz", model_id="fixture", category="dress",
            query_ids=["q"], gallery_ids=["ref", "target"],
            reference_positions=[0], target_positions=[1])
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete native rank capture was accepted")
