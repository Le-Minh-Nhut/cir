"""Tests for HINT replay cache isolation contract."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.common import isolated_cache_root
from workbench.replay.hint import evaluate_categories


@pytest.fixture
def dummy_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "fashioniq_dataset"
    root.mkdir()
    (root / "captions").mkdir()
    (root / "image_splits").mkdir()
    (root / "resized_image").mkdir()
    for cat in CATEGORIES:
        (root / "captions" / f"cap.{cat}.val.json").write_text("[]")
        (root / "captions" / f"cap.{cat}.train.json").write_text("[]")
        (root / "captions" / f"correction_dict_{cat}.json").write_text("{}")
        (root / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (root / "resized_image" / cat).mkdir()
    return root


def test_hint_stale_cache_in_canonical_root_not_reused(dummy_dataset_root: Path):
    """(a) Stale cache in the canonical root is NOT passed to or visible in the isolated cache."""
    # Place a poison stale cache in the canonical root
    stale_json = dummy_dataset_root / "fashion_iq_data.json"
    stale_json.write_text(json.dumps({"marker": "STALE_CACHE_POISON"}))
    stale_pkl = dummy_dataset_root / "test_queries_dress.pkl"
    stale_pkl.write_bytes(b"STALE_PKL_POISON")

    observed_cache_roots = []

    def fake_fashioniq(path: str, transform: any = None):
        cache_path = Path(path)
        observed_cache_roots.append(cache_path)
        # Verify the poison files do NOT exist in the path passed to FashionIQ
        assert not (cache_path / "fashion_iq_data.json").exists()
        assert not (cache_path / "test_queries_dress.pkl").exists()
        return SimpleNamespace()

    mock_torch = MagicMock()
    mock_datasets = MagicMock()
    mock_datasets.FashionIQ = fake_fashioniq
    mock_data_utils = MagicMock()
    mock_test_module = MagicMock()
    mock_test_module.test.return_value = [("dress_r1", 10.0), ("dress_r10", 20.0), ("dress_r50", 30.0)]
    mock_txt_processors = MagicMock()
    mock_load_model = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))

    with (
        patch("workbench.replay.hint._load_replay_modules", return_value=(mock_torch, mock_datasets, (mock_data_utils, mock_test_module), (mock_txt_processors, "patched:test-fixture"))),
        patch("workbench.replay.hint.load_trusted_whole_model", return_value=MagicMock()),
    ):
        results, agg = evaluate_categories(
            checkpoint_root=dummy_dataset_root,
            dataset_root=dummy_dataset_root,
            source_root=dummy_dataset_root,
            categories=("dress",),
        )

    assert len(observed_cache_roots) == 1
    assert observed_cache_roots[0] != dummy_dataset_root


def test_hint_clean_root_works(dummy_dataset_root: Path):
    """(b) A clean dataset root works without error and provides the symlinked inputs."""
    with isolated_cache_root(dummy_dataset_root) as cache_root:
        assert (cache_root / "captions").is_dir()
        assert (cache_root / "image_splits").is_dir()
        assert (cache_root / "resized_image").is_dir()
        # Verify no cache files exist in fresh cache root
        assert not (cache_root / "fashion_iq_data.json").exists()


def test_hint_canonical_root_left_unmodified(dummy_dataset_root: Path):
    """(c) Caches created during evaluation are written to temp root, leaving canonical root untouched."""
    canonical_files_before = set(dummy_dataset_root.rglob("*"))

    def fake_fashioniq_writing_cache(path: str, transform: any = None):
        cache_path = Path(path)
        # Simulate native dataset writing cache files to self.path
        (cache_path / "fashion_iq_data.json").write_text(json.dumps([{"item": 1}]))
        (cache_path / "test_queries_dress.pkl").write_bytes(b"DATA")
        return SimpleNamespace()

    mock_torch = MagicMock()
    mock_datasets = MagicMock()
    mock_datasets.FashionIQ = fake_fashioniq_writing_cache
    mock_data_utils = MagicMock()
    mock_test_module = MagicMock()
    mock_test_module.test.return_value = [("dress_r1", 10.0), ("dress_r10", 20.0), ("dress_r50", 30.0)]
    mock_txt_processors = MagicMock()
    mock_load_model = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))

    with (
        patch("workbench.replay.hint._load_replay_modules", return_value=(mock_torch, mock_datasets, (mock_data_utils, mock_test_module), (mock_txt_processors, "patched:test-fixture"))),
        patch("workbench.replay.hint.load_trusted_whole_model", return_value=MagicMock()),
    ):
        evaluate_categories(
            checkpoint_root=dummy_dataset_root,
            dataset_root=dummy_dataset_root,
            source_root=dummy_dataset_root,
            categories=("dress",),
        )

    canonical_files_after = set(dummy_dataset_root.rglob("*"))
    assert canonical_files_before == canonical_files_after
    assert not (dummy_dataset_root / "fashion_iq_data.json").exists()
    assert not (dummy_dataset_root / "test_queries_dress.pkl").exists()


def test_hint_isolated_root_lives_through_evaluation(dummy_dataset_root: Path):
    """(d) The isolated root directory exists and remains alive through evaluation."""
    active_cache_roots = []

    def fake_test(params, model, dataset, category, txt_processors):
        cache_root = dataset._cache_root
        active_cache_roots.append((category, cache_root, cache_root.exists()))
        return [(f"{category}_r1", 1.0), (f"{category}_r10", 2.0), (f"{category}_r50", 3.0)]

    def fake_fashioniq(path: str, transform: any = None):
        ds = SimpleNamespace()
        ds._cache_root = Path(path)
        return ds

    mock_torch = MagicMock()
    mock_datasets = MagicMock()
    mock_datasets.FashionIQ = fake_fashioniq
    mock_data_utils = MagicMock()
    mock_test_module = MagicMock()
    mock_test_module.test = fake_test
    mock_txt_processors = MagicMock()
    mock_load_model = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))

    with (
        patch("workbench.replay.hint._load_replay_modules", return_value=(mock_torch, mock_datasets, (mock_data_utils, mock_test_module), (mock_txt_processors, "patched:test-fixture"))),
        patch("workbench.replay.hint.load_trusted_whole_model", return_value=MagicMock()),
    ):
        results, agg = evaluate_categories(
            checkpoint_root=dummy_dataset_root,
            dataset_root=dummy_dataset_root,
            source_root=dummy_dataset_root,
            categories=("dress", "shirt", "toptee"),
        )

    assert len(active_cache_roots) == 3
    for cat, cache_root, exists in active_cache_roots:
        assert exists, f"Cache root for {cat} must exist during test() call"
    # Check that after evaluation exits, the temp root was cleaned up
    assert not active_cache_roots[0][1].exists()
