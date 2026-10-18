"""Tests for TG-CIR protocol parity and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.tgcir import (
    MODEL,
    preflight,
)


def test_tgcir_protocol_is_val_split():
    assert MODEL["native_protocol"] == "fashioniq_val_split"


def test_tgcir_test_function_signature_and_gallery_policy():
    """Verify test.py:6 signature (params, model, testset, category) and -10e10 reference exclusion."""
    tgcir_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "TGCIR"
    test_py = tgcir_source / "test.py"
    assert test_py.is_file()

    content = test_py.read_text()
    assert "def test(params, model, testset, category):" in content
    assert "sims[i, test_targets_id.index(t['source_img_id'])] = -10e10" in content


def test_tgcir_datasets_pair_union():
    """Verify datasets.py constructs pair union gallery (endpoints only, not full split)."""
    tgcir_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "TGCIR"
    datasets_py = tgcir_source / "datasets.py"
    assert datasets_py.is_file()

    content = datasets_py.read_text()
    # Check that test_targets_id is constructed from source_img_id and target_img_id in test_queries
    assert "test_targets_id = []" in content
    assert "i['source_img_id'] not in test_targets_id" in content
    assert "i['target_img_id'] not in test_targets_id" in content


def test_tgcir_preflight_blocks_empty_environment(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(source_dir, data_dir, ckpt_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("raw input missing" in b.lower() for b in blockers)
    assert any("mapping unresolved" in b.lower() for b in blockers)
