"""Tests for CLVC-Net replay wrapper and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.clvc_net import (
    EXPECTED_CATEGORY_FILES,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_clvc_net_protocol_is_val_split():
    assert PROTOCOL == "fashioniq_val_split"


def test_clvc_net_requires_six_checkpoint_files():
    assert len(EXPECTED_CATEGORY_FILES) == 3
    for cat in CATEGORIES:
        assert len(EXPECTED_CATEGORY_FILES[cat]) == 2
        assert EXPECTED_CATEGORY_FILES[cat] == (f"{cat}_local_model.pt", f"{cat}_global_model.pt")


def test_clvc_net_preflight_fails_on_empty_paths(tmp_path: Path):
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
    assert any("unresolved" in b.lower() for b in blockers)


def test_clvc_net_test_function_signature():
    """Verify CLVC-Net test.py test() signature is (params, local_model, global_model, testset, dataname) - 5 args."""
    clvc_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "CLVCNet"
    test_py = clvc_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, local_model, global_model, testset, dataname):"
            break
    else:
        pytest.fail("def test(...) not found in CLVCNet test.py")
