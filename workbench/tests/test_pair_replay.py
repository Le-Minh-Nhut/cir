"""Tests for PAIR replay wrapper and preflight validation."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.pair import (
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_pair_protocol_is_val_split():
    assert PROTOCOL == "fashioniq_val_split"


def test_pair_checkpoint_variants_registered():
    assert "pair_b1" in CHECKPOINTS
    assert "pair_b2" in CHECKPOINTS


def test_pair_preflight_fails_on_empty_paths(tmp_path: Path):
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


def test_pair_checkpoint_unverified_mapping(tmp_path: Path):
    fake_ckpt = tmp_path / DEFAULT_CHECKPOINT["filename"]
    fake_ckpt.write_bytes(b"dummy checkpoint data")
    blockers = validate_checkpoint(tmp_path)
    assert any("unverified" in b.lower() for b in blockers)


def test_pair_test_function_signature():
    """Verify PAIR test.py:6 test() signature is (params, model, testset, category) - 4 args."""
    pair_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "PAIR"
    test_py = pair_source / "test.py"
    assert test_py.is_file()
    # Read function definition line from test.py
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category):"
            break
    else:
        pytest.fail("def test(...) not found in PAIR test.py")
