"""Tests for ENCODER replay wrapper and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.replay.encoder import (
    AUXILIARY_ASSET_NAME,
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    MODEL,
    PROTOCOL,
    preflight,
    validate_auxiliary_assets,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_encoder_protocol_is_val_split():
    assert PROTOCOL == "fashioniq_val_split"


def test_encoder_checkpoint_variants_registered():
    assert "fashioniq" in CHECKPOINTS


def test_encoder_preflight_fails_on_empty_paths(tmp_path: Path):
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
    assert any("auxiliary asset" in b.lower() for b in blockers)


def test_encoder_auxiliary_asset_validation(tmp_path: Path):
    blockers = validate_auxiliary_assets(tmp_path)
    assert len(blockers) == 1
    assert AUXILIARY_ASSET_NAME in blockers[0]

    asset = tmp_path / AUXILIARY_ASSET_NAME
    asset.write_bytes(b"dummy open_clip model")
    assert validate_auxiliary_assets(tmp_path) == []


def test_encoder_test_function_signature():
    encoder_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "ENCODER"
    test_py = encoder_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category):"
            break
    else:
        pytest.fail("def test(...) not found in ENCODER test.py")
