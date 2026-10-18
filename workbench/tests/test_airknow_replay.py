"""Tests for Air-Know replay wrapper and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.airknow import (
    CHECKPOINTS,
    DEFAULT_VARIANT,
    KNOWN_AUTHOR_SHA256,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_airknow_protocol_and_model_id():
    assert MODEL["model_id"] == "airknow"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert DEFAULT_VARIANT == "fiq_n05"


def test_airknow_checkpoint_variants():
    assert "fiq_n05" in CHECKPOINTS
    assert "fiq_n08" in CHECKPOINTS
    assert "fiq_n02" not in CHECKPOINTS
    assert len(KNOWN_AUTHOR_SHA256) == 2
    for var in CHECKPOINTS.values():
        assert var["checkpoint_training_noise_pct"] in (50, 80)
        assert var["evaluation_noise_pct"] == 0


def test_airknow_preflight_fails_on_empty_paths(tmp_path: Path):
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


def test_airknow_validate_source_pinned():
    airknow_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "AirKnow"
    blockers = validate_source(airknow_source)
    assert blockers == []


def test_airknow_native_signature_and_train_mode():
    airknow_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "AirKnow"
    test_py = airknow_source / "test_BLIP2.py"
    assert test_py.is_file()
    lines = test_py.read_text().splitlines()
    found_sig = False
    for i, line in enumerate(lines):
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category, txt_processors):"
            found_sig = True
            # verify model.train() is called on the next line
            assert lines[i + 1].strip() == "model.train()"
            break
    assert found_sig, "def test(...) not found in test_BLIP2.py"
