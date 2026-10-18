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
