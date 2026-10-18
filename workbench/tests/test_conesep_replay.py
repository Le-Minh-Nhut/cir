"""Tests for ConeSep replay wrapper and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.conesep import (
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


def test_conesep_protocol_and_model_id():
    assert MODEL["model_id"] == "conesep"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert DEFAULT_VARIANT == "fiq_n02"


def test_conesep_checkpoint_variants():
    assert "fiq_n02" in CHECKPOINTS
    assert "fiq_n05" in CHECKPOINTS
    assert "fiq_n08" in CHECKPOINTS
    assert len(KNOWN_AUTHOR_SHA256) == 3
    for var in CHECKPOINTS.values():
        assert var["checkpoint_training_noise_pct"] in (20, 50, 80)
        assert var["evaluation_noise_pct"] == 0


def test_conesep_preflight_fails_on_empty_paths(tmp_path: Path):
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


def test_conesep_validate_source_pinned():
    conesep_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "ConeSep"
    blockers = validate_source(conesep_source)
    assert blockers == []


def test_conesep_native_signature():
    conesep_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "ConeSep"
    test_py = conesep_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category, txt_processors):"
            break
    else:
        pytest.fail("def test(...) signature not found in ConeSep test.py")
