"""Tests for DCNet replay preflight and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.dcnet import (
    MODEL,
    PROTOCOL,
    REQUIRED_RUN_FILES,
    preflight,
    validate_dataset,
    validate_run_directory,
    validate_source,
)


def test_dcnet_protocol_and_model_id():
    assert MODEL["model_id"] == "dcnet"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert "config.json" in REQUIRED_RUN_FILES
    assert "trained_model.pth" in REQUIRED_RUN_FILES


def test_dcnet_validate_source_pinned():
    dcnet_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "DCNet"
    blockers = validate_source(dcnet_source)
    assert blockers == []


def test_dcnet_run_dir_fails_on_missing_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    blockers = validate_run_directory(run_dir)
    assert any("config.json" in b for b in blockers)
    assert any("trained_model.pth" in b for b in blockers)


def test_dcnet_dataset_fails_on_missing_glove_and_resized_images(tmp_path: Path):
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    (ds_dir / "captions").mkdir()
    (ds_dir / "image_splits").mkdir()
    for cat in CATEGORIES:
        (ds_dir / "image_splits" / f"split.{cat}.val.json").write_text("[]")

    blockers = validate_dataset(ds_dir)
    assert any("resized_images" in b for b in blockers)
    assert any("glove" in b for b in blockers)
