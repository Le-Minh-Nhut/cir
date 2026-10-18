"""Tests for CLIP4Cir / Combiner dimensions, flag alignment, and retrieval logic."""
from __future__ import annotations

from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
CLIP4CIR_DIR = REPO / "workbench" / "third_party" / "CLIP4Cir"
EVALUATOR = CLIP4CIR_DIR / "src" / "validate.py"
COMBINER = CLIP4CIR_DIR / "src" / "combiner.py"


def test_rn50x4_combiner_dimensions() -> None:
    assert EVALUATOR.is_file(), f"Missing {EVALUATOR}"
    content = EVALUATOR.read_text(encoding="utf-8")
    # For RN50x4, feature_dim is 640.
    # projection_dim is 640 * 4 = 2560.
    # hidden_dim is 640 * 8 = 5120.
    assert 'default=640 * 4' in content
    assert 'default=640 * 8' in content
    assert 'default="RN50x4"' in content


def test_evaluator_flag_conventions() -> None:
    content = EVALUATOR.read_text(encoding="utf-8")
    for flag in [
        "--dataset",
        "--combining-function",
        "--combiner-path",
        "--clip-model-path",
        "--projection-dim",
        "--hidden-dim",
        "--clip-model-name",
        "--target-ratio",
        "--transform",
    ]:
        assert f'"{flag}"' in content, f"Missing flag {flag}"


def test_combiner_module_structure() -> None:
    assert COMBINER.is_file(), f"Missing {COMBINER}"
    content = COMBINER.read_text(encoding="utf-8")
    assert "class Combiner(nn.Module):" in content
    assert "self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)" in content
    assert "self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)" in content
    assert "self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)" in content
    assert "self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)" in content
