"""Tests for PTHA+MTST source integrity, evaluator flags, and preflight blockers."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.ptha_mtst import preflight
MTST_DIR = REPO / "workbench" / "third_party" / "MTST_PTHA"
DATA_UTILS = MTST_DIR / "MTST_PTHA" / "src" / "data_utils.py"
MODELS_INIT = MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "__init__.py"
EVALUATOR = MTST_DIR / "MTST_PTHA" / "src" / "blip_validate.py"


def test_mtst_data_root_fallback() -> None:
    assert DATA_UTILS.is_file(), f"Missing {DATA_UTILS}"
    content = DATA_UTILS.read_text(encoding="utf-8")
    assert 'os.environ.get("MTST_DATA_ROOT", "path_to_data_dir")' in content, (
        "data_utils.py lacks MTST_DATA_ROOT fallback"
    )


def test_guarded_import_in_models_init() -> None:
    assert MODELS_INIT.is_file(), f"Missing {MODELS_INIT}"
    content = MODELS_INIT.read_text(encoding="utf-8")
    assert "Blip2Qformer4TGIRCESIMZYN = None" in content, (
        "models/__init__.py lacks guarded import for Blip2Qformer4TGIRCESIMZYN"
    )
    assert "@registry.register_model(\"blip2_tgir_ce_sim\")" in (
        MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "blip2_models" / "blip2_qformer_tgir_concatembedding_sim.py"
    ).read_text(encoding="utf-8")


def test_evaluator_flags() -> None:
    assert EVALUATOR.is_file(), f"Missing {EVALUATOR}"
    content = EVALUATOR.read_text(encoding="utf-8")
    for flag in ["--dataset", "--blip-model-name", "--backbone", "--model-path"]:
        assert f'"{flag}"' in content, f"Missing flag {flag} in {EVALUATOR}"


def test_preflight_blocks_on_missing_fashioniq_weights(tmp_path: Path) -> None:
    blockers = preflight(
        source_root=MTST_DIR,
        dataset_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    assert any("FashionIQ fine-tuned checkpoint missing" in b for b in blockers)
    assert any("tuned_mtst_cirr.pt" in b for b in blockers)
