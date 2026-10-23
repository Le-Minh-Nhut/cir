"""Tests for PTHA+MTST source integrity, evaluator flags, and preflight blockers."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.ptha_mtst import evaluator_path, preflight
MTST_DIR = REPO / "workbench" / "third_party" / "MTST_PTHA"
DATA_UTILS = MTST_DIR / "MTST_PTHA" / "src" / "data_utils.py"
MODELS_INIT = MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "__init__.py"
EVALUATOR = MTST_DIR / "MTST_PTHA" / "src" / "validate_blip.py"
DISPATCHER = MTST_DIR / "MTST_PTHA" / "src" / "blip_validate.py"


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
    assert DISPATCHER.is_file(), f"Missing {DISPATCHER}"
    content = DISPATCHER.read_text(encoding="utf-8")
    for flag in ["--dataset", "--blip-model-name", "--backbone", "--model-path"]:
        assert f'"{flag}"' in content, f"Missing flag {flag} in {DISPATCHER}"


def test_validate_blip_cli_only_dispatches_cirr() -> None:
    content = EVALUATOR.read_text(encoding="utf-8")
    main = content[content.index("def main():"):]
    assert "if args.dataset == 'CIRR':" in main
    assert "args.dataset == 'fashionIQ'" not in main


def _dataset_root(root: Path) -> Path:
    for category in ("dress", "shirt", "toptee"):
        for relative in (
            f"captions/cap.{category}.val.json",
            f"image_splits/split.{category}.val.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


def test_preflight_uses_pinned_validate_blip_module() -> None:
    assert evaluator_path(MTST_DIR) == EVALUATOR
    assert EVALUATOR.is_file()
    assert DISPATCHER.is_file()
    blockers = preflight(MTST_DIR, REPO / "data" / "fashionIQ_dataset", REPO / "workbench" / "artifacts" / "checkpoints")
    assert not any("pinned evaluator missing" in blocker for blocker in blockers)


def test_cirr_checkpoint_requires_explicit_transfer_opt_in(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    model_root = checkpoint_root / "ptha_mtst"
    model_root.mkdir(parents=True)
    (model_root / "tuned_mtst_cirr.pt").write_bytes(b"placeholder")
    dataset_root = _dataset_root(tmp_path / "fashioniq")

    normal = preflight(MTST_DIR, dataset_root, checkpoint_root)
    assert any("CIRR weights require explicit transfer opt-in" in blocker for blocker in normal)

    transfer = preflight(
        MTST_DIR,
        dataset_root,
        checkpoint_root,
        transfer_cirr_to_fashioniq=True,
    )
    assert not any("CIRR weights require explicit transfer opt-in" in blocker for blocker in transfer)
    assert not any("CIRR checkpoint missing" in blocker for blocker in transfer)


def test_preflight_blocks_on_missing_fashioniq_weights(tmp_path: Path) -> None:
    blockers = preflight(
        source_root=MTST_DIR,
        dataset_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    assert any("FashionIQ fine-tuned checkpoint missing" in blocker for blocker in blockers)
    assert any("tuned_mtst_cirr.pt" in blocker for blocker in blockers)
