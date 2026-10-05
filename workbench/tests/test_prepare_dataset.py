from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fashioniq_root(tmp_path: Path) -> Path:
    root = tmp_path / "FashionIQ"
    for category in ("dress", "shirt", "toptee"):
        for path in (
            root / "captions" / f"cap.{category}.val.json",
            root / "image_splits" / f"split.{category}.val.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]")
    (root / "images").mkdir()
    return root


def csmcir_source(tmp_path: Path) -> Path:
    source = tmp_path / "CSMCIR"
    for category in ("dress", "shirt", "toptee"):
        for path in (
            source / "qwen_captions" / f"{category}_cot_val.json",
            source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]")
    return source


def test_valid_layout_without_model_only_checks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)

    assert module.main(["--dataset-root", str(root)]) == 0
    assert "[OK] FashionIQ layout valid" in capsys.readouterr().out


def test_missing_fashioniq_validation_file_blocks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    (root / "captions" / "cap.dress.val.json").unlink()

    assert module.main(["--dataset-root", str(root)]) == 1
    assert "[BLOCKED] required path missing" in capsys.readouterr().out


def test_csmcir_dry_run_does_not_create_link(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "CSMCIR_ROOT", source)

    assert module.main(["--dataset-root", str(root), "--model", "csmcir", "--dry-run"]) == 0
    assert not (source / "fashionIQ_dataset").exists()
    assert "[OK] dry-run: would link" in capsys.readouterr().out


def test_csmcir_creates_safe_canonical_link(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "CSMCIR_ROOT", source)

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 0
    destination = source / "fashionIQ_dataset"
    assert destination.is_symlink()
    assert destination.resolve() == root.resolve()


def test_csmcir_refuses_conflicting_destination(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    (source / "fashionIQ_dataset").mkdir()
    monkeypatch.setattr(module, "CSMCIR_ROOT", source)

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 1
    assert "[BLOCKED] CSMCIR dataset destination already exists" in capsys.readouterr().out


def test_csmcir_refuses_different_dataset_link(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    other = fashioniq_root(tmp_path / "other")
    source = csmcir_source(tmp_path)
    (source / "fashionIQ_dataset").symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(module, "CSMCIR_ROOT", source)

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 1
    assert "[BLOCKED] CSMCIR dataset link resolves to" in capsys.readouterr().out
