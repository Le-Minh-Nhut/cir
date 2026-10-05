from __future__ import annotations

import importlib.util
from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig


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



def config(tmp_path: Path, root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(tmp_path, tmp_path / "data", root, "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")


def csmcir_source(tmp_path: Path) -> Path:
    source = tmp_path / "third_party" / "CSMCIR"
    source.mkdir(parents=True)
    return source


def test_valid_layout_without_model_only_checks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)

    assert module.main(["--dataset-root", str(root)]) == 0
    assert "[OK] FashionIQ base layout valid" in capsys.readouterr().out


def test_missing_fashioniq_validation_file_blocks(tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    (root / "captions" / "cap.dress.val.json").unlink()

    assert module.main(["--dataset-root", str(root)]) == 1
    assert "[BLOCKED] required path missing" in capsys.readouterr().out


def test_dataset_root_cli_overrides_config(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    configured_root = tmp_path / "configured-FashionIQ"
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, configured_root))

    assert module.main(["--dataset-root", str(root)]) == 0
    assert str(root) in capsys.readouterr().out


def test_csmcir_dry_run_does_not_create_link_without_auxiliary_assets(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir", "--dry-run"]) == 0
    assert not (source / "fashionIQ_dataset").exists()
    assert "[OK] dry-run: would link" in capsys.readouterr().out


def test_csmcir_creates_safe_canonical_link_without_auxiliary_assets(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    source = csmcir_source(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 0
    destination = source / "fashionIQ_dataset"
    assert destination.is_symlink()
    assert destination.resolve() == root.resolve()




def test_csmcir_refuses_different_dataset_link(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    root = fashioniq_root(tmp_path)
    other = fashioniq_root(tmp_path / "other")
    source = csmcir_source(tmp_path)
    (source / "fashionIQ_dataset").symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(module, "resolve_config", lambda: config(tmp_path, root))

    assert module.main(["--dataset-root", str(root), "--model", "csmcir"]) == 1
    assert "[BLOCKED] CSMCIR dataset link resolves to" in capsys.readouterr().out
