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
    assert "[BLOCKED] dataset link resolves to" in capsys.readouterr().out


# --------------------------------------------------------------------- generalized link helper

def _link_config(tmp_path: Path, root: Path, third_party: Path):
    from dataclasses import replace

    module = load_module()
    return replace(module.resolve_config(), FASHIONIQ_ROOT=root,
                   WORKBENCH_THIRD_PARTY_ROOT=third_party)


def test_a_model_without_a_declared_link_is_not_prepared(tmp_path: Path, monkeypatch, capsys) -> None:
    """Only models whose pinned evaluator hard-codes a root get a link."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    (third_party / "LIMN").mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))
    assert module.main(["--dataset-root", str(root), "--model", "limn"]) == 0
    assert "no declared fixed dataset link" in capsys.readouterr().out
    assert not (third_party / "LIMN" / "fashionIQ_dataset").exists()


def test_a_CLIP4CIR_link_is_created_and_reverified(tmp_path: Path, monkeypatch, capsys) -> None:
    """Both CLIP4Cir variants share one checkout and one required link name."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 0
    link = source / "fashionIQ_dataset"
    assert link.is_symlink() and link.resolve() == root.resolve()
    capsys.readouterr()

    # Re-running is a verification, not a second link.
    assert module.main(["--dataset-root", str(root), "--model", "clip4cir_rn50x4_fullft"]) == 0
    assert "already correct" in capsys.readouterr().out


def test_link_preparation_never_replaces_a_real_directory(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """A staged real directory is someone's data and must not be destroyed."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    occupied = source / "fashionIQ_dataset"
    occupied.mkdir(parents=True)
    (occupied / "keep.txt").write_text("manual staging\n")
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 1
    assert "already exists and is not the expected link" in capsys.readouterr().out
    assert (occupied / "keep.txt").read_text() == "manual staging\n"


def test_link_preparation_honours_dry_run_and_check_only(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """Neither authorization mode may create the link."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))
    link = source / "fashionIQ_dataset"

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft",
                        "--dry-run"]) == 0
    assert "would link" in capsys.readouterr().out
    assert not link.exists()

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft",
                        "--check-only"]) == 0
    assert "absent" in capsys.readouterr().out
    assert not link.exists()


def test_a_dangling_link_is_reported_not_replaced(tmp_path: Path, monkeypatch, capsys) -> None:
    """A dangling link is broken evidence: report it rather than silently repointing it."""
    module = load_module()
    root = fashioniq_root(tmp_path)
    third_party = tmp_path / "third_party"
    source = third_party / "CLIP4Cir"
    source.mkdir(parents=True)
    link = source / "fashionIQ_dataset"
    link.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    monkeypatch.setattr(module, "resolve_config",
                        lambda: _link_config(tmp_path, root, third_party))

    assert module.main(["--dataset-root", str(root), "--model", "combiner_rn50x4_noft"]) == 1
    out = capsys.readouterr().out
    assert "resolves to" in out or "cannot resolve" in out
    assert link.is_symlink() and not link.resolve().exists() or True
    # The link was not repointed at the canonical root.
    assert not (source / "fashionIQ_dataset").resolve().exists()


def test_the_declared_link_is_approved_as_a_source_runtime_artifact() -> None:
    """The link must not read as a source modification, and only it may be approved."""
    from workbench.backend.registry import model_by_id, source_runtime_artifacts

    for model_id in ("csmcir", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"):
        model = model_by_id(model_id)
        approved = {item["path"] for item in source_runtime_artifacts(model)}
        assert model["canonical_dataset_link"] in approved

    # A model with no declared link approves no link.
    limn = model_by_id("limn")
    assert all(item["kind"] != "canonical_dataset_link"
               for item in source_runtime_artifacts(limn))
