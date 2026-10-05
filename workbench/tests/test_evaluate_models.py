from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from workbench.backend.registry import load_registry


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_models", Path("workbench/scripts/evaluate_models.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def records():
    return {model["model_id"]: model for model in load_registry()["models"]}


def prepare_source(monkeypatch, module, tmp_path: Path, model: dict) -> tuple[Path, Path, Path]:
    source = tmp_path / "third_party" / model["source_dir"]
    source.mkdir(parents=True)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = model["checkpoint_variants"][0]
    checkpoint_file = checkpoint_root / model["model_id"] / checkpoint["filename"]
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"checkpoint")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CHECKPOINT_ROOT", checkpoint_root)
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])
    return source, checkpoint_file, checkpoint_root


def test_unaudited_adapter_is_skipped() -> None:
    module = load_module()
    model = records()["hint"]
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)
    assert plan is None
    assert reasons == ["SKIPPED: adapter command not audited"]


def test_missing_checkpoint_is_blocked(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["csmcir"]
    source = tmp_path / "third_party" / model["source_dir"]
    (source / "fashionIQ_dataset").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], source / "fashionIQ_dataset", 200)

    assert plan is None
    assert any(reason.startswith("checkpoint missing:") for reason in reasons)


def test_csmcir_dry_run_constructs_official_command(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    model = records()["csmcir"]
    source, checkpoint_file, _ = prepare_source(monkeypatch, module, tmp_path, model)
    dataset_root = source / "fashionIQ_dataset"
    for path in (
        dataset_root / "captions", dataset_root / "image_splits", dataset_root / "images",
        source / "COT_ours2" / "bert_captions" / "fashioniq", source / "COT_ours2" / "fashioniq",
    ):
        path.mkdir(parents=True)
    script = source / "src" / "validate_blip_csmcir.py"
    script.parent.mkdir()
    script.write_text("")
    import workbench.backend.adapters.base as base
    import workbench.backend.adapters.models as adapters

    monkeypatch.setattr(base, "checkpoint_path", lambda *_: checkpoint_file)
    monkeypatch.setattr(adapters, "ROOT", tmp_path)
    monkeypatch.setattr(adapters, "checkpoint_path", lambda *_: checkpoint_file)
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)
    result = module.main([
        "--model", "csmcir", "--checkpoint", "fashioniq", "--protocol", model["native_protocol"],
        "--dataset-root", str(dataset_root), "--dry-run",
    ])

    assert result == 0
    assert reasons == []
    assert plan is not None
    assert plan.cwd == source / "src"
    assert plan.command == ["python", str(script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint_file)]
    assert "[RUN] command:" in capsys.readouterr().out


def test_encoder_requires_external_openclip_asset(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    (source / "evaluate_model.py").write_text("")
    dataset_root = tmp_path / "FashionIQ"
    dataset_root.mkdir()

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    assert reasons == [f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}"]
